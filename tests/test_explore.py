import beartype
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped

import sax.explore


def test_smoke():
    pass


def test_push_empty():
    n_neurons = 8
    top_k = 4
    scores = jnp.full((n_neurons, top_k), jnp.inf)
    indices = jnp.full((n_neurons, top_k), -1)

    incoming_scores = jnp.array([-1.0, -2, -3, -4, -5, -6, -7, -8])
    incoming_index = jnp.array(3)

    actual_scores, actual_indices = sax.explore.push(
        scores, indices, incoming_scores, incoming_index
    )

    expected_scores = scores.at[:, 0].set(incoming_scores)
    expected_indices = indices.at[:, 0].set(incoming_index)

    np.testing.assert_allclose(actual_scores, expected_scores)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_push_actual_values():
    scores = jnp.array([
        [-4.0, -2],
        [-3, -1],
        [-5, -4],
        [-3, -1],
    ])
    indices = jnp.array([
        [12, 17],
        [17, 14],
        [2, 3],
        [4, 5],
    ])

    incoming_scores = jnp.array([-1.0, -2, -3, -4])
    incoming_index = jnp.array(42)

    actual_scores, actual_indices = sax.explore.push(
        scores, indices, incoming_scores, incoming_index
    )

    expected_scores = jnp.array([
        [-4.0, -2],
        [-3, -2],
        [-5, -4],
        [-4, -3],
    ])

    expected_indices = jnp.array([
        [12, 17],
        [17, 42],
        [2, 3],
        [42, 4],
    ])

    np.testing.assert_allclose(actual_scores, expected_scores)
    np.testing.assert_allclose(actual_indices, expected_indices)


@jaxtyped(typechecker=beartype.beartype)
class DummySae(eqx.Module):
    features: Float[Array, " n_neurons"]

    def __init__(self, features: Float[Array, " n_neurons"]):
        self.features = features

    def __call__(
        self, x: Float[Array, " n_features"]
    ) -> tuple[None, Float[Array, " n_neurons"]]:
        return None, self.features


def test_inference_empty():
    batch_size = 2
    n_features = 4
    n_neurons = 5
    top_k = 3
    sae = DummySae(jnp.zeros(n_neurons).at[0].set(1))
    batch = jnp.zeros((batch_size, n_features))

    scores = jnp.full((n_neurons, top_k), jnp.inf)
    indices = jnp.full((n_neurons, top_k), -1)

    start = jnp.array(0)

    actual_scores, actual_indices = sax.explore.inference(
        sae, batch, scores, indices, start
    )

    expected_scores = jnp.array([
        [-1, -1, jnp.inf],
        [-0, -0, jnp.inf],
        [-0, -0, jnp.inf],
        [-0, -0, jnp.inf],
        [-0, -0, jnp.inf],
    ])
    expected_indices = jnp.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    np.testing.assert_allclose(actual_scores, expected_scores)
    np.testing.assert_allclose(actual_indices, expected_indices)


def test_inference_progress():
    batch_size = 2
    n_features = 4
    # n_neurons = 5
    # top_k = 3
    sae = DummySae(jnp.array([0.0, 1, -1, 2, 3]))
    batch = jnp.zeros((batch_size, n_features))

    scores = jnp.array([
        [-1.0, 1, 4],
        [-3, -2, 0],
        [-17, 0, 18],
        [0, 4, 7],
        [-1, 5, 6],
    ])
    indices = jnp.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
    ])

    start = jnp.array(20)

    actual_scores, actual_indices = sax.explore.inference(
        sae, batch, scores, indices, start
    )

    expected_scores = jnp.array([
        [-1.0, 0, 0],
        [-3, -2, -1],
        [-17, 0, 1],
        [-2, -2, 0],
        [-3, -3, -1],
    ])
    expected_indices = jnp.array([
        [1, 21, 20],
        [4, 5, 21],
        [7, 8, 21],
        [21, 20, 10],
        [21, 20, 13],
    ])

    np.testing.assert_allclose(actual_scores, expected_scores)
    np.testing.assert_allclose(actual_indices, expected_indices)
