import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
def push(
    scores: Float[Array, "n_features top_k"],
    indices: Int[Array, "n_features top_k"],
    incoming_scores: Float[Array, " n_features"],
    incoming_index: Int[Array, ""],
) -> tuple[Float[Array, "n_features top_k"], Int[Array, "n_features top_k"]]:
    at = jax.vmap(jnp.searchsorted)(scores, incoming_scores)
    scores = jax.vmap(jnp.insert)(scores, at, incoming_scores)[:, :-1]
    indices = jax.vmap(jnp.insert, in_axes=(0, 0, None))(indices, at, incoming_index)[
        :, :-1
    ]
    return scores, indices


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit
def inference(
    sae: eqx.Module,
    batch: Float[Array, "batch_size n_features"],
    scores: Float[Array, "n_neurons top_k"],
    indices: Int[Array, "n_neurons top_k"],
    start: Int[Array, ""],
) -> tuple[Float[Array, "n_neurons top_k"], Int[Array, "n_neurons top_k"]]:
    _, f_x = jax.vmap(sae)(batch)

    example_norms = jnp.linalg.norm(batch, axis=1, keepdims=True)
    batch = batch / example_norms

    batch_size, _ = batch.shape

    @jaxtyped(typechecker=beartype.beartype)
    def body(
        carry: tuple[Float[Array, "n_neurons top_k"], Int[Array, "n_neurons top_k"]],
        f_x_i: tuple[Float[Array, " n_neurons"], Int[Array, ""]],
    ) -> tuple[
        tuple[Float[Array, "n_neurons top_k"], Int[Array, "n_neurons top_k"]], None
    ]:
        f_x, i = f_x_i
        scores, indices = carry
        scores, indices = push(scores, indices, -f_x, start + i)
        return (scores, indices), None

    (scores, indices), _ = jax.lax.scan(
        body, (scores, indices), xs=(f_x, jnp.arange(batch_size))
    )
    return scores, indices
