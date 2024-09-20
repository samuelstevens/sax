import typing

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

bandwidth = 0.001


@jaxtyped(typechecker=beartype.beartype)
def rectangle(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return ((x > -0.5) & (x < 0.5)).astype(x.dtype)


@jax.custom_vjp
def heaviside(x, threshold):
    return (x > threshold).astype(x.dtype)


def heaviside_fwd(x, threshold):
    out = heaviside(x, threshold)
    cache = x, threshold  # Saved for use in the backward pass
    return out, cache


def heaviside_bwd(cache, output_grad):
    x, threshold = cache
    x_grad = 0.0 * output_grad  # We don’t apply STE to x input
    threshold_grad = (
        -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * output_grad
    )
    return x_grad, threshold_grad


heaviside.defvjp(heaviside_fwd, heaviside_bwd)


@jax.custom_vjp
def jumprelu(x: Float[Array, " d"], threshold: Float[Array, " d"]):
    return x * (x > threshold)


@jaxtyped(typechecker=beartype.beartype)
def jumprelu_fwd(x: Float[Array, " d"], threshold: Float[Array, " d"]):
    out = jumprelu(x, threshold)
    cache = x, threshold  # Saved for use in the backward pass
    return out, cache


@jaxtyped(typechecker=beartype.beartype)
def jumprelu_bwd(cache, output_grad):
    x, threshold = cache
    x_grad = (x > threshold) * output_grad
    threshold_grad = (
        -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * output_grad
    )
    return x_grad, threshold_grad


jumprelu.defvjp(jumprelu_fwd, jumprelu_bwd)


@jaxtyped(typechecker=beartype.beartype)
class ReluSAE(eqx.Module):
    w_enc: Float[Array, "d_hidden d_in"]
    b_enc: Float[Array, " d_hidden"]
    w_dec: Float[Array, "d_in d_hidden"]
    b_dec: Float[Array, " d_in"]

    def __init__(self, d_in: int, d_hidden: int, *, key: chex.PRNGKey):
        init_fn = jax.nn.initializers.he_uniform()
        w_dec = init_fn(key, (d_in, d_hidden), jnp.float32)
        # Re-scale each latent vector to have unit norm. w_dec is [n_features, d_model], so we want to take the norm along axis=1 (0-indexed).
        self.w_dec = w_dec / jnp.linalg.norm(w_dec, axis=1, keepdims=True)

        # Initialize w_enc to transpose of w_dec, but do not tie.
        self.w_enc = self.w_dec.T

        # Initialize biases to 0.
        self.b_enc = jnp.zeros((d_hidden,))
        self.b_dec = jnp.zeros((d_in,))

    def __call__(
        self, x: Float[Array, " d_in"]
    ) -> tuple[Float[Array, " d_in"], Float[Array, " d_hidden"]]:
        x = self.w_enc @ x + self.b_enc
        f_x = jax.nn.relu(x)
        x_hat = self.w_dec @ f_x + self.b_dec
        return x_hat, f_x

    @staticmethod
    def loss(
        model: typing.Self, x: Float[Array, "batch d_in"], sparsity_coeff: float
    ) -> Float[Array, ""]:
        x_hat, f_x = jax.vmap(model)(x)

        reconstruct_err = x - x_hat
        reconstruct_loss = jnp.sum(reconstruct_err**2, axis=-1)

        l1 = jnp.linalg.norm(f_x, ord=1, axis=-1)
        sparsity_loss = sparsity_coeff * l1

        return jnp.mean(reconstruct_loss + sparsity_loss)


class ReparamInvariantReluSAE(ReluSAE):
    @staticmethod
    def loss(
        model: typing.Self, x: Float[Array, "batch d_in"], sparsity_coeff: float
    ) -> Float[Array, ""]:
        x_hat, f_x = jax.vmap(model)(x)

        reconstruct_err = x - x_hat
        reconstruct_loss = jnp.sum(reconstruct_err**2, axis=-1)

        # Reparameterization-invariant L1
        # sum_^M f_x_i * ||d_i||2
        ri_l1 = f_x @ jnp.linalg.norm(model.w_dec, axis=0)
        sparsity_loss = sparsity_coeff * ri_l1

        return jnp.mean(reconstruct_loss + sparsity_loss)
