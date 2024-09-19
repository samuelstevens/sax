"""
Trains a sparse autoencoder (SAE) on activations.

"""

import dataclasses
import logging

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import tyro
from jaxtyping import Array, Float, jaxtyped

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations")


bandwidth = 0.001


@jaxtyped(typechecker=beartype.beartype)
def rectangle(x: Float[Array, " d"]) -> Float[Array, " d"]:
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
class SparseAutoencoder(eqx.Module):
    enc: eqx.nn.Linear
    log_threshold: Float[Array, " d_hidden"]
    dec: eqx.nn.Linear

    def __init__(self, d_model: int, d_hidden: int, key: chex.PRNGKey):
        key1, key2 = jax.random.split(key)
        self.enc = eqx.nn.Linear(d_model, d_hidden, key=key1)
        self.log_threshold = jnp.log(jnp.full((d_hidden,), 0.001, dtype=jnp.float32))
        self.dec = eqx.nn.Linear(d_hidden, d_model, key=key2)

    def __call__(self, x: Float[Array, " d_model"]) -> Float[Array, " d_model"]:
        # if use_pre_enc_bias:
        #     x = x - self.dec.bias

        x = jax.nn.relu(self.enc(x))
        threshold = jnp.exp(self.log_threshold)
        f_x = jumprelu(x, threshold)
        x_hat = self.dec(f_x)
        return x_hat, f_x

    def loss(
        self, x: Float[Array, " d_model"], sparsity_coeff: float
    ) -> Float[Array, ""]:
        x_hat, f_x = self(x)

        reconstruct_err = x - x_hat
        reconstruct_loss = jnp.sum(reconstruct_err**2, axis=-1)

        threshold = jnp.exp(self.log_threshold)
        l0 = jnp.sum(heaviside(f_x, threshold), axis=-1)
        sparsity_loss = sparsity_coeff * l0

        return reconstruct_loss + sparsity_loss


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Args:
    activations: str = "/fs/scratch/PAS2136/samuelstevens/datasets/sae/hf-hub_imageomics_BioCLIP/activations-local.bin"
    """activations file."""
    d_model: int = 768
    """model dimension."""
    n_layers: int = 12
    """number of model layers."""
    n_activations: int = 491016
    """number of activations"""
    expansion_factor: int = 64
    """how many times larger the SAE's hidden layer should be."""
    seed: int = 42
    """random seed."""


def make_dataloader(args: Args):
    pass


def step(model: eqx.Module, optim, state, batch: Float[Array, "batch d_model"]):
    pass


def main(args: Args):
    key = jax.random.key(seed=args.seed)

    # 1. Define the model and optimizers.
    key, subkey = jax.random.split(key)
    sae = SparseAutoencoder(args.d_model, args.expansion_factor * args.d_model, subkey)

    # 2. Load the data.
    dataloader = make_dataloader(args)

    # 3. Train the model.
    for b, batch in enumerate(dataloader):
        sae, state = step(sae, optim, state, batch)
        if (b + 1) % args.log_every == 0:
            logger.info("Finished a batch!")

    # 4.
    breakpoint()


if __name__ == "__main__":
    main(tyro.cli(Args))
