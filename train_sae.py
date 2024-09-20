"""
Trains a sparse autoencoder (SAE) on activations.

"""

import time
import os.path
import collections.abc
import dataclasses
import logging
import sys

import aim
import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, jaxtyped

import sax.nn
import sax.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations")


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
    sparsity_coeff: float = 1.0
    """how much to upweight loss_sparsity."""

    batch_size: int = 4096
    """batch size."""
    learning_rate: float = 7e-5
    """peak learning rate."""
    beta1: float = 0.9
    beta2: float = 0.999
    n_warmup_steps: int = 1_000
    """number of warmup steps."""
    n_grad_accum_steps: int = 1
    """number of steps to accumulate gradients for. `1` implies no accumulation."""
    grad_clip: float = 0.0
    """maximum gradient norm. 0.0 indicates no maximum."""
    n_epochs: int = 1
    """number of epochs."""
    pre_enc_bias: bool = False
    """whether to subtract b_dec before sae."""

    seed: int = 42
    """random seed."""
    log_every: int = 100
    """how often to log metrics."""
    track: bool = True
    """whether to track metrics using Aim."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store model checkpoints."""


@jaxtyped(typechecker=beartype.beartype)
def make_dataloader(
    args: Args, key: chex.PRNGKey
) -> collections.abc.Iterable[Float[Array, "batch d_model"]]:
    shape = (args.n_activations, args.n_layers, args.d_model)
    data = np.memmap(args.activations, dtype=np.float32, mode="r", shape=shape)
    data = data.reshape(args.n_activations * args.n_layers, args.d_model)

    i = jax.random.permutation(key, args.n_activations)
    start = 0
    while True:
        end = min(start + args.batch_size, args.n_activations)
        yield jnp.array(data[i[start:end]])
        start = end
        if start >= args.n_activations:
            break


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    batch: Float[Array, "batch d_model"],
    sparsity_coeff: float,
) -> tuple[eqx.Module, optax.OptState | optax.MultiStepsState, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(model.loss)(model, batch, sparsity_coeff)
    updates, new_state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def main(args: Args):
    key = jax.random.key(seed=args.seed)

    # 1. Define the model and optimizers.
    key, subkey = jax.random.split(key)
    model_kwargs = {
        "d_in": args.d_model,
        "d_hidden": args.expansion_factor * args.d_model,
    }
    sae = sax.nn.ReparamInvariantReluSAE(**model_kwargs, key=subkey)
    n_steps = jnp.ceil(
        args.n_activations * args.n_layers * args.n_epochs / args.batch_size
    )
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, args.n_warmup_steps, n_steps // args.n_grad_accum_steps
    )
    optim = optax.adamw(
        learning_rate=lr_schedule, b1=args.beta1, b2=args.beta2, weight_decay=0.0
    )
    if args.grad_clip > 0:
        optim = optax.chain(optim, optax.clip_by_global_norm(args.grad_clip))
    if args.n_grad_accum_steps > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=args.n_grad_accum_steps)

    state = optim.init(eqx.filter(sae, eqx.is_inexact_array))

    if args.track:
        run = aim.Run(experiment="train")
        run["hparams"] = {k: sax.helpers.to_aim_value(v) for k, v in vars(args).items()}
        run["hparams"]["cmd"] = " ".join([sys.executable] + sys.argv)
    else:
        run = sax.helpers.DummyAimRun()

    # 3. Train the model.
    global_step = 0
    start_time = time.time()

    for e in range(args.n_epochs):
        # 2. Make dataloader
        key, subkey = jax.random.split(key)
        dataloader = make_dataloader(args, subkey)

        for batch in dataloader:
            sae, state, loss = step(sae, optim, state, batch, args.sparsity_coeff)
            global_step += 1

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                metrics = {
                    "train_loss": loss.item(),
                    "step_per_sec": step_per_sec,
                    "learning_rate": lr_schedule(
                        global_step // args.n_grad_accum_steps
                    ).item(),
                }
                run.track(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

    # 4. Save model.
    sax.helpers.save(os.path.join(args.ckpt_dir, f"{run.hash}.eqx"), model_kwargs, sae)


if __name__ == "__main__":
    main(tyro.cli(Args))
