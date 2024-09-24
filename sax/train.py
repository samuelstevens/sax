import collections.abc
import dataclasses
import logging
import os.path
import time

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, jaxtyped

import wandb

from . import helpers, nn, tracking


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    train_activations: str = "activations-train.bin"
    """training activations."""
    val_activations: str = "activations-val.bin"
    """validation activations."""
    d_model: int = 768
    """model dimension."""
    n_layers: int = 12
    """number of model layers."""
    n_train: int = 1281167
    """number of train activations"""
    n_val: int = 50_000
    """number of val activations"""

    expansion_factor: int = 64
    """how many times larger the SAE's hidden layer should be."""
    sparsity_coeff: float = 1e-3
    """how much to upweight loss_sparsity."""
    batch_size: int = 4096
    """batch size."""
    learning_rate: float = 7e-5
    """peak learning rate."""
    beta1: float = 0.9
    beta2: float = 0.999
    n_lr_warmup: int = 1_000
    """number of learning rate warmup steps."""
    n_sparsity_warmup: int = 10_000
    # """number of sparsity coefficient warmup steps."""
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
    tags: list[str] = dataclasses.field(default_factory=list)
    """any tags for this specific run."""


@jaxtyped(typechecker=beartype.beartype)
def make_dataloader(
    args: Args, key: chex.PRNGKey, *, is_train: bool
) -> collections.abc.Iterable[Float[Array, "batch d_model"]]:
    n_activations = args.n_train if is_train else args.n_val
    activations_path = args.train_activations if is_train else args.val_activations
    shape = (n_activations, args.n_layers, args.d_model)
    data = np.memmap(activations_path, dtype=np.float32, mode="r", shape=shape)
    data = data.reshape(n_activations * args.n_layers, args.d_model)

    # Double batch size for validation (no gradients)
    batch_size = args.batch_size + is_train * args.batch_size

    i = jax.random.permutation(key, n_activations)
    start = 0
    while True:
        end = min(start + batch_size, n_activations)
        yield jnp.array(data[i[start:end]])
        start = end
        if start >= n_activations:
            break


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    batch: Float[Array, "batch d_model"],
    sparsity_coeff: float,
) -> tuple[
    eqx.Module,
    optax.OptState | optax.MultiStepsState,
    Float[Array, ""],
    Float[Array, ""],
]:
    (loss, l0), grads = eqx.filter_value_and_grad(model.loss, has_aux=True)(
        model, batch, sparsity_coeff
    )
    updates, new_state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss, l0


@jaxtyped(typechecker=beartype.beartype)
def evaluate(args: Args, sae: eqx.Module, key: chex.PRNGKey) -> dict[str, float]:
    @jaxtyped(typechecker=beartype.beartype)
    @eqx.filter_jit(donate="all-except-first")
    def _compute_loss(
        model: eqx.Module, batch: Float[Array, "b d"]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        loss, l0 = model.loss(model, batch, args.sparsity_coeff)
        return loss, l0

    dataloader = make_dataloader(args, key, is_train=False)
    losses, l0s = [], []
    for batch in dataloader:
        loss, l0 = _compute_loss(sae, batch)
        losses.append(loss)
        l0s.append(l0)
    return {
        "val_loss": jnp.mean(jnp.array(losses)).item(),
        "val_l0": jnp.mean(jnp.array(l0s)).item(),
    }


@beartype.beartype
def train(args: Args) -> str:
    key = jax.random.key(seed=args.seed)
    logger = logging.getLogger("sax.train")

    # 1. Define the model and optimizers.
    key, subkey = jax.random.split(key)
    model_kwargs = {
        "d_in": args.d_model,
        "d_hidden": args.expansion_factor * args.d_model,
    }
    sae = nn.ReparamInvariantReluSAE(**model_kwargs, key=subkey)
    n_steps = jnp.ceil(args.n_train * args.n_layers * args.n_epochs / args.batch_size)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, args.n_lr_warmup, n_steps
    )
    optim = optax.adamw(
        learning_rate=lr_schedule, b1=args.beta1, b2=args.beta2, weight_decay=0.0
    )
    if args.grad_clip > 0:
        optim = optax.chain(optim, optax.clip_by_global_norm(args.grad_clip))

    state = optim.init(eqx.filter(sae, eqx.is_inexact_array))

    mode = "online" if args.track else "disabled"
    hparams = {k: helpers.to_aim_value(v) for k, v in vars(args).items()}
    run = wandb.init(
        project="sax",
        entity="samuelstevens",
        config=hparams,
        tags=args.tags,
        mode=mode,
        reinit=True,
    )

    # Train the model.
    global_step = 0
    start_time = time.time()

    for epoch in range(args.n_epochs):
        # Make dataloader
        key, subkey = jax.random.split(key)
        train_dataloader = make_dataloader(args, subkey, is_train=True)

        # Iterate through all examples in random order.
        for batch in train_dataloader:
            sae, state, loss, l0 = step(sae, optim, state, batch, args.sparsity_coeff)
            global_step += 1

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                metrics = {
                    "train_loss": loss.item(),
                    "step_per_sec": step_per_sec,
                    "learning_rate": lr_schedule(global_step).item(),
                    "train_l0": l0.item(),
                    "epoch": epoch,
                }
                run.log(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

        # Evaluate
        key, subkey = jax.random.split(key)
        metrics = evaluate(args, sae, subkey)
        metrics["epoch"] = epoch
        run.log(metrics, step=global_step)
        logger.info(
            ", ".join(f"{key}: {value}" for key, value in sorted(metrics.items()))
        )

    # 4. Save model.
    helpers.save(os.path.join(args.ckpt_dir, f"{run.id}.eqx"), model_kwargs, sae)
    tracking.save(run.id, hparams, dict(run.summary))
    run.finish()
    return run.id
