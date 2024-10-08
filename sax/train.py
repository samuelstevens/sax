import collections.abc
import dataclasses
import logging
import os.path
import time
import typing

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, jaxtyped

import wandb

from . import helpers, nn, tracking

DataNorm = typing.Literal[None, "example", "batch"]


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
    data_norm: DataNorm = None
    """how to normalize activations before auto-encoding."""

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
    n_lr_warmup: int = 300
    """number of learning rate warmup steps."""
    n_sparsity_warmup: int = 1_000
    """number of sparsity coefficient warmup steps."""
    grad_clip: float = 0.0
    """maximum gradient norm. 0.0 indicates no maximum."""
    n_epochs: int = 4
    """number of epochs."""
    pre_enc_bias: bool = True
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
    n_examples = n_activations * args.n_layers
    data = data.reshape(n_examples, args.d_model)

    # Double batch size for validation (no gradients)
    batch_size = args.batch_size + is_train * args.batch_size

    i = jax.random.permutation(key, n_examples)
    start = 0
    while True:
        end = min(start + batch_size, n_examples)
        yield jnp.array(data[i[start:end]])
        start = end
        if start >= n_examples:
            break


@jaxtyped(typechecker=beartype.beartype)
def normalize(
    kind: DataNorm, batch: Float[Array, "batch d_model"]
) -> Float[Array, "batch d_model"]:
    if kind == "batch":
        breakpoint()
        batch_norm = None
        batch = batch / batch_norm
    if kind == "example":
        example_norms = jnp.linalg.norm(batch, axis=1, keepdims=True)
        return batch / example_norms
    if kind is None:
        return batch
    typing.assert_never(kind)


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    batch: Float[Array, "batch d_model"],
    sparsity_coeff: Float[Array, ""],
    data_norm: DataNorm,
) -> tuple[eqx.Module, optax.OptState | optax.MultiStepsState, nn.Loss]:
    batch = normalize(data_norm, batch)

    @jaxtyped(typechecker=beartype.beartype)
    def _compute_loss(model: eqx.Module) -> tuple[Float[Array, ""], nn.Loss]:
        """Compute loss. Return the complete loss (loss.loss) as the first term for optimization and the pytree `loss` for metrics."""
        loss = model.loss(model, batch, sparsity_coeff)
        return loss.loss, loss

    (_, loss), grads = eqx.filter_value_and_grad(_compute_loss, has_aux=True)(model)
    updates, new_state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def evaluate(args: Args, sae: eqx.Module, key: chex.PRNGKey) -> dict[str, float]:
    """
    Evaluates a sparse autoencoder on the validation datset, computing mean L2 loss, mean L0 sparsity, and the trivial validation loss using the mean batch activation.

    Args:
        args: run configuration.
        sae: the sparse autoencoder under evaluation.

    Returns:
        A dictionary of metrics.
    """

    @jaxtyped(typechecker=beartype.beartype)
    @eqx.filter_jit(donate="all-except-first")
    def _compute_loss(model: eqx.Module, batch: Float[Array, "b d"]) -> nn.Loss:
        batch = normalize(args.data_norm, batch)

        loss = model.loss(model, batch, args.sparsity_coeff)

        return loss

    # We need to shuffle the validation data because we use the batch mean to calculate the trivial loss, so we don't want the examples to be correlated.
    dataloader = make_dataloader(args, key, is_train=False)
    losses = []
    for batch in dataloader:
        loss = _compute_loss(sae, batch)
        losses.append(loss)

    mean = jax.tree.map(lambda *x: jnp.array(x).mean(), *losses)

    return {f"val/{key}": value for key, value in mean.to_dict().items()}


@beartype.beartype
def train(args: Args) -> str:
    key = jax.random.key(seed=args.seed)
    logger = logging.getLogger(__file__)

    # 1. Define the model and optimizers.
    key, subkey = jax.random.split(key)
    model_kwargs = {
        "d_in": args.d_model,
        "d_hidden": args.expansion_factor * args.d_model,
        "pre_enc_bias": args.pre_enc_bias,
    }
    sae = nn.ReparamInvariantReluSAE(**model_kwargs, key=subkey)
    sparsity_schedule = optax.schedules.warmup_constant_schedule(
        0.0, args.sparsity_coeff, args.n_sparsity_warmup
    )
    n_steps = jnp.ceil(
        args.n_train * args.n_layers * args.n_epochs / args.batch_size
    ).astype(int)
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
            sae, state, loss = step(
                sae, optim, state, batch, sparsity_schedule(global_step), args.data_norm
            )
            global_step += 1

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                metrics = {
                    **loss.to_dict(),
                    "step_per_sec": step_per_sec,
                    "learning_rate": lr_schedule(global_step).item(),
                    "sparsity_coeff": sparsity_schedule(global_step).item(),
                    "epoch": epoch,
                }
                run.log(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.loss.item(),
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


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    train(tyro.cli(Args))
