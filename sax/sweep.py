import collections.abc
import typing
import jax
import jax.numpy as jnp

import beartype

Primitive = float | int | bool | str

Distribution = typing.TypedDict(
    "Distribution", {"min": float, "max": float, "dist": typing.Literal["loguniform"]}
)


@beartype.beartype
def _expand_discrete(
    config: dict[str, Primitive | list[Primitive] | Distribution],
) -> collections.abc.Iterator[dict[str, Primitive]]:
    """
    Expands any list values in `config`
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}


@beartype.beartype
def expand(
    config: dict[str, Primitive | list[Primitive] | Distribution], *, n: int
) -> collections.abc.Iterator[dict[str, Primitive]]:
    discrete_configs = list(_expand_discrete(config))
    n_discrete_configs = len(discrete_configs)

    if n < n_discrete_configs:
        msg = f"You requested {n} total configs, but there are {n_discrete_configs} discrete configs."
        raise RuntimeError(msg)

    if n % n_discrete_configs != 0:
        msg = f"You requested {n} total configs, but there are {n_discrete_configs} discrete configs, which would lead to uneven sampling from the discrete configs."
        raise RuntimeError(msg)

    n_per_discrete = n // n_discrete_configs
    for config in discrete_configs:
        yield from _sample_from(config, n=n_per_discrete)


@beartype.beartype
def _sample_from(
    config: dict[str, Primitive | Distribution], *, n: int
) -> collections.abc.Iterator[dict[str, Primitive]]:
    # 1. Count the number of distributions
    # 2. Sample for each distribution
    # 3. Scale each sample based on the min/max/dist.
    # 4. Return the sampled config.
    pass


def _newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = jax.lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(
    num_points: int,
    dim: int,
    root_iters: int = 10_000,
    complement_basis: bool = True,
    shuffle: bool = False,
    key: jax.typing.ArrayLike | None = None,
    dtype=float,
):
    """
    Returns the Roberts sequence, a low-discrepancy quasi-random sequence:
    Low-discrepancy sequences are useful for quasi-Monte Carlo methods.
    Reference:
    Martin Roberts. The Unreasonable Effectiveness of Quasirandom Sequences.
    extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
    Args:
      num_points: Number of points to return.
      dim: The dimensionality of each point in the sequence.
      root_iters: Number of iterations to use to find the root.
      complement_basis: Complement the basis to improve precision, as described
        in https://www.martysmods.com/a-better-r2-sequence.
      key: a PRNG key.
      shuffle: Shuffle the elements of the sequence before returning them.
        Warning: This degrades the low-discrepancy property for prefixes of
        the output sequence.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
    Returns:
      An array of shape (num_points, dim) containing the sequence.

    From https://github.com/jax-ml/jax/pull/23808
    """

    def f(x):
        return x ** (dim + 1) - x - 1

    root = _newton_raphson(f, jnp.astype(1.0, dtype), root_iters)

    basis = 1 / root ** (1 + jnp.arange(dim, dtype=dtype))

    if complement_basis:
        basis = 1 - basis

    n = jnp.arange(num_points, dtype=dtype)
    x = n[:, None] * basis[None, :]

    x, _ = jnp.modf(x)

    if shuffle:
        if key is None:
            raise ValueError("key cannot be None when shuffle=True")
        x = jax.random.permutation(key, x)

    return x
