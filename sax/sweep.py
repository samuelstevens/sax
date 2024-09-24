import collections.abc
import typing

import beartype
import jax
import jax.numpy as jnp

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
    config: dict[str, Primitive | list[Primitive] | Distribution],
    *,
    n_per_discrete: int,
) -> collections.abc.Iterator[dict[str, Primitive]]:
    discrete_configs = list(_expand_discrete(config))
    for config in discrete_configs:
        yield from _sample_from(config, n=n_per_discrete)


@beartype.beartype
def _sample_from(
    config: dict[str, Primitive | Distribution], *, n: int
) -> collections.abc.Iterator[dict[str, Primitive]]:
    # 1. Count the number of distributions and collect random fields
    random_fields = {k: v for k, v in config.items() if isinstance(v, dict)}
    dim = len(random_fields)

    # 2. Sample for each distribution
    values = roberts_sequence(n, dim, perturb=True, key=jax.random.key(seed=0))

    # 3. Scale each sample based on the min/max/dist
    scaled_values = {}
    for (key, dist), column in zip(random_fields.items(), values.T):
        if dist["dist"] == "loguniform":
            scaled = jnp.exp(
                jnp.log(dist["min"])
                + column * (jnp.log(dist["max"]) - jnp.log(dist["min"]))
            )
        elif dist["dist"] == "uniform":
            scaled = dist["min"] + column * (dist["max"] - dist["min"])
        else:
            typing.assert_never(dist["dist"])

        scaled_values[key] = scaled

    # 4. Return the sampled configs
    for i in range(n):
        yield {
            **{k: v for k, v in config.items() if not isinstance(v, dict)},
            **{k: v[i].item() for k, v in scaled_values.items()},
        }


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
    perturb: bool = True,
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

    if perturb:
        if key is None:
            raise ValueError("key cannot be None when perturb=True")
        key, subkey = jax.random.split(key)
        x += jax.random.uniform(subkey, [dim])

    x, _ = jnp.modf(x)

    return x
