import logging

import beartype
from jaxtyping import jaxtyped, Float, Array, Bool
import tyro


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")

colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]


@jaxtyped(typechecker=beartype.beartype)
def is_pareto_efficient(points: Float[Array, "n 2"]) -> Bool[Array, " n"]:
    """ """
    import jax.numpy as jnp

    # Sort points by x-value.
    i_sorted = jnp.argsort(points[:, 0])
    points = points[i_sorted]
    is_efficient = jnp.zeros(len(points), dtype=bool)
    min_y = jnp.inf
    for i, (x, y) in enumerate(points):
        if y < min_y:
            min_y = y
            is_efficient = is_efficient.at[i].set(True)

    # Un-sort is_efficient to match original points order.
    undo = jnp.zeros(len(points), dtype=int).at[i_sorted].set(jnp.arange(len(points)))
    return is_efficient[undo]


@beartype.beartype
def plot_exp(ax, name: str, color: str):
    import jax.numpy as jnp
    import sax.tracking

    exps = list(sax.tracking.load_by_tag(name))
    points = jnp.unique(
        jnp.array([[exp.data["val_l0"], exp.data["val_loss"]] for exp in exps]), axis=0
    )

    if len(points) > 2:
        mask = is_pareto_efficient(points)
        pareto_points = points[mask]
        other_points = points[~mask]

        ax.scatter(
            other_points[:, 0],
            other_points[:, 1],
            alpha=0.2,
            label=f"{name} (hparam. sweep)",
            color=color,
        )

        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            color=color,
            label=f"{name} (optimal)",
            marker="o",
        )
    else:
        ax.scatter(points[:, 0], points[:, 1], label=f"{name} (all)", color=color)


@beartype.beartype
def ticker_fn(y: float, _) -> str:
    return "{:g}".format(y)


@beartype.beartype
def main(exp_names: list[str], /, out: str):
    """
    Args:
        exp_names: list of experiment names to plot.
        out: filename to save plot to.
    """

    if len(exp_names) > len(colors):
        print(f"Add more colors to {__file__}.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.ticker

    # Once I have these experiment IDs, what do I want to do?
    # -> Plot the sparsity-loss graph.
    # 1. Get all experiments by exp_name
    # 2. Get the most recent validation L0 and validation loss.
    # 3. Make a scatter plot.
    fig, ax = plt.subplots()

    for name, color in zip(exp_names, colors):
        plot_exp(ax, name, color)

    ax.set_title("L0 vs Loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L0")
    ax.set_ylabel("L2 Reconstruction Loss")
    ax.legend()

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker_fn))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker_fn))

    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    tyro.cli(main)
