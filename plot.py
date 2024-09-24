import logging

import beartype
import tyro


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")


@beartype.beartype
def plot_exp(ax, name: str):
    import jax.numpy as jnp
    import scipy.spatial
    import sax.tracking

    exps = list(sax.tracking.load_by_tag(name))
    points = jnp.unique(
        jnp.array([[exp.data["val_l0"], exp.data["val_loss"]] for exp in exps]), axis=0
    )

    chull = scipy.spatial.ConvexHull(points)
    chull_points = points[chull.vertices]

    mask = jnp.ones(len(points), dtype=bool)
    other_points = points[mask.at[chull.vertices].set(False)]

    ax.scatter(
        other_points[:, 0],
        other_points[:, 1],
        alpha=0.2,
        label=f"{name} (hparam. sweep)",
        color="tab:blue",
    )

    chull_points = chull_points[chull_points[:, 0].argsort()]
    ax.plot(
        chull_points[:, 0],
        chull_points[:, 1],
        color="tab:blue",
        label=f"{name} (optimal)",
        marker="o",
    )


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
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    # Once I have these experiment IDs, what do I want to do?
    # -> Plot the sparsity-loss graph.
    # 1. Get all experiments by exp_name
    # 2. Get the most recent validation L0 and validation loss.
    # 3. Make a scatter plot.
    fig, ax = plt.subplots()

    for name in exp_names:
        plot_exp(ax, name)

    ax.set_title("L0 vs Loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L0")
    ax.set_ylabel("L2 Reconstruction Loss")
    ax.legend()

    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker_fn))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(ticker_fn))

    fig.savefig(out)


if __name__ == "__main__":
    tyro.cli(main)
