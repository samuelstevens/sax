import logging
import tomllib

import beartype
import tyro

import sax.sweep
import sax.train
import submitit

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")


@beartype.beartype
def main(
    configs: list[str],
    /,
    n_per_discrete: int = 1,
    slurm: bool = False,
    override: sax.train.Args = sax.train.Args(),
):
    """
    Start a hyperparameter sweep of training runs using either a Slurm cluster or a local GPU. Results are written to an Aim repo, which can be queried for final metrics to make plots like those you see in SAE papers (comparing sparsity and reconstruction loss).

    Args:
        configs: list of config filepaths.
        n_per_discrete: number of random samples to draw for each *discrete* config.
        slurm: whether to use a slurm cluster for running jobs or a local GPU.
    """
    for file in configs:
        with open(file, "rb") as fd:
            config = tomllib.load(fd)
        sweep(config, n_per_discrete=n_per_discrete, slurm=slurm, override=override)


@beartype.beartype
def sweep(
    sweep_config: dict[
        str, sax.sweep.Primitive | list[sax.sweep.Primitive] | sax.sweep.Distribution
    ],
    *,
    n_per_discrete: int,
    slurm: bool,
    override: sax.train.Args,
) -> None:
    configs = list(sax.sweep.expand(sweep_config, n_per_discrete=n_per_discrete))
    logger.info("Sweep has %d experiments.", len(configs))
    sweep_args, errs = [], []
    for config in configs:
        try:
            sweep_args.append(sax.train.Args(**config))
        except Exception as err:
            errs.append(str(err))

    if errs:
        msg = "\n\n".join(errs)
        raise RuntimeError(msg)

    if slurm:
        executor = submitit.SlurmExecutor()
    else:
        executor = submitit.DebugExecutor(folder="logs")

    sweep_args = [overwrite(args, override) for args in sweep_args]
    jobs = executor.map_array(sax.train.train, sweep_args)
    for result in submitit.helpers.as_completed(jobs):
        breakpoint()


def overwrite(args: sax.train.Args, override: sax.train.Args) -> sax.train.Args:
    """
    If there are any non-default values in override, returns a copy of `args` with all those values included.

    Arguments:
        args: sweep args
        override: incoming args with zero or more non-default values.

    Returns:
        sax.train.Args
    """
    pass


if __name__ == "__main__":
    tyro.cli(main)
