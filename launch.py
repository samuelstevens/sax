import dataclasses
import logging
import os.path
import tomllib

import beartype
import submitit
import tyro

import sax.sweep
import sax.train

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")


@beartype.beartype
def main(
    config_file: str,
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
    with open(config_file, "rb") as fd:
        sweep_config = tomllib.load(fd)

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

    # Include filename in experiment tags.
    exp_name, _ = os.path.splitext(os.path.basename(config_file))
    sweep_args = [
        dataclasses.replace(overwrite(args, override), tags=args.tags + [exp_name])
        for args in sweep_args
    ]
    jobs = executor.map_array(sax.train.train, sweep_args)
    for i, result in enumerate(submitit.helpers.as_completed(jobs)):
        exp_id = result.result()
        logger.info("Finished task %s (%d/%d)", exp_id, i + 1, len(jobs))


@beartype.beartype
def overwrite(args: sax.train.Args, override: sax.train.Args) -> sax.train.Args:
    """
    If there are any non-default values in override, returns a copy of `args` with all those values included.

    Arguments:
        args: sweep args
        override: incoming args with zero or more non-default values.

    Returns:
        sax.train.Args
    """
    override_dict = {
        field.name: getattr(override, field.name)
        for field in dataclasses.fields(override)
        if getattr(override, field.name) != field.default
    }
    return dataclasses.replace(args, **override_dict)


if __name__ == "__main__":
    tyro.cli(main)
