import logging
import tomllib
import dataclasses

import beartype
import tyro

import sax.sweep
import sax.train

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")


@beartype.beartype
def main(configs: list[str], /, n_exps: int):
    """
    Start a hyperparameter sweep of training runs using either a Slurm cluster or a local GPU. Results are written to an Aim repo, which can be queried for final metrics to make plots like those you see in SAE papers (comparing sparsity and reconstruction loss).

    Args:
        configs: list of config filepaths.
    """
    for file in configs:
        with open(file, "rb") as fd:
            config = tomllib.load(fd)
        sweep(config, n=n_exps)


@beartype.beartype
def sweep(
    sweep_config: dict[
        str, sax.sweep.Primitive | list[sax.sweep.Primitive] | sax.sweep.Distribution
    ],
    *,
    n: int,
) -> None:
    configs = list(sax.sweep.expand(sweep_config, n=n))
    logger.info("Sweep has %d experiments.", len(configs))
    breakpoint()
    errs = []
    for config in configs:
        err = validate(config, cls=sax.train.Args)
        if err is not None:
            errs.append(err)

    if errs:
        raise RuntimeError(errs)

    for config in configs:
        executor.submit(config)


def validate(
    config: dict[str, sax.sweep.Primitive], *, cls: type = sax.train.Args
) -> str | None:
    errors = []
    cls_fields = {field.name: field.type for field in dataclasses.fields(cls)}
    
    for key, value in config.items():
        if key not in cls_fields:
            errors.append(f"Field '{key}' is not present in {cls.__name__}")
        else:
            expected_type = cls_fields[key]
            if not isinstance(value, expected_type):
                errors.append(f"Field '{key}' has type {type(value).__name__}, but {cls.__name__} expects {expected_type.__name__}")
    
    for key in cls_fields:
        if key not in config:
            errors.append(f"Required field '{key}' is missing from the config")
    
    if errors:
        return "\n".join(errors)
    return None


if __name__ == "__main__":
    tyro.cli(main)
