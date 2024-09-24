import dataclasses
import json
import os

import beartype
import equinox as eqx
import jax

#############
# Functions #
#############


@beartype.beartype
def to_aim_value(value: object):
    """
    Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.

    As a fallback, tries to call `to_aim_value()` on an object.
    """
    if value is None:
        return value

    if isinstance(value, (str, int, float)):
        return value

    if isinstance(value, list):
        return [to_aim_value(elem) for elem in value]

    if isinstance(value, dict):
        return {to_aim_value(k): to_aim_value(v) for k, v in value.items()}

    if dataclasses.is_dataclass(value):
        return to_aim_value(dataclasses.asdict(value))

    try:
        return value.tolist()
    except AttributeError:
        pass

    try:
        return value.to_aim_value()
    except AttributeError:
        pass

    raise ValueError(f"Could not convert value '{value}' to Aim-compatible value.")


# Using the trick from https://docs.kidger.site/equinox/examples/serialisation/ for save/load.


@beartype.beartype
def save(filename: str, model_kwargs: dict[str, object], model: eqx.Module):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as fd:
        kwargs_str = json.dumps(model_kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@beartype.beartype
def load(filename, cls: type[eqx.Module]) -> eqx.Module:
    with open(filename, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        model = cls(key=jax.random.key(seed=0), **kwargs)
        return eqx.tree_deserialise_leaves(fd, model)
