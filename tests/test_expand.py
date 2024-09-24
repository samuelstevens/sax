import dataclasses

import sax.sweep


def test_smoke():
    pass


@dataclasses.dataclass(frozen=True)
class DummyConfig:
    a: int
    b: int


def test_expand_one_list_field():
    sweep_config = {"a": [1, 2, 3], "b": 0}
    actual = list(sax.sweep.expand(sweep_config))
    expected = [
        {"a": 1, "b": 0},
        {"a": 2, "b": 0},
        {"a": 3, "b": 0},
    ]
    assert actual == expected


def test_expand_two_list_fields():
    sweep_config = {"a": [1, 2, 3], "b": ["a", "b"]}
    actual = list(sax.sweep.expand(sweep_config))
    expected = [
        {"a": 1, "b": "a"},
        {"a": 1, "b": "b"},
        {"a": 2, "b": "a"},
        {"a": 2, "b": "b"},
        {"a": 3, "b": "a"},
        {"a": 3, "b": "b"},
    ]
    assert actual == expected
