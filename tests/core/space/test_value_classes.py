# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default snapshot's contract on the engine's own value classes, and its checker.

The engine's value classes are those it states value semantics for: the type
token of a ``ValueSemantics`` some engine module declares. Its other frozen
dataclasses (results, inspection reports, declarations, the linked model) are
returned to callers or used by the engine itself; no semantics snapshots them.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Literal, TypeVar

from value_classes import is_frozen, mutable_fields

import finn.core.space as space
from finn.core.space import Located
from finn.core.space.semantics import ValueSemantics

T = TypeVar("T")

#: The engine value class fields that may hold a mutable value, and why.
EXEMPT = {
    # A Located holds the located node's own snapshot, under that node's semantics:
    # it is immutable exactly when the located value is (every kernel value is).
    f"{Located.__module__}.Located.value",
}


def value_classes() -> tuple[type, ...]:
    """The frozen dataclasses the engine states value semantics for."""

    found: set[type] = set()
    for info in pkgutil.walk_packages(space.__path__, "finn.core.space."):
        for value in vars(importlib.import_module(info.name)).values():
            token = value.type_token if isinstance(value, ValueSemantics) else None
            if isinstance(token, type) and token.__module__.startswith("finn.core.space"):
                found.add(token)
    return tuple(sorted((cls for cls in found if is_frozen(cls)), key=str))


def test_the_engines_frozen_value_classes_hold_immutable_values() -> None:
    assert value_classes() == (Located,)
    found = mutable_fields(value_classes())
    assert sorted(field.partition(":")[0] for field in found) == sorted(EXEMPT)


class Colour(Enum):
    RED = 1


@dataclass(frozen=True)
class Inner:
    count: int


@dataclass(frozen=True)
class Immutable(Generic[T]):
    name: str
    count: int | None
    ratio: float
    raw: bytes
    path: Path
    colour: Colour
    mode: Literal["a", "b"]
    inner: Inner
    nested: tuple[tuple[str, Inner], ...]
    members: frozenset[int]
    kind: type[Inner]
    rule: Callable[[int], bool]
    generic: Located[int]


class Opaque:
    """A class the checker cannot judge; a caller may name it immutable."""


@dataclass(frozen=True)
class Mutable(Generic[T]):
    items: list[int]
    table: Mapping[str, int]
    anything: object
    value: T
    pairs: tuple[tuple[str, list[int]], ...]
    maybe: dict[str, int] | None
    opaque: Opaque


def test_the_checker_names_each_mutable_field_and_no_immutable_one() -> None:
    assert mutable_fields((Immutable,)) == []

    def names(found: list[str]) -> list[str]:
        return [field.partition(":")[0].rpartition(".")[2] for field in found]

    found = names(mutable_fields((Mutable,)))
    assert found == ["items", "table", "anything", "value", "pairs", "maybe", "opaque"]
    # A type the caller names immutable, with its reason.
    assert "opaque" not in names(mutable_fields((Mutable,), {Opaque: "holds no state"}))
