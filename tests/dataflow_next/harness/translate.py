# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The one translation between the two packages' values: ``forward`` takes a value of
``finn.dataflow`` to the revised package's, ``back`` takes it home.

A value of either package (a dataclass or an enum member declared in one of its
modules) becomes the value of the class of the same module and name in the other: an
enum member by its name, a dataclass field by field, each field translated, without
running its constructor, so the value is carried exactly as it is, not rebuilt. Tuples,
lists, dicts and frozen sets are translated item by item; anything else (numbers,
strings, QONNX's datatypes and ranges, NumPy arrays, ``finn.dataflow.datatypes``' values,
which both packages share) is the same value on both sides.

A class the other package does not have, by module and name, is refused (``Untranslatable``):
a value of a type the revision dropped cannot be carried, and the harness names it.
"""

from __future__ import annotations

import dataclasses
import importlib
from enum import Enum
from typing import Any

LINE = "finn.dataflow"
NEXT = "finn.dataflow_next"
SHARED = ("finn.dataflow.datatypes",)
"""Modules both packages share: the revision imports them from the line."""


class Untranslatable(TypeError):
    """A value whose class the other package does not declare."""


def _home(module: str) -> str | None:
    """The package ``module`` belongs to, ``LINE`` or ``NEXT``, or None."""
    for package in (NEXT, LINE):
        if module == package or module.startswith(package + "."):
            if package == LINE and module in SHARED:
                return None
            return package
    return None


def _counterpart(cls: type, source: str, target: str) -> type:
    module = target + cls.__module__[len(source) :]
    try:
        found = importlib.import_module(module)
        for part in cls.__qualname__.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as error:
        raise Untranslatable(f"{target} has no {cls.__qualname__} in {module}") from error
    if not isinstance(found, type):
        raise Untranslatable(f"{module}.{cls.__qualname__} is not a class")
    return found


def _translate(value: Any, source: str, target: str) -> Any:
    kind = type(value)
    if kind in (tuple, list, frozenset, set):
        return kind(_translate(item, source, target) for item in value)
    if kind is dict:
        return {
            _translate(key, source, target): _translate(item, source, target)
            for key, item in value.items()
        }
    if _home(kind.__module__) != source:
        return value
    other: Any = _counterpart(kind, source, target)
    if isinstance(value, Enum):
        return other[value.name]
    if dataclasses.is_dataclass(value):
        made: Any = object.__new__(other)
        for field in dataclasses.fields(value):
            object.__setattr__(
                made, field.name, _translate(getattr(value, field.name), source, target)
            )
        return made
    raise Untranslatable(
        f"{kind.__module__}.{kind.__qualname__} is neither a dataclass nor an enum"
    )


def forward(value: Any) -> Any:
    """``value``, with every ``finn.dataflow`` value in it as the revised package's."""
    return _translate(value, LINE, NEXT)


def back(value: Any) -> Any:
    """``value``, with every revised package's value in it as ``finn.dataflow``'s."""
    return _translate(value, NEXT, LINE)


def counterpart(cls: type) -> type:
    """The revised package's class of the line's ``cls`` (or the line's, of the revision's)."""
    home = _home(cls.__module__)
    if home is None:
        raise Untranslatable(f"{cls.__module__}.{cls.__qualname__} belongs to neither package")
    return _counterpart(cls, home, NEXT if home == LINE else LINE)
