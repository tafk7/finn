# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default snapshot's contract, checked: a frozen value class holds immutable values.

Under the Space engine's default value semantics an instance of a frozen
dataclass is its own snapshot (``finn.core.space.semantics.default_semantics``):
a frozen value class holds immutable values; one that holds a mutable field
states its own semantics. ``mutable_fields`` reads each class's field
annotations and names every field whose type is not immutable, so a field that
would make sharing unsound fails a gate where the class is declared, not a
replay far from it.

Immutable: ``None``, ``bool``, ``int``, ``float``, ``complex``, ``str``,
``bytes``, ``pathlib`` paths, enums, frozen dataclasses (each checked where it
is declared), ``Literal``; ``tuple`` and ``frozenset`` of immutable types, and
unions of them; a class (``type[...]``) and a callable, which a deep copy
shares as well, so sharing them changes nothing. Not immutable: ``object``,
``Any``, a bare type variable, ``Mapping`` and every other type, unless the
caller names it immutable, with its reason.

A test helper, imported as ``value_classes`` (the gates put ``tests`` on
``PYTHONPATH``); the standard library only.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import importlib
import pathlib
import pkgutil
import types
import typing
from collections.abc import Iterable, Mapping

_ATOMIC: tuple[type, ...] = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    pathlib.PurePath,
)


def frozen_dataclasses(package: str) -> tuple[type, ...]:
    """Every frozen dataclass declared in ``package`` or a module below it."""

    root = importlib.import_module(package)
    names = [package]
    if hasattr(root, "__path__"):
        names += [found.name for found in pkgutil.walk_packages(root.__path__, f"{package}.")]
    classes: list[type] = []
    for name in names:
        module = importlib.import_module(name)
        classes.extend(
            value
            for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == name and is_frozen(value)
        )
    return tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__)))


def is_frozen(cls: type) -> bool:
    params = getattr(cls, "__dataclass_params__", None)
    return dataclasses.is_dataclass(cls) and params is not None and bool(params.frozen)


def mutable_fields(
    classes: Iterable[type], immutable: Mapping[object, str] | None = None
) -> list[str]:
    """``module.Class.field: annotation`` for each field whose annotation is not an
    immutable type. ``immutable`` names further types (classes or protocols) that are,
    each with its reason."""

    also = tuple(immutable or {})
    found: list[str] = []
    for cls in classes:
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            annotation = hints[field.name]
            if not _immutable(annotation, also):
                found.append(f"{cls.__module__}.{cls.__qualname__}.{field.name}: {annotation}")
    return found


def _immutable(annotation: object, also: tuple[object, ...]) -> bool:
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is None:
        if isinstance(annotation, typing.TypeVar):
            bounds = annotation.__constraints__ or (
                (annotation.__bound__,) if annotation.__bound__ is not None else ()
            )
            return bool(bounds) and all(_immutable(bound, also) for bound in bounds)
        if annotation in also:
            return True
        if not isinstance(annotation, type):
            return False
        return (
            issubclass(annotation, _ATOMIC)
            or issubclass(annotation, enum.Enum)
            or is_frozen(annotation)
        )
    if origin is typing.Literal or origin is type or origin is collections.abc.Callable:
        return True
    if origin is typing.Annotated:
        return _immutable(arguments[0], also)
    if origin in (tuple, frozenset, typing.Union, types.UnionType):
        return all(argument is Ellipsis or _immutable(argument, also) for argument in arguments)
    # A parameterized frozen dataclass (``Available[T]``): its fields are checked
    # where it is declared.
    return isinstance(origin, type) and is_frozen(origin)
