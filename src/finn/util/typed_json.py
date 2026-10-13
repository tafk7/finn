# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""JSON values read and written by their Python type annotations.

``decoded`` turns a value as JSON states it into a value of an annotated type, and
``encoded`` turns a value back into what JSON can state. Each refusal names its place
in the document (``configuration.target.period_ns: ...``).
"""

from __future__ import annotations

import collections.abc
import os
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Union


def decoded_fields(kind: type, stated: Mapping[str, Any], place: str) -> dict[str, Any]:
    """The fields of the dataclass ``kind`` that ``stated`` gives, each decoded by its
    annotation; refuses a required field ``stated`` does not give."""
    missing = [
        item.name
        for item in fields(kind)
        if item.default is MISSING and item.default_factory is MISSING and item.name not in stated
    ]
    if missing:
        raise ValueError(f"{place}: no {', '.join(missing)}, which {kind.__name__} requires")
    hints = typing.get_type_hints(kind)
    return {name: decoded(hints[name], value, f"{place}.{name}") for name, value in stated.items()}


def _text(kind: Any) -> str:
    """``kind`` as a refusal names it."""
    if isinstance(kind, type) and not typing.get_args(kind):
        return kind.__name__
    return str(kind).replace("typing.", "").replace("collections.abc.", "")


def decoded(kind: Any, stated: Any, place: str) -> Any:
    """``stated`` (as JSON states it, or as Python gives it) as a value of the type
    ``kind``: a bool, an int, a float (an int accepted), a str, a path (from a str), an
    enum (from its value), a list, tuple or sequence, a dict or mapping with str keys, a
    dataclass (from an object of its fields), a union (``X | None``), a callable, a
    class, ``Any``, or an instance of any other class. Refuses, naming ``place``, a null
    where ``kind`` is not optional, a value of another type, an unknown enum value and a
    key a dataclass does not declare."""
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if kind is Any:
        return stated
    if origin in (Union, types.UnionType):
        if stated is None and type(None) in args:
            return None
        if stated is None:
            raise ValueError(f"{place}: null, and it is not optional")
        members = [each for each in args if each is not type(None)]
        if len(members) == 1:
            return decoded(members[0], stated, place)
        for each in members:
            try:
                return decoded(each, stated, place)
            except ValueError:
                continue
        raise ValueError(f"{place}: {_text(kind)}, not {stated!r}")
    if stated is None:
        raise ValueError(f"{place}: null, and it is not optional")
    if origin is collections.abc.Callable or kind is collections.abc.Callable:
        if not callable(stated):
            raise ValueError(f"{place}: a function, not {stated!r}")
        return stated
    if origin is type:
        if not (isinstance(stated, type) and issubclass(stated, args[0])):
            raise ValueError(f"{place}: a {_text(args[0])} class, not {stated!r}")
        return stated
    if isinstance(kind, type) and is_dataclass(kind):
        if isinstance(stated, kind):
            return stated
        if not isinstance(stated, Mapping):
            raise ValueError(f"{place}: an object of {kind.__name__}'s fields, not {stated!r}")
        unknown = sorted(set(stated) - {item.name for item in fields(kind)})
        if unknown:
            raise ValueError(f"{place}: keys {kind.__name__} does not declare: {unknown}")
        values = decoded_fields(kind, stated, place)
        try:
            return kind(**values)
        except ValueError as refused:
            raise ValueError(f"{place}: {refused}") from None
    if isinstance(kind, type) and issubclass(kind, Enum):
        if isinstance(stated, kind):
            return stated
        try:
            return kind(stated)
        except ValueError:
            allowed = [each.value for each in kind]
            raise ValueError(f"{place}: {stated!r}; one of {allowed}") from None
    if origin in (list, tuple, collections.abc.Sequence):
        if not isinstance(stated, (list, tuple)):
            raise ValueError(f"{place}: a list, not {stated!r}")
        each = args[0] if args else Any
        items = [decoded(each, value, f"{place}[{index}]") for index, value in enumerate(stated)]
        return tuple(items) if origin is tuple else items
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(stated, Mapping):
            raise ValueError(f"{place}: an object, not {stated!r}")
        value_kind = args[1] if args else Any
        for key in stated:
            if not isinstance(key, str):
                raise ValueError(f"{place}: keys are strings, not {key!r}")
        return {key: decoded(value_kind, value, f"{place}.{key}") for key, value in stated.items()}
    if isinstance(kind, type) and issubclass(kind, PurePath):
        if not isinstance(stated, (str, os.PathLike)):
            raise ValueError(f"{place}: a path, not {stated!r}")
        return kind(stated)
    if kind is float and type(stated) is int:
        return float(stated)
    if kind is int and isinstance(stated, bool):
        raise ValueError(f"{place}: an int, not {stated!r}")
    if isinstance(kind, type) and not isinstance(stated, kind):
        name = _text(kind)
        article = "an" if name[0] in "aeiouAEIOU" else "a"
        raise ValueError(f"{place}: {article} {name}, not {stated!r}")
    return stated


def encoded(value: Any, place: str) -> Any:
    """``value`` as JSON states it: a dataclass as an object of its fields, an enum as
    its value, a path as a str, a tuple as a list. Refuses, naming ``place``, any other
    object."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: encoded(getattr(value, item.name), f"{place}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [encoded(each, f"{place}[{index}]") for index, each in enumerate(value)]
    if isinstance(value, Mapping):
        return {key: encoded(each, f"{place}.{key}") for key, each in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"{place}: {value!r} has no JSON form")


__all__ = ["decoded", "decoded_fields", "encoded"]
