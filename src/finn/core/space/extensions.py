# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A Space class built from data: ``composite`` names generated members as one class.

Where a family of Space classes is generated from data rather than written (the
candidates of a Decision over nodes built from a table, as ``finn.kernels.adapters``
builds its adapter chains), its members are ordinary values: nodes built in loops
and joined by assigning their formals (``current.width_in = previous.width_out``),
and declarations. ``composite`` names them as the members of a new Space class,
exactly as a class body would; Python's class construction places each node
(``__set_name__``) and the Space class is collected once to report definition
errors early. It is also the way to a root with structure: ``design_space`` takes
a root of plain values, so a structure built at run time is a class built here.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, TypeVar, overload

from ._configuration import Space
from .collection import collect_space
from .declarations import ViewKey, local_name
from .errors import DefinitionError

_OWNER = "space.extensions"

S = TypeVar("S", bound=Space)


@overload
def composite(
    name: str,
    members: Mapping[str, object],
    *,
    annotations: Mapping[str, object] | None = None,
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[Space]: ...


@overload
def composite(
    name: str,
    members: Mapping[str, object],
    *,
    base: type[S],
    annotations: Mapping[str, object] | None = None,
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[S]: ...


def composite(
    name: str,
    members: Mapping[str, object],
    *,
    base: type[Space] = Space,
    annotations: Mapping[str, object] | None = None,
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[Space]:
    """A new Space class whose members are ``members``: nodes and declarations.

    For a Space class generated from data. Equivalent to a class statement with
    those attributes, so every rule of a class body applies: each node is placed
    once, and names are one segment.
    ``annotations`` are the class body's annotations: the value type of each
    Param and Decision member (``{"choice": int}``).
    """

    local_name(name, "composite Space class name")
    if not isinstance(base, type) or not issubclass(base, Space):
        raise DefinitionError.broken(
            "composite-base-not-space",
            _OWNER,
            f"{name}: composite requires a Space base class",
            declaration=name,
        )
    for member in members:
        local_name(member, f"{name} member name")
    frame = sys._getframe(1)
    namespace: dict[str, object] = {"__module__": frame.f_globals.get("__name__", __name__)}
    namespace.update(members)
    if annotations is not None:
        unknown = sorted(annotations.keys() - members.keys())
        if unknown:
            raise DefinitionError.broken(
                "composite-annotation-unknown",
                _OWNER,
                f"{name}: annotations name no member {unknown}",
                declaration=name,
            )
        namespace["__annotations__"] = dict(annotations)
    if exports is not None:
        namespace["exports"] = dict(exports)
    try:
        space_type = type(name, (base,), namespace)
    except DefinitionError:
        raise
    except RuntimeError as cause:  # __set_name__ failures are wrapped by type()
        if isinstance(cause.__cause__, DefinitionError):
            raise cause.__cause__ from None
        raise
    collect_space(space_type)
    return space_type


__all__ = ["composite"]
