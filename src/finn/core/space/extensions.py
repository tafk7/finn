# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Graphs as data: name nodes built in plain Python as one composite family.

Nodes are ordinary values: build them in loops, keep them in lists, and join
them by assigning their formals (``current.width_in = previous.width_out``).
``composite`` names them as the members of a new family, exactly as a class
body would; Python's class construction places each node (``__set_name__``)
and the family is collected once to report definition errors early. There is
no builder state to seal: the nodes stay assignable until the family is prepared.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, TypeVar, overload

from ._configuration import Space
from .collection import collect_space
from .declarations import ViewKey, local_name
from .errors import DefinitionError

S = TypeVar("S", bound=Space)


@overload
def composite(
    name: str,
    members: Mapping[str, object],
    *,
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[Space]: ...


@overload
def composite(
    name: str,
    members: Mapping[str, object],
    *,
    base: type[S],
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[S]: ...


def composite(
    name: str,
    members: Mapping[str, object],
    *,
    base: type[Space] = Space,
    exports: Mapping[ViewKey[Any], object] | None = None,
) -> type[Space]:
    """A new family whose members are ``members``: nodes and declarations.

    Equivalent to a class statement with those attributes, so every rule of
    a class body applies: each node is placed once, and names are one segment.
    """

    local_name(name, "composite family name")
    if not isinstance(base, type) or not issubclass(base, Space):
        raise DefinitionError(f"{name}: composite requires a Space base family")
    for member in members:
        local_name(member, f"{name} member name")
    frame = sys._getframe(1)
    namespace: dict[str, object] = {"__module__": frame.f_globals.get("__name__", __name__)}
    namespace.update(members)
    if exports is not None:
        namespace["exports"] = dict(exports)
    try:
        family = type(name, (base,), namespace)
    except DefinitionError:
        raise
    except RuntimeError as cause:  # __set_name__ failures are wrapped by type()
        if isinstance(cause.__cause__, DefinitionError):
            raise cause.__cause__ from None
        raise
    collect_space(family)
    return family


__all__ = ["composite"]
