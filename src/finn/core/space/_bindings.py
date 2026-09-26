# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How each formal of one declared node is supplied."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from ._nodes import FamilyFormal, NodeDecl, family_formals
from .declarations import MISSING, Const, Decision, Param, ValueRef, View, at
from .errors import DefinitionError

if TYPE_CHECKING:
    from ._configuration import Space

BindingKind = Literal["literal", "reference", "local-decision", "node", "parameter"]


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    supplier: object
    kind: BindingKind


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    bindings: Mapping[str, PlacementBinding]
    # Formals left open at the call (OPEN, or optional and unbound): a Bind of
    # the parent may supply them; an optional one nobody supplies stays
    # unsupplied, and one with a value default falls back to that default.
    open: tuple[str, ...] = ()


def placement_plan(
    family: type[Space],
    record: NodeDecl | None,
    *,
    root: bool,
    formals: Mapping[str, object] | None = None,
) -> PlacementPlan:
    """Classify every formal of one node; the root's literal values stay runtime inputs."""
    formals = family_formals(family) if formals is None else formals
    supplied = record.bindings if record is not None else {}
    bindings: dict[str, PlacementBinding] = {}
    opened: list[str] = []
    for name, formal in formals.items():
        if name in supplied:
            value = supplied[name]
            kind: BindingKind
            if isinstance(formal, FamilyFormal):
                kind = "node"
            elif isinstance(value, Decision) and value.owner is None:
                kind = "local-decision"
            elif isinstance(value, Param) and value.owner is None:
                raise DefinitionError(
                    f"{family.__qualname__}.{name}{at(value.origin)}: an inline Param cannot "
                    "supply a formal; declare the formal on the enclosing family and bind it"
                )
            elif isinstance(value, Const) and value.owner is None:
                value, kind = value.value, "parameter" if root else "literal"
            elif isinstance(value, (ValueRef, View)):
                kind = "reference"
            else:
                kind = "parameter" if root else "literal"
            bindings[name] = PlacementBinding(value, kind)
        elif isinstance(formal, FamilyFormal) or root:
            continue
        else:
            opened.append(name)
    if record is not None:
        # An explicitly OPEN formal is open even though it is required.
        opened.extend(name for name in record.open if name not in opened)
    return PlacementPlan(MappingProxyType(bindings), tuple(opened))


def open_default(formal: object) -> object:
    """The fallback of an open formal: a value, UNSUPPLIED, or MISSING if required."""
    if isinstance(formal, Param):
        return cast(Param[object], formal).default
    return MISSING


__all__ = ["PlacementBinding", "PlacementPlan", "open_default", "placement_plan"]
