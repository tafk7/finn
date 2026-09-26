# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How each formal of one declared node is supplied."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from ._nodes import FamilyFormal, NodeDecl, family_formals
from .declarations import MISSING, UNSUPPLIED, Const, Decision, Param, ValueRef, View, at
from .errors import DefinitionError

if TYPE_CHECKING:
    from ._configuration import Space

BindingKind = Literal["literal", "reference", "local-decision", "node", "parameter"]


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    supplier: object
    kind: BindingKind
    # The scope whose body wrote the supplier; None is the placed node's own
    # source scope (a binding at the call or by direct assignment).
    source_scope: int | None = None
    origin: str | None = None


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    bindings: Mapping[str, PlacementBinding]
    # Formals nothing supplies: an optional one falls back to its default or
    # stays unsupplied; a required one is a definition error when preparing.
    unsupplied: tuple[str, ...] = ()


# A supplier written through a path by an enclosing node: (value, scope, origin).
Supply = tuple[object, int, "str | None"]


def placement_plan(
    family: type[Space],
    record: NodeDecl | None,
    *,
    root: bool,
    formals: Mapping[str, object] | None = None,
    nested: Mapping[str, Supply] | None = None,
) -> PlacementPlan:
    """Classify every formal of one node; the root's literal values stay runtime inputs."""
    formals = family_formals(family) if formals is None else formals
    supplied: dict[str, tuple[object, int | None, str | None]] = {}
    if record is not None:
        for name, value in record.bindings.items():
            supplied[name] = (value, None, record.supplied_at.get(name))
    for name, supply in (nested or {}).items():
        supplied[name] = supply
    bindings: dict[str, PlacementBinding] = {}
    unsupplied: list[str] = []
    for name, formal in formals.items():
        if name not in supplied:
            if not root or isinstance(formal, FamilyFormal):
                unsupplied.append(name)
            continue
        value, written, origin = supplied[name]
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
            value, kind = value.value, "parameter" if root and written is None else "literal"
        elif isinstance(value, (ValueRef, View)):
            kind = "reference"
        else:
            kind = "parameter" if root and written is None else "literal"
        bindings[name] = PlacementBinding(value, kind, written, origin)
    return PlacementPlan(MappingProxyType(bindings), tuple(unsupplied))


def fallback(formal: object) -> object:
    """The default of an unsupplied formal: a value, UNSUPPLIED, or MISSING if required."""
    if isinstance(formal, Param):
        return cast(Param[object], formal).default
    assert isinstance(formal, FamilyFormal)
    return MISSING if formal.required else UNSUPPLIED


__all__ = ["PlacementBinding", "PlacementPlan", "Supply", "fallback", "placement_plan"]
