# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit child suppliers, nested parameter targets, and placement plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ._configuration import Space
from .declarations import (
    Decision,
    DecisionRef,
    Declaration,
    Net,
    Param,
    Port,
    ScopedValueRef,
    Subspace,
    ValueRef,
    class_namespace,
)
from .errors import DefinitionError


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    supplier: object
    kind: Literal["literal", "reference", "exposed-param", "local-decision"]


@dataclass(frozen=True, slots=True)
class PlacementTarget:
    path: tuple[Subspace[Space], ...]
    member: Param[object]
    name: str


@dataclass(frozen=True, slots=True)
class NestedBinding:
    reference: ValueRef[object]
    target: PlacementTarget
    binding: PlacementBinding


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    bindings: Mapping[str, PlacementBinding]
    nested_bindings: tuple[NestedBinding, ...] = ()
    ports: Mapping[str, Net[Space]] = MappingProxyType({})


class _TargetResolver:
    """Resolve concrete typed paths without recursive template compilation."""

    def __init__(self) -> None:
        self.tables: dict[type[Space], tuple[dict[str, object], dict[int, str]]] = {}

    def member(self, space_type: type[Space], reference: object) -> tuple[str, object]:
        if space_type not in self.tables:
            names = {
                id(value): name
                for base in reversed(space_type.__mro__)
                for name, value in vars(base).items()
                if isinstance(value, Declaration)
            }
            self.tables[space_type] = (class_namespace(space_type), names)
        namespace, names = self.tables[space_type]
        name = names.get(id(reference))
        if name is None:
            raise DefinitionError("binding reference is not a member of this child family")
        return name, namespace[name]

    def resolve(
        self,
        space_type: type[Space],
        reference: object,
    ) -> PlacementTarget:
        path: list[Subspace[Space]] = []
        seen: set[tuple[type[Space], int]] = set()
        while isinstance(reference, ScopedValueRef):
            if isinstance(reference, DecisionRef):
                raise DefinitionError("binding targets require Param references, not DecisionRef")
            identity = (space_type, id(reference))
            if identity in seen:
                raise DefinitionError("cyclic scoped parameter reference")
            seen.add(identity)
            _, placement = self.member(space_type, reference.placement)
            if not isinstance(placement, Subspace):
                raise DefinitionError("binding targets require concrete child placements")
            path.append(placement)
            space_type, reference = placement.space_type, reference.member
        name, member = self.member(space_type, reference)
        if not isinstance(member, Param):
            raise DefinitionError("binding target is not a Param declaration")
        return PlacementTarget(tuple(path), member, name)


def _placement_binding(parameter: Param[object], supplier: object, label: str) -> PlacementBinding:
    kind: Literal["literal", "reference", "exposed-param", "local-decision"]
    if isinstance(supplier, Param) and supplier.owner is None:
        kind = "exposed-param"
    elif isinstance(supplier, Decision) and supplier.owner is None:
        kind = "local-decision"
    elif isinstance(supplier, ValueRef):
        kind = "reference"
    else:
        kind = "literal"
    assert parameter.semantics is not None
    if isinstance(supplier, ValueRef):
        if supplier.semantics is not None and not parameter.semantics.is_compatible_with(
            supplier.semantics
        ):
            raise DefinitionError(f"{label}: binding has incompatible value semantics")
    else:
        try:
            supplier = parameter.semantics.freeze(supplier)
        except (TypeError, ValueError) as error:
            raise DefinitionError(f"{label}: {error}") from error
    return PlacementBinding(supplier, kind)


def collect_placement(placement: Subspace[Space]) -> PlacementPlan:
    """Validate named/direct bindings and explicitly targeted nested Params.

    Nested target exposure is checked against the allocated scopes before any
    callbacks are linked. It is not inferred by flattening child definitions.
    """
    namespace = class_namespace(placement.space_type)
    parameters = {name: value for name, value in namespace.items() if isinstance(value, Param)}
    label = placement.name or placement.space_type.__qualname__
    named = dict(placement.bindings)
    # A port is attached, never supplied: its binding names one of the parent's nets.
    ports: dict[str, Net[Space]] = {}
    for name in [name for name in named if isinstance(namespace.get(name), Port)]:
        net = named.pop(name)
        if not isinstance(net, Net):
            raise DefinitionError(f"{label}.{name}: a port binds to a Net declaration")
        ports[name] = net
    extra = named.keys() - parameters.keys()
    if extra:
        raise DefinitionError(f"{label}: unknown child parameter bindings {sorted(extra)}")
    resolver = _TargetResolver()
    nested: list[NestedBinding] = []
    targets: set[tuple[tuple[Subspace[Space], ...], str]] = set()
    for reference, supplier in placement.parameter_bindings.items():
        target = resolver.resolve(placement.space_type, reference)
        identity = (target.path, target.name)
        if identity in targets:
            raise DefinitionError(f"{label}: duplicate parameter binding")
        targets.add(identity)
        if not target.path:
            if target.name in named:
                raise DefinitionError(f"{label}.{target.name}: duplicate named and mapped binding")
            named[target.name] = supplier
        else:
            nested.append(
                NestedBinding(
                    reference,
                    target,
                    _placement_binding(target.member, supplier, f"{label}.{target.name}"),
                )
            )
    missing = parameters.keys() - named.keys()
    if missing:
        raise DefinitionError(f"{label}: missing child parameter bindings {sorted(missing)}")
    bindings = {
        name: _placement_binding(parameter, named[name], f"{label}.{name}")
        for name, parameter in parameters.items()
    }
    return PlacementPlan(MappingProxyType(bindings), tuple(nested), MappingProxyType(ports))


class PlacementPlans:
    """Normalized placements owned by one collection/preparation session."""

    def __init__(self) -> None:
        self._plans: dict[Subspace[Space], PlacementPlan] = {}

    def get(self, placement: Subspace[Space]) -> PlacementPlan:
        if placement not in self._plans:
            self._plans[placement] = collect_placement(placement)
        return self._plans[placement]
