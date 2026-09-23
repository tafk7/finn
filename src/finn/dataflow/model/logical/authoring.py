# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical Region and topology declarations for Kernel authors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter as _SignatureParameter, Signature, signature
from typing import cast

from finn.kernels._engine import ABSENT
from finn.dataflow.model.children import KernelChoice, choice_role, kernel_choice_members
from finn.dataflow.model.logical.composition import (
    CompositionError,
    ImplementationPath,
    LogicalResult,
    NetworkResult,
    ParentBoundary,
    ParentConnection,
    RegionResult,
    compose_network,
    qualify_logical,
)
from finn.dataflow.model.logical.network import DataflowNetwork, PositionMap
from finn.dataflow.model.logical.region import DataflowRegion, RegionRefused
from finn.dataflow.model.logical.semantics import (
    DATAFLOW_LOGICAL_RESULT_SEMANTICS,
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
)
from finn.kernels.space.declarations import (
    AuthoringError,
    Derived,
    Space,
    ValueSource,
    _declaration_name,
    allow_absent,
    allow_inapplicable,
    reject,
    semantics_for,
)


@dataclass(frozen=True, slots=True, eq=False, init=False, kw_only=True)
class RegionDeclaration(Derived[DataflowRegion]):
    """A reusable Region construction recipe."""

    family: str
    version: str
    construct: Callable[..., DataflowRegion]

    def __init__(
        self,
        *,
        family: str,
        version: str,
        construct: Callable[..., DataflowRegion],
        name: str | None = None,
        **dependencies: ValueSource[object],
    ) -> None:
        if not family:
            raise AuthoringError("a Region declaration needs a non-empty family")
        if not version:
            raise AuthoringError("a Region declaration needs a non-empty version")
        if not callable(construct):
            raise AuthoringError("a Region declaration needs a callable constructor")
        _check_constructor(family, construct, tuple(dependencies))

        @wraps(construct)
        def evaluate(**values: object) -> object:
            try:
                return construct(**values)
            except RegionRefused as error:
                return reject(
                    "kernel-region-refused",
                    f"{family} cannot be constructed from these facts: {error}",
                    values={"family": family, "version": version},
                )

        object.__setattr__(self, "value_semantics", semantics_for(DATAFLOW_REGION_SEMANTICS))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a Region"))
        object.__setattr__(self, "dependencies", tuple(dependencies.items()))
        object.__setattr__(self, "evaluate", evaluate)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "construct", construct)


def _check_constructor(
    family: str,
    construct: Callable[..., DataflowRegion],
    dependencies: tuple[str, ...],
) -> None:
    parameters = signature(construct).parameters
    if any(
        parameter.kind
        in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD, parameter.POSITIONAL_ONLY)
        for parameter in parameters.values()
    ):
        raise AuthoringError(f"the {family} Region constructor must take only named parameters")
    accepted = set(parameters)
    declared = set(dependencies)
    if accepted != declared:
        missing = sorted(declared - accepted)
        extra = sorted(accepted - declared)
        raise AuthoringError(
            f"the {family} Region constructor signature does not match its dependency "
            f"mapping; unused dependencies {missing}, unbound parameters {extra}"
        )


@dataclass(frozen=True, slots=True, eq=False)
class KernelEndpoint:
    child: KernelChoice
    boundary_id: str
    output: bool

    def __post_init__(self) -> None:
        if not self.boundary_id:
            raise AuthoringError("a Kernel endpoint needs a non-empty boundary id")


@dataclass(frozen=True, slots=True, eq=False)
class EdgeSink:
    endpoint: KernelEndpoint
    position_map: ValueSource[PositionMap] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, KernelEndpoint) or self.endpoint.output:
            raise AuthoringError("an EdgeSink names an input Kernel endpoint")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkEdge:
    source: KernelEndpoint
    sinks: tuple[EdgeSink, ...]
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        source: KernelEndpoint,
        *sinks: EdgeSink,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(source, KernelEndpoint) or not source.output:
            raise AuthoringError("a NetworkEdge source names an output Kernel endpoint")
        if not sinks or any(not isinstance(sink, EdgeSink) for sink in sinks):
            raise AuthoringError("a NetworkEdge needs one or more EdgeSink declarations")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sinks", tuple(sinks))
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkEdge"))
        object.__setattr__(self, "when", when)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NetworkBoundary:
    endpoint: KernelEndpoint
    stable_name: str | None
    when: ValueSource[bool] | None

    def __init__(
        self,
        endpoint: KernelEndpoint,
        *,
        name: str | None = None,
        when: ValueSource[bool] | None = None,
    ) -> None:
        if not isinstance(endpoint, KernelEndpoint):
            raise AuthoringError("a NetworkBoundary exposes a Kernel endpoint")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "stable_name", _declaration_name(name, "a NetworkBoundary"))
        object.__setattr__(self, "when", when)


TopologyDeclaration = NetworkEdge | NetworkBoundary
TOPOLOGY_TYPES: tuple[type, ...] = (NetworkEdge, NetworkBoundary)


def topology_members(kernel_type: type[Space]) -> tuple[tuple[str, TopologyDeclaration], ...]:
    ordered: dict[str, TopologyDeclaration] = {}
    for base in reversed(kernel_type.__mro__):
        if not issubclass(base, Space) or base is Space:
            continue
        for name, value in base.__dict__.items():
            if isinstance(value, TOPOLOGY_TYPES):
                ordered[name] = cast(TopologyDeclaration, value)
            elif name in ordered:
                raise AuthoringError(
                    f"{base.__name__}.{name} replaces topology with {type(value).__name__}"
                )
    return tuple(ordered.items())


def composite_logical_property(kernel_type: type[Space]) -> Derived[LogicalResult]:
    choices = kernel_choice_members(kernel_type)
    topology = topology_members(kernel_type)
    dependencies: list[tuple[str, ValueSource[object]]] = []
    choice_keys: dict[int, tuple[str, str, str]] = {}
    for index, (member_name, declaration) in enumerate(choices):
        role = choice_role(member_name, declaration)
        node_id = declaration.node_id or role
        key = f"child_{index}"
        choice_keys[id(declaration)] = (role, node_id, key)
        dependencies.append(
            (key, allow_inapplicable(cast("ValueSource[object]", declaration.logical_result)))
        )
    topology_plan: list[tuple[str, TopologyDeclaration, str | None, tuple[str | None, ...]]] = []
    for index, (member_name, topology_declaration) in enumerate(topology):
        identity = topology_declaration.stable_name or member_name
        when_key = None
        if topology_declaration.when is not None:
            when_key = f"topology_when_{index}"
            dependencies.append(
                (when_key, allow_absent(cast("ValueSource[object]", topology_declaration.when)))
            )
        map_keys: list[str | None] = []
        if isinstance(topology_declaration, NetworkEdge):
            for sink_index, sink in enumerate(topology_declaration.sinks):
                if sink.position_map is None:
                    map_keys.append(None)
                else:
                    key = f"topology_map_{index}_{sink_index}"
                    dependencies.append((key, cast("ValueSource[object]", sink.position_map)))
                    map_keys.append(key)
        topology_plan.append((identity, topology_declaration, when_key, tuple(map_keys)))

    def boundary_id(endpoint: KernelEndpoint) -> str:
        try:
            _role, node_id, _key = choice_keys[id(endpoint.child)]
        except KeyError:
            raise CompositionError("topology names a Kernel child outside the class") from None
        return f"{node_id}/{endpoint.boundary_id}"

    def evaluate(**values: object) -> object:
        children = []
        for _role, node_id, key in choice_keys.values():
            value = values[key]
            if value is ABSENT:
                continue
            if not isinstance(value, (RegionResult, NetworkResult)):
                return reject(
                    "kernel-logical-result-type", "a child returned an invalid logical value"
                )
            children.append(qualify_logical(ImplementationPath((node_id,)), value))
        connections = []
        boundaries = []
        try:
            for identity, declaration, when_key, map_keys in topology_plan:
                if when_key is not None and (
                    values[when_key] is ABSENT or not bool(values[when_key])
                ):
                    continue
                if isinstance(declaration, NetworkEdge):
                    connections.append(
                        ParentConnection(
                            identity,
                            boundary_id(declaration.source),
                            tuple(boundary_id(sink.endpoint) for sink in declaration.sinks),
                            tuple(
                                None if key is None else cast(PositionMap, values[key])
                                for key in map_keys
                            ),
                        )
                    )
                else:
                    boundaries.append(ParentBoundary(identity, boundary_id(declaration.endpoint)))
            return compose_network(
                children=tuple(children),
                connections=tuple(connections),
                boundaries=tuple(boundaries),
            )
        except (CompositionError, KeyError, TypeError, ValueError) as error:
            return reject("kernel-composition-refused", str(error))

    evaluate.__signature__ = Signature(  # type: ignore[attr-defined]
        [
            _SignatureParameter(name, _SignatureParameter.KEYWORD_ONLY)
            for name, _source in dependencies
        ]
    )
    return Derived(
        semantics_for(DATAFLOW_LOGICAL_RESULT_SEMANTICS), None, tuple(dependencies), evaluate
    )


def network_property(logical: ValueSource[LogicalResult]) -> Derived[DataflowNetwork]:
    def evaluate(*, logical: LogicalResult) -> object:
        if not isinstance(logical, NetworkResult):
            return reject(
                "kernel-network-unavailable", "a composite logical result needs a Network"
            )
        return logical.network

    return Derived(
        semantics_for(DATAFLOW_NETWORK_SEMANTICS),
        None,
        (("logical", cast("ValueSource[object]", logical)),),
        evaluate,
    )


__all__ = [
    "EdgeSink",
    "KernelEndpoint",
    "NetworkBoundary",
    "NetworkEdge",
    "RegionDeclaration",
    "topology_members",
]
