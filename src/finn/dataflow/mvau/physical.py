# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider-independent physical records for production MVAU elaboration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.authoring.design import DesignRealization
from finn.dataflow.design import Finding, QualifiedPath
from finn.dataflow.mvau.associations import MVAUNetworkRef
from finn.dataflow.mvau.source import (
    MVAU_DECLARATION_FAMILY_VERSION,
    MVAUResolvedDesign,
    mvau_problem_fingerprint,
)


class MVAUPhysicalDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class MVAUPhysicalNumericProtocol(str, Enum):
    AXI_STREAM = "axi_stream"


class MVAUPhysicalControlKind(str, Enum):
    CLOCK = "clock"
    RESET = "reset"
    CONFIGURATION = "configuration"


PhysicalParameterValue = bool | int | float | str


@dataclass(frozen=True, order=True)
class MVAUSemanticPortRef:
    region_id: str
    port_id: str


@dataclass(frozen=True)
class MVAUElaborationOrigin:
    """Exact selected-point identity from which physical elaboration was derived."""

    declaration_family_version: str
    problem_fingerprint: str
    assignments: tuple[tuple[QualifiedPath, object], ...]
    kernel_ids: tuple[str, ...]
    provider_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalComponent:
    id: str
    implementation_id: str
    parent_id: str | None = None
    parameters: tuple[tuple[str, PhysicalParameterValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))


@dataclass(frozen=True)
class MVAUPhysicalNumericInterface:
    id: str
    component_id: str
    direction: MVAUPhysicalDirection
    protocol: MVAUPhysicalNumericProtocol
    logical_width_bits: int
    physical_width_bits: int
    data_signal: str
    valid_signal: str
    ready_signal: str
    semantic_ports: tuple[MVAUSemanticPortRef, ...]


@dataclass(frozen=True)
class MVAUPhysicalControlInterface:
    id: str
    component_id: str
    kind: MVAUPhysicalControlKind
    signal: str


@dataclass(frozen=True)
class MVAUPhysicalConnection:
    id: str
    interface_ids: tuple[str, ...]
    semantic_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalBoundary:
    id: str
    interface_id: str
    semantic_port: MVAUSemanticPortRef


@dataclass(frozen=True)
class MVAUPhysicalAssociation:
    physical_id: str
    source_owner_ids: tuple[str, ...] = ()
    semantic_region_ids: tuple[str, ...] = ()
    semantic_ports: tuple[MVAUSemanticPortRef, ...] = ()
    semantic_edge_ids: tuple[str, ...] = ()
    decision_paths: tuple[QualifiedPath, ...] = ()
    kernel_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MVAUPhysicalElaboration:
    """Typed physical representation of one production MVAU Network."""

    source_scope_id: str
    origin: MVAUElaborationOrigin
    semantic_result: MVAUNetworkRef
    target_fpga_part: str
    target_clock_period_ns: float
    components: tuple[MVAUPhysicalComponent, ...]
    numeric_interfaces: tuple[MVAUPhysicalNumericInterface, ...]
    control_interfaces: tuple[MVAUPhysicalControlInterface, ...]
    connections: tuple[MVAUPhysicalConnection, ...]
    boundaries: tuple[MVAUPhysicalBoundary, ...]
    associations: tuple[MVAUPhysicalAssociation, ...]

    def __post_init__(self) -> None:
        components = tuple(sorted(self.components, key=lambda item: item.id))
        numeric = tuple(sorted(self.numeric_interfaces, key=lambda item: item.id))
        controls = tuple(sorted(self.control_interfaces, key=lambda item: item.id))
        connections = tuple(sorted(self.connections, key=lambda item: item.id))
        boundaries = tuple(sorted(self.boundaries, key=lambda item: item.id))
        associations = tuple(sorted(self.associations, key=lambda item: item.physical_id))
        for values, label in (
            (tuple(item.id for item in components), "component"),
            (tuple(item.id for item in numeric), "numeric interface"),
            (tuple(item.id for item in controls), "control interface"),
            (tuple(item.id for item in connections), "connection"),
            (tuple(item.id for item in boundaries), "boundary"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} identities must be unique")
        component_ids = {item.id for item in components}
        interface_ids = {item.id for item in numeric} | {item.id for item in controls}
        physical_ids = component_ids | interface_ids | {item.id for item in connections}
        if any(item.component_id not in component_ids for item in numeric) or any(
            item.component_id not in component_ids for item in controls
        ):
            raise ValueError("every physical interface must name a component")
        if any(
            component.parent_id is not None and component.parent_id not in component_ids
            for component in components
        ):
            raise ValueError("every physical component parent must name another component")
        for component in components:
            ancestors = set()
            parent = component.parent_id
            while parent is not None:
                if parent == component.id or parent in ancestors:
                    raise ValueError("physical component parent relationships must be acyclic")
                ancestors.add(parent)
                parent = next(item.parent_id for item in components if item.id == parent)
        if any(
            not isinstance(interface.direction, MVAUPhysicalDirection)
            or interface.protocol is not MVAUPhysicalNumericProtocol.AXI_STREAM
            or interface.logical_width_bits <= 0
            or interface.physical_width_bits < interface.logical_width_bits
            or interface.physical_width_bits % 8
            or not interface.data_signal
            or not interface.valid_signal
            or not interface.ready_signal
            for interface in numeric
        ):
            raise ValueError("numeric interfaces must have complete byte-aligned AXI-stream shapes")
        if any(
            not isinstance(interface.kind, MVAUPhysicalControlKind) or not interface.signal
            for interface in controls
        ):
            raise ValueError("control interfaces must have a declared kind and signal")
        if any(
            len(item.interface_ids) != 2 or len(set(item.interface_ids)) != 2
            for item in connections
        ):
            raise ValueError("physical connections must name exactly two distinct endpoints")
        if any(
            endpoint not in interface_ids for item in connections for endpoint in item.interface_ids
        ):
            raise ValueError("every physical connection endpoint must name an interface")
        if any(item.interface_id not in interface_ids for item in boundaries):
            raise ValueError("every physical boundary must name an interface")
        if any(item.physical_id not in physical_ids for item in associations):
            raise ValueError("every physical association must name a physical object")
        semantic_regions = {node.id: node.region for node in self.semantic_result.network.nodes}
        semantic_edges = {edge.id for edge in self.semantic_result.network.edges}
        semantic_ports = {
            MVAUSemanticPortRef(region_id, interface.port.id)
            for region_id, region in semantic_regions.items()
            for interface in region.interfaces
        }
        if any(
            port not in semantic_ports for interface in numeric for port in interface.semantic_ports
        ):
            raise ValueError("numeric interface references an unknown semantic port")
        if any(boundary.semantic_port not in semantic_ports for boundary in boundaries):
            raise ValueError("physical boundary references an unknown semantic port")
        if any(
            region_id not in semantic_regions
            for association in associations
            for region_id in association.semantic_region_ids
        ):
            raise ValueError("physical association references an unknown semantic region")
        if any(
            port not in semantic_ports
            for association in associations
            for port in association.semantic_ports
        ):
            raise ValueError("physical association references an unknown semantic port")
        if any(
            edge_id not in semantic_edges
            for association in associations
            for edge_id in association.semantic_edge_ids
        ):
            raise ValueError("physical association references an unknown semantic edge")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "numeric_interfaces", numeric)
        object.__setattr__(self, "control_interfaces", controls)
        object.__setattr__(self, "connections", connections)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "associations", associations)

    def component(self, component_id: str) -> MVAUPhysicalComponent:
        matches = tuple(item for item in self.components if item.id == component_id)
        if len(matches) != 1:
            raise KeyError(f"expected one physical component {component_id!r}")
        return matches[0]


class MVAUElaborationError(ValueError):
    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = tuple(
            sorted(findings, key=lambda item: (item.path, item.kind.value, item.code))
        )
        super().__init__(f"MVAU elaboration failed with {len(self.findings)} finding(s)")


def mvau_elaboration_origin(
    resolved: MVAUResolvedDesign,
    realization: DesignRealization,
) -> MVAUElaborationOrigin:
    """Construct the immutable identity of one production design realization."""

    return MVAUElaborationOrigin(
        MVAU_DECLARATION_FAMILY_VERSION,
        mvau_problem_fingerprint(resolved.point.problem),
        tuple(sorted(resolved.point.assignments.items(), key=lambda item: item[0])),
        tuple(realization.kernel(name).kernel_id for name in realization.kernels),
    )


__all__ = [
    "MVAUElaborationError",
    "MVAUElaborationOrigin",
    "MVAUPhysicalAssociation",
    "MVAUPhysicalBoundary",
    "MVAUPhysicalComponent",
    "MVAUPhysicalConnection",
    "MVAUPhysicalControlInterface",
    "MVAUPhysicalControlKind",
    "MVAUPhysicalDirection",
    "MVAUPhysicalElaboration",
    "MVAUPhysicalNumericInterface",
    "MVAUPhysicalNumericProtocol",
    "MVAUSemanticPortRef",
    "PhysicalParameterValue",
    "mvau_elaboration_origin",
]
