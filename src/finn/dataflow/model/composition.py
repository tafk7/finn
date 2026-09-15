# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified hierarchical authoring over the existing flat Network model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import TypeAlias

from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    DirectConnection,
    Edge,
    EdgeTransport,
    FanoutMode,
    NetworkNode,
    PassCorrespondence,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.region import DataflowRegion, InputInterface


_ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


class CompositionError(ValueError):
    """A hierarchy cannot be lowered to one canonical flat Network."""


@dataclass(frozen=True, slots=True)
class ImplementationPath:
    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or any(not _ATOM.fullmatch(segment) for segment in segments):
            raise ValueError("an implementation path needs non-empty ASCII identifier segments")
        object.__setattr__(self, "segments", segments)

    def child(self, segment: str) -> ImplementationPath:
        return ImplementationPath((*self.segments, segment))

    @property
    def value(self) -> str:
        return "/".join(self.segments)


@dataclass(frozen=True, slots=True)
class RegionResult:
    region: DataflowRegion

    def __post_init__(self) -> None:
        if not isinstance(self.region, DataflowRegion):
            raise TypeError("RegionResult contains one DataflowRegion")


@dataclass(frozen=True, slots=True)
class QualifiedChildResult:
    use_path: ImplementationPath
    result: RegionResult | NetworkResult


@dataclass(frozen=True, slots=True)
class NetworkResult:
    network: DataflowNetwork
    children: tuple[QualifiedChildResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.network, DataflowNetwork):
            raise TypeError("NetworkResult contains one DataflowNetwork")
        children = tuple(self.children)
        if any(not isinstance(child, QualifiedChildResult) for child in children):
            raise TypeError("NetworkResult children must be QualifiedChildResult values")
        object.__setattr__(self, "children", children)


LogicalResult: TypeAlias = RegionResult | NetworkResult


@dataclass(frozen=True, slots=True)
class NetworkFragment:
    use_path: ImplementationPath
    network: DataflowNetwork
    children: tuple[QualifiedChildResult, ...]


@dataclass(frozen=True, slots=True)
class ParentConnection:
    edge_id: str
    source_boundary: str
    sink_boundaries: tuple[str, ...]
    position_maps: tuple[PositionMap | None, ...] = ()
    fanout: FanoutMode = FanoutMode.REPLICATE
    pass_correspondence: PassCorrespondence = PassCorrespondence.ONE_TO_ONE
    transport: EdgeTransport = DirectConnection()

    def __post_init__(self) -> None:
        if not _ATOM.fullmatch(self.edge_id):
            raise ValueError("a parent edge id must be one identifier atom")
        sinks = tuple(self.sink_boundaries)
        maps = tuple(self.position_maps) or (None,) * len(sinks)
        if not sinks or len(maps) != len(sinks):
            raise ValueError("a parent connection needs one map slot per sink")
        object.__setattr__(self, "sink_boundaries", sinks)
        object.__setattr__(self, "position_maps", maps)


@dataclass(frozen=True, slots=True)
class ParentBoundary:
    boundary_id: str
    child_boundary: str

    def __post_init__(self) -> None:
        if not _ATOM.fullmatch(self.boundary_id):
            raise ValueError("a parent boundary id must be one identifier atom")


def _prefix(path: ImplementationPath, value: str) -> str:
    return f"{path.value}/{value}"


def _endpoint(path: ImplementationPath, endpoint: RegionEndpoint) -> RegionEndpoint:
    return RegionEndpoint(_prefix(path, endpoint.node_id), endpoint.port_id)


def qualify_region(use_path: ImplementationPath, result: RegionResult) -> NetworkFragment:
    node_id = use_path.value
    boundaries = []
    for item in result.region.inputs:
        if isinstance(item, InputInterface):
            boundaries.append(
                BoundaryContract(
                    _prefix(use_path, item.port.id),
                    RegionEndpoint(node_id, item.port.id),
                    item.port.beat_sequence,
                )
            )
    boundaries.extend(
        BoundaryContract(
            _prefix(use_path, item.port.id),
            RegionEndpoint(node_id, item.port.id),
            item.port.beat_sequence,
        )
        for item in result.region.outputs
    )
    return NetworkFragment(
        use_path,
        DataflowNetwork((NetworkNode(node_id, result.region),), (), tuple(boundaries)),
        (QualifiedChildResult(use_path, result),),
    )


def qualify_network(use_path: ImplementationPath, result: NetworkResult) -> NetworkFragment:
    network = result.network
    qualified = DataflowNetwork(
        tuple(NetworkNode(_prefix(use_path, node.id), node.region) for node in network.nodes),
        tuple(
            replace(
                edge,
                id=_prefix(use_path, edge.id),
                source=_endpoint(use_path, edge.source),
                sinks=tuple(
                    replace(sink, endpoint=_endpoint(use_path, sink.endpoint))
                    for sink in edge.sinks
                ),
            )
            for edge in network.edges
        ),
        tuple(
            replace(
                boundary,
                id=_prefix(use_path, boundary.id),
                endpoint=_endpoint(use_path, boundary.endpoint),
            )
            for boundary in network.boundaries
        ),
    )
    nested = tuple(
        QualifiedChildResult(
            ImplementationPath((*use_path.segments, *child.use_path.segments)),
            child.result,
        )
        for child in result.children
    )
    return NetworkFragment(
        use_path,
        qualified,
        (QualifiedChildResult(use_path, result), *nested),
    )


def qualify_logical(use_path: ImplementationPath, result: LogicalResult) -> NetworkFragment:
    if isinstance(result, RegionResult):
        return qualify_region(use_path, result)
    if isinstance(result, NetworkResult):
        return qualify_network(use_path, result)
    raise TypeError("logical composition requires RegionResult or NetworkResult")


def _boundary_direction(network: DataflowNetwork, boundary: BoundaryContract) -> bool:
    """Return True for output, False for input, refusing ambiguity."""

    node = network.node(boundary.endpoint.node_id)
    inputs = tuple(
        item
        for item in node.region.inputs
        if isinstance(item, InputInterface) and item.port.id == boundary.endpoint.port_id
    )
    outputs = tuple(
        item for item in node.region.outputs if item.port.id == boundary.endpoint.port_id
    )
    if len(inputs) + len(outputs) != 1:
        raise CompositionError(f"boundary {boundary.id!r} has an ambiguous endpoint")
    return bool(outputs)


def compose_network(
    *,
    children: tuple[NetworkFragment, ...],
    connections: tuple[ParentConnection, ...] = (),
    boundaries: tuple[ParentBoundary, ...] = (),
    disposed_boundaries: tuple[str, ...] = (),
) -> NetworkResult:
    """Flatten qualified children and explicitly consume every child boundary."""

    nodes = tuple(node for child in children for node in child.network.nodes)
    edges = [edge for child in children for edge in child.network.edges]
    child_boundaries = {
        boundary.id: boundary for child in children for boundary in child.network.boundaries
    }
    if len(child_boundaries) != sum(len(child.network.boundaries) for child in children):
        raise CompositionError("qualified child boundary ids collide")
    consumed: set[str] = set()
    for connection in connections:
        try:
            source = child_boundaries[connection.source_boundary]
            sinks = tuple(child_boundaries[name] for name in connection.sink_boundaries)
        except KeyError as error:
            raise CompositionError(f"parent connection names unknown boundary {error.args[0]!r}")
        if not _boundary_direction(DataflowNetwork(nodes, tuple(edges), ()), source):
            raise CompositionError("a parent connection source must be an output boundary")
        if any(
            _boundary_direction(DataflowNetwork(nodes, tuple(edges), ()), sink) for sink in sinks
        ):
            raise CompositionError("a parent connection sink must be an input boundary")
        if source.id in consumed or any(sink.id in consumed for sink in sinks):
            raise CompositionError("a child boundary may be consumed only once")
        source_port = (
            DataflowNetwork(nodes, tuple(edges), ())
            .node(source.endpoint.node_id)
            .region.output_interface(source.endpoint.port_id)
            .port
        )
        sink_contracts = []
        for sink, position_map in zip(sinks, connection.position_maps):
            sink_contracts.append(
                SinkContract(
                    sink.endpoint,
                    position_map or PositionMap.identity(source_port.beat_sequence.image_set),
                )
            )
        edges.append(
            Edge(
                connection.edge_id,
                source.endpoint,
                tuple(sink_contracts),
                connection.fanout,
                connection.pass_correspondence,
                connection.transport,
            )
        )
        consumed.add(source.id)
        consumed.update(sink.id for sink in sinks)
    parent_boundaries = []
    for exported in boundaries:
        try:
            child = child_boundaries[exported.child_boundary]
        except KeyError as error:
            raise CompositionError(f"parent boundary names unknown child {error.args[0]!r}")
        if child.id in consumed:
            raise CompositionError("a consumed child boundary cannot also be re-exported")
        consumed.add(child.id)
        parent_boundaries.append(replace(child, id=exported.boundary_id))
    disposed = set(disposed_boundaries)
    if disposed - set(child_boundaries):
        raise CompositionError("disposed child boundary is unknown")
    if disposed & consumed:
        raise CompositionError("a child boundary cannot be consumed and disposed")
    consumed.update(disposed)
    unconsumed = sorted(set(child_boundaries) - consumed)
    if unconsumed:
        raise CompositionError(f"child boundaries are neither connected nor exported: {unconsumed}")
    network = DataflowNetwork(nodes, tuple(edges), tuple(parent_boundaries))
    report = validate_network(network)
    if report.issues:
        raise CompositionError(
            "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        )
    return NetworkResult(
        network,
        tuple(child_result for child in children for child_result in child.children),
    )


__all__ = [
    "CompositionError",
    "ImplementationPath",
    "LogicalResult",
    "NetworkFragment",
    "NetworkResult",
    "ParentBoundary",
    "ParentConnection",
    "QualifiedChildResult",
    "RegionResult",
    "compose_network",
    "qualify_logical",
    "qualify_network",
    "qualify_region",
]
