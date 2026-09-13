# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable normalized values for flat acyclic dataflow networks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum

from finn.dataflow.model.maps import (
    AffineRankMap,
    CoordinateSet,
    ExplicitCoordinateMap,
    IdentityCoordinateMap,
    MapCapabilityError,
    MaterializationRequired,
    RectangularDomain,
    check_materialization_budget,
    normalize_coordinate,
)
from finn.dataflow.model.region import BeatSequence, Coordinate, DataflowRegion, Port


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


class PassCorrespondence(str, Enum):
    """Supported correspondence between complete region passes."""

    ONE_TO_ONE = "one_to_one"


class FanoutMode(str, Enum):
    """Supported logical fan-out behavior."""

    REPLICATE = "replicate"


@dataclass(frozen=True, order=True)
class RegionEndpoint:
    """Qualified reference to one region-owned interface."""

    node_id: str
    port_id: str

    def __post_init__(self) -> None:
        _require_string(self.node_id, "node_id")
        _require_string(self.port_id, "port_id")


@dataclass(frozen=True, init=False, eq=False)
class PositionMap:
    """Declared source-to-sink position function with explicit/compact backends."""

    _explicit_map: ExplicitCoordinateMap | None
    _identity_map: IdentityCoordinateMap | None
    _affine_map: AffineRankMap | None

    def __init__(
        self,
        entries: Mapping[Coordinate, Coordinate] | Iterable[tuple[Coordinate, Coordinate]],
    ) -> None:
        object.__setattr__(self, "_explicit_map", ExplicitCoordinateMap(entries))
        object.__setattr__(self, "_identity_map", None)
        object.__setattr__(self, "_affine_map", None)

    @classmethod
    def _from_compact(cls, value: IdentityCoordinateMap | AffineRankMap) -> PositionMap:
        result = object.__new__(cls)
        object.__setattr__(
            result, "_identity_map", value if isinstance(value, IdentityCoordinateMap) else None
        )
        object.__setattr__(
            result, "_affine_map", value if isinstance(value, AffineRankMap) else None
        )
        object.__setattr__(result, "_explicit_map", None)
        return result

    @classmethod
    def from_coordinate_map(
        cls, value: ExplicitCoordinateMap | IdentityCoordinateMap | AffineRankMap
    ) -> PositionMap:
        """Construct from one decoded map without losing its declared domains."""

        if isinstance(value, ExplicitCoordinateMap):
            result = object.__new__(cls)
            object.__setattr__(result, "_explicit_map", value)
            object.__setattr__(result, "_identity_map", None)
            object.__setattr__(result, "_affine_map", None)
            return result
        if isinstance(value, (IdentityCoordinateMap, AffineRankMap)):
            return cls._from_compact(value)
        raise TypeError("value must be a supported coordinate map")

    @property
    def is_explicit(self) -> bool:
        return self._explicit_map is not None

    @property
    def coordinate_map(
        self,
    ) -> ExplicitCoordinateMap | IdentityCoordinateMap | AffineRankMap:
        value = self._explicit_map or self._identity_map or self._affine_map
        assert value is not None
        return value

    @property
    def entries(self) -> tuple[tuple[Coordinate, Coordinate], ...]:
        if self._explicit_map is None:
            raise MaterializationRequired(
                "compact PositionMap.entries requires materialize_entries(max_entries=...)"
            )
        return self._explicit_map.entries

    def bind_domains(
        self,
        source_domain: RectangularDomain,
        target_domain: RectangularDomain,
    ) -> PositionMap:
        if self._explicit_map is not None:
            result = object.__new__(type(self))
            object.__setattr__(
                result,
                "_explicit_map",
                self._explicit_map.bind_domains(source_domain, target_domain),
            )
            object.__setattr__(result, "_identity_map", None)
            object.__setattr__(result, "_affine_map", None)
            return result
        if self._identity_map is not None:
            if self._identity_map.domain.ambient != source_domain:
                raise ValueError("identity position map is bound to incompatible domains")
            return self._from_compact(
                IdentityCoordinateMap(self._identity_map.domain, target_domain)
            )
        assert self._affine_map is not None
        if self._affine_map.source != source_domain or self._affine_map.target != target_domain:
            raise ValueError("affine position map is bound to incompatible domains")
        return self

    @property
    def domain(self) -> frozenset[Coordinate]:
        raise MaterializationRequired(
            "PositionMap.domain requires source_set or bounded materialization"
        )

    @property
    def codomain(self) -> frozenset[Coordinate]:
        raise MaterializationRequired(
            "PositionMap.codomain requires sink_set or bounded materialization"
        )

    @property
    def source_set(self) -> CoordinateSet:
        if self._explicit_map is not None:
            return self._explicit_map.source_set
        if self._identity_map is not None:
            return self._identity_map.domain
        assert self._affine_map is not None
        return CoordinateSet.full(self._affine_map.source)

    @property
    def sink_set(self) -> CoordinateSet:
        if self._explicit_map is not None:
            return self._explicit_map.target_set
        if self._identity_map is not None:
            return self._identity_map.target
        assert self._affine_map is not None
        return self._affine_map.image_set

    def mapped(self, position: Coordinate) -> Coordinate:
        candidate = normalize_coordinate(position, "position")
        try:
            return self.coordinate_map.mapped(candidate)
        except ValueError as exc:
            raise KeyError(candidate) from exc

    @classmethod
    def identity(cls, positions: CoordinateSet | Iterable[Coordinate]) -> PositionMap:
        if isinstance(positions, CoordinateSet):
            return cls._from_compact(IdentityCoordinateMap(positions))
        return cls(
            (normalize_coordinate(position, "identity position"),) * 2 for position in positions
        )

    @classmethod
    def row_major_reshape(cls, source: RectangularDomain, sink: RectangularDomain) -> PositionMap:
        return cls._from_compact(AffineRankMap.row_major_reshape(source, sink))

    @classmethod
    def affine(
        cls,
        source: RectangularDomain,
        *,
        view_extents: Iterable[int],
        sink: RectangularDomain,
        offset: int,
        coefficients: Iterable[int],
    ) -> PositionMap:
        affine_map = AffineRankMap.from_mixed_radix(
            source,
            view_extents=view_extents,
            target=sink,
            offset=offset,
            coefficients=coefficients,
        )
        if affine_map.is_in_bounds:
            try:
                affine_map.image_set
            except MapCapabilityError as exc:
                raise MapCapabilityError(
                    "compact position map requires an exact gap-free image"
                ) from exc
        return cls._from_compact(affine_map)

    def materialize_entries(self, *, max_entries: int) -> tuple[tuple[Coordinate, Coordinate], ...]:
        if self._explicit_map is not None:
            check_materialization_budget(
                len(self._explicit_map.entries),
                max_entries,
                field_name="max_entries",
                unit="entries",
            )
            return self._explicit_map.entries
        source_set = self.source_set
        check_materialization_budget(
            source_set.cardinality,
            max_entries,
            field_name="max_entries",
            unit="entries",
        )
        return tuple((source, self.mapped(source)) for source in source_set.iter_coordinates())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PositionMap):
            return NotImplemented
        return bool(self.coordinate_map == other.coordinate_map)

    def __hash__(self) -> int:
        return hash("position-map")


@dataclass(frozen=True)
class SinkContract:
    """One input endpoint and its source-to-sink tensor-position map."""

    endpoint: RegionEndpoint
    position_map: PositionMap


@dataclass(frozen=True)
class DirectConnection:
    """Direct logical transport with no added semantic behavior."""


@dataclass(frozen=True)
class ChannelSpec:
    """Ordered channel identity without physical capacity or timing fields."""

    channel_id: str
    preserve_order: bool = True
    pass_correspondence: PassCorrespondence = PassCorrespondence.ONE_TO_ONE

    def __post_init__(self) -> None:
        _require_string(self.channel_id, "channel_id")


@dataclass(frozen=True)
class OrderedChannel:
    """Logical ordered-channel transport."""

    specification: ChannelSpec


EdgeTransport = DirectConnection | OrderedChannel


@dataclass(frozen=True)
class NetworkNode:
    """One stable network node and its independently scheduled region."""

    id: str
    region: DataflowRegion

    def __post_init__(self) -> None:
        _require_string(self.id, "node id")
        if not isinstance(self.region, DataflowRegion):
            raise TypeError("region must be a DataflowRegion")


@dataclass(frozen=True)
class Edge:
    """One producer output replicated to an ordered list of input sinks."""

    id: str
    source: RegionEndpoint
    sinks: tuple[SinkContract, ...]
    fanout: FanoutMode = FanoutMode.REPLICATE
    pass_correspondence: PassCorrespondence = PassCorrespondence.ONE_TO_ONE
    transport: EdgeTransport = DirectConnection()

    def __post_init__(self) -> None:
        _require_string(self.id, "edge id")
        object.__setattr__(self, "sinks", tuple(self.sinks))


@dataclass(frozen=True)
class BoundaryContract:
    """One stable external boundary associated with a region endpoint."""

    id: str
    endpoint: RegionEndpoint
    external_beat_sequence: BeatSequence
    pass_correspondence: PassCorrespondence = PassCorrespondence.ONE_TO_ONE

    def __post_init__(self) -> None:
        _require_string(self.id, "boundary id")


@dataclass(frozen=True)
class DataflowNetwork:
    """Flat graph of independently scheduled regions and exact edge contracts."""

    nodes: tuple[NetworkNode, ...]
    edges: tuple[Edge, ...]
    boundaries: tuple[BoundaryContract, ...]

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.nodes, key=lambda item: item.id))
        if not all(isinstance(node, NetworkNode) for node in nodes):
            raise TypeError("nodes must contain only NetworkNode values")

        def resolve_port(endpoint: RegionEndpoint, *, output: bool) -> Port | None:
            matches = tuple(node for node in nodes if node.id == endpoint.node_id)
            if len(matches) != 1:
                return None
            try:
                interface = (
                    matches[0].region.output_interface(endpoint.port_id)
                    if output
                    else matches[0].region.input_interface(endpoint.port_id)
                )
            except KeyError:
                return None
            return interface.port

        edges = []
        for edge in self.edges:
            source = resolve_port(edge.source, output=True)
            sinks = []
            for sink_contract in edge.sinks:
                sink = resolve_port(sink_contract.endpoint, output=False)
                position_map = sink_contract.position_map
                if source is not None and sink is not None:
                    try:
                        position_map = position_map.bind_domains(
                            source.operand.position_domain,
                            sink.operand.position_domain,
                        )
                    except (ValueError, MapCapabilityError):
                        pass
                sinks.append(replace(sink_contract, position_map=position_map))
            edges.append(replace(edge, sinks=tuple(sinks)))

        boundaries = []
        for boundary in self.boundaries:
            input_port = resolve_port(boundary.endpoint, output=False)
            output_port = resolve_port(boundary.endpoint, output=True)
            port = input_port if input_port is not None else output_port
            sequence = boundary.external_beat_sequence
            if port is not None and not (input_port is not None and output_port is not None):
                try:
                    sequence = sequence.bind_position_domain(port.operand.position_domain)
                except ValueError:
                    pass
            boundaries.append(replace(boundary, external_beat_sequence=sequence))

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", tuple(sorted(edges, key=lambda item: item.id)))
        object.__setattr__(self, "boundaries", tuple(sorted(boundaries, key=lambda item: item.id)))

    def node(self, node_id: str) -> NetworkNode:
        matches = tuple(node for node in self.nodes if node.id == node_id)
        if len(matches) != 1:
            raise KeyError(f"expected one node {node_id!r}, found {len(matches)}")
        return matches[0]


__all__ = [
    "BoundaryContract",
    "ChannelSpec",
    "DataflowNetwork",
    "DirectConnection",
    "Edge",
    "EdgeTransport",
    "FanoutMode",
    "NetworkNode",
    "OrderedChannel",
    "PassCorrespondence",
    "PositionMap",
    "RegionEndpoint",
    "SinkContract",
]
