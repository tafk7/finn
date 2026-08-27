# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable normalized values for flat acyclic dataflow networks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from finn.dataflow.region import BeatSequence, Coordinate, DataflowRegion


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


@dataclass(frozen=True, init=False)
class PositionMap:
    """Concrete declared correspondence between source and sink positions."""

    entries: tuple[tuple[Coordinate, Coordinate], ...]

    def __init__(
        self,
        entries: Mapping[Coordinate, Coordinate] | Iterable[tuple[Coordinate, Coordinate]],
    ) -> None:
        values = entries.items() if isinstance(entries, Mapping) else entries
        normalized = tuple((tuple(source), tuple(sink)) for source, sink in values)
        object.__setattr__(self, "entries", tuple(sorted(normalized)))

    @property
    def domain(self) -> frozenset[Coordinate]:
        return frozenset(source for source, _sink in self.entries)

    @property
    def codomain(self) -> frozenset[Coordinate]:
        return frozenset(sink for _source, sink in self.entries)

    def mapped(self, position: Coordinate) -> Coordinate:
        values = tuple(sink for source, sink in self.entries if source == position)
        if len(values) != 1:
            raise KeyError(f"expected one mapping for {position!r}, found {len(values)}")
        return values[0]

    @classmethod
    def identity(cls, positions: Iterable[Coordinate]) -> PositionMap:
        return cls((tuple(position), tuple(position)) for position in positions)


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
        object.__setattr__(self, "nodes", tuple(sorted(self.nodes, key=lambda item: item.id)))
        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda item: item.id)))
        object.__setattr__(
            self, "boundaries", tuple(sorted(self.boundaries, key=lambda item: item.id))
        )

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
