# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Detached logical-to-physical binding and relation records."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

from finn.dataflow.artifacts.build import ModuleBuildRequirements
from finn.dataflow.model.logical.network import DataflowNetwork
from finn.dataflow.model.physical.layout import PackedBeatLayout, PeriodicLast
from finn.dataflow.model.physical.structure import PhysicalStructure


@dataclass(frozen=True)
class KernelStreamBinding:
    region_port_id: str
    abi_bus_id: str
    payload: PackedBeatLayout
    framing: PeriodicLast | None = None

    def __post_init__(self) -> None:
        if type(self.region_port_id) is not str or type(self.abi_bus_id) is not str:
            raise TypeError("stream binding identities must be strings")
        if not self.region_port_id or not self.abi_bus_id:
            raise ValueError("stream bindings require local port and bus identities")
        if not isinstance(self.payload, PackedBeatLayout):
            raise TypeError("stream bindings require an immutable packed layout")
        if self.framing is not None and not isinstance(self.framing, PeriodicLast):
            raise TypeError("framing must be PeriodicLast or None")


@dataclass(frozen=True)
class KernelRealizationFacts:
    requirements: ModuleBuildRequirements
    streams: tuple[KernelStreamBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "streams", tuple(self.streams))
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("Kernel realization requires model-free module requirements")
        if any(not isinstance(item, KernelStreamBinding) for item in self.streams):
            raise TypeError("Kernel realization requires authored stream bindings")


@dataclass(frozen=True, slots=True)
class SemanticPortBinding:
    node_id: str
    instance_id: str
    local: KernelStreamBinding

    def __post_init__(self) -> None:
        if not self.node_id or not self.instance_id:
            raise ValueError("semantic and physical instance identities must be non-empty")
        if not isinstance(self.local, KernelStreamBinding):
            raise TypeError("a semantic port binding contains one KernelStreamBinding")


@dataclass(frozen=True, slots=True)
class BoundaryBinding:
    boundary_id: str
    top_bus_id: str
    payload: PackedBeatLayout
    instance_id: str
    child_bus_id: str

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.boundary_id,
                self.top_bus_id,
                self.instance_id,
                self.child_bus_id,
            )
        ):
            raise ValueError("boundary binding identities must be non-empty")
        if not isinstance(self.payload, PackedBeatLayout):
            raise TypeError("a boundary binding contains one PackedBeatLayout")


@dataclass(frozen=True, slots=True)
class EdgeBinding:
    edge_id: str
    source_instance: str
    source_bus: str
    sink_instance: str
    sink_bus: str

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.edge_id,
                self.source_instance,
                self.source_bus,
                self.sink_instance,
                self.sink_bus,
            )
        ):
            raise ValueError("edge binding identities must be non-empty")


@dataclass(frozen=True, slots=True)
class CompositePhysicalFacts:
    requirements: ModuleBuildRequirements
    port_bindings: tuple[SemanticPortBinding, ...]
    boundary_bindings: tuple[BoundaryBinding, ...]
    edge_bindings: tuple[EdgeBinding, ...]
    structure: PhysicalStructure | None = dataclass_field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "port_bindings", tuple(self.port_bindings))
        object.__setattr__(self, "boundary_bindings", tuple(self.boundary_bindings))
        object.__setattr__(self, "edge_bindings", tuple(self.edge_bindings))


@dataclass(frozen=True, slots=True)
class LogicalPhysicalRelation:
    network: DataflowNetwork
    physical: CompositePhysicalFacts

    def __post_init__(self) -> None:
        if not isinstance(self.network, DataflowNetwork):
            raise TypeError("a physical relation contains one DataflowNetwork")
        if not isinstance(self.physical, CompositePhysicalFacts):
            raise TypeError("a physical relation contains CompositePhysicalFacts")


__all__ = [
    "BoundaryBinding",
    "CompositePhysicalFacts",
    "EdgeBinding",
    "KernelRealizationFacts",
    "KernelStreamBinding",
    "LogicalPhysicalRelation",
    "SemanticPortBinding",
]
