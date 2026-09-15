# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-authored logical port to physical stream bindings.

These values retain local semantic port names for compiler correspondence.
Only their checked, model-free physical structure reaches artifact preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from finn.dataflow._engine import Decided
from finn.dataflow.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.dataflow.artifacts.build import ModuleABIRequirements, ModuleBuildRequirements
from finn.dataflow.model.region import DataflowRegion, InputInterface, Port, element_width

if TYPE_CHECKING:
    from finn.dataflow.kernels.kernel import Kernel  # noqa: PLC0415


def _natural(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} integer")


class UnusedBitPolicy(Enum):
    DRIVE_ZERO = "drive_zero"
    IGNORE_ON_RECEIVE = "ignore_on_receive"


@dataclass(frozen=True)
class FieldPlacement:
    field_index: int
    bit_offset: int
    bit_width: int

    def __post_init__(self) -> None:
        _natural(self.field_index, "field index")
        _natural(self.bit_offset, "field offset")
        _natural(self.bit_width, "field width", positive=True)


@dataclass(frozen=True)
class UnusedBitRange:
    bit_offset: int
    bit_width: int
    policy: UnusedBitPolicy

    def __post_init__(self) -> None:
        _natural(self.bit_offset, "unused offset")
        _natural(self.bit_width, "unused width", positive=True)
        if not isinstance(self.policy, UnusedBitPolicy):
            raise TypeError("unused bits require an explicit policy")


@dataclass(frozen=True)
class PackedBeatLayout:
    fields: tuple[FieldPlacement, ...]
    unused: tuple[UnusedBitRange, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "unused", tuple(self.unused))
        if any(not isinstance(x, FieldPlacement) for x in self.fields):
            raise TypeError("payload fields must be FieldPlacement values")
        if any(not isinstance(x, UnusedBitRange) for x in self.unused):
            raise TypeError("payload padding must be UnusedBitRange values")


@dataclass(frozen=True)
class PeriodicLast:
    member: str
    period_beats: int
    asserted_index: int

    def __post_init__(self) -> None:
        if self.member != "tlast":
            raise ValueError("first-profile framing requires tlast")
        _natural(self.period_beats, "last period", positive=True)
        _natural(self.asserted_index, "last index")
        if self.asserted_index >= self.period_beats:
            raise ValueError("last index must be within period")


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


def region_ports(region: DataflowRegion) -> tuple[tuple[Port, Endpoint], ...]:
    return tuple(
        (item.port, Endpoint.TARGET) for item in region.inputs if isinstance(item, InputInterface)
    ) + tuple((item.port, Endpoint.INITIATOR) for item in region.outputs)


def validate_kernel_stream_bindings(
    region: DataflowRegion,
    abi: ModuleABIRequirements,
    bindings: tuple[KernelStreamBinding, ...],
) -> None:
    """Require full logical coverage and a partition of every payload carrier."""
    ports = {port.id: (port, direction) for port, direction in region_ports(region)}
    buses = {port.name: port for port in abi.ports if isinstance(port, Bus)}
    if len({x.region_port_id for x in bindings}) != len(bindings):
        raise ValueError("duplicate logical port binding")
    if len({x.abi_bus_id for x in bindings}) != len(bindings):
        raise ValueError("duplicate physical bus binding")
    if {x.region_port_id for x in bindings} != set(ports):
        raise ValueError("stream bindings must cover every logical port exactly once")
    if {x.abi_bus_id for x in bindings} != set(buses):
        raise ValueError("stream bindings must cover every physical bus exactly once")
    for binding in bindings:
        port, direction = ports[binding.region_port_id]
        bus = buses[binding.abi_bus_id]
        if bus.protocol is not StandardProtocol.AXIS or bus.endpoint is not direction:
            raise ValueError("logical port direction/protocol disagrees with physical bus")
        members = {m.logical: m for m in bus.signals}
        if set(members) != {"tdata", "tvalid", "tready"} | (
            {"tlast"} if binding.framing is not None else set()
        ):
            raise ValueError("unsupported or unbound stream sideband")
        if any(members[m].width != 1 for m in set(members) - {"tdata"}):
            raise ValueError("stream handshake/framing pins must be one bit")
        width = members["tdata"].width
        fields = binding.payload.fields
        if sorted(x.field_index for x in fields) != list(
            range(port.beat_sequence.elements_per_beat)
        ):
            raise ValueError("payload must bind each ordered logical field exactly once")
        if any(x.bit_width != element_width(port.operand.element_type) for x in fields):
            raise ValueError("payload field width disagrees with scalar encoding")
        occupied: set[int] = set()
        spans: tuple[FieldPlacement | UnusedBitRange, ...] = (*fields, *binding.payload.unused)
        for span in spans:
            bits = set(range(span.bit_offset, span.bit_offset + span.bit_width))
            if max(bits) >= width or bits & occupied:
                raise ValueError("payload ranges overlap or exceed carrier")
            occupied |= bits
        if occupied != set(range(width)):
            raise ValueError("payload carrier bits need an explicit field/padding disposition")
        expected_policy = (
            UnusedBitPolicy.IGNORE_ON_RECEIVE
            if direction is Endpoint.TARGET
            else UnusedBitPolicy.DRIVE_ZERO
        )
        if any(x.policy is not expected_policy for x in binding.payload.unused):
            raise ValueError("padding policy disagrees with stream direction")
        if (
            binding.framing is not None
            and port.beat_sequence.beat_count % binding.framing.period_beats
        ):
            raise ValueError("framing period must divide the logical pass")


def low_fields_binding(
    *,
    region: DataflowRegion,
    abi: ModuleABIRequirements,
    region_port_id: str,
    abi_bus_id: str,
    framing: PeriodicLast | None = None,
) -> KernelStreamBinding:
    """Author a low-field-first layout with an explicit high-padding policy."""
    port, direction = next((p, d) for p, d in region_ports(region) if p.id == region_port_id)
    bus = next(p for p in abi.ports if isinstance(p, Bus) and p.name == abi_bus_id)
    carrier = next(m.width for m in bus.signals if m.logical == "tdata")
    scalar = element_width(port.operand.element_type)
    logical = scalar * port.beat_sequence.elements_per_beat
    if logical > carrier:
        raise ValueError("logical payload exceeds physical carrier")
    return KernelStreamBinding(
        region_port_id,
        abi_bus_id,
        PackedBeatLayout(
            tuple(
                FieldPlacement(i, i * scalar, scalar)
                for i in range(port.beat_sequence.elements_per_beat)
            ),
            ()
            if logical == carrier
            else (
                UnusedBitRange(
                    logical,
                    carrier - logical,
                    UnusedBitPolicy.IGNORE_ON_RECEIVE
                    if direction is Endpoint.TARGET
                    else UnusedBitPolicy.DRIVE_ZERO,
                ),
            ),
        ),
        framing,
    )


def capture_kernel_realization(kernel: Kernel) -> KernelRealizationFacts:
    """Capture requirements and authored mappings from the same child point."""
    from finn.dataflow.kernels.kernel import PhysicallyUnsupported  # noqa: PLC0415

    physical = kernel.physical.accepted_answer
    if not isinstance(physical, Decided):
        raise PhysicallyUnsupported(f"Kernel physical projection is not accepted: {physical}")
    bindings = kernel.answer(type(kernel).physical_streams)
    if not isinstance(bindings, Decided):
        raise PhysicallyUnsupported(f"Kernel stream bindings are not available: {bindings}")
    return KernelRealizationFacts(physical.value, bindings.value)
