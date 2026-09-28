# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete generator interfaces and packed stream construction data."""

from __future__ import annotations

from dataclasses import dataclass, field
from finn.kernels.artifacts.abi import Bus, Endpoint, StandardProtocol
from finn.kernels.artifacts.build import ModuleABIRequirements, ModuleBuildRequirements
from finn.parked.dataflow.logical_values.region import DataflowRegion, InputInterface, Port, element_width
from finn.kernels.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    PeriodicLast,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.kernels.physical.structure import PhysicalStructure


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


@dataclass(frozen=True)
class PhysicalPort:
    """An actual top-level stream presents this public operand role."""

    role: str
    bus_id: str
    payload: PackedBeatLayout
    framing: PeriodicLast | None = None

    def __post_init__(self) -> None:
        if type(self.role) is not str or type(self.bus_id) is not str:
            raise TypeError("physical port identities must be strings")
        if not self.role or not self.bus_id:
            raise ValueError("physical ports require public role and actual bus identities")
        if not isinstance(self.payload, PackedBeatLayout):
            raise TypeError("physical ports require an immutable packed layout")
        if self.framing is not None and not isinstance(self.framing, PeriodicLast):
            raise TypeError("physical framing must be PeriodicLast or None")


@dataclass(frozen=True)
class PhysicalResult:
    """Detached codegen requirements and the concrete interface needed by users."""

    requirements: ModuleBuildRequirements
    ports: tuple[PhysicalPort, ...]
    required_values: tuple[str, ...] = ()
    structure: PhysicalStructure | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "required_values", tuple(self.required_values))
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("a physical result requires detached module requirements")
        if any(not isinstance(port, PhysicalPort) for port in self.ports):
            raise TypeError("physical ports must be PhysicalPort records")
        roles = tuple(port.role for port in self.ports) + self.required_values
        if len(set(roles)) != len(roles) or any(
            type(role) is not str or not role for role in roles
        ):
            raise ValueError("physical operand roles must be nonempty and unique")
        buses = {port.name: port for port in self.requirements.abi.ports if isinstance(port, Bus)}
        if len({port.bus_id for port in self.ports}) != len(self.ports):
            raise ValueError("physical ports bind each bus once")
        if {port.bus_id for port in self.ports} != set(buses):
            raise ValueError("physical ports must cover every top data bus")
        for port in self.ports:
            validate_payload(buses[port.bus_id], port.payload, port.framing)


def validate_payload(bus: Bus, layout: PackedBeatLayout, framing: PeriodicLast | None) -> None:
    """Check supported stream protocol, sidebands and every packed carrier bit."""
    if bus.protocol is not StandardProtocol.AXIS:
        raise ValueError("physical stream requires AXIS")
    if not layout.fields:
        raise ValueError("a public physical stream must present at least one value field")
    members = {member.logical: member for member in bus.signals}
    if set(members) != {"tdata", "tvalid", "tready"} | (
        {"tlast"} if framing is not None else set()
    ):
        raise ValueError("unsupported or unbound stream sideband")
    if any(member.width != 1 for name, member in members.items() if name != "tdata"):
        raise ValueError("handshake and framing members must be one bit")
    if sorted(item.field_index for item in layout.fields) != list(range(len(layout.fields))):
        raise ValueError("payload must contain every ordered field once")
    occupied: set[int] = set()
    spans: tuple[FieldPlacement | UnusedBitRange, ...] = (*layout.fields, *layout.unused)
    for span in spans:
        bits = set(range(span.bit_offset, span.bit_offset + span.bit_width))
        if not bits or max(bits) >= members["tdata"].width or occupied & bits:
            raise ValueError("payload ranges overlap or exceed carrier")
        occupied |= bits
    if occupied != set(range(members["tdata"].width)):
        raise ValueError("every carrier bit needs an explicit field/padding disposition")
    policies = (
        (UnusedBitPolicy.IGNORE_ON_RECEIVE,)
        if bus.endpoint is Endpoint.TARGET
        else (UnusedBitPolicy.DRIVE_ZERO, UnusedBitPolicy.UNSPECIFIED)
    )
    if any(item.policy not in policies for item in layout.unused):
        raise ValueError("padding policy disagrees with stream direction")


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
    if len({item.region_port_id for item in bindings}) != len(bindings):
        raise ValueError("duplicate logical port binding")
    if len({item.abi_bus_id for item in bindings}) != len(bindings):
        raise ValueError("duplicate physical bus binding")
    if {item.region_port_id for item in bindings} != set(ports):
        raise ValueError("stream bindings must cover every logical port exactly once")
    if {item.abi_bus_id for item in bindings} != set(buses):
        raise ValueError("stream bindings must cover every physical bus exactly once")
    for binding in bindings:
        port, direction = ports[binding.region_port_id]
        bus = buses[binding.abi_bus_id]
        if bus.protocol is not StandardProtocol.AXIS or bus.endpoint is not direction:
            raise ValueError("logical port direction/protocol disagrees with physical bus")
        members = {member.logical: member for member in bus.signals}
        if set(members) != {"tdata", "tvalid", "tready"} | (
            {"tlast"} if binding.framing is not None else set()
        ):
            raise ValueError("unsupported or unbound stream sideband")
        if any(members[name].width != 1 for name in set(members) - {"tdata"}):
            raise ValueError("stream handshake/framing pins must be one bit")
        width = members["tdata"].width
        fields = binding.payload.fields
        if sorted(item.field_index for item in fields) != list(
            range(port.beat_sequence.elements_per_beat)
        ):
            raise ValueError("payload must bind each ordered logical field exactly once")
        if any(item.bit_width != element_width(port.operand.element_type) for item in fields):
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
        expected_policies = (
            (UnusedBitPolicy.IGNORE_ON_RECEIVE,)
            if direction is Endpoint.TARGET
            else (UnusedBitPolicy.DRIVE_ZERO, UnusedBitPolicy.UNSPECIFIED)
        )
        if any(item.policy not in expected_policies for item in binding.payload.unused):
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
    """Author a low-field-first layout with explicit high-padding policy."""

    port, direction = next(
        (port, side) for port, side in region_ports(region) if port.id == region_port_id
    )
    bus = next(port for port in abi.ports if isinstance(port, Bus) and port.name == abi_bus_id)
    carrier = next(member.width for member in bus.signals if member.logical == "tdata")
    scalar = element_width(port.operand.element_type)
    logical = scalar * port.beat_sequence.elements_per_beat
    if logical > carrier:
        raise ValueError("logical payload exceeds physical carrier")
    return KernelStreamBinding(
        region_port_id,
        abi_bus_id,
        PackedBeatLayout(
            tuple(
                FieldPlacement(index, index * scalar, scalar)
                for index in range(port.beat_sequence.elements_per_beat)
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


__all__ = [
    "KernelStreamBinding",
    "KernelRealizationFacts",
    "PhysicalPort",
    "PhysicalResult",
    "region_ports",
    "validate_kernel_stream_bindings",
    "validate_payload",
    "low_fields_binding",
]
