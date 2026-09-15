# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Checked physical composition below a selected :class:`DataflowDesign`.

The values in this module divide two responsibilities deliberately.  A
``DesignPhysicalFacts`` retains the semantic correspondence needed by the Op
that selected the Design.  Its ``ModuleBuildRequirements`` contains only the
physical, reusable component description that artifact preparation may see.

The first supported composition is the external-weight decomposed dot product:
one ReplayBuffer feeding one DotpAxi.  It is intentionally strict.  There is no
generic fan-out, width converter, clock generator, or implicit Verilog sizing
hidden in the lowerer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessAssessment,
    Unresolved,
)
from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Data,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
)
from finn.dataflow.artifacts.contributions import CopiedSource, DataSlot
from finn.dataflow.artifacts.derivation import ProducerIdentity, Scalar
from finn.dataflow.kernels.kernel import PhysicallyUnsupported
from finn.dataflow.kernels.physical import (
    FieldPlacement,
    KernelRealizationFacts,
    KernelStreamBinding,
    PackedBeatLayout,
    PeriodicLast,
    UnusedBitPolicy,
    UnusedBitRange,
    capture_kernel_realization,
    validate_kernel_stream_bindings,
)
from finn.dataflow.model.network import (
    DataflowNetwork,
    DirectConnection,
    PassCorrespondence,
    PositionMap,
)
from finn.dataflow.model.region import InputInterface, Port, element_width
from finn.dataflow.space.occurrence import ProjectionAssessment, layer_runtime

if TYPE_CHECKING:
    from finn.dataflow.designs.design import DataflowDesign


class PhysicalCompositionError(ValueError):
    """A selected physical structure cannot realize its logical contract."""


def _identity(value: str, label: str) -> None:
    if not value:
        raise PhysicalCompositionError(f"{label} must be non-empty")


def _natural(value: int, label: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        kind = "positive" if positive else "nonnegative"
        raise PhysicalCompositionError(f"{label} must be a {kind} integer")


@dataclass(frozen=True, slots=True)
class SemanticPortBinding:
    node_id: str
    instance_id: str
    local: KernelStreamBinding

    def __post_init__(self) -> None:
        _identity(self.node_id, "semantic node id")
        _identity(self.instance_id, "physical instance id")
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
        for value, label in (
            (self.boundary_id, "boundary id"),
            (self.top_bus_id, "top bus id"),
            (self.instance_id, "physical instance id"),
            (self.child_bus_id, "child bus id"),
        ):
            _identity(value, label)
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
        for value, label in (
            (self.edge_id, "edge id"),
            (self.source_instance, "source instance"),
            (self.source_bus, "source bus"),
            (self.sink_instance, "sink instance"),
            (self.sink_bus, "sink bus"),
        ):
            _identity(value, label)


@dataclass(frozen=True, slots=True)
class PhysicalPin:
    instance_id: str | None
    signal_id: str

    def __post_init__(self) -> None:
        if self.instance_id == "":
            raise PhysicalCompositionError("a child pin has a non-empty instance id")
        _identity(self.signal_id, "physical signal id")


@dataclass(frozen=True, slots=True)
class PinSlice:
    pin: PhysicalPin
    bit_offset: int
    bit_width: int

    def __post_init__(self) -> None:
        if not isinstance(self.pin, PhysicalPin):
            raise TypeError("a pin slice names one PhysicalPin")
        _natural(self.bit_offset, "pin-slice offset")
        _natural(self.bit_width, "pin-slice width", positive=True)


@dataclass(frozen=True, slots=True)
class ConstantBits:
    bit_width: int
    value: int

    def __post_init__(self) -> None:
        _natural(self.bit_width, "constant width", positive=True)
        _natural(self.value, "constant value")
        if self.value >= 1 << self.bit_width:
            raise PhysicalCompositionError("a constant value must fit its declared width")


@dataclass(frozen=True, slots=True)
class PhysicalWire:
    destination: PinSlice
    source: PinSlice | ConstantBits
    invert: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.destination, PinSlice):
            raise TypeError("a physical wire destination is one PinSlice")
        if not isinstance(self.source, (PinSlice, ConstantBits)):
            raise TypeError("a physical wire source is one PinSlice or ConstantBits")
        if self.destination.bit_width != self.source.bit_width:
            raise PhysicalCompositionError("a physical wire connects equal-width slices")
        if self.invert and (
            not isinstance(self.source, PinSlice) or self.destination.bit_width != 1
        ):
            raise PhysicalCompositionError("only a one-bit pin-to-pin wire may invert")


@dataclass(frozen=True, slots=True)
class ModuleInstance:
    instance_id: str
    requirements: ModuleBuildRequirements

    def __post_init__(self) -> None:
        _identity(self.instance_id, "module instance id")
        if not isinstance(self.requirements, ModuleBuildRequirements):
            raise TypeError("a module instance contains ModuleBuildRequirements")


@dataclass(frozen=True, slots=True)
class UnusedOutput:
    pin: PhysicalPin
    reason: str

    def __post_init__(self) -> None:
        if self.pin.instance_id is None:
            raise PhysicalCompositionError("only a child output may be explicitly unused")
        _identity(self.reason, "unused-output reason")


@dataclass(frozen=True, slots=True)
class PhysicalStructure:
    top_abi: ModuleABIRequirements
    instances: tuple[ModuleInstance, ...]
    wires: tuple[PhysicalWire, ...]
    unused_outputs: tuple[UnusedOutput, ...]
    ignored_top_input_bits: tuple[PinSlice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "wires", tuple(self.wires))
        object.__setattr__(self, "unused_outputs", tuple(self.unused_outputs))
        object.__setattr__(self, "ignored_top_input_bits", tuple(self.ignored_top_input_bits))
        validate_physical_structure(self)


@dataclass(frozen=True, slots=True)
class DesignPhysicalFacts:
    requirements: ModuleBuildRequirements
    port_bindings: tuple[SemanticPortBinding, ...]
    boundary_bindings: tuple[BoundaryBinding, ...]
    edge_bindings: tuple[EdgeBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "port_bindings", tuple(self.port_bindings))
        object.__setattr__(self, "boundary_bindings", tuple(self.boundary_bindings))
        object.__setattr__(self, "edge_bindings", tuple(self.edge_bindings))


@dataclass(frozen=True, slots=True)
class _PinInfo:
    direction: Direction
    width: int
    role: object
    bus_id: str | None = None
    member: str | None = None


def _abi_pins(abi: ModuleABIRequirements) -> Mapping[str, _PinInfo]:
    pins: dict[str, _PinInfo] = {}
    for port in abi.ports:
        if isinstance(port, Signal):
            pins[port.name] = _PinInfo(port.direction, port.width, port.role)
            continue
        directions = dict(port.member_directions())
        for member in port.signals:
            pins[member.physical] = _PinInfo(
                directions[member.physical], member.width, port.role, port.name, member.logical
            )
    return MappingProxyType(pins)


def _validate_bus_domains(abi: ModuleABIRequirements) -> None:
    clocks = {
        port.name for port in abi.ports if isinstance(port, Signal) and isinstance(port.role, Clock)
    }
    resets = {
        port.name for port in abi.ports if isinstance(port, Signal) and isinstance(port.role, Reset)
    }
    for port in abi.ports:
        if not isinstance(port, Bus):
            continue
        if port.associated_clock not in clocks:
            raise PhysicalCompositionError(f"bus {port.name!r} has no declared associated clock")
        if port.associated_reset not in resets:
            raise PhysicalCompositionError(f"bus {port.name!r} has no declared associated reset")


def _slice_bits(value: PinSlice, info: _PinInfo) -> set[int]:
    end = value.bit_offset + value.bit_width
    if end > info.width:
        raise PhysicalCompositionError(
            f"slice {value.pin.signal_id}[{end - 1}:{value.bit_offset}] exceeds "
            f"its {info.width}-bit pin"
        )
    return set(range(value.bit_offset, end))


def _pin_info(
    pin: PhysicalPin,
    *,
    top: Mapping[str, _PinInfo],
    children: Mapping[str, Mapping[str, _PinInfo]],
) -> _PinInfo:
    inventory = top if pin.instance_id is None else children.get(pin.instance_id)
    if inventory is None or pin.signal_id not in inventory:
        owner = "top" if pin.instance_id is None else pin.instance_id
        raise PhysicalCompositionError(f"{owner!r} has no physical pin {pin.signal_id!r}")
    return inventory[pin.signal_id]


def _is_source(pin: PhysicalPin, info: _PinInfo) -> bool:
    return (
        info.direction is Direction.IN
        if pin.instance_id is None
        else info.direction is Direction.OUT
    )


def _is_destination(pin: PhysicalPin, info: _PinInfo) -> bool:
    return (
        info.direction is Direction.OUT
        if pin.instance_id is None
        else info.direction is Direction.IN
    )


def _all_bits(pin: PhysicalPin, info: _PinInfo) -> set[tuple[PhysicalPin, int]]:
    return {(pin, bit) for bit in range(info.width)}


def validate_physical_structure(structure: PhysicalStructure) -> None:
    """Check total physical coverage and every direction/width relation."""

    if not isinstance(structure.top_abi, ModuleABIRequirements):
        raise TypeError("a physical structure has ModuleABIRequirements for its top")
    instance_ids = tuple(instance.instance_id for instance in structure.instances)
    if len(instance_ids) != len(set(instance_ids)):
        raise PhysicalCompositionError("a physical structure names one instance twice")
    if any(
        not isinstance(instance.requirements.abi.entry_point, FixedModuleName)
        for instance in structure.instances
    ):
        raise PhysicalCompositionError(
            "the first composition profile instantiates fixed-name child modules"
        )

    top = _abi_pins(structure.top_abi)
    _validate_bus_domains(structure.top_abi)
    children = {
        instance.instance_id: _abi_pins(instance.requirements.abi)
        for instance in structure.instances
    }
    for instance in structure.instances:
        _validate_bus_domains(instance.requirements.abi)
    destinations: set[tuple[PhysicalPin, int]] = set()
    sources: dict[tuple[PhysicalPin, int], int] = {}

    for wire in structure.wires:
        destination_info = _pin_info(wire.destination.pin, top=top, children=children)
        if not _is_destination(wire.destination.pin, destination_info):
            raise PhysicalCompositionError(
                f"wire destination {wire.destination.pin} is not driven by the wrapper"
            )
        destination_bits = _slice_bits(wire.destination, destination_info)
        qualified_destination = {(wire.destination.pin, bit) for bit in destination_bits}
        overlap = destinations & qualified_destination
        if overlap:
            raise PhysicalCompositionError("a physical destination bit has more than one driver")
        destinations |= qualified_destination

        if isinstance(wire.source, PinSlice):
            source_info = _pin_info(wire.source.pin, top=top, children=children)
            if not _is_source(wire.source.pin, source_info):
                raise PhysicalCompositionError(
                    f"wire source {wire.source.pin} is not driven toward the wrapper"
                )
            source_bits = _slice_bits(wire.source, source_info)
            for bit in source_bits:
                qualified_source = (wire.source.pin, bit)
                sources[qualified_source] = sources.get(qualified_source, 0) + 1
            if wire.invert and not (
                isinstance(source_info.role, Reset) and isinstance(destination_info.role, Reset)
            ):
                raise PhysicalCompositionError("only reset-polarity routing may invert")
            if isinstance(source_info.role, Reset) and isinstance(destination_info.role, Reset):
                polarity_changes = source_info.role.active_low != destination_info.role.active_low
                if wire.invert != polarity_changes:
                    raise PhysicalCompositionError("reset polarity and inversion disagree")

    ignored: set[tuple[PhysicalPin, int]] = set()
    for ignored_slice in structure.ignored_top_input_bits:
        if ignored_slice.pin.instance_id is not None:
            raise PhysicalCompositionError("ignored input padding belongs to the composed top")
        info = _pin_info(ignored_slice.pin, top=top, children=children)
        if (
            info.direction is not Direction.IN
            or info.member != "tdata"
            or not isinstance(info.role, Data)
        ):
            raise PhysicalCompositionError("only top input payload padding may be ignored")
        qualified_ignored = {(ignored_slice.pin, bit) for bit in _slice_bits(ignored_slice, info)}
        if ignored & qualified_ignored:
            raise PhysicalCompositionError("ignored top input ranges overlap")
        if any(bit in sources for bit in qualified_ignored):
            raise PhysicalCompositionError("an ignored top input bit is also consumed")
        ignored |= qualified_ignored

    unused: set[PhysicalPin] = set()
    for disposition in structure.unused_outputs:
        info = _pin_info(disposition.pin, top=top, children=children)
        if disposition.pin.instance_id is None or info.direction is not Direction.OUT:
            raise PhysicalCompositionError("an unused endpoint must be one child output pin")
        if disposition.pin in unused:
            raise PhysicalCompositionError("a child output is disposed more than once")
        if any((disposition.pin, bit) in sources for bit in range(info.width)):
            raise PhysicalCompositionError("a used child output cannot also be disposed")
        unused.add(disposition.pin)

    required_destinations: set[tuple[PhysicalPin, int]] = set()
    required_sources: set[tuple[PhysicalPin, int]] = set()
    for name, info in top.items():
        pin = PhysicalPin(None, name)
        if info.direction is Direction.OUT:
            required_destinations |= _all_bits(pin, info)
        elif info.direction is Direction.IN:
            required_sources |= _all_bits(pin, info)
        else:
            raise PhysicalCompositionError("the first profile does not support inout top pins")
    for instance_id, inventory in children.items():
        for name, info in inventory.items():
            pin = PhysicalPin(instance_id, name)
            if info.direction is Direction.IN:
                required_destinations |= _all_bits(pin, info)
            elif info.direction is Direction.OUT:
                if pin not in unused:
                    required_sources |= _all_bits(pin, info)
            else:
                raise PhysicalCompositionError(
                    "the first profile does not support inout child pins"
                )

    if destinations != required_destinations:
        missing = required_destinations - destinations
        extra = destinations - required_destinations
        raise PhysicalCompositionError(
            "physical destination coverage is incomplete "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    accounted_sources = set(sources) | ignored
    if accounted_sources != required_sources:
        missing = required_sources - accounted_sources
        extra = accounted_sources - required_sources
        raise PhysicalCompositionError(
            f"physical source coverage is incomplete (missing={len(missing)}, extra={len(extra)})"
        )
    for qualified, count in sources.items():
        info = _pin_info(qualified[0], top=top, children=children)
        if count > 1 and not isinstance(info.role, (Clock, Reset)):
            raise PhysicalCompositionError("the first profile does not physically fan out data")


def _bus(abi: ModuleABIRequirements, name: str, endpoint: Endpoint) -> Bus:
    matches = tuple(
        port
        for port in abi.ports
        if isinstance(port, Bus) and port.name == name and port.endpoint is endpoint
    )
    if len(matches) != 1 or matches[0].protocol is not StandardProtocol.AXIS:
        raise PhysicalCompositionError(f"expected one {endpoint.value} AXIS bus {name!r}")
    return matches[0]


def _member(bus: Bus, logical: str) -> Member:
    matches = tuple(member for member in bus.signals if member.logical == logical)
    if len(matches) != 1:
        raise PhysicalCompositionError(f"AXIS bus {bus.name!r} must have one {logical!r} member")
    return matches[0]


def _signal(
    abi: ModuleABIRequirements,
    name: str,
    direction: Direction,
    role: type[object],
) -> Signal:
    matches = tuple(
        port
        for port in abi.ports
        if isinstance(port, Signal)
        and port.name == name
        and port.direction is direction
        and isinstance(port.role, role)
    )
    if len(matches) != 1 or matches[0].width != 1:
        raise PhysicalCompositionError(
            f"expected one one-bit {direction.value} {role.__name__} signal {name!r}"
        )
    return matches[0]


def _validate_decomposed_control_contract(
    top_abi: ModuleABIRequirements,
    replay_abi: ModuleABIRequirements,
    compute_abi: ModuleABIRequirements,
) -> None:
    inventories = (
        (
            tuple(port.name for port in top_abi.ports),
            ("ap_clk", "ap_clk2x", "ap_rst_n", "in0_V", "in1_V", "out0_V"),
            "top",
        ),
        (
            tuple(port.name for port in replay_abi.ports),
            ("clk", "rst", "in0", "out0", "ofin"),
            "Replay",
        ),
        (
            tuple(port.name for port in compute_abi.ports),
            (
                "ap_clk",
                "ap_clk2x",
                "ap_rst_n",
                "s_axis_weights",
                "s_axis_input",
                "m_axis_output",
            ),
            "Dotp",
        ),
    )
    for actual_ports, expected_ports, label in inventories:
        if actual_ports != expected_ports:
            raise PhysicalCompositionError(
                f"{label} pin/interface inventory differs from the supported profile"
            )
    bus_members = (
        (
            top_abi,
            {
                "in0_V": {
                    "tdata": "in0_V_tdata",
                    "tvalid": "in0_V_tvalid",
                    "tready": "in0_V_tready",
                },
                "in1_V": {
                    "tdata": "in1_V_tdata",
                    "tvalid": "in1_V_tvalid",
                    "tready": "in1_V_tready",
                },
                "out0_V": {
                    "tdata": "out0_V_tdata",
                    "tvalid": "out0_V_tvalid",
                    "tready": "out0_V_tready",
                },
            },
            "top",
        ),
        (
            replay_abi,
            {
                "in0": {"tdata": "idat", "tvalid": "ivld", "tready": "irdy"},
                "out0": {
                    "tdata": "odat",
                    "tvalid": "ovld",
                    "tready": "ordy",
                    "tlast": "olast",
                },
            },
            "Replay",
        ),
        (
            compute_abi,
            {
                "s_axis_weights": {
                    "tdata": "s_axis_weights_tdata",
                    "tvalid": "s_axis_weights_tvalid",
                    "tready": "s_axis_weights_tready",
                },
                "s_axis_input": {
                    "tdata": "s_axis_input_tdata",
                    "tvalid": "s_axis_input_tvalid",
                    "tready": "s_axis_input_tready",
                    "tlast": "s_axis_input_tlast",
                },
                "m_axis_output": {
                    "tdata": "m_axis_output_tdata",
                    "tvalid": "m_axis_output_tvalid",
                    "tready": "m_axis_output_tready",
                },
            },
            "Dotp",
        ),
    )
    for abi, expected_members, label in bus_members:
        actual_members = {
            port.name: {member.logical: member.physical for member in port.signals}
            for port in abi.ports
            if isinstance(port, Bus)
        }
        if actual_members != expected_members:
            raise PhysicalCompositionError(
                f"{label} stream-member inventory differs from the supported profile"
            )
    if top_abi.entry_point != GeneratedModuleName("finn_mvau_external"):
        raise PhysicalCompositionError("the decomposed top uses its canonical generated name")
    if top_abi.parameters:
        raise PhysicalCompositionError("the decomposed top has no external HDL parameters")
    top_clock = _signal(top_abi, "ap_clk", Direction.IN, Clock)
    top_clock2x = _signal(top_abi, "ap_clk2x", Direction.IN, Clock)
    top_reset = _signal(top_abi, "ap_rst_n", Direction.IN, Reset)
    expected_reset = Reset(
        active_low=True,
        synchronous=True,
        synchronous_to=("ap_clk", "ap_clk2x"),
    )
    if top_clock.role != Clock(Free()) or top_clock2x.role != Clock(Derived("ap_clk", 2)):
        raise PhysicalCompositionError("the top requires free ap_clk and derived ap_clk2x")
    if top_reset.role != expected_reset:
        raise PhysicalCompositionError("the top reset must be synchronous to both clocks")
    if top_abi.clock_alignments != (ClockAlignment("ap_clk", "ap_clk2x"),):
        raise PhysicalCompositionError("the top requires the aligned-2x-v1 clock relation")

    replay_clock = _signal(replay_abi, "clk", Direction.IN, Clock)
    replay_reset = _signal(replay_abi, "rst", Direction.IN, Reset)
    if replay_clock.role != Clock(Free()) or replay_abi.clock_alignments:
        raise PhysicalCompositionError("Replay requires one free, unaligned clock")
    if replay_reset.role != Reset(active_low=False, synchronous=True, synchronous_to=("clk",)):
        raise PhysicalCompositionError("Replay reset must be active-high synchronous to clk")

    compute_clock = _signal(compute_abi, "ap_clk", Direction.IN, Clock)
    compute_clock2x = _signal(compute_abi, "ap_clk2x", Direction.IN, Clock)
    compute_reset = _signal(compute_abi, "ap_rst_n", Direction.IN, Reset)
    if compute_clock.role != Clock(Free()) or compute_clock2x.role != Clock(Derived("ap_clk", 2)):
        raise PhysicalCompositionError("Dotp requires free ap_clk and derived ap_clk2x")
    if compute_reset.role != expected_reset:
        raise PhysicalCompositionError(
            "Dotp reset must be active-low synchronous to ap_clk and ap_clk2x"
        )
    if compute_abi.clock_alignments != (ClockAlignment("ap_clk", "ap_clk2x"),):
        raise PhysicalCompositionError("Dotp requires the aligned-2x-v1 clock relation")

    for abi, clock, reset, label in (
        (top_abi, "ap_clk", "ap_rst_n", "top"),
        (replay_abi, "clk", "rst", "Replay"),
        (compute_abi, "ap_clk", "ap_rst_n", "Dotp"),
    ):
        if any(
            isinstance(port, Bus)
            and (port.associated_clock != clock or port.associated_reset != reset)
            for port in abi.ports
        ):
            raise PhysicalCompositionError(
                f"every {label} stream must use its declared base clock and reset"
            )


def _stream(facts: KernelRealizationFacts, port_id: str) -> KernelStreamBinding:
    matches = tuple(item for item in facts.streams if item.region_port_id == port_id)
    if len(matches) != 1:
        raise PhysicalCompositionError(f"expected one stream binding for {port_id!r}")
    return matches[0]


def _low_field_layout(layout: PackedBeatLayout, *, label: str) -> int:
    offset = 0
    for index, field in enumerate(sorted(layout.fields, key=lambda item: item.field_index)):
        if field.field_index != index or field.bit_offset != offset:
            raise PhysicalCompositionError(
                f"{label} requires low, field-order-preserving payload placement"
            )
        offset += field.bit_width
    return offset


def _layout_for_carrier(
    layout: PackedBeatLayout,
    width: int,
    *,
    policy: UnusedBitPolicy,
) -> PackedBeatLayout:
    logical = _low_field_layout(layout, label="the first composition profile")
    if logical > width:
        raise PhysicalCompositionError("logical payload exceeds the composed carrier")
    return PackedBeatLayout(
        layout.fields,
        () if logical == width else (UnusedBitRange(logical, width - logical, policy),),
    )


def _axis(
    name: str,
    width: int,
    endpoint: Endpoint,
) -> Bus:
    return Bus(
        name,
        StandardProtocol.AXIS,
        (
            Member("tdata", f"{name}_tdata", width),
            Member("tvalid", f"{name}_tvalid"),
            Member("tready", f"{name}_tready"),
        ),
        endpoint=endpoint,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


def top_boundary_layout(
    top_abi: ModuleABIRequirements,
    top_bus_id: str,
    local: KernelStreamBinding,
) -> PackedBeatLayout:
    """Place one local logical payload in its declared composed-top carrier."""

    bus = _bus(top_abi, top_bus_id, _bus_endpoint(top_abi, top_bus_id))
    width = _member(bus, "tdata").width
    policy = (
        UnusedBitPolicy.IGNORE_ON_RECEIVE
        if bus.endpoint is Endpoint.TARGET
        else UnusedBitPolicy.DRIVE_ZERO
    )
    return _layout_for_carrier(local.payload, width, policy=policy)


def _bus_endpoint(abi: ModuleABIRequirements, name: str) -> Endpoint:
    matches = tuple(port for port in abi.ports if isinstance(port, Bus) and port.name == name)
    if len(matches) != 1:
        raise PhysicalCompositionError(f"expected one top bus {name!r}")
    return matches[0].endpoint


def _slice(instance: str | None, signal: str, width: int, offset: int = 0) -> PinSlice:
    return PinSlice(PhysicalPin(instance, signal), offset, width)


def _copy_fields(
    destination_instance: str | None,
    destination_signal: str,
    destination: PackedBeatLayout,
    source_instance: str | None,
    source_signal: str,
    source: PackedBeatLayout,
) -> tuple[PhysicalWire, ...]:
    source_fields = {field.field_index: field for field in source.fields}
    destination_fields = {field.field_index: field for field in destination.fields}
    if set(source_fields) != set(destination_fields):
        raise PhysicalCompositionError("connected payloads do not name the same logical fields")
    wires = []
    for index in sorted(source_fields):
        left = source_fields[index]
        right = destination_fields[index]
        if left.bit_width != right.bit_width:
            raise PhysicalCompositionError("connected logical fields have different widths")
        wires.append(
            PhysicalWire(
                _slice(destination_instance, destination_signal, right.bit_width, right.bit_offset),
                _slice(source_instance, source_signal, left.bit_width, left.bit_offset),
            )
        )
    return tuple(wires)


def _drive_target_padding(
    instance: str, signal: str, layout: PackedBeatLayout
) -> tuple[PhysicalWire, ...]:
    wires = []
    for unused in layout.unused:
        if unused.policy is not UnusedBitPolicy.IGNORE_ON_RECEIVE:
            raise PhysicalCompositionError("a child target padding range must ignore on receive")
        wires.append(
            PhysicalWire(
                _slice(instance, signal, unused.bit_width, unused.bit_offset),
                ConstantBits(unused.bit_width, 0),
            )
        )
    return tuple(wires)


def compose_decomposed(
    *, replay: KernelRealizationFacts, compute: KernelRealizationFacts
) -> PhysicalStructure:
    """Compose the checked ReplayBuffer -> DotpAxi external-weight profile."""

    if replay.requirements.implementation_id != "replay_buffer":
        raise PhysicalCompositionError("the replay role requires replay_buffer")
    if compute.requirements.implementation_id != "dotp_axi":
        raise PhysicalCompositionError("the compute role requires external dotp_axi")
    replay_abi = replay.requirements.abi
    compute_abi = compute.requirements.abi
    if replay_abi.entry_point != FixedModuleName("replay_buffer"):
        raise PhysicalCompositionError("the replay child must expose replay_buffer")
    if compute_abi.entry_point != FixedModuleName("dotp_axi"):
        raise PhysicalCompositionError("the compute child must expose dotp_axi")

    replay_in = _stream(replay, "activation_in")
    replay_out = _stream(replay, "activation_out")
    compute_activation = _stream(compute, "activation")
    compute_weight = _stream(compute, "weight")
    compute_output = _stream(compute, "output")

    replay_in_bus = _bus(replay_abi, replay_in.abi_bus_id, Endpoint.TARGET)
    replay_out_bus = _bus(replay_abi, replay_out.abi_bus_id, Endpoint.INITIATOR)
    compute_activation_bus = _bus(compute_abi, compute_activation.abi_bus_id, Endpoint.TARGET)
    compute_weight_bus = _bus(compute_abi, compute_weight.abi_bus_id, Endpoint.TARGET)
    compute_output_bus = _bus(compute_abi, compute_output.abi_bus_id, Endpoint.INITIATOR)

    if replay_out.framing is None or replay_out.framing != compute_activation.framing:
        raise PhysicalCompositionError("Replay and Dotp activation framing must match exactly")
    if (
        replay_in.framing is not None
        or compute_weight.framing is not None
        or compute_output.framing is not None
    ):
        raise PhysicalCompositionError("the first profile has framing only on its internal edge")

    activation_width = ((_low_field_layout(replay_in.payload, label="activation") + 7) // 8) * 8
    weight_width = _member(compute_weight_bus, "tdata").width
    output_width = _member(compute_output_bus, "tdata").width
    activation_top_layout = _layout_for_carrier(
        replay_in.payload, activation_width, policy=UnusedBitPolicy.IGNORE_ON_RECEIVE
    )
    weight_top_layout = _layout_for_carrier(
        compute_weight.payload, weight_width, policy=UnusedBitPolicy.IGNORE_ON_RECEIVE
    )
    output_top_layout = _layout_for_carrier(
        compute_output.payload, output_width, policy=UnusedBitPolicy.DRIVE_ZERO
    )
    if output_top_layout.unused:
        raise PhysicalCompositionError("the first profile exposes no unused output bits")

    top_abi = ModuleABIRequirements(
        GeneratedModuleName("finn_mvau_external"),
        (
            Signal("ap_clk", Direction.IN, 1, Clock(Free())),
            Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
            Signal(
                "ap_rst_n",
                Direction.IN,
                1,
                Reset(
                    active_low=True,
                    synchronous=True,
                    synchronous_to=("ap_clk", "ap_clk2x"),
                ),
            ),
            _axis("in0_V", activation_width, Endpoint.TARGET),
            _axis("in1_V", weight_width, Endpoint.TARGET),
            _axis("out0_V", output_width, Endpoint.INITIATOR),
        ),
        (),
        (ClockAlignment("ap_clk", "ap_clk2x"),),
    )
    _validate_decomposed_control_contract(top_abi, replay_abi, compute_abi)

    replay_input_data = _member(replay_in_bus, "tdata").physical
    replay_output_data = _member(replay_out_bus, "tdata").physical
    compute_activation_data = _member(compute_activation_bus, "tdata").physical
    compute_weight_data = _member(compute_weight_bus, "tdata").physical
    compute_output_data = _member(compute_output_bus, "tdata").physical
    wires = (
        PhysicalWire(_slice("u_replay", "clk", 1), _slice(None, "ap_clk", 1)),
        PhysicalWire(_slice("u_replay", "rst", 1), _slice(None, "ap_rst_n", 1), invert=True),
        *_copy_fields(
            "u_replay",
            replay_input_data,
            replay_in.payload,
            None,
            "in0_V_tdata",
            activation_top_layout,
        ),
        PhysicalWire(
            _slice("u_replay", _member(replay_in_bus, "tvalid").physical, 1),
            _slice(None, "in0_V_tvalid", 1),
        ),
        PhysicalWire(
            _slice(None, "in0_V_tready", 1),
            _slice("u_replay", _member(replay_in_bus, "tready").physical, 1),
        ),
        PhysicalWire(_slice("u_compute", "ap_clk", 1), _slice(None, "ap_clk", 1)),
        PhysicalWire(_slice("u_compute", "ap_clk2x", 1), _slice(None, "ap_clk2x", 1)),
        PhysicalWire(_slice("u_compute", "ap_rst_n", 1), _slice(None, "ap_rst_n", 1)),
        *_copy_fields(
            "u_compute",
            compute_weight_data,
            compute_weight.payload,
            None,
            "in1_V_tdata",
            weight_top_layout,
        ),
        *_drive_target_padding("u_compute", compute_weight_data, compute_weight.payload),
        PhysicalWire(
            _slice("u_compute", _member(compute_weight_bus, "tvalid").physical, 1),
            _slice(None, "in1_V_tvalid", 1),
        ),
        PhysicalWire(
            _slice(None, "in1_V_tready", 1),
            _slice("u_compute", _member(compute_weight_bus, "tready").physical, 1),
        ),
        *_copy_fields(
            "u_compute",
            compute_activation_data,
            compute_activation.payload,
            "u_replay",
            replay_output_data,
            replay_out.payload,
        ),
        *_drive_target_padding("u_compute", compute_activation_data, compute_activation.payload),
        PhysicalWire(
            _slice("u_compute", _member(compute_activation_bus, "tvalid").physical, 1),
            _slice("u_replay", _member(replay_out_bus, "tvalid").physical, 1),
        ),
        PhysicalWire(
            _slice("u_replay", _member(replay_out_bus, "tready").physical, 1),
            _slice("u_compute", _member(compute_activation_bus, "tready").physical, 1),
        ),
        PhysicalWire(
            _slice("u_compute", _member(compute_activation_bus, "tlast").physical, 1),
            _slice("u_replay", _member(replay_out_bus, "tlast").physical, 1),
        ),
        *_copy_fields(
            None,
            "out0_V_tdata",
            output_top_layout,
            "u_compute",
            compute_output_data,
            compute_output.payload,
        ),
        PhysicalWire(
            _slice(None, "out0_V_tvalid", 1),
            _slice("u_compute", _member(compute_output_bus, "tvalid").physical, 1),
        ),
        PhysicalWire(
            _slice("u_compute", _member(compute_output_bus, "tready").physical, 1),
            _slice(None, "out0_V_tready", 1),
        ),
    )
    return PhysicalStructure(
        top_abi,
        (
            ModuleInstance("u_replay", replay.requirements),
            ModuleInstance("u_compute", compute.requirements),
        ),
        wires,
        (
            UnusedOutput(
                PhysicalPin("u_replay", "ofin"),
                "olast frames each Dotp accumulation; the composed ABI exposes "
                "no run-complete signal",
            ),
        ),
        tuple(
            _slice(None, bus, unused.bit_width, unused.bit_offset)
            for bus, layout in (
                ("in0_V_tdata", activation_top_layout),
                ("in1_V_tdata", weight_top_layout),
            )
            for unused in layout.unused
        ),
    )


def _port_for(network: DataflowNetwork, node_id: str, port_id: str, *, output: bool) -> Port:
    try:
        region = network.node(node_id).region
        return (
            region.output_interface(port_id).port
            if output
            else region.input_interface(port_id).port
        )
    except KeyError as error:
        raise PhysicalCompositionError(
            f"network endpoint {node_id}.{port_id} is not a {'source' if output else 'sink'}"
        ) from error


def _layout_fields(layout: PackedBeatLayout) -> tuple[tuple[int, int], ...]:
    return tuple(
        (field.field_index, field.bit_width)
        for field in sorted(layout.fields, key=lambda item: item.field_index)
    )


_BitSource = tuple[str, object, bool]


def _wire_map(structure: PhysicalStructure) -> Mapping[tuple[PhysicalPin, int], _BitSource]:
    result: dict[tuple[PhysicalPin, int], _BitSource] = {}
    for wire in structure.wires:
        for offset in range(wire.destination.bit_width):
            destination = (wire.destination.pin, wire.destination.bit_offset + offset)
            if isinstance(wire.source, ConstantBits):
                source: _BitSource = (
                    "constant",
                    (wire.source.value >> offset) & 1,
                    wire.invert,
                )
            else:
                source = (
                    "pin",
                    (wire.source.pin, wire.source.bit_offset + offset),
                    wire.invert,
                )
            result[destination] = source
    return MappingProxyType(result)


def _expect_pin_copy(
    wiring: Mapping[tuple[PhysicalPin, int], _BitSource],
    destination: PhysicalPin,
    source: PhysicalPin,
    *,
    destination_offset: int = 0,
    source_offset: int = 0,
    width: int = 1,
    invert: bool = False,
) -> None:
    for index in range(width):
        actual = wiring.get((destination, destination_offset + index))
        expected: _BitSource = (
            "pin",
            (source, source_offset + index),
            invert,
        )
        if actual != expected:
            raise PhysicalCompositionError(
                f"physical wire for {destination.signal_id}[{destination_offset + index}] "
                "does not preserve the required source bit"
            )


def _expect_field_copy(
    wiring: Mapping[tuple[PhysicalPin, int], _BitSource],
    destination: PhysicalPin,
    destination_layout: PackedBeatLayout,
    source: PhysicalPin,
    source_layout: PackedBeatLayout,
) -> None:
    destination_fields = {item.field_index: item for item in destination_layout.fields}
    source_fields = {item.field_index: item for item in source_layout.fields}
    if set(destination_fields) != set(source_fields):
        raise PhysicalCompositionError("physical payloads name different logical fields")
    for index in sorted(source_fields):
        left = source_fields[index]
        right = destination_fields[index]
        if left.bit_width != right.bit_width:
            raise PhysicalCompositionError("physical payload fields have different widths")
        _expect_pin_copy(
            wiring,
            destination,
            source,
            destination_offset=right.bit_offset,
            source_offset=left.bit_offset,
            width=left.bit_width,
        )


def _expect_zero_padding(
    wiring: Mapping[tuple[PhysicalPin, int], _BitSource],
    destination: PhysicalPin,
    layout: PackedBeatLayout,
) -> None:
    for unused in layout.unused:
        if unused.policy is not UnusedBitPolicy.IGNORE_ON_RECEIVE:
            raise PhysicalCompositionError("a child target's unused bits must ignore on receive")
        for bit in range(unused.bit_offset, unused.bit_offset + unused.bit_width):
            if wiring.get((destination, bit)) != ("constant", 0, False):
                raise PhysicalCompositionError("every unused child input bit is driven to zero")


def _check_payload(
    port: Port,
    bus: Bus,
    layout: PackedBeatLayout,
) -> None:
    width = _member(bus, "tdata").width
    fields = tuple(sorted(layout.fields, key=lambda item: item.field_index))
    if tuple(item.field_index for item in fields) != tuple(
        range(port.beat_sequence.elements_per_beat)
    ):
        raise PhysicalCompositionError("a payload must place every logical field once")
    scalar = element_width(port.operand.element_type)
    if any(item.bit_width != scalar for item in fields):
        raise PhysicalCompositionError("a payload field width differs from its logical scalar")
    occupied: set[int] = set()
    spans: tuple[FieldPlacement | UnusedBitRange, ...] = (*layout.fields, *layout.unused)
    for span in spans:
        bits = set(range(span.bit_offset, span.bit_offset + span.bit_width))
        if not bits or max(bits) >= width or occupied & bits:
            raise PhysicalCompositionError(
                "payload fields and unused ranges must partition a carrier"
            )
        occupied |= bits
    if occupied != set(range(width)):
        raise PhysicalCompositionError("payload fields and unused ranges must cover the carrier")
    policy = (
        UnusedBitPolicy.IGNORE_ON_RECEIVE
        if bus.endpoint is Endpoint.TARGET
        else UnusedBitPolicy.DRIVE_ZERO
    )
    if any(item.policy is not policy for item in layout.unused):
        raise PhysicalCompositionError("payload padding policy disagrees with endpoint direction")


def validate_design_physical_facts(
    network: DataflowNetwork,
    structure: PhysicalStructure,
    facts: DesignPhysicalFacts,
) -> None:
    """Check semantic port, boundary, and edge coverage against one Network."""

    instances = {instance.instance_id: instance for instance in structure.instances}
    if set(instances) != {"u_replay", "u_compute"}:
        raise PhysicalCompositionError("the decomposed profile has u_replay and u_compute")
    _validate_decomposed_control_contract(
        structure.top_abi,
        instances["u_replay"].requirements.abi,
        instances["u_compute"].requirements.abi,
    )
    if facts.requirements != lower_module_structure(
        structure,
        producer=DECOMPOSED_PRODUCER,
        wrapper_template=DECOMPOSED_WRAPPER_TEMPLATE,
    ):
        raise PhysicalCompositionError("Design facts do not contain this structure's requirements")
    ports = tuple(facts.port_bindings)
    if {(item.node_id, item.instance_id) for item in ports} != {
        ("replay", "u_replay"),
        ("compute", "u_compute"),
    }:
        raise PhysicalCompositionError("semantic nodes must map to their canonical instances")
    keys = tuple((item.node_id, item.local.region_port_id) for item in ports)
    if len(keys) != len(set(keys)):
        raise PhysicalCompositionError("a semantic Region port is bound more than once")
    physical_keys = tuple((item.instance_id, item.local.abi_bus_id) for item in ports)
    if len(physical_keys) != len(set(physical_keys)):
        raise PhysicalCompositionError("a physical child bus is bound more than once")
    if {item.node_id for item in ports} != {node.id for node in network.nodes}:
        raise PhysicalCompositionError("semantic bindings and Network nodes differ")
    for item in ports:
        instance = instances.get(item.instance_id)
        if instance is None:
            raise PhysicalCompositionError("a semantic binding names no module instance")
        node = network.node(item.node_id)
        node_bindings = tuple(binding.local for binding in ports if binding.node_id == item.node_id)
        validate_kernel_stream_bindings(node.region, instance.requirements.abi, node_bindings)
        _low_field_layout(item.local.payload, label="the first composition profile")

    by_port = {(item.node_id, item.local.region_port_id): item for item in ports}
    boundaries = {boundary.id: boundary for boundary in network.boundaries}
    supplied_boundaries = {item.boundary_id: item for item in facts.boundary_bindings}
    if len(supplied_boundaries) != len(facts.boundary_bindings) or set(supplied_boundaries) != set(
        boundaries
    ):
        raise PhysicalCompositionError("physical boundaries cover every Network boundary once")
    top_buses = {port.name: port for port in structure.top_abi.ports if isinstance(port, Bus)}
    if {item.top_bus_id for item in facts.boundary_bindings} != set(top_buses):
        raise PhysicalCompositionError("boundary bindings cover every top data bus once")
    for boundary_id, boundary in boundaries.items():
        boundary_binding = supplied_boundaries[boundary_id]
        endpoint = boundary.endpoint
        local = by_port.get((endpoint.node_id, endpoint.port_id))
        if local is None or (boundary_binding.instance_id, boundary_binding.child_bus_id) != (
            local.instance_id,
            local.local.abi_bus_id,
        ):
            raise PhysicalCompositionError("a boundary binding disagrees with its Network endpoint")
        input_port = any(
            isinstance(interface, InputInterface) and interface.port.id == endpoint.port_id
            for interface in network.node(endpoint.node_id).region.inputs
        )
        top_bus = top_buses[boundary_binding.top_bus_id]
        expected_endpoint = Endpoint.TARGET if input_port else Endpoint.INITIATOR
        if top_bus.endpoint is not expected_endpoint:
            raise PhysicalCompositionError("a top bus direction disagrees with its boundary")
        port = _port_for(network, endpoint.node_id, endpoint.port_id, output=not input_port)
        _check_payload(port, top_bus, boundary_binding.payload)
        logical_width = _low_field_layout(
            boundary_binding.payload, label="the first composition profile"
        )
        if _member(top_bus, "tdata").width != ((logical_width + 7) // 8) * 8:
            raise PhysicalCompositionError("a top payload carrier must be byte-aligned exactly")
        if _layout_fields(boundary_binding.payload) != _layout_fields(local.local.payload):
            raise PhysicalCompositionError("a boundary changes logical field order or width")
        if port.beat_sequence != boundary.external_beat_sequence:
            raise PhysicalCompositionError("a physical boundary cannot adapt its logical sequence")
        if boundary.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
            raise PhysicalCompositionError("the first profile requires one-to-one boundaries")

    edges = {edge.id: edge for edge in network.edges}
    supplied_edges = {item.edge_id: item for item in facts.edge_bindings}
    if len(supplied_edges) != len(facts.edge_bindings) or set(supplied_edges) != set(edges):
        raise PhysicalCompositionError("physical edges cover every Network edge once")
    for edge_id, edge in edges.items():
        if len(edge.sinks) != 1:
            raise PhysicalCompositionError("the first physical profile does not implement fan-out")
        if not isinstance(edge.transport, DirectConnection):
            raise PhysicalCompositionError("the first physical profile requires direct transport")
        if edge.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
            raise PhysicalCompositionError("the first physical profile requires one-to-one edges")
        source_binding = by_port.get((edge.source.node_id, edge.source.port_id))
        sink_endpoint = edge.sinks[0].endpoint
        sink_binding = by_port.get((sink_endpoint.node_id, sink_endpoint.port_id))
        if source_binding is None or sink_binding is None:
            raise PhysicalCompositionError("an edge endpoint has no semantic port binding")
        edge_binding = supplied_edges[edge_id]
        if (
            edge_binding.source_instance,
            edge_binding.source_bus,
            edge_binding.sink_instance,
            edge_binding.sink_bus,
        ) != (
            source_binding.instance_id,
            source_binding.local.abi_bus_id,
            sink_binding.instance_id,
            sink_binding.local.abi_bus_id,
        ):
            raise PhysicalCompositionError("a physical edge disagrees with its Network endpoints")
        if _layout_fields(source_binding.local.payload) != _layout_fields(
            sink_binding.local.payload
        ):
            raise PhysicalCompositionError("a physical edge changes logical field order or width")
        if source_binding.local.framing != sink_binding.local.framing:
            raise PhysicalCompositionError("a physical edge changes stream framing")
        source_port = _port_for(network, edge.source.node_id, edge.source.port_id, output=True)
        sink_port = _port_for(network, sink_endpoint.node_id, sink_endpoint.port_id, output=False)
        if source_port.beat_sequence != sink_port.beat_sequence:
            raise PhysicalCompositionError("a physical edge cannot adapt logical beat order")
        expected_map = PositionMap.identity(source_port.beat_sequence.image_set)
        if edge.sinks[0].position_map != expected_map:
            raise PhysicalCompositionError(
                "the first physical profile does not implement a logical map"
            )
        sink_region = network.node(sink_endpoint.node_id).region
        sf_levels = tuple(
            level.extent for level in sink_region.schedule.levels if level.name == "sf"
        )
        if len(sf_levels) != 1 or source_binding.local.framing != PeriodicLast(
            "tlast", sf_levels[0], sf_levels[0] - 1
        ):
            raise PhysicalCompositionError("internal tlast must frame each synapse-fold group")

    wiring = _wire_map(structure)
    ignored = {
        (item.pin, bit)
        for item in structure.ignored_top_input_bits
        for bit in range(item.bit_offset, item.bit_offset + item.bit_width)
    }
    expected_ignored: set[tuple[PhysicalPin, int]] = set()
    for boundary_binding in facts.boundary_bindings:
        top_bus = top_buses[boundary_binding.top_bus_id]
        local_binding = next(
            item
            for item in ports
            if item.instance_id == boundary_binding.instance_id
            and item.local.abi_bus_id == boundary_binding.child_bus_id
        ).local
        child_bus = _bus(
            instances[boundary_binding.instance_id].requirements.abi,
            boundary_binding.child_bus_id,
            top_bus.endpoint,
        )
        top_data = PhysicalPin(None, _member(top_bus, "tdata").physical)
        child_data = PhysicalPin(boundary_binding.instance_id, _member(child_bus, "tdata").physical)
        if top_bus.endpoint is Endpoint.TARGET:
            _expect_field_copy(
                wiring,
                child_data,
                local_binding.payload,
                top_data,
                boundary_binding.payload,
            )
            _expect_zero_padding(wiring, child_data, local_binding.payload)
            for unused in boundary_binding.payload.unused:
                expected_ignored.update(
                    (top_data, bit)
                    for bit in range(unused.bit_offset, unused.bit_offset + unused.bit_width)
                )
            _expect_pin_copy(
                wiring,
                PhysicalPin(boundary_binding.instance_id, _member(child_bus, "tvalid").physical),
                PhysicalPin(None, _member(top_bus, "tvalid").physical),
            )
            _expect_pin_copy(
                wiring,
                PhysicalPin(None, _member(top_bus, "tready").physical),
                PhysicalPin(boundary_binding.instance_id, _member(child_bus, "tready").physical),
            )
        else:
            _expect_field_copy(
                wiring,
                top_data,
                boundary_binding.payload,
                child_data,
                local_binding.payload,
            )
            _expect_pin_copy(
                wiring,
                PhysicalPin(None, _member(top_bus, "tvalid").physical),
                PhysicalPin(boundary_binding.instance_id, _member(child_bus, "tvalid").physical),
            )
            _expect_pin_copy(
                wiring,
                PhysicalPin(boundary_binding.instance_id, _member(child_bus, "tready").physical),
                PhysicalPin(None, _member(top_bus, "tready").physical),
            )
    if ignored != expected_ignored:
        raise PhysicalCompositionError("ignored top bits do not match boundary padding")

    for edge_binding in facts.edge_bindings:
        source_stream = next(
            item.local
            for item in ports
            if item.instance_id == edge_binding.source_instance
            and item.local.abi_bus_id == edge_binding.source_bus
        )
        sink_stream = next(
            item.local
            for item in ports
            if item.instance_id == edge_binding.sink_instance
            and item.local.abi_bus_id == edge_binding.sink_bus
        )
        source_bus = _bus(
            instances[edge_binding.source_instance].requirements.abi,
            edge_binding.source_bus,
            Endpoint.INITIATOR,
        )
        sink_bus = _bus(
            instances[edge_binding.sink_instance].requirements.abi,
            edge_binding.sink_bus,
            Endpoint.TARGET,
        )
        source_data = PhysicalPin(
            edge_binding.source_instance, _member(source_bus, "tdata").physical
        )
        sink_data = PhysicalPin(edge_binding.sink_instance, _member(sink_bus, "tdata").physical)
        _expect_field_copy(
            wiring, sink_data, sink_stream.payload, source_data, source_stream.payload
        )
        _expect_zero_padding(wiring, sink_data, sink_stream.payload)
        _expect_pin_copy(
            wiring,
            PhysicalPin(edge_binding.sink_instance, _member(sink_bus, "tvalid").physical),
            PhysicalPin(edge_binding.source_instance, _member(source_bus, "tvalid").physical),
        )
        _expect_pin_copy(
            wiring,
            PhysicalPin(edge_binding.source_instance, _member(source_bus, "tready").physical),
            PhysicalPin(edge_binding.sink_instance, _member(sink_bus, "tready").physical),
        )
        if source_stream.framing is not None:
            _expect_pin_copy(
                wiring,
                PhysicalPin(
                    edge_binding.sink_instance,
                    _member(sink_bus, source_stream.framing.member).physical,
                ),
                PhysicalPin(
                    edge_binding.source_instance,
                    _member(source_bus, source_stream.framing.member).physical,
                ),
            )

    _expect_pin_copy(wiring, PhysicalPin("u_replay", "clk"), PhysicalPin(None, "ap_clk"))
    _expect_pin_copy(
        wiring,
        PhysicalPin("u_replay", "rst"),
        PhysicalPin(None, "ap_rst_n"),
        invert=True,
    )
    _expect_pin_copy(wiring, PhysicalPin("u_compute", "ap_clk"), PhysicalPin(None, "ap_clk"))
    _expect_pin_copy(wiring, PhysicalPin("u_compute", "ap_clk2x"), PhysicalPin(None, "ap_clk2x"))
    _expect_pin_copy(wiring, PhysicalPin("u_compute", "ap_rst_n"), PhysicalPin(None, "ap_rst_n"))
    expected_unused = {PhysicalPin("u_replay", "ofin")}
    if {item.pin for item in structure.unused_outputs} != expected_unused:
        raise PhysicalCompositionError("only u_replay.ofin may be deliberately open")


_PORT_DECLARATIONS = "PORT_DECLARATIONS"
_NET_DECLARATIONS = "NET_DECLARATIONS"
_ASSIGNMENTS = "ASSIGNMENTS"
_INSTANCES = "INSTANCES"
DECOMPOSED_WRAPPER_TEMPLATE = RenderedSourceRequirement(
    EntryPointSourceName(".sv"),
    "decomposed_wrapper.sv.j2",
    (_PORT_DECLARATIONS, _NET_DECLARATIONS, _ASSIGNMENTS, _INSTANCES),
    SELF_CONTAINED_JINJA_RENDERER,
    requires=("module:dotp_axi", "module:replay_buffer"),
    provides_entry_point=True,
)
DECOMPOSED_PRODUCER = ProducerIdentity("finn.mvau.decomposed.external", "1")


def _sv_width(width: int) -> str:
    return "" if width == 1 else f" [{width - 1}:0]"


def _sv_port_declarations(abi: ModuleABIRequirements) -> str:
    declarations: list[str] = []
    for port in abi.ports:
        if isinstance(port, Signal):
            declarations.append(
                f"    {port.direction.value} logic{_sv_width(port.width)} {port.name}"
            )
            continue
        directions = dict(port.member_directions())
        for member in port.signals:
            declarations.append(
                f"    {directions[member.physical].value} logic"
                f"{_sv_width(member.width)} {member.physical}"
            )
    return ",\n".join(declarations)


def _net_name(pin: PhysicalPin) -> str:
    if pin.instance_id is None:
        return pin.signal_id
    return f"n__{pin.instance_id}__{pin.signal_id}"


def _sv_slice(value: PinSlice, *, pin_width: int) -> str:
    base = _net_name(value.pin)
    if value.bit_width == 1:
        return base if pin_width == 1 else f"{base}[{value.bit_offset}]"
    return f"{base}[{value.bit_offset + value.bit_width - 1}:{value.bit_offset}]"


def _sv_constant(value: ConstantBits) -> str:
    if value.value == 0:
        return f"{value.bit_width}'b" + "0" * value.bit_width
    return f"{value.bit_width}'h{value.value:x}"


def _sv_net_declarations(structure: PhysicalStructure) -> str:
    unused = {item.pin for item in structure.unused_outputs}
    lines = []
    for instance in structure.instances:
        for name, info in _abi_pins(instance.requirements.abi).items():
            pin = PhysicalPin(instance.instance_id, name)
            if pin in unused:
                continue
            lines.append(f"    logic{_sv_width(info.width)} {_net_name(pin)};")
    return "\n".join(lines)


def _sv_assignments(structure: PhysicalStructure) -> str:
    top = _abi_pins(structure.top_abi)
    children = {
        instance.instance_id: _abi_pins(instance.requirements.abi)
        for instance in structure.instances
    }

    def render_slice(value: PinSlice) -> str:
        info = _pin_info(value.pin, top=top, children=children)
        return _sv_slice(value, pin_width=info.width)

    lines = []
    for wire in structure.wires:
        source = (
            render_slice(wire.source)
            if isinstance(wire.source, PinSlice)
            else _sv_constant(wire.source)
        )
        if wire.invert:
            source = f"!{source}"
        lines.append(f"    assign {render_slice(wire.destination)} = {source};")
    return "\n".join(lines)


def _sv_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if hasattr(value, "value"):
        return str(cast(object, value).value)  # type: ignore[attr-defined]
    return str(value)


def _sv_instances(structure: PhysicalStructure) -> str:
    unused = {item.pin for item in structure.unused_outputs}
    blocks = []
    for instance in structure.instances:
        name = cast(FixedModuleName, instance.requirements.abi.entry_point).value
        parameters = instance.requirements.parameters
        parameter_block = ""
        if parameters:
            parameter_block = (
                " #(\n"
                + ",\n".join(f"        .{key}({_sv_scalar(value)})" for key, value in parameters)
                + "\n    )"
            )
        connections = []
        for signal in _abi_pins(instance.requirements.abi):
            pin = PhysicalPin(instance.instance_id, signal)
            connection = "" if pin in unused else _net_name(pin)
            connections.append(f"        .{signal}({connection})")
        blocks.append(
            f"    {name}{parameter_block} {instance.instance_id} (\n"
            + ",\n".join(connections)
            + "\n    );"
        )
    return "\n\n".join(blocks)


def _flatten_contributions(
    instances: Sequence[ModuleInstance],
) -> tuple[CopiedSource, ...]:
    result: list[CopiedSource] = []
    destinations: dict[tuple[str, str], CopiedSource] = {}
    for instance in instances:
        if instance.requirements.render_inputs:
            raise PhysicalCompositionError(
                "the first composition profile does not nest a rendered child"
            )
        for contribution in instance.requirements.contributions:
            if isinstance(contribution, DataSlot):
                raise PhysicalCompositionError(
                    "the external decomposed profile has no child data slot"
                )
            if not isinstance(contribution, CopiedSource):
                raise PhysicalCompositionError(
                    "the first composition profile flattens copied child sources only"
                )
            coordinate = (contribution.library, contribution.path)
            previous = destinations.get(coordinate)
            if previous is not None:
                if previous != contribution:
                    raise PhysicalCompositionError(
                        f"child sources stage incompatible declarations at {coordinate!r}"
                    )
                continue
            destinations[coordinate] = contribution
            result.append(contribution)
    return tuple(result)


def lower_module_structure(
    structure: PhysicalStructure,
    *,
    producer: ProducerIdentity,
    wrapper_template: RenderedSourceRequirement,
) -> ModuleBuildRequirements:
    """Lower one validated structure to model-free generated-module inputs."""

    validate_physical_structure(structure)
    if not isinstance(structure.top_abi.entry_point, GeneratedModuleName):
        raise PhysicalCompositionError("a composed module requires a generated top name")
    if wrapper_template.renderer != SELF_CONTAINED_JINJA_RENDERER:
        raise PhysicalCompositionError("the composed wrapper uses the self-contained renderer")
    if not isinstance(wrapper_template.output, EntryPointSourceName):
        raise PhysicalCompositionError("the composed wrapper output follows its generated name")
    if not wrapper_template.provides_entry_point:
        raise PhysicalCompositionError("the composed wrapper provides the generated entry point")
    expected_arguments = {
        _PORT_DECLARATIONS,
        _NET_DECLARATIONS,
        _ASSIGNMENTS,
        _INSTANCES,
    }
    if set(wrapper_template.arguments) != expected_arguments:
        raise PhysicalCompositionError(
            "the composed wrapper declares the canonical fragment inputs"
        )

    render_inputs: tuple[tuple[str, Scalar], ...] = (
        (_PORT_DECLARATIONS, _sv_port_declarations(structure.top_abi)),
        (_NET_DECLARATIONS, _sv_net_declarations(structure)),
        (_ASSIGNMENTS, _sv_assignments(structure)),
        (_INSTANCES, _sv_instances(structure)),
    )
    return ModuleBuildRequirements(
        producer.producer_id,
        producer.contract_version,
        (),
        structure.top_abi,
        (*_flatten_contributions(structure.instances), wrapper_template),
        render_inputs,
    )


def design_physical_refusal(design: DataflowDesign, reason: str) -> Absent:
    namespace = layer_runtime(design).compiled.namespace
    return Absent(
        (
            Finding(
                FindingKind.REJECTION,
                "design-physically-unsupported",
                QualifiedPath(f"{namespace}.physical"),
                reason,
                (("design", type(design).id or type(design).__name__),),
            ),
        )
    )


def assess_design_physical(design: DataflowDesign) -> ProjectionAssessment[DesignPhysicalFacts]:
    """Evaluate a Design's selected physical hook without touching alternatives."""

    name = f"{layer_runtime(design).compiled.namespace}.physical"
    logical = design.dataflow
    if not isinstance(logical.accepted_answer, Decided):
        blocked = cast("Answer[DesignPhysicalFacts]", logical.accepted_answer)
        return ProjectionAssessment(name, logical.readiness, logical.constraints, blocked, blocked)
    answer = design.physical_implementation()
    if not isinstance(answer, (Decided, Absent, Unresolved)):
        raise TypeError("DataflowDesign.physical_implementation must return an Answer")
    marker = QualifiedPath(name)
    answers = dict(logical.readiness.answers)
    answers[marker] = cast("Answer[object]", answer)
    readiness = ReadinessAssessment(
        name,
        MappingProxyType(dict(sorted(answers.items()))),
        None if isinstance(answer, Unresolved) else True,
    )
    return ProjectionAssessment(name, readiness, logical.constraints, answer, answer)


def selected_kernel_realization(
    design: DataflowDesign, role: str
) -> Answer[KernelRealizationFacts]:
    """Capture one selected child only, preserving its original point answer."""

    selected = design.kernel(role)
    if not isinstance(selected, Decided):
        return cast("Answer[KernelRealizationFacts]", selected)
    kernel = selected.value
    physical = kernel.physical.accepted_answer
    if not isinstance(physical, Decided):
        return cast("Answer[KernelRealizationFacts]", physical)
    streams = kernel.answer(type(kernel).physical_streams)
    if not isinstance(streams, Decided):
        return cast("Answer[KernelRealizationFacts]", streams)
    try:
        return Decided(capture_kernel_realization(kernel))
    except PhysicallyUnsupported as error:  # defensive against a changing child point
        return cast(
            "Answer[KernelRealizationFacts]",
            design_physical_refusal(design, str(error)),
        )


__all__ = [
    "BoundaryBinding",
    "ConstantBits",
    "DECOMPOSED_PRODUCER",
    "DECOMPOSED_WRAPPER_TEMPLATE",
    "DesignPhysicalFacts",
    "EdgeBinding",
    "ModuleInstance",
    "PhysicalCompositionError",
    "PhysicalPin",
    "PhysicalStructure",
    "PhysicalWire",
    "PinSlice",
    "SemanticPortBinding",
    "UnusedOutput",
    "assess_design_physical",
    "compose_decomposed",
    "design_physical_refusal",
    "lower_module_structure",
    "selected_kernel_realization",
    "top_boundary_layout",
    "validate_design_physical_facts",
    "validate_physical_structure",
]
