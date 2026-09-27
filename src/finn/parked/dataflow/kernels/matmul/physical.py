# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical assembly rules for the external Replay-to-Dotp matmul family."""

from __future__ import annotations

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import (
    EntryPointSourceName,
    FixedModuleName,
    GeneratedModuleName,
    ModuleABIRequirements,
    RenderedSourceRequirement,
    SELF_CONTAINED_JINJA_RENDERER,
)
from finn.kernels.artifacts.derivation import ProducerIdentity
from finn.kernels.physical.layout import (
    PackedBeatLayout,
    PeriodicLast,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.kernels.physical.structure import (
    ConstantBits,
    ModuleInstance,
    PhysicalPin,
    PhysicalStructure,
    PhysicalWire,
    PinSlice,
    UnusedOutput,
    PhysicalStructureError,
)
from finn.parked.dataflow.model.physical.interface import (
    KernelRealizationFacts,
    KernelStreamBinding,
)


PhysicalCompositionError = PhysicalStructureError


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

    period = dict(replay.requirements.parameters)["LEN"]
    if not isinstance(period, int) or isinstance(period, bool):
        raise PhysicalCompositionError("replay length must be an integer")
    if replay_out.framing != PeriodicLast("tlast", period, period - 1):
        raise PhysicalCompositionError("internal tlast must frame each synapse-fold group")

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
    for label, layout, carrier in (
        ("weights", compute_weight.payload, weight_width),
        ("result", compute_output.payload, output_width),
    ):
        logical_width = _low_field_layout(layout, label=label)
        if carrier != ((logical_width + 7) // 8) * 8:
            raise PhysicalCompositionError("a top payload carrier must be byte-aligned exactly")
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


def validate_decomposed_structure(
    structure: PhysicalStructure, replay: KernelRealizationFacts, compute: KernelRealizationFacts
) -> None:
    """Optional construction diagnostic; no logical View or paired model is involved.

    Runtime construction already uses these same checked field/control helpers.
    Tests can independently corrupt a structure and check the actual wiring.
    """
    _validate_decomposed_control_contract(
        structure.top_abi, replay.requirements.abi, compute.requirements.abi
    )
    expected = compose_decomposed(replay=replay, compute=compute)
    if structure.top_abi != expected.top_abi or structure.instances != expected.instances:
        raise PhysicalCompositionError("physical modules and ABI differ from construction inputs")
    if structure.unused_outputs != expected.unused_outputs:
        raise PhysicalCompositionError("only u_replay.ofin may be deliberately open")
    if structure.ignored_top_input_bits != expected.ignored_top_input_bits:
        raise PhysicalCompositionError("ignored top bits must match declared input padding")

    def bits(value: PhysicalStructure) -> dict[tuple[PhysicalPin, int], object]:
        result: dict[tuple[PhysicalPin, int], object] = {}
        for wire in value.wires:
            for index in range(wire.destination.bit_width):
                source = (
                    ("constant", (wire.source.value >> index) & 1, wire.invert)
                    if isinstance(wire.source, ConstantBits)
                    else ("pin", wire.source.pin, wire.source.bit_offset + index, wire.invert)
                )
                result[wire.destination.pin, wire.destination.bit_offset + index] = source
        return result

    actual_bits, expected_bits = bits(structure), bits(expected)
    for destination, source in expected_bits.items():
        if actual_bits.get(destination) != source:
            raise PhysicalCompositionError(
                "physical wire does not preserve its required source bit or padding driven to zero"
            )
    if actual_bits.keys() != expected_bits.keys():
        raise PhysicalCompositionError("physical wire destinations differ from construction inputs")


PORT_DECLARATIONS = "PORT_DECLARATIONS"
NET_DECLARATIONS = "NET_DECLARATIONS"
ASSIGNMENTS = "ASSIGNMENTS"
INSTANCES = "INSTANCES"
DECOMPOSED_WRAPPER_TEMPLATE = RenderedSourceRequirement(
    EntryPointSourceName(".sv"),
    "decomposed_wrapper.sv.j2",
    (PORT_DECLARATIONS, NET_DECLARATIONS, ASSIGNMENTS, INSTANCES),
    SELF_CONTAINED_JINJA_RENDERER,
    requires=("module:dotp_axi", "module:replay_buffer"),
    provides_entry_point=True,
)
DECOMPOSED_PRODUCER = ProducerIdentity("finn.mvau.decomposed.external", "1")


__all__ = [
    "DECOMPOSED_PRODUCER",
    "DECOMPOSED_WRAPPER_TEMPLATE",
    "PhysicalCompositionError",
    "compose_decomposed",
    "top_boundary_layout",
    "validate_decomposed_structure",
]
