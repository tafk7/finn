# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Validation for logical-to-physical stream bindings."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from finn.dataflow.artifacts.abi import Bus, Endpoint, Member, StandardProtocol
from finn.dataflow.artifacts.build import ModuleABIRequirements
from finn.dataflow.model.logical.network import (
    DataflowNetwork,
    DirectConnection,
    PassCorrespondence,
    PositionMap,
)
from finn.dataflow.model.logical.region import DataflowRegion, InputInterface, Port, element_width
from finn.dataflow.model.physical.layout import (
    FieldPlacement,
    PackedBeatLayout,
    UnusedBitPolicy,
    UnusedBitRange,
)
from finn.dataflow.model.physical.structure import (
    ConstantBits,
    PhysicalPin,
    PhysicalStructure,
    PhysicalStructureError,
)
from finn.dataflow.model.physical.validation import validate_physical_structure
from finn.dataflow.model.relations.values import CompositePhysicalFacts, KernelStreamBinding


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
        expected_policy = (
            UnusedBitPolicy.IGNORE_ON_RECEIVE
            if direction is Endpoint.TARGET
            else UnusedBitPolicy.DRIVE_ZERO
        )
        if any(item.policy is not expected_policy for item in binding.payload.unused):
            raise ValueError("padding policy disagrees with stream direction")
        if (
            binding.framing is not None
            and port.beat_sequence.beat_count % binding.framing.period_beats
        ):
            raise ValueError("framing period must divide the logical pass")


def _bus(abi: ModuleABIRequirements, name: str, endpoint: Endpoint) -> Bus:
    matches = tuple(
        port
        for port in abi.ports
        if isinstance(port, Bus) and port.name == name and port.endpoint is endpoint
    )
    if len(matches) != 1 or matches[0].protocol is not StandardProtocol.AXIS:
        raise PhysicalStructureError(f"expected one {endpoint.value} AXIS bus {name!r}")
    return matches[0]


def _member(bus: Bus, logical: str) -> Member:
    matches = tuple(member for member in bus.signals if member.logical == logical)
    if len(matches) != 1:
        raise PhysicalStructureError(f"AXIS bus {bus.name!r} must have one {logical!r} member")
    return matches[0]


def _port_for(network: DataflowNetwork, node_id: str, port_id: str, *, output: bool) -> Port:
    try:
        region = network.node(node_id).region
        return (
            region.output_interface(port_id).port
            if output
            else region.input_interface(port_id).port
        )
    except KeyError as error:
        raise PhysicalStructureError(
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
            source: _BitSource
            if isinstance(wire.source, ConstantBits):
                source = ("constant", (wire.source.value >> offset) & 1, wire.invert)
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
) -> None:
    for index in range(width):
        expected: _BitSource = (
            "pin",
            (source, source_offset + index),
            False,
        )
        if wiring.get((destination, destination_offset + index)) != expected:
            raise PhysicalStructureError(
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
        raise PhysicalStructureError("physical payloads name different logical fields")
    for index in sorted(source_fields):
        left = source_fields[index]
        right = destination_fields[index]
        if left.bit_width != right.bit_width:
            raise PhysicalStructureError("physical payload fields have different widths")
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
            raise PhysicalStructureError("a child target's unused bits must ignore on receive")
        for bit in range(unused.bit_offset, unused.bit_offset + unused.bit_width):
            if wiring.get((destination, bit)) != ("constant", 0, False):
                raise PhysicalStructureError("every unused child input bit is driven to zero")


def _check_payload(port: Port, bus: Bus, layout: PackedBeatLayout) -> None:
    width = _member(bus, "tdata").width
    fields = tuple(sorted(layout.fields, key=lambda item: item.field_index))
    if tuple(item.field_index for item in fields) != tuple(
        range(port.beat_sequence.elements_per_beat)
    ):
        raise PhysicalStructureError("a payload must place every logical field once")
    scalar = element_width(port.operand.element_type)
    if any(item.bit_width != scalar for item in fields):
        raise PhysicalStructureError("a payload field width differs from its logical scalar")
    occupied: set[int] = set()
    spans: tuple[FieldPlacement | UnusedBitRange, ...] = (*layout.fields, *layout.unused)
    for span in spans:
        bits = set(range(span.bit_offset, span.bit_offset + span.bit_width))
        if not bits or max(bits) >= width or occupied & bits:
            raise PhysicalStructureError(
                "payload fields and unused ranges must partition a carrier"
            )
        occupied |= bits
    if occupied != set(range(width)):
        raise PhysicalStructureError("payload fields and unused ranges must cover the carrier")
    policy = (
        UnusedBitPolicy.IGNORE_ON_RECEIVE
        if bus.endpoint is Endpoint.TARGET
        else UnusedBitPolicy.DRIVE_ZERO
    )
    if any(item.policy is not policy for item in layout.unused):
        raise PhysicalStructureError("payload padding policy disagrees with endpoint direction")


def validate_logical_physical_relation(
    network: DataflowNetwork,
    structure: PhysicalStructure,
    facts: CompositePhysicalFacts,
) -> None:
    """Check generic semantic port, boundary, edge, and payload correspondence."""

    validate_physical_structure(structure)
    instances = {instance.instance_id: instance for instance in structure.instances}
    ports = tuple(facts.port_bindings)
    keys = tuple((item.node_id, item.local.region_port_id) for item in ports)
    if len(keys) != len(set(keys)):
        raise PhysicalStructureError("a semantic Region port is bound more than once")
    physical_keys = tuple((item.instance_id, item.local.abi_bus_id) for item in ports)
    if len(physical_keys) != len(set(physical_keys)):
        raise PhysicalStructureError("a physical child bus is bound more than once")
    if {item.node_id for item in ports} != {node.id for node in network.nodes}:
        raise PhysicalStructureError("semantic bindings and Network nodes differ")
    for item in ports:
        instance = instances.get(item.instance_id)
        if instance is None:
            raise PhysicalStructureError("a semantic binding names no module instance")
        node = network.node(item.node_id)
        node_bindings = tuple(binding.local for binding in ports if binding.node_id == item.node_id)
        validate_kernel_stream_bindings(node.region, instance.requirements.abi, node_bindings)

    by_port = {(item.node_id, item.local.region_port_id): item for item in ports}
    boundaries = {boundary.id: boundary for boundary in network.boundaries}
    supplied_boundaries = {item.boundary_id: item for item in facts.boundary_bindings}
    if len(supplied_boundaries) != len(facts.boundary_bindings) or set(supplied_boundaries) != set(
        boundaries
    ):
        raise PhysicalStructureError("physical boundaries cover every Network boundary once")
    top_buses = {port.name: port for port in structure.top_abi.ports if isinstance(port, Bus)}
    if {item.top_bus_id for item in facts.boundary_bindings} != set(top_buses):
        raise PhysicalStructureError("boundary bindings cover every top data bus once")
    for boundary_id, boundary in boundaries.items():
        binding = supplied_boundaries[boundary_id]
        endpoint = boundary.endpoint
        local = by_port.get((endpoint.node_id, endpoint.port_id))
        if local is None or (binding.instance_id, binding.child_bus_id) != (
            local.instance_id,
            local.local.abi_bus_id,
        ):
            raise PhysicalStructureError("a boundary binding disagrees with its Network endpoint")
        input_port = any(
            isinstance(interface, InputInterface) and interface.port.id == endpoint.port_id
            for interface in network.node(endpoint.node_id).region.inputs
        )
        top_bus = top_buses[binding.top_bus_id]
        expected_endpoint = Endpoint.TARGET if input_port else Endpoint.INITIATOR
        if top_bus.endpoint is not expected_endpoint:
            raise PhysicalStructureError("a top bus direction disagrees with its boundary")
        port = _port_for(network, endpoint.node_id, endpoint.port_id, output=not input_port)
        _check_payload(port, top_bus, binding.payload)
        if _layout_fields(binding.payload) != _layout_fields(local.local.payload):
            raise PhysicalStructureError("a boundary changes logical field order or width")
        if port.beat_sequence != boundary.external_beat_sequence:
            raise PhysicalStructureError("a physical boundary cannot adapt its logical sequence")

    edges = {edge.id: edge for edge in network.edges}
    supplied_edges = {item.edge_id: item for item in facts.edge_bindings}
    if len(supplied_edges) != len(facts.edge_bindings) or set(supplied_edges) != set(edges):
        raise PhysicalStructureError("physical edges cover every Network edge once")
    for edge_id, edge in edges.items():
        if len(edge.sinks) != 1 or not isinstance(edge.transport, DirectConnection):
            raise PhysicalStructureError(
                "the physical relation requires direct point-to-point edges"
            )
        if edge.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
            raise PhysicalStructureError("the physical relation requires one-to-one edges")
        source_binding = by_port.get((edge.source.node_id, edge.source.port_id))
        sink_endpoint = edge.sinks[0].endpoint
        sink_binding = by_port.get((sink_endpoint.node_id, sink_endpoint.port_id))
        if source_binding is None or sink_binding is None:
            raise PhysicalStructureError("an edge endpoint has no semantic port binding")
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
            raise PhysicalStructureError("a physical edge disagrees with its Network endpoints")
        if _layout_fields(source_binding.local.payload) != _layout_fields(
            sink_binding.local.payload
        ):
            raise PhysicalStructureError("a physical edge changes logical field order or width")
        if source_binding.local.framing != sink_binding.local.framing:
            raise PhysicalStructureError("a physical edge changes stream framing")
        source_port = _port_for(network, edge.source.node_id, edge.source.port_id, output=True)
        sink_port = _port_for(network, sink_endpoint.node_id, sink_endpoint.port_id, output=False)
        if source_port.beat_sequence != sink_port.beat_sequence:
            raise PhysicalStructureError("a physical edge cannot adapt logical beat order")
        if edge.sinks[0].position_map != PositionMap.identity(source_port.beat_sequence.image_set):
            raise PhysicalStructureError("the physical relation does not implement a logical map")

    wiring = _wire_map(structure)
    ignored = {
        (item.pin, bit)
        for item in structure.ignored_top_input_bits
        for bit in range(item.bit_offset, item.bit_offset + item.bit_width)
    }
    expected_ignored: set[tuple[PhysicalPin, int]] = set()
    for binding in facts.boundary_bindings:
        top_bus = top_buses[binding.top_bus_id]
        local_binding = next(
            item.local
            for item in ports
            if item.instance_id == binding.instance_id
            and item.local.abi_bus_id == binding.child_bus_id
        )
        child_bus = _bus(
            instances[binding.instance_id].requirements.abi,
            binding.child_bus_id,
            top_bus.endpoint,
        )
        top_data = PhysicalPin(None, _member(top_bus, "tdata").physical)
        child_data = PhysicalPin(binding.instance_id, _member(child_bus, "tdata").physical)
        if top_bus.endpoint is Endpoint.TARGET:
            _expect_field_copy(wiring, child_data, local_binding.payload, top_data, binding.payload)
            _expect_zero_padding(wiring, child_data, local_binding.payload)
            for unused in binding.payload.unused:
                expected_ignored.update(
                    (top_data, bit)
                    for bit in range(unused.bit_offset, unused.bit_offset + unused.bit_width)
                )
            _expect_pin_copy(
                wiring,
                PhysicalPin(binding.instance_id, _member(child_bus, "tvalid").physical),
                PhysicalPin(None, _member(top_bus, "tvalid").physical),
            )
            _expect_pin_copy(
                wiring,
                PhysicalPin(None, _member(top_bus, "tready").physical),
                PhysicalPin(binding.instance_id, _member(child_bus, "tready").physical),
            )
        else:
            _expect_field_copy(wiring, top_data, binding.payload, child_data, local_binding.payload)
            _expect_pin_copy(
                wiring,
                PhysicalPin(None, _member(top_bus, "tvalid").physical),
                PhysicalPin(binding.instance_id, _member(child_bus, "tvalid").physical),
            )
            _expect_pin_copy(
                wiring,
                PhysicalPin(binding.instance_id, _member(child_bus, "tready").physical),
                PhysicalPin(None, _member(top_bus, "tready").physical),
            )
    if ignored != expected_ignored:
        raise PhysicalStructureError("ignored top bits do not match boundary padding")
    for edge_binding_item in facts.edge_bindings:
        source_stream = next(
            item.local
            for item in ports
            if item.instance_id == edge_binding_item.source_instance
            and item.local.abi_bus_id == edge_binding_item.source_bus
        )
        sink_stream = next(
            item.local
            for item in ports
            if item.instance_id == edge_binding_item.sink_instance
            and item.local.abi_bus_id == edge_binding_item.sink_bus
        )
        source_bus = _bus(
            instances[edge_binding_item.source_instance].requirements.abi,
            edge_binding_item.source_bus,
            Endpoint.INITIATOR,
        )
        sink_bus = _bus(
            instances[edge_binding_item.sink_instance].requirements.abi,
            edge_binding_item.sink_bus,
            Endpoint.TARGET,
        )
        source_data = PhysicalPin(
            edge_binding_item.source_instance, _member(source_bus, "tdata").physical
        )
        sink_data = PhysicalPin(
            edge_binding_item.sink_instance, _member(sink_bus, "tdata").physical
        )
        _expect_field_copy(
            wiring, sink_data, sink_stream.payload, source_data, source_stream.payload
        )
        _expect_zero_padding(wiring, sink_data, sink_stream.payload)
        _expect_pin_copy(
            wiring,
            PhysicalPin(edge_binding_item.sink_instance, _member(sink_bus, "tvalid").physical),
            PhysicalPin(edge_binding_item.source_instance, _member(source_bus, "tvalid").physical),
        )
        _expect_pin_copy(
            wiring,
            PhysicalPin(edge_binding_item.source_instance, _member(source_bus, "tready").physical),
            PhysicalPin(edge_binding_item.sink_instance, _member(sink_bus, "tready").physical),
        )
        if source_stream.framing is not None:
            _expect_pin_copy(
                wiring,
                PhysicalPin(
                    edge_binding_item.sink_instance,
                    _member(sink_bus, source_stream.framing.member).physical,
                ),
                PhysicalPin(
                    edge_binding_item.source_instance,
                    _member(source_bus, source_stream.framing.member).physical,
                ),
            )


__all__ = [
    "region_ports",
    "validate_kernel_stream_bindings",
    "validate_logical_physical_relation",
]
