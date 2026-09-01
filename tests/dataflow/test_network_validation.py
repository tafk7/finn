# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataclasses import replace
from typing import cast

from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.network import (
    BoundaryContract,
    ChannelSpec,
    DataflowNetwork,
    DirectConnection,
    Edge,
    NetworkNode,
    OrderedChannel,
    PassCorrespondence,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]


def _compute(*, interleaved: bool) -> DataflowRegion:
    if interleaved:
        return construct_batch_interleaved_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2, 3)
    return construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)


def _network(*, interleaved: bool, source_port: Port | None = None) -> DataflowNetwork:
    compute = _compute(interleaved=interleaved)
    weight = compute.input_interface("weight").port
    delivery = construct_cyclic_parameter_region(source_port or weight)
    position_map = PositionMap.identity(
        delivery.output_interface("weight").port.beat_sequence.image
    )
    return DataflowNetwork(
        (
            NetworkNode("compute", compute),
            NetworkNode("delivery", delivery),
        ),
        (
            Edge(
                "weight",
                RegionEndpoint("delivery", "weight"),
                (SinkContract(RegionEndpoint("compute", "weight"), position_map),),
                transport=OrderedChannel(ChannelSpec("weight_channel")),
            ),
        ),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint("compute", "activation"),
                compute.input_interface("activation").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint("compute", "output"),
                compute.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def test_full_tile_and_chunked_delivery_networks_are_structurally_valid() -> None:
    assert validate_network(_network(interleaved=False)) == NetworkValidationReport()
    assert validate_network(_network(interleaved=True)) == NetworkValidationReport()


def test_full_tile_and_chunked_sequences_are_not_directly_compatible() -> None:
    standard = _compute(interleaved=False)
    wrong_source = (
        construct_cyclic_parameter_region(standard.input_interface("weight").port)
        .output_interface("weight")
        .port
    )
    network = _network(interleaved=True, source_port=wrong_source)
    codes = {issue.code for issue in validate_network(network).issues}
    assert "edge.element_count_mismatch" in codes


def test_equal_width_with_wrong_field_order_is_rejected() -> None:
    compute = _compute(interleaved=False)
    weight = compute.input_interface("weight").port
    reversed_port = Port(
        "weight",
        weight.operand,
        BeatSequence(
            weight.beat_sequence.elements_per_beat,
            tuple(tuple(reversed(beat)) for beat in weight.beat_sequence.beats),
        ),
    )
    report = validate_network(_network(interleaved=False, source_port=reversed_port))
    assert "edge.beat_sequence_mismatch" in {issue.code for issue in report.issues}


def test_removing_edge_and_exposing_endpoints_preserves_port_sequences() -> None:
    connected = _network(interleaved=False)
    compute = connected.node("compute").region
    delivery = connected.node("delivery").region
    exposed = DataflowNetwork(
        connected.nodes,
        (),
        (
            *connected.boundaries,
            BoundaryContract(
                "delivered_weight",
                RegionEndpoint("delivery", "weight"),
                delivery.output_interface("weight").port.beat_sequence,
            ),
            BoundaryContract(
                "compute_weight",
                RegionEndpoint("compute", "weight"),
                compute.input_interface("weight").port.beat_sequence,
            ),
        ),
    )
    assert validate_network(exposed) == NetworkValidationReport()
    assert (
        exposed.node("delivery").region.output_interface("weight").port.beat_sequence
        == connected.node("delivery").region.output_interface("weight").port.beat_sequence
    )
    assert (
        exposed.node("compute").region.input_interface("weight").port.beat_sequence
        == connected.node("compute").region.input_interface("weight").port.beat_sequence
    )


def test_every_endpoint_must_be_connected_or_exposed_exactly_once() -> None:
    network = _network(interleaved=False)
    missing_output = DataflowNetwork(network.nodes, network.edges, network.boundaries[:-1])
    duplicate_input = DataflowNetwork(
        network.nodes,
        network.edges,
        (
            *network.boundaries,
            BoundaryContract(
                "activation_again",
                RegionEndpoint("compute", "activation"),
                network.node("compute").region.input_interface("activation").port.beat_sequence,
            ),
        ),
    )
    assert "endpoint.output_ownership" in {
        issue.code for issue in validate_network(missing_output).issues
    }
    assert "endpoint.input_ownership" in {
        issue.code for issue in validate_network(duplicate_input).issues
    }


def test_network_structural_validation_is_independent_of_binding_realizability() -> None:
    network = _network(interleaved=False)
    assert validate_network(network) == NetworkValidationReport()
    assert not hasattr(network, "binding")


def test_duplicate_node_edge_and_boundary_identities_are_reported() -> None:
    network = _network(interleaved=False)
    duplicate = DataflowNetwork(
        (*network.nodes, network.nodes[0]),
        (*network.edges, network.edges[0]),
        (*network.boundaries, network.boundaries[0]),
    )
    codes = {issue.code for issue in validate_network(duplicate).issues}
    assert {"node.id_duplicate", "edge.id_duplicate", "boundary.id_duplicate"} <= codes


def test_missing_and_wrong_direction_edge_endpoints_are_reported() -> None:
    network = _network(interleaved=False)
    edge = network.edges[0]
    wrong = replace(
        edge,
        source=RegionEndpoint("compute", "activation"),
        sinks=(
            SinkContract(RegionEndpoint("delivery", "weight"), edge.sinks[0].position_map),
            SinkContract(RegionEndpoint("missing", "port"), edge.sinks[0].position_map),
        ),
    )
    report = validate_network(DataflowNetwork(network.nodes, (wrong,), network.boundaries))
    codes = {issue.code for issue in report.issues}
    assert "edge.source_missing_or_not_output" in codes
    assert "edge.sink_missing_or_not_input" in codes


def test_empty_and_duplicate_sink_lists_are_reported() -> None:
    network = _network(interleaved=False)
    edge = network.edges[0]
    empty = replace(edge, sinks=())
    duplicate = replace(edge, sinks=(edge.sinks[0], edge.sinks[0]))
    assert "edge.sinks_empty" in {
        issue.code
        for issue in validate_network(
            DataflowNetwork(network.nodes, (empty,), network.boundaries)
        ).issues
    }
    assert "edge.sink_duplicate" in {
        issue.code
        for issue in validate_network(
            DataflowNetwork(network.nodes, (duplicate,), network.boundaries)
        ).issues
    }


def test_non_functional_and_non_bijective_position_maps_are_reported() -> None:
    network = _network(interleaved=False)
    edge = network.edges[0]
    entries = edge.sinks[0].position_map.entries
    non_functional = PositionMap((*entries, (entries[0][0], entries[1][1])))
    non_bijective = PositionMap(
        tuple(
            (source, entries[0][1] if index == 1 else sink)
            for index, (source, sink) in enumerate(entries)
        )
    )
    reports = (
        validate_network(
            DataflowNetwork(
                network.nodes,
                (replace(edge, sinks=(replace(edge.sinks[0], position_map=non_functional),)),),
                network.boundaries,
            )
        ),
        validate_network(
            DataflowNetwork(
                network.nodes,
                (replace(edge, sinks=(replace(edge.sinks[0], position_map=non_bijective),)),),
                network.boundaries,
            )
        ),
    )
    assert "position_map.source_not_function" in {issue.code for issue in reports[0].issues}
    assert "position_map.not_injective" in {issue.code for issue in reports[1].issues}


def test_element_type_and_beat_order_mismatches_are_both_reported() -> None:
    compute = _compute(interleaved=False)
    weight = compute.input_interface("weight").port
    wrong_source = Port(
        "weight",
        Operand("W", DataType["UINT8"], weight.operand.shape),
        BeatSequence(
            weight.beat_sequence.elements_per_beat,
            tuple(tuple(reversed(beat)) for beat in weight.beat_sequence.beats),
        ),
    )
    codes = {
        issue.code
        for issue in validate_network(_network(interleaved=False, source_port=wrong_source)).issues
    }
    assert {"edge.element_type_mismatch", "edge.beat_sequence_mismatch"} <= codes


def test_boundary_pass_and_sequence_mismatches_are_reported() -> None:
    network = _network(interleaved=False)
    boundary = network.boundaries[0]
    wrong_sequence = replace(
        boundary.external_beat_sequence,
        beats=tuple(reversed(boundary.external_beat_sequence.beats)),
    )
    invalid = replace(
        boundary,
        external_beat_sequence=wrong_sequence,
        pass_correspondence=cast(PassCorrespondence, "not_one_to_one"),
    )
    report = validate_network(
        DataflowNetwork(network.nodes, network.edges, (invalid, *network.boundaries[1:]))
    )
    codes = {issue.code for issue in report.issues}
    assert {"boundary.pass_correspondence_unsupported", "boundary.beat_sequence_mismatch"} <= codes


def test_invalid_ordered_channel_contract_is_reported() -> None:
    network = _network(interleaved=False)
    invalid_edge = replace(
        network.edges[0],
        transport=OrderedChannel(ChannelSpec("weight_channel", preserve_order=False)),
    )
    report = validate_network(DataflowNetwork(network.nodes, (invalid_edge,), network.boundaries))
    assert "channel.contract_unsupported" in {issue.code for issue in report.issues}


def _identity_region() -> DataflowRegion:
    operand = Operand("value", INT8, (1,))
    sequence = BeatSequence(1, (((0,),),))
    availability: dict[Coordinate, Coordinate] = {(0,): ()}
    return DataflowRegion(
        LogicalSchedule(()),
        (InputInterface(Port("input", operand, sequence), ScheduledInputRequirements()),),
        (
            OutputInterface(
                Port("output", operand, sequence),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def test_directed_cycles_are_reported() -> None:
    region = _identity_region()
    position_map = PositionMap.identity(((0,),))
    network = DataflowNetwork(
        (NetworkNode("a", region), NetworkNode("b", region)),
        (
            Edge(
                "a_to_b",
                RegionEndpoint("a", "output"),
                (SinkContract(RegionEndpoint("b", "input"), position_map),),
                transport=DirectConnection(),
            ),
            Edge(
                "b_to_a",
                RegionEndpoint("b", "output"),
                (SinkContract(RegionEndpoint("a", "input"), position_map),),
                transport=DirectConnection(),
            ),
        ),
        (),
    )
    assert "network.cycle" in {issue.code for issue in validate_network(network).issues}
