# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from finn.dataflow.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.network import (
    BoundaryContract,
    ChannelSpec,
    DataflowNetwork,
    Edge,
    NetworkNode,
    OrderedChannel,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import BeatSequence, DataflowRegion, NumericElementType, Port

INT8 = NumericElementType("int", 8)
INT16 = NumericElementType("int", 16)


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
