# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataclasses import replace
from typing import cast

from finn.dataflow.ops.mvau.regions import (
    construct_batch_interleaved_streamed_mvau_region,
    construct_standard_streamed_mvau_region,
)
from finn.dataflow.model.network import (
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
from finn.dataflow.model.network_validation import (
    NetworkValidationBudget,
    NetworkValidationReport,
    validate_network,
)
from finn.dataflow.model.maps import (
    CoordinateSet,
    ExplicitCoordinateMap,
    RectangularDomain,
    ValidationCapabilityError,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.space.dataflow_value_semantics import DATAFLOW_NETWORK_SEMANTICS
from finn.dataflow.model.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    InternalInput,
)

INT8 = DataType["INT8"]
INT16 = DataType["INT16"]

#: One position, required once at the one iteration point.  Spelled out as
#: annotated constants because the literal dict infers `tuple[int]` rather than
#: the `tuple[int, ...]` coordinate the model declares.
SINGLE_USE: dict[RequirementKey, int] = {((0,), (0,)): 1}
SINGLE_RESULT: dict[Coordinate, Coordinate] = {(0,): (0,)}


def _compute(*, interleaved: bool) -> DataflowRegion:
    if interleaved:
        return construct_batch_interleaved_streamed_mvau_region(6, 4, 6, INT8, INT8, INT16, 3, 2, 3)
    return construct_standard_streamed_mvau_region(2, 4, 4, INT8, INT8, INT16, 2, 2)


def _network(*, interleaved: bool, source_port: Port | None = None) -> DataflowNetwork:
    compute = _compute(interleaved=interleaved)
    weight = compute.input_interface("weight").port
    delivery = construct_cyclic_parameter_region(source_port or weight)
    position_map = PositionMap.identity(
        delivery.output_interface("weight").port.beat_sequence.image_set
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
            tuple(
                tuple(reversed(beat))
                for beat in weight.beat_sequence.materialize_beats(
                    max_fields=weight.beat_sequence.delivered_field_count
                )
            ),
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
    entries = edge.sinks[0].position_map.materialize_entries(
        max_entries=weight_position_count(network)
    )
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
            tuple(
                tuple(reversed(beat))
                for beat in weight.beat_sequence.materialize_beats(
                    max_fields=weight.beat_sequence.delivered_field_count
                )
            ),
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
    sequence = boundary.external_beat_sequence
    wrong_sequence = BeatSequence(
        sequence.elements_per_beat,
        tuple(reversed(sequence.materialize_beats(max_fields=sequence.delivered_field_count))),
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


def weight_position_count(network: DataflowNetwork) -> int:
    return network.node("compute").region.input_interface("weight").port.operand.position_count


def _compact_edge_network(
    source_sequence: BeatSequence,
    sink_sequence: BeatSequence,
    position_map: PositionMap,
) -> DataflowNetwork:
    source_affine = source_sequence.affine_map
    sink_affine = sink_sequence.affine_map
    assert source_affine is not None and sink_affine is not None
    source_operand = Operand("source", INT8, source_affine.target.extents)
    sink_operand = Operand("sink", INT8, sink_affine.target.extents)
    source_schedule = LogicalSchedule(())
    source_region = DataflowRegion(
        source_schedule,
        (),
        (
            OutputInterface(
                Port("output", source_operand, source_sequence),
                ScheduledOutputAvailability.affine(
                    source_operand.position_domain,
                    source_schedule.iteration_domain,
                    view_extents=source_operand.shape,
                    offset=0,
                    coefficients=(0,) * source_operand.rank,
                ),
            ),
        ),
    )
    sink_region = DataflowRegion(
        LogicalSchedule(()),
        (
            InputInterface(
                Port("input", sink_operand, sink_sequence),
                ScheduledInputRequirements(),
            ),
        ),
        (),
    )
    return DataflowNetwork(
        (NetworkNode("source", source_region), NetworkNode("sink", sink_region)),
        (
            Edge(
                "edge",
                RegionEndpoint("source", "output"),
                (SinkContract(RegionEndpoint("sink", "input"), position_map),),
            ),
        ),
        (),
    )


def test_explicit_reversal_map_over_million_compact_occurrences_is_algebraic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = RectangularDomain((2,))
    beat_count = 1_000_000
    source = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=beat_count,
        view_extents=(beat_count // 2, 2, 1),
        offset=0,
        coefficients=(0, 1, 0),
    )
    sink = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=beat_count,
        view_extents=(beat_count // 2, 2, 1),
        offset=1,
        coefficients=(0, -1, 0),
    )
    network = _compact_edge_network(
        source,
        sink,
        PositionMap((((0,), (1,)), ((1,), (0,)))),
    )

    monkeypatch.setattr(
        BeatSequence,
        "position_at",
        lambda *_args, **_kwargs: pytest.fail("algebraic dispatch enumerated beats"),
    )
    assert validate_network(network) == NetworkValidationReport()


def test_unrecognized_compact_edge_requires_an_explicit_budget() -> None:
    domain = RectangularDomain((2, 2))
    source = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=4,
        view_extents=(4, 1),
        offset=0,
        coefficients=(1, 0),
    )
    sink = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=4,
        view_extents=(2, 2, 1),
        offset=0,
        coefficients=(1, 2, 0),
    )
    transpose = PositionMap.affine(
        domain,
        view_extents=(2, 2),
        sink=domain,
        offset=0,
        coefficients=(1, 2),
    )
    network = _compact_edge_network(source, sink, transpose)

    with pytest.raises(ValidationCapabilityError):
        validate_network(network)
    with pytest.raises(ValidationCapabilityError):
        validate_network(
            network,
            expansion_budget=NetworkValidationBudget(max_generated_fields=3),
        )
    assert (
        validate_network(
            network,
            expansion_budget=NetworkValidationBudget(max_generated_fields=4),
        )
        == NetworkValidationReport()
    )


def _explicit_edge_network(
    count: int,
    position_map: PositionMap,
    *,
    sink_extent: int | None = None,
) -> tuple[DataflowNetwork, BeatSequence]:
    sink_extent = count if sink_extent is None else sink_extent
    sequence = BeatSequence(1, tuple(((index,),) for index in range(count)))
    source_operand = Operand("source", INT8, (count,))
    sink_operand = Operand("sink", INT8, (sink_extent,))
    availability: dict[Coordinate, Coordinate] = {(index,): () for index in range(count)}
    source = DataflowRegion(
        LogicalSchedule(()),
        (),
        (
            OutputInterface(
                Port("output", source_operand, sequence),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )
    sink = DataflowRegion(
        LogicalSchedule(()),
        (InputInterface(Port("input", sink_operand, sequence), ScheduledInputRequirements()),),
        (),
    )
    return (
        DataflowNetwork(
            (NetworkNode("source", source), NetworkNode("sink", sink)),
            (
                Edge(
                    "edge",
                    RegionEndpoint("source", "output"),
                    (SinkContract(RegionEndpoint("sink", "input"), position_map),),
                ),
            ),
            (),
        ),
        sequence,
    )


def test_network_construction_persistently_binds_explicit_maps_for_equality() -> None:
    domain = RectangularDomain((2,))
    explicit, _ = _explicit_edge_network(2, PositionMap((((0,), (0,)), ((1,), (1,)))))
    compact, _ = _explicit_edge_network(2, PositionMap.identity(CoordinateSet.full(domain)))

    assert validate_network(explicit) == NetworkValidationReport()
    assert validate_network(compact) == NetworkValidationReport()
    assert explicit == compact
    assert DATAFLOW_NETWORK_SEMANTICS.values_equal(explicit, compact)
    assert explicit.edges[0].sinks[0].position_map.source_set.cardinality == 2


def test_duplicate_explicit_map_network_is_not_equal_to_a_valid_compact_network() -> None:
    domain = RectangularDomain((2,))
    malformed, _ = _explicit_edge_network(2, PositionMap((((0,), (0,)), ((0,), (0,)))))
    valid, _ = _explicit_edge_network(2, PositionMap.identity(CoordinateSet.full(domain)))

    malformed_codes = {issue.code for issue in validate_network(malformed)}
    assert "position_map.source_not_function" in malformed_codes
    assert "position_map.source_domain_mismatch" in malformed_codes
    assert validate_network(valid) == NetworkValidationReport()
    assert malformed != valid
    assert not DATAFLOW_NETWORK_SEMANTICS.values_equal(malformed, valid)
    assert len({malformed, valid}) == len({valid, malformed}) == 2


def test_network_construction_binds_an_external_explicit_boundary_sequence() -> None:
    network, unbound_sequence = _explicit_edge_network(2, PositionMap((((0,), (0,)), ((1,), (1,)))))
    source = network.node("source").region
    boundary = BoundaryContract(
        "source",
        RegionEndpoint("source", "output"),
        unbound_sequence,
    )
    exposed = DataflowNetwork(
        network.nodes,
        (),
        (
            boundary,
            BoundaryContract(
                "sink",
                RegionEndpoint("sink", "input"),
                network.node("sink").region.input_interface("input").port.beat_sequence,
            ),
        ),
    )

    assert validate_network(exposed) == NetworkValidationReport()
    assert (
        exposed.boundaries[1].external_beat_sequence
        == source.output_interface("output").port.beat_sequence
    )


def test_compact_identity_is_typed_against_both_different_operand_ambients() -> None:
    source_domain = RectangularDomain((2,))
    explicit, _ = _explicit_edge_network(
        2,
        PositionMap((((0,), (0,)), ((1,), (1,)))),
        sink_extent=3,
    )
    compact, _ = _explicit_edge_network(
        2,
        PositionMap.identity(CoordinateSet.full(source_domain)),
        sink_extent=3,
    )

    assert validate_network(explicit) == NetworkValidationReport()
    assert validate_network(compact) == NetworkValidationReport()
    assert explicit == compact
    assert compact.edges[0].sinks[0].position_map.sink_set.ambient == RectangularDomain((3,))


def test_all_explicit_validation_uses_one_lookup_not_linear_map_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 2_000
    network, _ = _explicit_edge_network(
        count,
        PositionMap(tuple(((index,), (index,)) for index in range(count))),
    )

    monkeypatch.setattr(
        ExplicitCoordinateMap,
        "mapped",
        lambda *_args, **_kwargs: pytest.fail("explicit validation rescanned map entries"),
    )
    assert validate_network(network) == NetworkValidationReport()


def test_algebraic_mismatch_without_a_witness_uses_a_map_level_path() -> None:
    domain = RectangularDomain((4,))
    ordered = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=4,
        view_extents=(4,),
        offset=0,
        coefficients=(1,),
    )
    transposed = BeatSequence.affine(
        domain,
        elements_per_beat=1,
        beat_count=4,
        view_extents=(2, 2),
        offset=0,
        coefficients=(1, 2),
    )
    network = _compact_edge_network(
        ordered,
        transposed,
        PositionMap.identity(CoordinateSet.full(domain)),
    )

    mismatches = [
        issue for issue in validate_network(network) if issue.code == "edge.beat_sequence_mismatch"
    ]
    assert len(mismatches) == 1
    assert mismatches[0].path == "edge['edge'].sinks[0].beat_sequence"


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


def test_an_internal_input_is_not_a_network_endpoint() -> None:
    """Endpoint ownership ranges over ports, and an internal input has none.

    Requiring it to be consumed or exposed exactly once would refuse every
    region that declares one, which is every region that says it consumes an
    operand no port carries.
    """

    operand = Operand("w", DataType["INT8"], (1,))
    activation = Operand("x", DataType["INT8"], (1,))
    region = DataflowRegion(
        LogicalSchedule((ScheduleLevel("step", 1),)),
        (
            InputInterface(
                Port("x_in", activation, BeatSequence(1, (((0,),),))),
                ScheduledInputRequirements(SINGLE_USE),
            ),
            InternalInput(operand, ScheduledInputRequirements(SINGLE_USE)),
        ),
        (
            OutputInterface(
                Port("y_out", activation, BeatSequence(1, (((0,),),))),
                ScheduledOutputAvailability(SINGLE_RESULT),
            ),
        ),
    )
    network = DataflowNetwork(
        (NetworkNode("only", region),),
        (),
        (
            BoundaryContract(
                "in",
                RegionEndpoint("only", "x_in"),
                region.input_interface("x_in").port.beat_sequence,
            ),
            BoundaryContract(
                "out",
                RegionEndpoint("only", "y_out"),
                region.output_interface("y_out").port.beat_sequence,
            ),
        ),
    )

    assert not validate_network(network)
