# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from collections import Counter

import pytest

from finn.dataflow.design import (
    Absent,
    Decided,
    DependencyRef,
    DesignSpaceSpec,
    Engine,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import NO_KERNEL, KernelSelection, SelectedKernel
from finn.dataflow.mvau.regions import (
    construct_batch_interleaved_mvau_weight_port,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.mvau.weight_adapter import (
    construct_weight_sequence_adapter_region,
    weight_sequence_adapter_applicable,
)
from finn.dataflow.mvau.weight_adapter_kernel import (
    FULL_TILE_TO_CHUNKED,
    build_mvau_weight_adapter_selection,
)
from finn.dataflow.network import (
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import validate_network
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Port,
    ScheduledInputRequirements,
)
from finn.dataflow.region_validation import validate_region
from finn.dataflow.spec_algebra import assemble_specs

INT8 = DataType["INT8"]


def _ports() -> tuple[Port, Port]:
    full = construct_standard_mvau_weight_port(2, 4, 4, INT8, 2, 2)
    chunked = construct_batch_interleaved_mvau_weight_port(2, 4, 4, INT8, 2, 2, 2)
    return full, chunked


def test_full_tile_to_chunked_adapter_has_exact_schedule_and_relations() -> None:
    full, chunked = _ports()

    adapter = construct_weight_sequence_adapter_region(full, chunked)

    assert not validate_region(adapter)
    assert adapter.schedule.level_names == ("output_beat",)
    assert adapter.schedule.extents == (chunked.beat_sequence.beat_count,)
    assert adapter.input_interface("weight_in").port.beat_sequence == full.beat_sequence
    assert adapter.output_interface("weight_out").port.beat_sequence == chunked.beat_sequence
    requirements = adapter.input_interface("weight_in").requirements
    for ordinal, beat in enumerate(chunked.beat_sequence.beats):
        expected = Counter(beat)
        for position in chunked.beat_sequence.image:
            assert requirements.required((ordinal,), position) == expected[position]
    availability = adapter.output_interface("weight_out").availability
    assert availability.domain == chunked.beat_sequence.image
    for position in availability.domain:
        first_beat = next(
            ordinal for ordinal, beat in enumerate(chunked.beat_sequence.beats) if position in beat
        )
        assert availability.available_at(position) == (first_beat,)


def test_chunked_to_full_adapter_preserves_exact_boundary_sequences() -> None:
    full, chunked = _ports()

    adapter = construct_weight_sequence_adapter_region(chunked, full)

    assert adapter.input_interface("weight_in").port.beat_sequence == chunked.beat_sequence
    assert adapter.output_interface("weight_out").port.beat_sequence == full.beat_sequence
    assert adapter.input_interface("weight_in").port.operand == (
        adapter.output_interface("weight_out").port.operand
    )


def test_equal_width_with_different_field_order_is_not_directly_equal() -> None:
    full, _chunked = _ports()
    reordered = Port(
        "weight",
        full.operand,
        BeatSequence(
            full.beat_sequence.elements_per_beat,
            tuple(tuple(reversed(beat)) for beat in full.beat_sequence.beats),
        ),
    )

    assert full.beat_type.logical_bit_width == reordered.beat_type.logical_bit_width
    assert full.beat_sequence != reordered.beat_sequence
    assert weight_sequence_adapter_applicable(full, reordered)
    source = construct_cyclic_parameter_region(full)
    sink = DataflowRegion(
        LogicalSchedule(()),
        (InputInterface(reordered, ScheduledInputRequirements()),),
        (),
    )
    report = validate_network(
        DataflowNetwork(
            (NetworkNode("source", source), NetworkNode("sink", sink)),
            (
                Edge(
                    "weight",
                    RegionEndpoint("source", "weight"),
                    (
                        SinkContract(
                            RegionEndpoint("sink", "weight"),
                            PositionMap.identity(full.beat_sequence.image),
                        ),
                    ),
                ),
            ),
            (),
        )
    )
    assert "edge.beat_sequence_mismatch" in {issue.code for issue in report.issues}


def test_adapter_rejects_different_tensor_images() -> None:
    full, chunked = _ports()
    invalid = Port(
        "weight",
        chunked.operand,
        BeatSequence(
            chunked.beat_sequence.elements_per_beat,
            chunked.beat_sequence.beats[:-1],
        ),
    )

    assert not weight_sequence_adapter_applicable(full, invalid)
    with pytest.raises(ValueError):
        construct_weight_sequence_adapter_region(full, invalid)


def test_the_adapter_is_an_ordinary_optional_kernel() -> None:
    full, chunked = _ports()
    source = QualifiedPath("problem.test.adapter_source")
    sink = QualifiedPath("problem.test.adapter_sink")
    port_semantics = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
    selection = build_mvau_weight_adapter_selection(
        DependencyRef.problem("source_port", source, port_semantics),
        DependencyRef.problem("sink_port", sink, port_semantics),
    )
    spec = assemble_specs(
        (
            selection.build_spec(),
            DesignSpaceSpec(
                ProblemSchema(
                    (ProblemField(source, port_semantics), ProblemField(sink, port_semantics))
                )
            ),
        )
    )
    engine = Engine()
    point = engine.start(engine.validate(spec), {source: full, sink: chunked})

    selected = engine.commit_assignments(
        point, {selection.paths.kernel: FULL_TILE_TO_CHUNKED}
    ).point
    region = engine.query_property(selected, selection.paths.region)
    assert isinstance(region, Decided)
    assert isinstance(region.value, DataflowRegion)
    assert not validate_region(region.value)
    assert engine.query_property(selected, selection.paths.selected_kernel) == Decided(
        SelectedKernel(selection.name, FULL_TILE_TO_CHUNKED, "1")
    )
    assert (
        engine.evaluate_constraint_set(selected, selection.feasibility_constraint_set).verdict
        is True
    )

    # The adapter is never implied: leaving it unselected is representable.
    unselected = engine.commit_assignments(point, {selection.paths.kernel: NO_KERNEL}).point
    assert isinstance(engine.query_property(unselected, selection.paths.region), Absent)


def test_the_adapter_refuses_endpoints_it_cannot_relate() -> None:
    full, chunked = _ports()
    mismatched = Port(
        "weight",
        chunked.operand,
        BeatSequence(chunked.beat_sequence.elements_per_beat, chunked.beat_sequence.beats[:-1]),
    )
    source = QualifiedPath("problem.test.adapter_source")
    sink = QualifiedPath("problem.test.adapter_sink")
    port_semantics = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
    selection = build_mvau_weight_adapter_selection(
        DependencyRef.problem("source_port", source, port_semantics),
        DependencyRef.problem("sink_port", sink, port_semantics),
    )
    spec = assemble_specs(
        (
            selection.build_spec(),
            DesignSpaceSpec(
                ProblemSchema(
                    (ProblemField(source, port_semantics), ProblemField(sink, port_semantics))
                )
            ),
        )
    )
    engine = Engine()
    point = engine.commit_assignments(
        engine.start(engine.validate(spec), {source: full, sink: mismatched}),
        {selection.paths.kernel: FULL_TILE_TO_CHUNKED},
    ).point
    assessment = engine.evaluate_constraint_set(point, selection.feasibility_constraint_set)
    assert assessment.verdict is False


def test_placing_the_adapter_keeps_its_region_and_separates_its_paths() -> None:
    full, chunked = _ports()
    source = QualifiedPath("problem.test.adapter_source")
    sink = QualifiedPath("problem.test.adapter_sink")
    port_semantics = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
    base = build_mvau_weight_adapter_selection(
        DependencyRef.problem("source_port", source, port_semantics),
        DependencyRef.problem("sink_port", sink, port_semantics),
    )
    placed = KernelSelection(
        f"scope.adapter0.{base.name}",
        (base.kernel(FULL_TILE_TO_CHUNKED).place("scope.adapter0"),),
        optional=True,
    )
    spec = assemble_specs(
        (
            base.build_spec(),
            placed.build_spec(),
            DesignSpaceSpec(
                ProblemSchema(
                    (ProblemField(source, port_semantics), ProblemField(sink, port_semantics))
                )
            ),
        )
    )
    engine = Engine()
    point = engine.commit_assignments(
        engine.start(engine.validate(spec), {source: full, sink: chunked}),
        {base.paths.kernel: FULL_TILE_TO_CHUNKED, placed.paths.kernel: FULL_TILE_TO_CHUNKED},
    ).point
    assert base.paths.region != placed.paths.region
    assert engine.query_property(point, base.paths.region) == engine.query_property(
        point, placed.paths.region
    )
