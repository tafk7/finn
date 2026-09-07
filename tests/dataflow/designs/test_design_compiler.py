# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD4: Design topology resolves to one canonical, validated Network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import subprocess
import sys
from typing import cast


import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, Engine, QualifiedPath, Unresolved
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.space.dataflow_value_semantics import POSITION_MAP_SEMANTICS
from finn.dataflow.space.compiler import _Ref, _compile_space
from finn.dataflow.space.occurrence import is_attached_occurrence
from finn.dataflow.space.declarations import (
    AuthoringError,
    ConstraintGroup,
    Decision,
    Input,
    SubspaceChoice,
    Problem,
    Space,
    Subspace,
    constraint,
    derived,
    divisors_of,
)
from finn.dataflow.designs.design import (
    NetworkBoundary,
    NetworkEdge,
    DataflowDesign,
    KernelChoice,
    EdgeSink,
    design_dataflow,
)
from finn.dataflow.space.occurrence import ChoiceView
from finn.dataflow.kernels.kernel import Kernel, ModuleParameter, RegionDeclaration
from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    PositionMap,
    RegionEndpoint,
)
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.space.spec_algebra import assemble_specs

PRODUCE = ComputationContract("test.produce")
CONSUME = ComputationContract("test.consume")


def _sequence(extent: int, lanes: int) -> BeatSequence:
    folds = extent // lanes
    return BeatSequence(
        lanes,
        tuple(tuple((fold * lanes + lane,) for lane in range(lanes)) for fold in range(folds)),
    )


def _requirements(extent: int, lanes: int) -> ScheduledInputRequirements:
    folds = extent // lanes
    return ScheduledInputRequirements(
        {((fold,), (fold * lanes + lane,)): 1 for fold in range(folds) for lane in range(lanes)}
    )


def _availability(extent: int, lanes: int) -> ScheduledOutputAvailability:
    folds = extent // lanes
    return ScheduledOutputAvailability(
        {(fold * lanes + lane,): (fold,) for fold in range(folds) for lane in range(lanes)}
    )


def _producer_region(extent: int, lanes: int) -> DataflowRegion:
    """One input, one output; the output is what a consumer reads."""

    operand = Operand("value", DataType["INT8"], (extent,))
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("fold", extent // lanes),)),
        (
            InputInterface(
                Port("source", operand, _sequence(extent, lanes)), _requirements(extent, lanes)
            ),
        ),
        (
            OutputInterface(
                Port("stream", operand, _sequence(extent, lanes)), _availability(extent, lanes)
            ),
        ),
    )


def _consumer_region(extent: int, lanes: int) -> DataflowRegion:
    operand = Operand("value", DataType["INT8"], (extent,))
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("fold", extent // lanes),)),
        (
            InputInterface(
                Port("stream", operand, _sequence(extent, lanes)), _requirements(extent, lanes)
            ),
        ),
        (
            OutputInterface(
                Port("result", operand, _sequence(extent, lanes)), _availability(extent, lanes)
            ),
        ),
    )


class ProducerKernel(Kernel):
    id = "producer"
    computation = PRODUCE
    extent = Input(int)
    lanes = Input(int)
    region = RegionDeclaration(
        family="test.produce", version="1", construct=_producer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("producer", ())


class ConsumerKernel(Kernel):
    id = "consumer"
    computation = CONSUME
    extent = Input(int)
    lanes = Input(int)
    region = RegionDeclaration(
        family="test.consume", version="1", construct=_consumer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("consumer", ())


class PipelinedConsumerKernel(Kernel):
    """Same computation and Region, one extra physical decision of its own."""

    id = "pipelined_consumer"
    computation = CONSUME
    extent = Input(int)
    lanes = Input(int)
    stages = Decision(int, values=(1, 2))
    region = RegionDeclaration(
        family="test.consume", version="1", construct=_consumer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("pipelined_consumer", ())


class Chain(DataflowDesign):
    """producer -> consumer, with the remaining endpoints exposed."""

    id = "chain"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    produce = KernelChoice(
        Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
    )
    consume = KernelChoice(
        Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
    )

    stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))

    source = NetworkBoundary(produce.input("source"))
    result = NetworkBoundary(consume.output("result"))


class Harness(Space):
    extent = Problem(int)
    lanes = Decision(int, domain=divisors_of(extent))


def _compiled(design_type: type[DataflowDesign]):
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    design = _compile_space(
        design_type,
        "root.design",
        {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")},
        _allow_problem=False,
    )
    return harness, design


def _started(design_type: type[DataflowDesign], extent: int = 8, lanes: int = 2):
    harness, design = _compiled(design_type)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, design.spec))),
        {"problem.root.extent": extent},
    )
    return engine, engine.commit_assignments(point, {"root.lanes": lanes}).point, design


def _network(engine: Engine, point: object) -> DataflowNetwork:
    answer = engine.query_property(cast("object", point), "semantic.root.design.network")  # type: ignore[arg-type]
    assert isinstance(answer, Decided), answer
    return cast(DataflowNetwork, answer.value)


def _valid(engine: Engine, point: object) -> bool:
    assessment = engine.evaluate_constraints(
        point,  # type: ignore[arg-type]
        (
            "constraint.root.design.network_structurally_valid",
            "constraint.root.design.segments_match_network",
        ),
    )
    return assessment.verdict is True


def _findings(engine: Engine, point: object) -> set[str]:
    assessment = engine.evaluate_constraints(
        point,  # type: ignore[arg-type]
        (
            "constraint.root.design.network_structurally_valid",
            "constraint.root.design.segments_match_network",
        ),
    )
    return {
        finding.code
        for answer in assessment.answers.values()
        if isinstance(answer, (Absent, Unresolved))
        for finding in answer.findings
    }


# -- the selected Network -----------------------------------------------------


def test_the_design_generates_one_network_property_and_two_constraints() -> None:
    _harness, design = _compiled(Chain)
    assert "semantic.root.design.network" in {str(item.path) for item in design.spec.properties}
    assert {str(item.path) for item in design.spec.constraints} >= {
        "constraint.root.design.network_structurally_valid",
        "constraint.root.design.segments_match_network",
    }


def test_a_direct_connection_becomes_one_canonical_edge() -> None:
    engine, point, _design = _started(Chain)
    network = _network(engine, point)
    assert tuple(node.id for node in network.nodes) == ("consume", "produce")
    assert network.node("produce").region == _producer_region(8, 2)
    assert network.node("consume").region == _consumer_region(8, 2)
    assert tuple(edge.id for edge in network.edges) == ("stream",)
    edge = network.edges[0]
    assert edge.source == RegionEndpoint("produce", "stream")
    assert tuple(sink.endpoint for sink in edge.sinks) == (RegionEndpoint("consume", "stream"),)
    assert _valid(engine, point)


def test_an_omitted_sink_map_is_the_identity_over_the_source_image() -> None:
    engine, point, _design = _started(Chain)
    source = _producer_region(8, 2).output_interface("stream").port
    assert _network(engine, point).edges[0].sinks[0].position_map == PositionMap.identity(
        source.beat_sequence.image
    )


def test_a_boundary_takes_the_exact_endpoint_beat_sequence() -> None:
    engine, point, _design = _started(Chain)
    network = _network(engine, point)
    assert network.boundaries == (
        BoundaryContract(
            "result",
            RegionEndpoint("consume", "result"),
            _consumer_region(8, 2).output_interface("result").port.beat_sequence,
        ),
        BoundaryContract(
            "source",
            RegionEndpoint("produce", "source"),
            _producer_region(8, 2).input_interface("source").port.beat_sequence,
        ),
    )


def test_an_explicit_position_map_is_used_verbatim() -> None:
    class Remapped(DataflowDesign):
        id = "remapped"
        version = "1"
        extent = Input(int)
        lanes = Input(int)

        @derived(POSITION_MAP_SEMANTICS, extent=extent)
        def reversed_map(*, extent: int) -> PositionMap:
            return PositionMap(((index,), (extent - 1 - index,)) for index in range(extent))

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"),
            EdgeSink(consume.input("stream"), position_map=reversed_map),
        )
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    engine, point, _design = _started(Remapped)
    network = _network(engine, point)
    assert network.edges[0].sinks[0].position_map == PositionMap(
        ((index,), (7 - index,)) for index in range(8)
    )
    # A reversal does not agree with the sink's beat order, and the canon says so.
    assert "design-network-edge.beat_sequence_mismatch" in _findings(engine, point)


def test_a_fan_out_edge_carries_one_contract_per_sink() -> None:
    class Fanout(DataflowDesign):
        id = "fanout"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        left = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="left"
        )
        right = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="right"
        )
        stream = NetworkEdge(
            produce.output("stream"),
            EdgeSink(left.input("stream")),
            EdgeSink(right.input("stream")),
        )
        source = NetworkBoundary(produce.input("source"))
        left_result = NetworkBoundary(left.output("result"))
        right_result = NetworkBoundary(right.output("result"))

    engine, point, _design = _started(Fanout)
    network = _network(engine, point)
    assert len(network.edges) == 1
    assert tuple(sink.endpoint for sink in network.edges[0].sinks) == (
        RegionEndpoint("left", "stream"),
        RegionEndpoint("right", "stream"),
    )
    assert _valid(engine, point)


def test_complementary_conditions_swap_an_edge_for_a_boundary() -> None:
    class Conditional(DataflowDesign):
        id = "conditional"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        supplied = Decision(bool, values=(False, True))

        @derived(bool, supplied=supplied)
        def external(*, supplied: bool) -> bool:
            return not supplied

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=supplied,
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream")), when=supplied
        )
        source = NetworkBoundary(produce.input("source"), when=supplied)
        supplied_stream = NetworkBoundary(consume.input("stream"), when=external)
        result = NetworkBoundary(consume.output("result"))

    engine, point, _design = _started(Conditional)
    with_supplier = engine.commit_assignments(point, {"root.design.supplied": True}).point
    network = _network(engine, with_supplier)
    assert tuple(node.id for node in network.nodes) == ("consume", "produce")
    assert tuple(edge.id for edge in network.edges) == ("stream",)
    assert tuple(item.id for item in network.boundaries) == ("result", "source")
    assert _valid(engine, with_supplier)

    external_only = engine.commit_assignments(point, {"root.design.supplied": False}).point
    network = _network(engine, external_only)
    assert tuple(node.id for node in network.nodes) == ("consume",)
    assert network.edges == ()
    assert tuple(item.id for item in network.boundaries) == ("result", "supplied_stream")
    assert _valid(engine, external_only)


# -- the negative matrix ------------------------------------------------------


def _refuses(design_type: type[DataflowDesign], code: str, **problem: int) -> None:
    engine, point, _design = _started(design_type, **problem)
    assert code in _findings(engine, point), _findings(engine, point)


def test_an_unknown_port_is_refused_by_canonical_validation() -> None:
    class WrongPort(DataflowDesign):
        id = "wrong_port"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("absent"), EdgeSink(consume.input("stream")))
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    _refuses(WrongPort, "design-network-edge.source_missing_or_not_output")


def test_a_reversed_direction_is_refused() -> None:
    class Reversed(DataflowDesign):
        id = "reversed"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("source"), EdgeSink(consume.input("stream")))
        result = NetworkBoundary(consume.output("result"))
        produced = NetworkBoundary(produce.output("stream"))

    _refuses(Reversed, "design-network-edge.source_missing_or_not_output")


def test_an_unaccounted_endpoint_is_refused() -> None:
    class Unaccounted(DataflowDesign):
        id = "unaccounted"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        result = NetworkBoundary(consume.output("result"))

    _refuses(Unaccounted, "design-network-endpoint.input_ownership")


def test_implicit_fan_in_is_refused() -> None:
    class FanIn(DataflowDesign):
        id = "fan_in"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        left = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE, role="left"
        )
        right = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE, role="right"
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        from_left = NetworkEdge(left.output("stream"), EdgeSink(consume.input("stream")))
        from_right = NetworkEdge(right.output("stream"), EdgeSink(consume.input("stream")))
        left_source = NetworkBoundary(left.input("source"))
        right_source = NetworkBoundary(right.input("source"))
        result = NetworkBoundary(consume.output("result"))

    _refuses(FanIn, "design-network-endpoint.input_ownership")


def test_an_endpoint_used_by_both_an_edge_and_a_boundary_is_refused() -> None:
    class Both(DataflowDesign):
        id = "both"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        also_external = NetworkBoundary(consume.input("stream"))
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    _refuses(Both, "design-network-endpoint.input_ownership")


def test_a_cyclic_topology_is_refused() -> None:
    class Cyclic(DataflowDesign):
        id = "cyclic"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        forward = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        backward = NetworkEdge(consume.output("result"), EdgeSink(produce.input("source")))

    _refuses(Cyclic, "design-network-network.cycle")


def test_active_topology_over_an_inactive_segment_is_refused() -> None:
    class Dangling(DataflowDesign):
        id = "dangling"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        present = Decision(bool, values=(False, True))
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        source = NetworkBoundary(produce.input("source"), when=present)
        result = NetworkBoundary(consume.output("result"))

    engine, point, _design = _started(Dangling)
    absent = engine.commit_assignments(point, {"root.design.present": False}).point
    assert "design-network-edge.source_missing_or_not_output" in _findings(engine, absent)


def test_a_point_with_no_active_segment_is_refused() -> None:
    class Optional(DataflowDesign):
        id = "optional"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        present = Decision(bool, values=(False, True))
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        source = NetworkBoundary(produce.input("source"), when=present)
        stream = NetworkBoundary(produce.output("stream"), when=present)

    engine, point, _design = _started(Optional)
    empty = engine.commit_assignments(point, {"root.design.present": False}).point
    assert _network(engine, empty) == DataflowNetwork((), (), ())
    assert "design-no-active-segment" in _findings(engine, empty)


def test_duplicate_edge_and_boundary_ids_are_authoring_errors() -> None:
    class DuplicateEdge(DataflowDesign):
        id = "duplicate_edge"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        other = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream")), name="stream"
        )

    with pytest.raises(AuthoringError, match="edge id 'stream' twice"):
        _compiled(DuplicateEdge)

    class DuplicateBoundary(DataflowDesign):
        id = "duplicate_boundary"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        source = NetworkBoundary(produce.input("source"))
        again = NetworkBoundary(produce.output("stream"), name="source")

    with pytest.raises(AuthoringError, match="boundary id 'source' twice"):
        _compiled(DuplicateBoundary)


def test_endpoint_direction_is_checked_where_it_is_declared() -> None:
    class Segment(DataflowDesign):
        id = "segment"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )

    with pytest.raises(AuthoringError, match="NetworkEdge source must name an output port"):
        NetworkEdge(Segment.produce.input("source"), EdgeSink(Segment.produce.input("source")))
    with pytest.raises(AuthoringError, match="EdgeSink must name an input port"):
        EdgeSink(Segment.produce.output("stream"))
    with pytest.raises(AuthoringError, match="at least one EdgeSink"):
        NetworkEdge(Segment.produce.output("stream"))


def test_an_endpoint_from_another_design_is_rejected() -> None:
    class Other(DataflowDesign):
        id = "other"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )

    class Borrower(DataflowDesign):
        id = "borrower"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(Other.produce.output("stream"), EdgeSink(consume.input("stream")))

    with pytest.raises(AuthoringError, match="Kernel segment outside the class"):
        _compiled(Borrower)


def test_a_repeated_sink_on_one_edge_is_refused() -> None:
    class Repeated(DataflowDesign):
        id = "repeated_sink"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"),
            EdgeSink(consume.input("stream")),
            EdgeSink(consume.input("stream")),
        )
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    _refuses(Repeated, "design-network-edge.sink_duplicate")


def test_a_non_bijective_position_map_is_refused() -> None:
    class Collapsing(DataflowDesign):
        id = "collapsing"
        version = "1"
        extent = Input(int)
        lanes = Input(int)

        @derived(POSITION_MAP_SEMANTICS, extent=extent)
        def collapsed(*, extent: int) -> PositionMap:
            return PositionMap(((index,), (0,)) for index in range(extent))

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream"), position_map=collapsed)
        )
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    _refuses(Collapsing, "design-network-position_map.not_injective")


def test_a_beat_count_mismatch_across_an_edge_is_refused() -> None:
    """Two segments folded differently cannot be wired directly."""

    class MisfoldedKernel(Kernel):
        id = "misfolded"
        computation = CONSUME
        extent = Input(int)
        lanes = Input(int)

        @derived(int, lanes=lanes)
        def halved(*, lanes: int) -> int:
            return max(1, lanes // 2)

        region = RegionDeclaration(
            family="test.consume",
            version="1",
            construct=_consumer_region,
            extent=extent,
            lanes=halved,
        )

        @classmethod
        def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
            return ComponentABI("misfolded", ())

    class Misfolded(DataflowDesign):
        id = "misfolded_design"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(MisfoldedKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    codes = _findings(*_started(Misfolded)[:2])
    assert "design-network-edge.beat_count_mismatch" in codes
    assert "design-network-edge.element_count_mismatch" in codes


def test_a_multiply_authored_fan_out_is_refused() -> None:
    class Twice(DataflowDesign):
        id = "twice_out"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        left = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="left"
        )
        right = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="right"
        )
        to_left = NetworkEdge(produce.output("stream"), EdgeSink(left.input("stream")))
        to_right = NetworkEdge(produce.output("stream"), EdgeSink(right.input("stream")))
        source = NetworkBoundary(produce.input("source"))
        left_result = NetworkBoundary(left.output("result"))
        right_result = NetworkBoundary(right.output("result"))

    _refuses(Twice, "design-network-endpoint.output_ownership")


def test_a_both_active_conditional_edge_and_boundary_is_refused() -> None:
    class BothActive(DataflowDesign):
        id = "both_active"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        supplied = Decision(bool, values=(False, True))
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream")), when=supplied
        )
        # Deliberately the *same* condition rather than its complement.
        external = NetworkBoundary(consume.input("stream"), when=supplied)
        source = NetworkBoundary(produce.input("source"))
        produced = NetworkBoundary(produce.output("stream"), when=supplied)
        result = NetworkBoundary(consume.output("result"))

    engine, point, _design = _started(BothActive)
    both = engine.commit_assignments(point, {"root.design.supplied": True}).point
    assert "design-network-endpoint.input_ownership" in _findings(engine, both)
    neither = engine.commit_assignments(point, {"root.design.supplied": False}).point
    assert "design-network-endpoint.input_ownership" in _findings(engine, neither)


# -- KD5: configured Design ---------------------------------------------------


def _placed(design_type: type[DataflowDesign]) -> type[Space]:
    """A root Space placing one Design at exactly the namespace `_started` uses.

    The engine paths are identical to the fragment form above -- ``root.lanes``,
    ``root.design....`` -- so a test may probe either way and mean the same
    point.  Written as a factory rather than one class per Design because the
    Design under test is the parameter.
    """

    class Root(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        design = Subspace(design_type, extent=extent, lanes=lanes)

    return Root


def _occurrence(
    design_type: type[DataflowDesign],
    extent: int = 8,
    lanes: int = 2,
    *,
    select: Mapping[str, str] | None = None,
    design_decisions: Sequence[tuple[Decision[object], object]] = (),
    kernel_decisions: Sequence[tuple[str, str, Decision[object], object]] = (),
) -> DataflowDesign:
    """One attached Design occurrence, specialized only through the public API.

    Every step is a descriptor or a lifecycle call: select an alternative
    through its segment's view, assign a Design Decision on the Design, assign a
    candidate's own Decision on that candidate.  Each returns a successor, and
    the Design at that successor is reached back through ``root`` -- which is
    the navigation the class-centered design intends and the reason no test here
    needs an engine path.
    """

    root_type = _placed(design_type)
    root = root_type.start({root_type.extent: extent}, namespace="root")
    design = cast(DataflowDesign, root.assign(root_type.lanes, lanes).design)
    for role, alternative in (select or {}).items():
        view = cast(ChoiceView, getattr(design, role))
        design = cast(DataflowDesign, view.select(alternative).root.design)
    for declaration, value in design_decisions:
        design = design.assign(declaration, value)
    for role, alternative, declaration, value in kernel_decisions:
        view = cast(ChoiceView, getattr(design, role))
        kernel = view.alternative(alternative)
        design = cast(DataflowDesign, kernel.assign(declaration, value).root.design)
    return design


class Selectable(DataflowDesign):
    """A two-segment Design whose consumer has two candidates."""

    id = "selectable"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    produce = KernelChoice(
        Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
    )
    consume = KernelChoice(
        Subspace(ConsumerKernel, extent=extent, lanes=lanes),
        Subspace(PipelinedConsumerKernel, extent=extent, lanes=lanes),
        computation=CONSUME,
    )

    stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
    source = NetworkBoundary(produce.input("source"))
    result = NetworkBoundary(consume.output("result"))


def test_a_design_generates_its_dataflow_projection_over_regions_alone() -> None:
    _harness, design = _compiled(Chain)
    assert "root.design.dataflow_accepts" in {item.name for item in design.spec.constraint_sets}
    profile = next(
        item for item in design.spec.readiness_profiles if item.name == "root.design.dataflow_ready"
    )
    assert {str(path) for path in profile.properties} == {
        "semantic.root.design.network",
        "semantic.root.design.produce.region",
        "semantic.root.design.consume.region",
    }
    projection = design.projection("dataflow")
    assert projection.name == "root.design.dataflow"
    assert projection.output.path == QualifiedPath("semantic.root.design.network")


def test_the_design_occurrence_answers_its_network_with_no_wrapper() -> None:
    design = _occurrence(Chain)
    assert isinstance(design, Chain)
    assert type(design).id == "chain"
    assessment = design.dataflow
    assert assessment.readiness.ready is True
    assert assessment.accepted_answer == Decided(_network(*_started(Chain)[:2]))
    assert set(design.roles) == {"produce", "consume"}
    assert design.selected("produce") == Decided("producer")
    assert design.selected("consume") == Decided("consumer")


def test_each_role_reports_its_own_region_node_and_computation() -> None:
    design = _occurrence(Chain)
    network = design.dataflow.accepted_answer
    assert isinstance(network, Decided)
    for role in design.roles:
        region = design.region(role)
        assert isinstance(region, Decided)
        assert network.value.node(design.node_id(role)).region == region.value
        assert design.computation(role) is (PRODUCE if role == "produce" else CONSUME)
    assert design.region_family("produce") == Decided(("test.produce", "1"))


def test_the_selected_kernel_is_reached_as_an_occurrence_not_a_copy() -> None:
    design = _occurrence(Chain)
    produce = design.kernel("produce")
    assert isinstance(produce, Decided)
    assert isinstance(produce.value, ProducerKernel)
    assert produce.value.dataflow.accepted_answer == design.region("produce")


def test_the_network_resolves_before_any_candidate_is_physically_ready() -> None:
    """U3's claim: a Region is not waiting on a parameter, an ABI or a target."""

    design = _occurrence(Selectable, select={"consume": "pipelined_consumer"})
    # `stages` is the chosen candidate's own physical Decision and is uncommitted.
    consume = design.kernel("consume")
    assert isinstance(consume, Decided)
    assert isinstance(consume.value.physical.accepted_answer, Unresolved)

    assessment = design.dataflow
    assert assessment.readiness.ready is True
    assert isinstance(assessment.accepted_answer, Decided)
    assert {node.id for node in assessment.accepted_answer.value.nodes} == {
        "produce",
        "consume",
    }


def test_an_uncommitted_selector_leaves_the_network_unresolved() -> None:
    design = _occurrence(Selectable)
    assessment = design.dataflow
    assert isinstance(assessment.accepted_answer, Unresolved)
    assert any(
        "root.design.consume.kernel" in str(finding.path)
        for finding in assessment.accepted_answer.findings
    )


def test_an_uncommitted_design_decision_leaves_the_network_unresolved() -> None:
    class Chosen(DataflowDesign):
        id = "chosen"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        spare = Decision(int, values=(1, 2))
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        source = NetworkBoundary(produce.input("source"))
        stream = NetworkBoundary(produce.output("stream"))

    assert isinstance(_occurrence(Chosen).dataflow.accepted_answer, Unresolved)
    complete = _occurrence(Chosen, design_decisions=((Chosen.spare, 1),))
    assert isinstance(complete.dataflow.accepted_answer, Decided)


def test_an_inactive_segment_contributes_no_node() -> None:
    class Optional(DataflowDesign):
        id = "optional_segment"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        present = Decision(bool, values=(False, True))

        @derived(bool, present=present)
        def external(*, present: bool) -> bool:
            return not present

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream")), when=present
        )
        source = NetworkBoundary(produce.input("source"), when=present)
        supplied = NetworkBoundary(consume.input("stream"), when=external)
        result = NetworkBoundary(consume.output("result"))

    design = _occurrence(Optional, design_decisions=((Optional.present, False),))
    network = design.dataflow.accepted_answer
    assert isinstance(network, Decided)
    assert tuple(node.id for node in network.value.nodes) == ("consume",)
    assert design.is_active("produce") == Decided(False)
    assert design.is_active("consume") == Decided(True)
    assert isinstance(design.region("produce"), Absent)


def test_two_occurrences_of_one_design_resolve_independently() -> None:
    class Root(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        left = Subspace(Chain, extent=extent, lanes=lanes)
        right = Subspace(Chain, extent=extent, lanes=lanes)

    root = Root.start({Root.extent: 8}, namespace="root").assign(Root.lanes, 2)
    first = cast(Chain, root.left)
    second = cast(Chain, root.right)
    assert first is not second
    assert first.kernel("produce") != second.kernel("produce")
    assert first.dataflow.accepted_answer == second.dataflow.accepted_answer


def test_a_structurally_invalid_topology_refuses_the_network() -> None:
    class Broken(DataflowDesign):
        id = "broken"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        result = NetworkBoundary(consume.output("result"))

    answer = _occurrence(Broken).dataflow.accepted_answer
    assert isinstance(answer, Absent)
    assert "design-network-endpoint.input_ownership" in {
        finding.code for finding in answer.findings
    }


def test_every_network_refusal_survives_python_o() -> None:
    script = (
        "from dataflow.designs.test_design_compiler import (\n"
        "    Selectable, _occurrence)\n"
        "from finn.dataflow._engine import Unresolved\n"
        "answer = _occurrence(Selectable).dataflow.accepted_answer\n"
        "assert type(answer) is Unresolved, answer\n"
        "print('refused')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "refused" in result.stdout


# -- decision ownership -------------------------------------------------------


class Folding(Space):
    """A Design-owned helper that happens to hold a Decision."""

    extent = Input(int)
    lanes = Decision(int, domain=divisors_of(extent))
    exports = (lanes,)


class NestedOwnership(DataflowDesign):
    """A Design whose semantic choice lives in a helper Space it owns."""

    id = "nested_ownership"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    fold = Subspace(Folding, extent=extent)

    produce = KernelChoice(
        Subspace(ProducerKernel, extent=extent, lanes=fold.lanes), computation=PRODUCE
    )
    consume = KernelChoice(
        Subspace(PipelinedConsumerKernel, extent=extent, lanes=fold.lanes), computation=CONSUME
    )

    stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
    source = NetworkBoundary(produce.input("source"))
    result = NetworkBoundary(consume.output("result"))


def _nested(stages: int = 1) -> DataflowDesign:
    design = _occurrence(NestedOwnership)
    fold = cast(Space, design.fold)
    design = cast(DataflowDesign, fold.assign(Folding.lanes, 2).root.design)
    return _occurrence_assign_candidate(design, "consume", "pipelined_consumer", stages)


def _occurrence_assign_candidate(
    design: DataflowDesign, role: str, alternative: str, stages: int
) -> DataflowDesign:
    view = cast(ChoiceView, getattr(design, role))
    kernel = view.alternative(alternative)
    return cast(DataflowDesign, kernel.assign(PipelinedConsumerKernel.stages, stages).root.design)


def test_a_decision_in_a_design_owned_helper_is_a_design_assignment() -> None:
    design = _nested()
    assert dict(design.assignments) == {QualifiedPath("root.design.fold.lanes"): 2}
    assert design.imported_decisions == (QualifiedPath("root.lanes"),)


def test_a_contained_kernels_decision_is_neither_retained_nor_imported() -> None:
    design = _nested(stages=2)
    stages = QualifiedPath("root.design.consume.pipelined_consumer.stages")
    assert stages not in design.assignments
    assert stages not in design.imported_decisions
    # It belongs to the Kernel that owns it, and is carried in its build unit.
    consume = design.kernel("consume")
    assert isinstance(consume, Decided)
    built = consume.value.physical.accepted_answer
    assert isinstance(built, Decided)
    assert dict(built.value.assignments) == {"stages": 2}


def test_a_selector_is_a_design_assignment_not_imported_provenance() -> None:
    design = _occurrence(
        Selectable,
        select={"consume": "pipelined_consumer"},
        kernel_decisions=(("consume", "pipelined_consumer", PipelinedConsumerKernel.stages, 1),),
    )
    assert QualifiedPath("root.design.consume.kernel") in design.assignments
    assert design.imported_decisions == (QualifiedPath("root.lanes"),)


def test_only_decisions_outside_the_design_are_imported() -> None:
    """Every path in `imported_decisions` names something the Design does not own."""

    _harness, compiled = _compiled(NestedOwnership)
    internal = {str(item.path) for item in compiled.spec.decisions}
    design = _nested()
    assert not {str(path) for path in design.imported_decisions} & internal


def test_an_inactive_connection_does_not_demand_its_position_map() -> None:
    """An inactive topology declaration is inactive, maps included."""

    class ConditionalMap(DataflowDesign):
        id = "conditional_map"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        supplied = Decision(bool, values=(False, True))
        reversal = Decision(bool, values=(False, True))

        @derived(bool, supplied=supplied)
        def external(*, supplied: bool) -> bool:
            return not supplied

        @derived(POSITION_MAP_SEMANTICS, extent=extent, reversal=reversal)
        def chosen_map(*, extent: int, reversal: bool) -> PositionMap:
            if reversal:
                return PositionMap(((index,), (extent - 1 - index,)) for index in range(extent))
            return PositionMap(((index,), (index,)) for index in range(extent))

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=supplied,
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"),
            EdgeSink(consume.input("stream"), position_map=chosen_map),
            when=supplied,
        )
        source = NetworkBoundary(produce.input("source"), when=supplied)
        external_stream = NetworkBoundary(consume.input("stream"), when=external)
        result = NetworkBoundary(consume.output("result"))

    engine, point, _design = _started(ConditionalMap)
    # `reversal` is deliberately never committed.
    inactive = engine.commit_assignments(point, {"root.design.supplied": False}).point
    network = _network(engine, inactive)
    assert network.edges == ()
    assert tuple(node.id for node in network.nodes) == ("consume",)
    assert _valid(engine, inactive)

    # Active, and now the map genuinely is required.
    active = engine.commit_assignments(point, {"root.design.supplied": True}).point
    assert isinstance(engine.query_property(active, "semantic.root.design.network"), Unresolved)
    resolved = engine.commit_assignments(active, {"root.design.reversal": False}).point
    assert len(_network(engine, resolved).edges) == 1
    assert _valid(engine, resolved)


def test_a_declared_position_map_is_forwarded_through_its_own_property() -> None:
    class Mapped(DataflowDesign):
        id = "mapped"
        version = "1"
        extent = Input(int)
        lanes = Input(int)

        @derived(POSITION_MAP_SEMANTICS, extent=extent)
        def identity_map(*, extent: int) -> PositionMap:
            return PositionMap(((index,), (index,)) for index in range(extent))

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(
            produce.output("stream"), EdgeSink(consume.input("stream"), position_map=identity_map)
        )
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    engine, point, design = _started(Mapped)
    assert "semantic.root.design.stream.position_map.0" in {
        str(item.path) for item in design.spec.properties
    }
    assert _valid(engine, point)


class Always(Space):
    """A branch case whose selected output is a Boolean the Design can gate on."""

    extent = Input(int)

    @derived(bool, extent=extent)
    def flag(*, extent: int) -> bool:
        return extent > 0

    exports = (flag,)


def test_a_branch_output_reaches_a_boundary_condition_and_a_kernel_parameter() -> None:
    class Chooser(Space):
        extent = Input(int)

        @derived(int, extent=extent)
        def depth(*, extent: int) -> int:
            return extent

        exports = (depth,)

    class Parameterized(Kernel):
        id = "parameterized"
        computation = CONSUME
        extent = Input(int)
        lanes = Input(int)
        choice = SubspaceChoice({"only": Subspace(Chooser, extent=extent)}, outputs=("depth",))
        region = RegionDeclaration(
            family="test.consume",
            version="1",
            construct=_consumer_region,
            extent=extent,
            lanes=lanes,
        )
        DEPTH = ModuleParameter(choice.depth)

        @classmethod
        def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
            return ComponentABI("parameterized", (), (("DEPTH", str(parameters["DEPTH"])),))

    class Gated(DataflowDesign):
        id = "gated_boundary"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        policy = SubspaceChoice({"always": Subspace(Always, extent=extent)}, outputs=("flag",))
        produce = KernelChoice(
            Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE
        )
        consume = KernelChoice(
            Subspace(Parameterized, extent=extent, lanes=lanes), computation=CONSUME
        )
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"), when=policy.flag)

    design = _occurrence(Gated)
    consume = design.kernel("consume")
    assert isinstance(consume, Decided)
    built = consume.value.physical.accepted_answer
    assert isinstance(built, Decided)
    assert dict(built.value.parameters) == {"DEPTH": 8}
    network = design.dataflow.accepted_answer
    assert isinstance(network, Decided)
    assert tuple(item.id for item in network.value.boundaries) == ("result", "source")


def test_the_design_occurrence_is_attached_and_answers_its_own_declarations() -> None:
    """One protocol now: values exist on an attached occurrence and nowhere else."""

    design = _occurrence(Chain)
    assert is_attached_occurrence(design)
    assert design.node_id("produce") == "produce"
    assert design.region_family("produce") == Decided(("test.produce", "1"))
    with pytest.raises(AttributeError, match="only on an attached Space occurrence"):
        _ = Chain.__new__(Chain).extent


# -- which of a candidate's constraints gate the Network -----------------------
#
# U3's claim has two halves and only one of them was implemented.  Excluding a
# candidate's ``physical_support`` from the Design's question means nothing
# unless the *rest* of that candidate's constraints are in it -- and they were
# not, so a Kernel that refused its own Region for a reason it had classified
# as semantic left the Network accepted.


def _consumer_with(name: str, verdict: bool, *, physical: bool):
    """One consumer Kernel whose single constraint is classified either way."""

    class Classified(Kernel):
        id = name
        computation = CONSUME
        extent = Input(int)
        lanes = Input(int)
        region = RegionDeclaration(
            family="test.consume",
            version="1",
            construct=_consumer_region,
            extent=extent,
            lanes=lanes,
        )

        @constraint(lanes=lanes)
        def supported(*, lanes: int) -> bool:
            return verdict

        if physical:
            physical_support = ConstraintGroup(supported, name="realizable")
        else:
            dataflow_support = ConstraintGroup(supported)

        @classmethod
        def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
            return ComponentABI(name, ())

    Classified.__name__ = name
    return Classified


def _chain_over(*candidates: type[Kernel]) -> type[DataflowDesign]:
    # The declarations are built here rather than in the class body because a
    # generator expression there cannot see the class namespace.
    declared_extent = Input(int)
    declared_lanes = Input(int)
    cases = tuple(
        Subspace(item, extent=declared_extent, lanes=declared_lanes) for item in candidates
    )

    class Classified(DataflowDesign):
        id = "classified_chain"
        version = "1"
        extent = declared_extent
        lanes = declared_lanes

        produce = KernelChoice(
            Subspace(ProducerKernel, extent=declared_extent, lanes=declared_lanes),
            computation=PRODUCE,
        )
        consume = KernelChoice(*cases, computation=CONSUME)
        stream = NetworkEdge(produce.output("stream"), EdgeSink(consume.input("stream")))
        source = NetworkBoundary(produce.input("source"))
        result = NetworkBoundary(consume.output("result"))

    return Classified


def _network_verdict(design_type: type[DataflowDesign], **assignments: object) -> object:
    engine, point, design = _started(design_type)
    if assignments:
        point = engine.commit_assignments(point, assignments).point
    return design_dataflow(engine, design, point)


def test_a_candidates_semantic_refusal_refuses_the_network() -> None:
    """A Kernel that says its Region is wrong here is not overruled by silence."""

    assessment = _network_verdict(_chain_over(_consumer_with("semantic", False, physical=False)))
    assert isinstance(assessment.accepted_answer, Absent)
    assert assessment.constraints[0].verdict is False


def test_a_candidates_build_refusal_does_not_refuse_the_network() -> None:
    """The other half, unchanged: an unbuildable Kernel still contributes a Region."""

    assessment = _network_verdict(_chain_over(_consumer_with("buildable", False, physical=True)))
    assert isinstance(assessment.accepted_answer, Decided)


def test_an_unselected_candidates_refusal_is_not_applicable_rather_than_a_refusal() -> None:
    """Every candidate's constraints are in the set; only the selected one answers."""

    accepting = _consumer_with("accepting", True, physical=False)
    refusing = _consumer_with("refusing", False, physical=False)
    design_type = _chain_over(accepting, refusing)

    chosen = _network_verdict(design_type, **{"root.design.consume.kernel": "accepting"})
    assert isinstance(chosen.accepted_answer, Decided)
    assessment = chosen.constraints[0]
    assert assessment.verdict is True
    assert any(
        path.value == "constraint.root.design.consume.refusing.supported"
        for path in assessment.not_applicable
    )

    other = _network_verdict(design_type, **{"root.design.consume.kernel": "refusing"})
    assert isinstance(other.accepted_answer, Absent)
