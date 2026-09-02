# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD4: Design topology resolves to one canonical, validated Network."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

from typing_extensions import Self

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, Engine, QualifiedPath, Unresolved
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design.region import POSITION_MAP_SEMANTICS
from finn.dataflow.model.compiler import _Ref, _compile_space
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    Decision,
    Input,
    Problem,
    Space,
    derived,
    divisors_of,
)
from finn.dataflow.model.design import (
    Boundary,
    Connection,
    DataflowDesign,
    Kernels,
    Sink,
    configure_design,
)
from finn.dataflow.model.kernel import Kernel, Region
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    PositionMap,
    RegionEndpoint,
)
from finn.dataflow.region import (
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
from finn.dataflow.spec_algebra import assemble_specs

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
    region = Region(
        family="test.produce", version="1", construct=_producer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("producer", ())


class ConsumerKernel(Kernel):
    id = "consumer"
    computation = CONSUME
    extent = Input(int)
    lanes = Input(int)
    region = Region(
        family="test.consume", version="1", construct=_consumer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("consumer", ())


class PipelinedConsumerKernel(Kernel):
    """Same computation and Region, one extra physical decision of its own."""

    id = "pipelined_consumer"
    computation = CONSUME
    extent = Input(int)
    lanes = Input(int)
    stages = Decision(int, values=(1, 2))
    region = Region(
        family="test.consume", version="1", construct=_consumer_region, extent=extent, lanes=lanes
    )

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI("pipelined_consumer", ())


class Chain(DataflowDesign):
    """producer -> consumer, with the remaining endpoints exposed."""

    id = "chain"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
    consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)

    stream = Connection(produce.output("stream"), Sink(consume.input("stream")))

    source = Boundary(produce.input("source"))
    result = Boundary(consume.output("result"))


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

        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(
            produce.output("stream"),
            Sink(consume.input("stream"), position_map=reversed_map),
        )
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

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
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        left = Kernels(
            Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="left"
        )
        right = Kernels(
            Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="right"
        )
        stream = Connection(
            produce.output("stream"),
            Sink(left.input("stream")),
            Sink(right.input("stream")),
        )
        source = Boundary(produce.input("source"))
        left_result = Boundary(left.output("result"))
        right_result = Boundary(right.output("result"))

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

        produce = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=supplied,
        )
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")), when=supplied)
        source = Boundary(produce.input("source"), when=supplied)
        supplied_stream = Boundary(consume.input("stream"), when=external)
        result = Boundary(consume.output("result"))

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
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("absent"), Sink(consume.input("stream")))
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

    _refuses(WrongPort, "design-network-edge.source_missing_or_not_output")


def test_a_reversed_direction_is_refused() -> None:
    class Reversed(DataflowDesign):
        id = "reversed"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("source"), Sink(consume.input("stream")))
        result = Boundary(consume.output("result"))
        produced = Boundary(produce.output("stream"))

    _refuses(Reversed, "design-network-edge.source_missing_or_not_output")


def test_an_unaccounted_endpoint_is_refused() -> None:
    class Unaccounted(DataflowDesign):
        id = "unaccounted"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        result = Boundary(consume.output("result"))

    _refuses(Unaccounted, "design-network-endpoint.input_ownership")


def test_implicit_fan_in_is_refused() -> None:
    class FanIn(DataflowDesign):
        id = "fan_in"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        left = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE, role="left"
        )
        right = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE, role="right"
        )
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        from_left = Connection(left.output("stream"), Sink(consume.input("stream")))
        from_right = Connection(right.output("stream"), Sink(consume.input("stream")))
        left_source = Boundary(left.input("source"))
        right_source = Boundary(right.input("source"))
        result = Boundary(consume.output("result"))

    _refuses(FanIn, "design-network-endpoint.input_ownership")


def test_an_endpoint_used_by_both_an_edge_and_a_boundary_is_refused() -> None:
    class Both(DataflowDesign):
        id = "both"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        also_external = Boundary(consume.input("stream"))
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

    _refuses(Both, "design-network-endpoint.input_ownership")


def test_a_cyclic_topology_is_refused() -> None:
    class Cyclic(DataflowDesign):
        id = "cyclic"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        forward = Connection(produce.output("stream"), Sink(consume.input("stream")))
        backward = Connection(consume.output("result"), Sink(produce.input("source")))

    _refuses(Cyclic, "design-network-network.cycle")


def test_active_topology_over_an_inactive_segment_is_refused() -> None:
    class Dangling(DataflowDesign):
        id = "dangling"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        present = Decision(bool, values=(False, True))
        produce = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        source = Boundary(produce.input("source"), when=present)
        result = Boundary(consume.output("result"))

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
        produce = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        source = Boundary(produce.input("source"), when=present)
        stream = Boundary(produce.output("stream"), when=present)

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
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        other = Connection(produce.output("stream"), Sink(consume.input("stream")), name="stream")

    with pytest.raises(AuthoringError, match="edge id 'stream' twice"):
        _compiled(DuplicateEdge)

    class DuplicateBoundary(DataflowDesign):
        id = "duplicate_boundary"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        source = Boundary(produce.input("source"))
        again = Boundary(produce.output("stream"), name="source")

    with pytest.raises(AuthoringError, match="boundary id 'source' twice"):
        _compiled(DuplicateBoundary)


def test_endpoint_direction_is_checked_where_it_is_declared() -> None:
    class Segment(DataflowDesign):
        id = "segment"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)

    with pytest.raises(AuthoringError, match="Connection source must name an output port"):
        Connection(Segment.produce.input("source"), Sink(Segment.produce.input("source")))
    with pytest.raises(AuthoringError, match="Sink must name an input port"):
        Sink(Segment.produce.output("stream"))
    with pytest.raises(AuthoringError, match="at least one Sink"):
        Connection(Segment.produce.output("stream"))


def test_an_endpoint_from_another_design_is_rejected() -> None:
    class Other(DataflowDesign):
        id = "other"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)

    class Borrower(DataflowDesign):
        id = "borrower"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(Other.produce.output("stream"), Sink(consume.input("stream")))

    with pytest.raises(AuthoringError, match="Kernel segment outside the class"):
        _compiled(Borrower)


def test_a_repeated_sink_on_one_edge_is_refused() -> None:
    class Repeated(DataflowDesign):
        id = "repeated_sink"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(
            produce.output("stream"),
            Sink(consume.input("stream")),
            Sink(consume.input("stream")),
        )
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

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

        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(
            produce.output("stream"), Sink(consume.input("stream"), position_map=collapsed)
        )
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

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

        region = Region(
            family="test.consume",
            version="1",
            construct=_consumer_region,
            extent=extent,
            lanes=halved,
        )

        @classmethod
        def component_abi(cls, configured: Self) -> ComponentABI:
            return ComponentABI("misfolded", ())

    class Misfolded(DataflowDesign):
        id = "misfolded_design"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(MisfoldedKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

    codes = _findings(*_started(Misfolded)[:2])
    assert "design-network-edge.beat_count_mismatch" in codes
    assert "design-network-edge.element_count_mismatch" in codes


def test_a_multiply_authored_fan_out_is_refused() -> None:
    class Twice(DataflowDesign):
        id = "twice_out"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        left = Kernels(
            Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="left"
        )
        right = Kernels(
            Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME, role="right"
        )
        to_left = Connection(produce.output("stream"), Sink(left.input("stream")))
        to_right = Connection(produce.output("stream"), Sink(right.input("stream")))
        source = Boundary(produce.input("source"))
        left_result = Boundary(left.output("result"))
        right_result = Boundary(right.output("result"))

    _refuses(Twice, "design-network-endpoint.output_ownership")


def test_a_both_active_conditional_edge_and_boundary_is_refused() -> None:
    class BothActive(DataflowDesign):
        id = "both_active"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        supplied = Decision(bool, values=(False, True))
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")), when=supplied)
        # Deliberately the *same* condition rather than its complement.
        external = Boundary(consume.input("stream"), when=supplied)
        source = Boundary(produce.input("source"))
        produced = Boundary(produce.output("stream"), when=supplied)
        result = Boundary(consume.output("result"))

    engine, point, _design = _started(BothActive)
    both = engine.commit_assignments(point, {"root.design.supplied": True}).point
    assert "design-network-endpoint.input_ownership" in _findings(engine, both)
    neither = engine.commit_assignments(point, {"root.design.supplied": False}).point
    assert "design-network-endpoint.input_ownership" in _findings(engine, neither)


# -- KD5: configured Design ---------------------------------------------------


def _configure(design_type: type[DataflowDesign], extent: int = 8, lanes: int = 2, **assignments):
    engine, point, design = _started(design_type, extent, lanes)
    if assignments:
        point = engine.commit_assignments(point, assignments).point
    return configure_design(engine, design, point)


class Selectable(DataflowDesign):
    """A two-segment Design whose consumer has two candidates."""

    id = "selectable"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
    consume = Kernels(
        Case(ConsumerKernel, extent=extent, lanes=lanes),
        Case(PipelinedConsumerKernel, extent=extent, lanes=lanes),
        computation=CONSUME,
    )

    stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
    source = Boundary(produce.input("source"))
    result = Boundary(consume.output("result"))


def test_a_design_generates_its_feasibility_set_and_readiness_profile() -> None:
    _harness, design = _compiled(Chain)
    assert "root.design.feasibility" in {item.name for item in design.spec.constraint_sets}
    profile = next(
        item for item in design.spec.readiness_profiles if item.name == "root.design.configured"
    )
    assert "semantic.root.design.network" in {str(path) for path in profile.properties}
    assert {
        "semantic.root.design.produce.region",
        "semantic.root.design.consume.region",
    } <= {str(path) for path in profile.properties}


def test_configuration_returns_an_instance_of_the_authored_class() -> None:
    answer = _configure(Chain)
    assert isinstance(answer, Decided)
    configured = answer.value
    assert isinstance(configured, Chain)
    assert type(configured).id == "chain"
    assert set(configured.kernels) == {"produce", "consume"}
    assert isinstance(configured.kernels["produce"], ProducerKernel)
    assert configured.produce is configured.kernels["produce"]
    assert configured.consume is configured.kernels["consume"]
    assert configured.selected_candidates == {"produce": "producer", "consume": "consumer"}
    assert configured.resolved_network == _network(*_started(Chain)[:2])


def test_the_configured_kernel_region_is_the_exact_network_node_region() -> None:
    answer = _configure(Chain)
    assert isinstance(answer, Decided)
    configured = answer.value
    for role, kernel in configured.kernels.items():
        node = configured.resolved_network.node(configured.node_id(role))
        assert kernel.resolved_region is node.region or kernel.resolved_region == node.region


def test_the_configured_design_retains_only_approved_state() -> None:
    answer = _configure(Selectable, **{"root.design.consume.kernel": "consumer"})
    assert isinstance(answer, Decided)
    configured = answer.value
    assert set(vars(configured)) == {
        "_compilation",
        "_values",
        "resolved_network",
        "kernels",
        "selected_candidates",
        "assignments",
        "imported_decisions",
    }
    values = list(vars(configured).values())
    assert not any(isinstance(value, Engine) for value in values)
    assert not any(hasattr(value, "design_space") for value in values)
    assert not any(
        hasattr(value, "assignments") and value is not configured.assignments
        for value in values
        if not isinstance(value, dict)
    )


def test_the_configured_design_records_selections_without_branch_baggage() -> None:
    answer = _configure(
        Selectable,
        **{
            "root.design.consume.kernel": "pipelined_consumer",
            "root.design.consume.pipelined_consumer.stages": 2,
        },
    )
    assert isinstance(answer, Decided)
    configured = answer.value
    assert configured.selected_candidates["consume"] == "pipelined_consumer"
    assert dict(configured.assignments) == {
        QualifiedPath("root.design.consume.kernel"): "pipelined_consumer"
    }
    assert not hasattr(configured, "catalog")
    assert not hasattr(configured, "branches")


def test_design_and_kernel_provenance_are_recorded() -> None:
    answer = _configure(Chain)
    assert isinstance(answer, Decided)
    configured = answer.value
    assert configured.imported_decisions == (QualifiedPath("root.lanes"),)
    assert configured.region_family("produce") == ("test.produce", "1")


def test_an_incomplete_selector_refuses_configuration() -> None:
    answer = _configure(Selectable)
    assert isinstance(answer, Unresolved)
    assert any("root.design.consume.kernel" in str(finding.path) for finding in answer.findings)


def test_an_incomplete_design_decision_refuses_configuration() -> None:
    class Chosen(DataflowDesign):
        id = "chosen"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        spare = Decision(int, values=(1, 2))
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        source = Boundary(produce.input("source"))
        stream = Boundary(produce.output("stream"))

    assert isinstance(_configure(Chosen), Unresolved)
    assert isinstance(_configure(Chosen, **{"root.design.spare": 1}), Decided)


def test_an_incomplete_kernel_physical_decision_refuses_configuration() -> None:
    pending = _configure(Selectable, **{"root.design.consume.kernel": "pipelined_consumer"})
    assert isinstance(pending, Unresolved)
    assert any(
        "root.design.consume.pipelined_consumer.stages" in str(finding.path)
        for finding in pending.findings
    )
    complete = _configure(
        Selectable,
        **{
            "root.design.consume.kernel": "pipelined_consumer",
            "root.design.consume.pipelined_consumer.stages": 2,
        },
    )
    assert isinstance(complete, Decided)


def test_an_inactive_case_is_never_configured() -> None:
    answer = _configure(Selectable, **{"root.design.consume.kernel": "consumer"})
    assert isinstance(answer, Decided)
    assert type(answer.value.kernels["consume"]) is ConsumerKernel


def test_an_inactive_segment_contributes_no_configured_kernel() -> None:
    class Optional(DataflowDesign):
        id = "optional_segment"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        present = Decision(bool, values=(False, True))

        @derived(bool, present=present)
        def external(*, present: bool) -> bool:
            return not present

        produce = Kernels(
            Case(ProducerKernel, extent=extent, lanes=lanes),
            computation=PRODUCE,
            when=present,
        )
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")), when=present)
        source = Boundary(produce.input("source"), when=present)
        supplied = Boundary(consume.input("stream"), when=external)
        result = Boundary(consume.output("result"))

    answer = _configure(Optional, **{"root.design.present": False})
    assert isinstance(answer, Decided)
    assert set(answer.value.kernels) == {"consume"}
    assert tuple(node.id for node in answer.value.resolved_network.nodes) == ("consume",)


def test_two_occurrences_of_one_design_configure_independently() -> None:
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    bindings = {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")}
    left = _compile_space(Chain, "root.left", bindings, _allow_problem=False)
    right = _compile_space(Chain, "root.right", bindings, _allow_problem=False)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, left.spec, right.spec))),
        {"problem.root.extent": 8},
    )
    point = engine.commit_assignments(point, {"root.lanes": 2}).point
    first = configure_design(engine, left, point)
    second = configure_design(engine, right, point)
    assert isinstance(first, Decided) and isinstance(second, Decided)
    assert first.value is not second.value
    assert first.value.kernels["produce"] is not second.value.kernels["produce"]
    assert first.value.resolved_network == second.value.resolved_network


def test_an_infeasible_point_refuses_configuration() -> None:
    class Broken(DataflowDesign):
        id = "broken"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Case(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(Case(ConsumerKernel, extent=extent, lanes=lanes), computation=CONSUME)
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        result = Boundary(consume.output("result"))

    answer = _configure(Broken)
    assert isinstance(answer, Unresolved)
    assert "design-network-endpoint.input_ownership" in {
        finding.code for finding in answer.findings
    }


def test_every_configuration_refusal_survives_python_o() -> None:
    script = (
        "from dataflow.model.test_design_compiler import (\n"
        "    Selectable, _configure)\n"
        "from finn.dataflow._engine import Unresolved\n"
        "answer = _configure(Selectable)\n"
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
