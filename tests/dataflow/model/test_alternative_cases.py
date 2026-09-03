# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD8: one branch declaration, several external specialization algorithms.

The algorithms are imported unchanged from the generic branch-inspection tests.
That import is the point of the phase: if a Kernel segment needed its own
selection code, the seam would not be a seam.  Nothing here adds a production
default algorithm, a `choose=` callback, a policy registry, or a measurement
service.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing import cast


import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow._engine import Absent, Decided, Engine, QualifiedPath, Unresolved
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.model.branching import BranchCatalog
from finn.dataflow.model.compiler import _Ref, _compile_space, compile_space_model
from finn.dataflow.model.declarations import (
    AuthoringError,
    Decision,
    Input,
    Problem,
    Space,
    Subspace,
    derived,
    divisors_of,
)
from finn.dataflow.designs.design import (
    Boundary,
    Connection,
    DataflowDesign,
    Kernels,
    Sink,
    configure_design,
)
from finn.dataflow.kernels.kernel import Kernel, Parameter, Region
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
from finn.dataflow.model.spec_algebra import assemble_specs

from dataflow.model.test_branch_inspection import (
    Assessment,
    Root as GenericRoot,
    _model as generic_model,
    assign_case,
    cheapest_case_by_property,
    exhaustive_trial,
    first_feasible_case,
    first_globally_feasible_case,
    resolve_recursively,
)
from dataflow.designs.test_design_compiler import (
    CONSUME,
    PRODUCE,
    ConsumerKernel,
    Harness,
    ProducerKernel,
    _consumer_region,
    _producer_region,
)

# -- the alternatives ---------------------------------------------------------


def _renamed_input_region(extent: int, lanes: int) -> DataflowRegion:
    """The consumer Region with its input port under a different name."""

    folds = extent // lanes
    operand = Operand("value", DataType["INT8"], (extent,))
    sequence = BeatSequence(
        lanes,
        tuple(tuple((fold * lanes + lane,) for lane in range(lanes)) for fold in range(folds)),
    )
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("fold", folds),)),
        (
            InputInterface(
                Port("elsewhere", operand, sequence),
                ScheduledInputRequirements(
                    {
                        ((fold,), (fold * lanes + lane,)): 1
                        for fold in range(folds)
                        for lane in range(lanes)
                    }
                ),
            ),
        ),
        (
            OutputInterface(
                Port("result", operand, sequence),
                ScheduledOutputAvailability(
                    {
                        (fold * lanes + lane,): (fold,)
                        for fold in range(folds)
                        for lane in range(lanes)
                    }
                ),
            ),
        ),
    )


class BufferedConsumerKernel(Kernel):
    """Same computation and Region, different everything physical.

    Candidate-specific Input names, one Kernel-local physical Decision, an extra
    physical Parameter, an extra source contribution, and its own configured
    type -- which is the whole set of things an alternative is allowed to differ
    in without touching the Region a peer reads.
    """

    id = "buffered_consumer"
    version = "1"
    computation = CONSUME

    width = Input(int)
    parallel_lanes = Input(int)
    buffered = Decision(bool, values=(False, True))

    region = Region(
        family="test.consume",
        version="1",
        construct=_consumer_region,
        extent=width,
        lanes=parallel_lanes,
    )

    @derived(int, buffered=buffered)
    def depth(*, buffered: bool) -> int:
        return 4 if buffered else 1

    # Deliberately independent of the local Decision: a cost a policy can read
    # before it has committed anything inside the candidate.
    @derived(int, lanes=parallel_lanes)
    def cost(*, lanes: int) -> int:
        return 2 * lanes

    LANES = Parameter(parallel_lanes)
    DEPTH = Parameter(depth)

    sources = (
        CopiedSource("fixture", "buffer.sv", provides=("module:buffer",)),
        CopiedSource(
            "fixture",
            "buffered_consumer.sv",
            provides=("module:buffered_consumer",),
            requires=("module:buffer",),
        ),
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI(
            "buffered_consumer",
            (),
            (("DEPTH", str(parameters["DEPTH"])), ("LANES", str(parameters["LANES"]))),
        )


class ProducingKernel(Kernel):
    """A candidate offering the wrong computation for a consumer segment."""

    id = "producing"
    computation = PRODUCE
    extent = Input(int)
    lanes = Input(int)
    region = Region(
        family="test.produce",
        version="1",
        construct=_producer_region,
        extent=extent,
        lanes=lanes,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("producing", ())


class MisportedConsumerKernel(Kernel):
    """Right computation, but a Region the declared topology cannot wire."""

    id = "misported_consumer"
    computation = CONSUME
    extent = Input(int)
    lanes = Input(int)
    region = Region(
        family="test.consume",
        version="1",
        construct=_renamed_input_region,
        extent=extent,
        lanes=lanes,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        return ComponentABI("misported_consumer", ())


class Alternatives(DataflowDesign):
    """One segment with three candidates over one unchanged topology."""

    id = "alternatives"
    version = "1"

    extent = Input(int)
    lanes = Input(int)

    produce = Kernels(Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
    consume = Kernels(
        Subspace(ConsumerKernel, extent=extent, lanes=lanes),
        Subspace(BufferedConsumerKernel, width=extent, parallel_lanes=lanes),
        Subspace(MisportedConsumerKernel, extent=extent, lanes=lanes),
        computation=CONSUME,
    )

    stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
    source = Boundary(produce.input("source"))
    result = Boundary(consume.output("result"))


SELECTOR = QualifiedPath("root.design.consume.kernel")


class ClosedRoot(Space):
    """A closed root: the Design is a child, so the whole thing compiles publicly."""

    extent = Problem(int)
    lanes = Decision(int, domain=divisors_of(extent))
    design = Subspace(Alternatives, extent=extent, lanes=lanes)


def _catalog(root_type: type[Space] = ClosedRoot) -> BranchCatalog:
    """The branch catalog through the supported entry point and nothing else."""

    return compile_space_model(root_type, "root", problem_namespace="problem.root").branches


def _compiled(design_type: type[DataflowDesign] = Alternatives):
    harness = _compile_space(Harness, "root", problem_namespace="problem.root")
    design = _compile_space(
        design_type,
        "root.design",
        {name: cast("_Ref[object]", harness.member(name)) for name in ("extent", "lanes")},
        _allow_problem=False,
    )
    return harness, design


def _started(design_type: type[DataflowDesign] = Alternatives, extent: int = 8, lanes: int = 2):
    harness, design = _compiled(design_type)
    engine = Engine()
    point = engine.start(
        engine.validate(assemble_specs((harness.spec, design.spec))),
        {"problem.root.extent": extent},
    )
    return engine, engine.commit_assignments(point, {"root.lanes": lanes}).point, design


# -- what an alternative may differ in ----------------------------------------


def test_an_alternative_keeps_the_region_and_changes_only_physical_facts() -> None:
    engine, point, design = _started()
    plain = engine.commit_assignments(point, {SELECTOR: "consumer"}).point
    buffered = engine.commit_assignments(
        point,
        {SELECTOR: "buffered_consumer", "root.design.consume.buffered_consumer.buffered": True},
    ).point
    region = "semantic.root.design.consume.region"
    assert engine.query_property(plain, region) == engine.query_property(buffered, region)

    first = configure_design(engine, design, plain)
    second = configure_design(engine, design, buffered)
    assert isinstance(first, Decided) and isinstance(second, Decided)
    assert first.value.consume.kernel_id == "consumer"
    assert second.value.consume.kernel_id == "buffered_consumer"
    assert dict(first.value.consume.parameters) == {}
    assert dict(second.value.consume.parameters) == {"LANES": 2, "DEPTH": 4}
    assert first.value.consume.contributions == ()
    assert len(second.value.consume.contributions) == 2
    assert first.value.resolved_network == second.value.resolved_network


def test_an_alternative_may_use_candidate_specific_input_names() -> None:
    branch = _catalog().branch("root.design.consume")
    assert tuple(case.id for case in branch.cases) == (
        "consumer",
        "buffered_consumer",
        "misported_consumer",
    )
    assert branch.case("buffered_consumer").decision_paths == (
        QualifiedPath("root.design.consume.buffered_consumer.buffered"),
    )
    assert branch.case("consumer").decision_paths == ()


def test_an_incompatible_computation_is_refused_at_authoring() -> None:
    class WrongOffer(DataflowDesign):
        id = "wrong_offer"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(
            Subspace(ConsumerKernel, extent=extent, lanes=lanes),
            Subspace(ProducingKernel, extent=extent, lanes=lanes),
            computation=CONSUME,
        )

    with pytest.raises(AuthoringError, match="requires computation test.consume"):
        _compiled(WrongOffer)


def test_a_candidate_whose_region_breaks_the_topology_is_selectable_but_infeasible() -> None:
    engine, point, design = _started()
    broken = engine.commit_assignments(point, {SELECTOR: "misported_consumer"}).point
    assessment = engine.evaluate_constraint_set(broken, "root.design.feasibility")
    assert assessment.verdict is False
    codes = {
        finding.code
        for answer in assessment.answers.values()
        if isinstance(answer, (Absent, Unresolved))
        for finding in answer.findings
    }
    # The canon names it, and no adapter is silently inserted.
    assert "design-network-edge.sink_missing_or_not_input" in codes
    assert isinstance(configure_design(engine, design, broken), Unresolved)


# -- external algorithms, imported unchanged ----------------------------------


def test_explicit_selection_by_case_id() -> None:
    engine, point, design = _started()
    branch = _catalog().branch("root.design.consume")
    chosen = assign_case(engine, point, branch, "buffered_consumer")
    assert dict(chosen.assignments)[SELECTOR] == "buffered_consumer"


def test_first_feasible_case() -> None:
    engine, point, _design = _started()
    branch = _catalog().branch("root.design.consume")
    chosen = first_feasible_case(engine, point, branch)
    assert chosen is not None
    assert chosen[0] == "consumer"


def test_a_design_wide_feasibility_algorithm_rejects_the_misported_candidate() -> None:
    """Case-owned constraints are not the whole story, and an algorithm can say so."""

    class MisportedFirst(DataflowDesign):
        id = "misported_first"
        version = "1"
        extent = Input(int)
        lanes = Input(int)
        produce = Kernels(Subspace(ProducerKernel, extent=extent, lanes=lanes), computation=PRODUCE)
        consume = Kernels(
            Subspace(MisportedConsumerKernel, extent=extent, lanes=lanes),
            Subspace(ConsumerKernel, extent=extent, lanes=lanes),
            computation=CONSUME,
        )
        stream = Connection(produce.output("stream"), Sink(consume.input("stream")))
        source = Boundary(produce.input("source"))
        result = Boundary(consume.output("result"))

    class MisportedRoot(Space):
        extent = Problem(int)
        lanes = Decision(int, domain=divisors_of(extent))
        design = Subspace(MisportedFirst, extent=extent, lanes=lanes)

    engine, point, design = _started(MisportedFirst)
    branch = _catalog(MisportedRoot).branch("root.design.consume")
    assert tuple(case.id for case in branch.cases) == ("misported_consumer", "consumer")

    # Its own constraints accept it; the composition does not.
    local = first_feasible_case(engine, point, branch)
    assert local is not None and local[0] == "misported_consumer"

    chosen = first_globally_feasible_case(engine, point, branch)
    assert chosen is not None and chosen[0] == "consumer"
    assert isinstance(configure_design(engine, design, chosen[1]), Decided)


def test_exhaustive_trial_reports_every_final_case_assessment() -> None:
    engine, point, design = _started()
    branch = _catalog().branch("root.design.consume")
    assessments = exhaustive_trial(engine, point, branch)
    assert tuple(item.case_id for item in assessments) == (
        "consumer",
        "buffered_consumer",
        "misported_consumer",
    )
    assert all(isinstance(item, Assessment) for item in assessments)
    # `exhaustive_trial` scores case-owned constraints, and by those the
    # misported candidate is fine -- its failure is the composition's, which is
    # what `first_globally_feasible_case` above is for.
    assert not any(item.refused for item in assessments)
    assert {item.case_id for item in assessments if not item.ready} == {"buffered_consumer"}


def test_nested_branch_traversal_reaches_a_kernel_segment() -> None:
    engine, point, design = _started()
    catalog = _catalog()
    for namespace in catalog.namespaces:
        point = resolve_recursively(engine, point, catalog, catalog.branch(namespace))
    assert dict(point.assignments)[SELECTOR] == "consumer"


def test_a_cost_guided_algorithm_reads_a_caller_designated_property() -> None:
    """The path comes from the catalog; the algorithm never builds one."""

    engine, point, _design = _started()
    branch = _catalog().branch("root.design.consume")
    assert branch.case("buffered_consumer").property_named("cost") == QualifiedPath(
        "semantic.root.design.consume.buffered_consumer.cost"
    )
    with pytest.raises(KeyError):
        branch.case("consumer").property_named("cost")
    chosen = cheapest_case_by_property(engine, point, branch, "cost")
    assert chosen is not None and chosen[0] == "buffered_consumer"
    assert dict(chosen[1].assignments)[SELECTOR] == "buffered_consumer"


# -- required behaviour -------------------------------------------------------


def test_a_rejected_trial_leaves_the_original_point_untouched() -> None:
    engine, point, design = _started()
    branch = _catalog().branch("root.design.consume")
    broken = assign_case(engine, point, branch, "misported_consumer")
    assert dict(point.assignments) == {QualifiedPath("root.lanes"): 2}
    assert broken is not point
    good = assign_case(engine, point, branch, "consumer")
    assert isinstance(configure_design(engine, design, good), Decided)


def test_an_unresolved_nested_decision_stays_visible_rather_than_refusing() -> None:
    engine, point, design = _started()
    pending = engine.commit_assignments(point, {SELECTOR: "buffered_consumer"}).point
    answer = configure_design(engine, design, pending)
    assert isinstance(answer, Unresolved)
    assert any("buffered_consumer.buffered" in str(finding.path) for finding in answer.findings)


def test_an_inactive_candidates_constraints_do_not_reject_a_sibling() -> None:
    engine, point, design = _started()
    chosen = engine.commit_assignments(point, {SELECTOR: "consumer"}).point
    assert engine.evaluate_constraint_set(chosen, "root.design.feasibility").verdict is True
    for namespace in ("buffered_consumer", "misported_consumer"):
        absent = engine.query_property(chosen, f"semantic.root.design.consume.{namespace}.region")
        assert isinstance(absent, Absent)
        assert not absent.is_rejection


def test_committing_the_selector_changes_only_what_is_allowed_to_vary() -> None:
    engine, point, design = _started()
    plain = configure_design(
        engine,
        design,
        assign_case(engine, point, _catalog().branch("root.design.consume"), "consumer"),
    )
    buffered = configure_design(
        engine,
        design,
        engine.commit_assignments(
            point,
            {
                SELECTOR: "buffered_consumer",
                "root.design.consume.buffered_consumer.buffered": False,
            },
        ).point,
    )
    assert isinstance(plain, Decided) and isinstance(buffered, Decided)
    assert plain.value.resolved_network == buffered.value.resolved_network
    assert plain.value.node_id("consume") == buffered.value.node_id("consume")
    assert plain.value.region_family("consume") == buffered.value.region_family("consume")
    assert plain.value.selected_candidates != buffered.value.selected_candidates
    assert plain.value.consume.kernel_id != buffered.value.consume.kernel_id


def test_the_configured_design_keeps_the_selection_but_not_the_catalog() -> None:
    engine, point, design = _started()
    answer = configure_design(
        engine,
        design,
        engine.commit_assignments(
            point,
            {
                SELECTOR: "buffered_consumer",
                "root.design.consume.buffered_consumer.buffered": True,
            },
        ).point,
    )
    assert isinstance(answer, Decided)
    configured = answer.value
    assert configured.selected_candidates["consume"] == "buffered_consumer"
    assert not any(isinstance(value, BranchCatalog) for value in vars(configured).values())
    assert not any(callable(value) for value in vars(configured).values())


def test_branch_inspection_stays_usable_outside_kernel_segments() -> None:
    """The same catalog type describes a plain Space branch and a Kernel one."""

    _engine, _point, generic = generic_model(GenericRoot)
    _harness, design = _compiled()
    kernel_branch = _catalog().branch("root.design.consume")
    generic_branch = generic.branch("root.top")
    assert type(kernel_branch) is type(generic_branch)
    assert type(kernel_branch.cases[0]) is type(generic_branch.cases[0])
