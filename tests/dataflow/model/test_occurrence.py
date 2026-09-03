# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U1: the occurrence lifecycle and the validated projection over one point.

Every Space below is synthetic.  That is the point of the phase: the claim is
about a generic lifecycle, and justifying it against MVAU would prove only that
the layer fits the one shape it was written beside.  Nothing here names a
Kernel, a Region, a Design, or a Network.

The suite is the U1 forcing matrix, in order:

```text
one root, two occurrences of one child class      Stage under `pair`
nested Use and OneOf                              Pair.inner inside Root.pair
child assignment ambiguity rejection              §ambiguity
readiness True, projection still rejected         §validated projections
one constraint in two projections                 Stage.fits in model and build
partial then complete successor occurrence        §specialization
stale source fingerprint                          §frozen context
no raw runtime through public methods             §capability
callback capability audit in a subprocess         §capability
```
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import pytest

from finn.dataflow._engine import (
    Absent,
    Answer,
    Decided,
    DesignPoint,
    DesignSpace,
    Engine,
    FindingKind,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.model.compiler import SpaceModel, compile_space_model
from finn.dataflow.model.declarations import (
    AuthoringError,
    Case,
    ConstraintGroup,
    Decision,
    Input,
    OneOf,
    Problem,
    Projection,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.occurrence import Diagnostic, Occurrence, ProjectionAssessment

# -- synthetic spaces ---------------------------------------------------------


class Stage(Space):
    """A leaf placed five times over, with two projections sharing a constraint."""

    width = Input(int)
    lanes = Decision(int, values=(1, 2, 3, 4))

    @derived(int, width=width, lanes=lanes)
    def throughput(*, width: int, lanes: int) -> int:
        return width * lanes

    @constraint(throughput=throughput)
    def fits(*, throughput: int) -> object:
        if throughput > 16:
            return reject("stage-too-wide", "this stage does not fit", values={"got": throughput})
        return True

    @constraint(lanes=lanes)
    def lanes_are_powers_of_two(*, lanes: int) -> object:
        if lanes not in (1, 2, 4):
            return reject("stage-odd-lanes", "a built stage folds by a power of two")
        return True

    ready = Readiness(decisions=(lanes,), properties=(throughput,), name="ready")
    model_checks = ConstraintGroup(fits, name="model_checks")
    build_checks = ConstraintGroup(fits, lanes_are_powers_of_two, name="build_checks")

    #: Two projections over one output.  ``fits`` belongs to both groups and so
    #: to both projections, which is the many-to-many membership the contract
    #: requires: one constraint can be a model obligation and a build obligation
    #: at once without being declared twice.
    model_view = Projection(throughput, readiness=ready, constraints=model_checks)
    build_view = Projection(throughput, readiness=ready, constraints=build_checks)

    exports = (throughput,)


class Fallback(Space):
    """A second alternative, exporting the same output under different terms."""

    width = Input(int)

    @derived(int, width=width)
    def throughput(*, width: int) -> int:
        return width

    exports = (throughput,)


class Pair(Space):
    """Two occurrences of one class, plus a branch nested inside a Use."""

    width = Input(int)
    first = Use(Stage, width=width)
    second = Use(Stage, width=width)
    inner = OneOf(
        Case(Stage, name="stage", width=width),
        Case(Fallback, name="fallback", width=width),
        outputs=("throughput",),
    )
    exports = ()


class Root(Space):
    width = Problem(int)
    supplied = Problem(bool)

    pair = Use(Pair, width=width)
    choice = OneOf(
        Case(Stage, name="stage", width=width),
        Case(Fallback, name="fallback", width=width),
        outputs=("throughput",),
    )
    selected = choice.throughput

    #: A conditional placement, so a projection can be *finally inapplicable*
    #: rather than merely unresolved.
    optional = Use(Stage, when=supplied, width=width)
    optional_value = optional.throughput

    chosen_ready = Readiness(properties=(selected,), name="chosen_ready")
    result = Projection(selected, readiness=chosen_ready, name="result")
    supply = Projection(optional_value, name="supply")


def model() -> SpaceModel:
    return compile_space_model(Root, "root", problem_namespace="problem.root")


def start(*, width: int = 4, supplied: bool = True) -> Occurrence:
    return model().start({"problem.root.width": width, "problem.root.supplied": supplied})


def first_stage(occurrence: Occurrence) -> Occurrence:
    return occurrence.child(Root.pair).child(Pair.first)


def codes(answer: Answer[object]) -> tuple[str, ...]:
    assert isinstance(answer, (Absent, Unresolved))
    return tuple(finding.code for finding in answer.findings)


# -- the shape of an occurrence -----------------------------------------------


def test_a_root_occurrence_names_its_class_namespace_and_scope() -> None:
    root = start()
    assert root.space_type is Root
    assert root.namespace == "root"
    assert root.scope == ("root",)
    assert repr(root) == "Occurrence(Root at root)"


def test_one_class_placed_twice_gives_two_distinct_child_occurrences() -> None:
    root = start()
    pair = root.child(Root.pair)
    left = pair.child(Pair.first)
    right = pair.child(Pair.second)
    assert left.space_type is right.space_type is Stage
    assert left.namespace == "root.pair.first"
    assert right.namespace == "root.pair.second"
    assert left.scope == ("root", "pair", "first")
    assert right.scope == ("root", "pair", "second")


def test_a_branch_nested_inside_a_use_is_reached_by_naming_both() -> None:
    root = start()
    inner = root.child(Root.pair).child(Pair.inner, "fallback")
    assert inner.space_type is Fallback
    assert inner.namespace == "root.pair.inner.fallback"
    assert inner.scope == ("root", "pair", "inner", "fallback")


def test_a_child_view_reaches_the_root_over_the_same_point() -> None:
    root = start()
    assert first_stage(root).root.namespace == "root"


def test_branch_inspection_is_reachable_without_a_path() -> None:
    root = start()
    info = root.branch(Root.choice)
    assert info.selector == QualifiedPath("root.choice.case")
    assert tuple(case.id for case in info.cases) == ("stage", "fallback")


def test_an_unselected_branch_refuses_to_guess_which_case_was_meant() -> None:
    root = start()
    with pytest.raises(RequestError) as raised:
        root.child(Root.choice)
    assert [finding.code for finding in raised.value.findings] == ["occurrence-branch-unselected"]


def test_omitting_the_case_follows_the_committed_selector() -> None:
    root = start().assign(Root.choice, "fallback")
    assert root.child(Root.choice).space_type is Fallback


# -- ambiguity ----------------------------------------------------------------


def test_a_declaration_of_a_class_placed_five_times_is_refused_at_the_root() -> None:
    """The whole reason a view is bound to a namespace rather than a class."""

    root = start()
    with pytest.raises(AuthoringError) as raised:
        root.assign(Stage.lanes, 2)
    message = str(raised.value)
    assert "not owned by Root at root" in message
    for namespace in (
        "root.pair.first",
        "root.pair.second",
        "root.pair.inner.stage",
        "root.choice.stage",
        "root.optional",
    ):
        assert namespace in message
    assert "never inferred from a Python class" in message


def test_the_same_declaration_is_accepted_through_the_view_that_owns_it() -> None:
    root = start()
    left = first_stage(root).assign(Stage.lanes, 2)
    assert left.answer(Stage.throughput) == Decided(8)
    # The sibling is a different occurrence and is untouched.
    assert isinstance(left.root.child(Root.pair).child(Pair.second).answer(Stage.lanes), Unresolved)


def test_a_parent_declaration_is_refused_from_a_child_view() -> None:
    root = start()
    with pytest.raises(AuthoringError, match="not owned by Stage at root.pair.first"):
        first_stage(root).answer(Root.width)


def test_a_declaration_from_another_model_entirely_is_refused_by_name() -> None:
    class Stranger(Space):
        value = Decision(int, values=(1,))

    root = start()
    with pytest.raises(AuthoringError, match="not declared by any Space in this model"):
        root.answer(Stranger.value)


def test_only_a_decision_or_a_branch_can_be_assigned() -> None:
    root = start()
    with pytest.raises(AuthoringError, match="only a Decision or a branch"):
        first_stage(root).assign(cast("Decision[int]", Stage.throughput), 3)


# -- specialization -----------------------------------------------------------


def test_one_occurrence_answers_more_as_its_point_becomes_complete() -> None:
    """Partial then complete, without crossing a public object-type boundary."""

    stage = first_stage(start())
    assert isinstance(stage.answer(Stage.throughput), Unresolved)
    complete = stage.assign(Stage.lanes, 2)
    assert type(complete) is type(stage) is Occurrence
    assert complete.answer(Stage.throughput) == Decided(8)


def test_assignment_returns_the_same_view_over_a_successor_point() -> None:
    stage = first_stage(start())
    successor = stage.assign(Stage.lanes, 2)
    assert successor is not stage
    assert successor.namespace == stage.namespace == "root.pair.first"
    assert isinstance(stage.answer(Stage.lanes), Unresolved)
    assert successor.answer(Stage.lanes) == Decided(2)


def test_the_successor_root_carries_the_child_assignment() -> None:
    stage = first_stage(start()).assign(Stage.lanes, 4)
    assert first_stage(stage.root).answer(Stage.lanes) == Decided(4)


def test_assigning_a_branch_commits_its_ordinary_selector() -> None:
    root = start().assign(Root.choice, "stage")
    assert root.child(Root.choice).space_type is Stage
    assert isinstance(root.answer(Root.selected), Unresolved)


def test_naming_the_one_case_of_a_singleton_commits_nothing_and_still_works() -> None:
    class Solo(Space):
        width = Problem(int)
        only = OneOf(Case(Fallback, name="fallback", width=width), outputs=("throughput",))
        value = only.throughput

    compiled = compile_space_model(Solo, "root", problem_namespace="problem.root")
    root = compiled.start({"problem.root.width": 5})
    assert root.branch(Solo.only).selector is None
    assert root.assign(Solo.only, "fallback").answer(Solo.value) == Decided(5)


def test_a_case_id_that_does_not_exist_is_refused_before_anything_is_committed() -> None:
    root = start()
    with pytest.raises(AuthoringError, match="has no case 'absent'"):
        root.assign(Root.choice, "absent")


def test_a_value_outside_the_declared_domain_is_refused_with_findings() -> None:
    stage = first_stage(start())
    with pytest.raises(RequestError) as raised:
        stage.assign(Stage.lanes, 7)
    assert raised.value.findings
    assert isinstance(stage.answer(Stage.lanes), Unresolved)


def test_a_query_never_commits_anything() -> None:
    root = start()
    root.assess(Root.result)
    root.project(Root.supply)
    first_stage(root).assess(Stage.model_view)
    assert isinstance(first_stage(root).answer(Stage.lanes), Unresolved)


# -- validated projections ----------------------------------------------------


def test_an_unmet_readiness_obligation_reduces_to_unresolved() -> None:
    assessment = first_stage(start()).assess(Stage.model_view)
    assert assessment.readiness is not None and assessment.readiness.ready is None
    assert isinstance(assessment.accepted_answer, Unresolved)
    assert "readiness-decision-unassigned" in codes(assessment.accepted_answer)


def test_a_final_output_with_accepting_constraints_reduces_to_decided() -> None:
    stage = first_stage(start()).assign(Stage.lanes, 2)
    assessment = stage.assess(Stage.model_view)
    assert assessment.readiness is not None and assessment.readiness.ready is True
    assert [group.verdict for group in assessment.constraints] == [True]
    assert assessment.output == Decided(8)
    assert assessment.accepted_answer == Decided(8)
    assert stage.project(Stage.model_view) == Decided(8)


def test_readiness_can_be_true_while_the_projection_is_still_rejected() -> None:
    """Readiness and validity are different questions, and this is the proof.

    Every obligation is final -- there is nothing left to decide and nothing
    unresolved -- and the output is a perfectly available ``Decided(20)``.  The
    projection is nevertheless ``Absent``, because a constraint answered a final
    refusal.  Collapsing the two into one Boolean is how a point no hardware
    could build gets handed on as merely incomplete.
    """

    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    assessment = stage.assess(Stage.model_view)
    assert assessment.readiness is not None and assessment.readiness.ready is True
    assert assessment.output == Decided(20)
    assert [group.verdict for group in assessment.constraints] == [False]
    assert isinstance(assessment.accepted_answer, Absent)
    assert codes(assessment.accepted_answer) == ("stage-too-wide",)


def test_one_constraint_participates_in_two_projections() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    shared = QualifiedPath("constraint.root.pair.first.fits")
    model_view = stage.assess(Stage.model_view)
    build_view = stage.assess(Stage.build_view)
    assert shared in model_view.constraints[0].answers
    assert shared in build_view.constraints[0].answers
    assert isinstance(model_view.accepted_answer, Absent)
    assert isinstance(build_view.accepted_answer, Absent)


def test_a_projection_can_own_an_obligation_the_other_does_not() -> None:
    """Same output, same readiness, different verdict -- because of one group."""

    stage = first_stage(start(width=2)).assign(Stage.lanes, 3)
    assert stage.project(Stage.model_view) == Decided(6)
    refused = stage.project(Stage.build_view)
    assert isinstance(refused, Absent)
    assert codes(refused) == ("stage-odd-lanes",)


def test_a_finally_inapplicable_output_reduces_to_absent_not_unresolved() -> None:
    root = start(supplied=False)
    answer = root.project(Root.supply)
    assert isinstance(answer, Absent)
    assert not isinstance(answer, Unresolved)


def test_the_same_projection_resolves_once_its_conditional_placement_applies() -> None:
    root = start(supplied=True)
    assert isinstance(root.project(Root.supply), Unresolved)
    committed = root.child(Root.optional).assign(Stage.lanes, 2).root
    assert committed.project(Root.supply) == Decided(8)


def test_a_projection_without_a_readiness_profile_still_reduces() -> None:
    assessment = start().assess(Root.supply)
    assert assessment.readiness is None
    assert assessment.constraints == ()
    assert isinstance(assessment.accepted_answer, Unresolved)


def test_the_assessment_keeps_the_raw_output_beside_the_accepted_answer() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    assessment = stage.assess(Stage.model_view)
    assert isinstance(assessment, ProjectionAssessment)
    assert assessment.name == "root.pair.first.model_view"
    assert assessment.output != assessment.accepted_answer


def test_a_projection_declaration_of_another_occurrence_is_refused() -> None:
    root = start()
    with pytest.raises(AuthoringError, match="not owned by Root at root"):
        root.assess(Stage.model_view)


def test_assess_refuses_anything_that_is_not_one_of_the_three_questions() -> None:
    root = start()
    with pytest.raises(AuthoringError, match="Readiness, a ConstraintGroup, or a Projection"):
        root.assess(cast("Readiness", Root.pair))


def test_readiness_and_constraint_groups_are_assessable_on_their_own() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    assert stage.assess(Stage.ready).ready is True
    assert stage.assess(Stage.ready).profile == "root.pair.first.ready"
    assert stage.assess(Stage.model_checks).verdict is False


# -- authoring rules for a projection ----------------------------------------


def test_a_projection_needs_a_value_declaration_as_its_output() -> None:
    with pytest.raises(AuthoringError, match="names one value declaration"):
        Projection(cast("Decision[int]", "throughput"))


def test_a_projection_readiness_must_be_a_readiness_declaration() -> None:
    with pytest.raises(AuthoringError, match="readiness= is one Readiness"):
        Projection(Stage.throughput, readiness=cast("Readiness", Stage.model_checks))


def test_a_projection_constraint_must_be_a_constraint_group() -> None:
    with pytest.raises(AuthoringError, match="constraints= are ConstraintGroup"):
        Projection(Stage.throughput, constraints=cast("ConstraintGroup", Stage.fits))


def test_a_projection_may_not_name_a_declaration_of_another_class() -> None:
    class Borrower(Space):
        width = Problem(int)

        @derived(int, width=width)
        def value(*, width: int) -> int:
            return width

        stolen = Projection(value, readiness=Stage.ready)

    with pytest.raises(AuthoringError, match="names a declaration outside the class"):
        compile_space_model(Borrower, "root", problem_namespace="problem.root")


def test_a_projection_may_not_name_one_constraint_group_twice() -> None:
    class Doubled(Space):
        width = Problem(int)

        @derived(int, width=width)
        def value(*, width: int) -> int:
            return width

        @constraint(width=width)
        def positive(*, width: int) -> object:
            return width > 0

        checks = ConstraintGroup(positive)
        view = Projection(value, constraints=(checks, checks))

    with pytest.raises(AuthoringError, match="names constraint group 'checks' twice"):
        compile_space_model(Doubled, "root", problem_namespace="problem.root")


def test_adding_a_projection_changes_no_engine_declaration() -> None:
    """A projection is a stored question over paths that already exist."""

    class Plain(Space):
        width = Problem(int)

        @derived(int, width=width)
        def value(*, width: int) -> int:
            return width

        @constraint(width=width)
        def positive(*, width: int) -> object:
            return width > 0

        checks = ConstraintGroup(positive)
        ready = Readiness(properties=(value,), name="ready")

    class Projected(Space):
        width = Problem(int)

        @derived(int, width=width)
        def value(*, width: int) -> int:
            return width

        @constraint(width=width)
        def positive(*, width: int) -> object:
            return width > 0

        checks = ConstraintGroup(positive)
        ready = Readiness(properties=(value,), name="ready")
        view = Projection(value, readiness=ready, constraints=checks)

    plain = compile_space_model(Plain, "root", problem_namespace="problem.root").specification
    projected = compile_space_model(
        Projected, "root", problem_namespace="problem.root"
    ).specification
    assert [item.path for item in plain.decisions] == [item.path for item in projected.decisions]
    assert [item.path for item in plain.properties] == [item.path for item in projected.properties]
    assert [item.path for item in plain.constraints] == [
        item.path for item in projected.constraints
    ]
    assert [item.name for item in plain.constraint_sets] == [
        item.name for item in projected.constraint_sets
    ]
    assert [item.name for item in plain.readiness_profiles] == [
        item.name for item in projected.readiness_profiles
    ]


def test_the_declared_namespaces_of_the_existing_stack_are_unchanged() -> None:
    """U1 renames, reparents, and removes no path."""

    paths = {str(item.path) for item in model().specification.decisions}
    assert paths == {
        "root.choice.case",
        "root.choice.stage.lanes",
        "root.optional.lanes",
        "root.pair.first.lanes",
        "root.pair.inner.case",
        "root.pair.inner.stage.lanes",
        "root.pair.second.lanes",
    }


# -- frozen context and stale detection ---------------------------------------


def test_two_identical_problems_have_one_fingerprint() -> None:
    compiled = model()
    left = compiled.fingerprint({"problem.root.width": 4, "problem.root.supplied": True})
    right = compiled.fingerprint({"problem.root.width": 4, "problem.root.supplied": True})
    assert left == right


def test_a_changed_declared_problem_fact_makes_the_occurrence_stale() -> None:
    root = start(width=4)
    assert root.is_stale_for({"problem.root.width": 8, "problem.root.supplied": True})
    assert not root.is_stale_for({"problem.root.width": 4, "problem.root.supplied": True})


def test_a_stale_occurrence_is_reconstructed_rather_than_refreshed() -> None:
    """Strict reconstruction: nothing is rebased, retained, or silently dropped."""

    old = first_stage(start(width=4)).assign(Stage.lanes, 2)
    assert old.answer(Stage.throughput) == Decided(8)
    fresh = start(width=8)
    assert fresh.fingerprint != old.fingerprint
    assert isinstance(first_stage(fresh).answer(Stage.lanes), Unresolved)
    # The old occurrence is untouched by the existence of the new one.
    assert old.answer(Stage.throughput) == Decided(8)


def test_a_recorded_fingerprint_from_another_problem_is_refused() -> None:
    compiled = model()
    recorded = compiled.fingerprint({"problem.root.width": 4, "problem.root.supplied": True})
    with pytest.raises(RequestError) as raised:
        compiled.start(
            {"problem.root.width": 9, "problem.root.supplied": True}, fingerprint=recorded
        )
    finding = raised.value.findings[0]
    assert finding.code == "occurrence-problem-fingerprint-mismatch"
    assert dict(finding.values)["expected"] == recorded


def test_a_matching_recorded_fingerprint_hydrates_without_complaint() -> None:
    compiled = model()
    problem = {"problem.root.width": 4, "problem.root.supplied": True}
    recorded = compiled.fingerprint(problem)
    assert compiled.start(problem, fingerprint=recorded).fingerprint == recorded


def test_a_mutation_of_the_caller_s_problem_mapping_does_not_reach_the_occurrence() -> None:
    """The problem is projected once and frozen; queries never reread it."""

    problem: dict[str, object] = {"problem.root.width": 4, "problem.root.supplied": True}
    root = start_from(problem)
    stage = first_stage(root).assign(Stage.lanes, 2)
    problem["problem.root.width"] = 9
    assert stage.answer(Stage.throughput) == Decided(8)
    assert root.is_stale_for(problem)


def start_from(problem: dict[str, object]) -> Occurrence:
    return model().start(problem)


def test_external_state_the_schema_does_not_declare_cannot_enter_the_problem() -> None:
    """The frozen problem is closed, so irrelevant context cannot perturb it."""

    compiled = model()
    root = compiled.start({"problem.root.width": 4, "problem.root.supplied": True})
    with pytest.raises(RequestError):
        compiled.fingerprint(
            {"problem.root.width": 4, "problem.root.supplied": True, "problem.root.mood": 1}
        )
    assert root.answer(Root.width) == Decided(4)


def test_a_problem_value_without_canonical_text_is_refused_rather_than_hashed() -> None:
    class Opaque:
        pass

    class Holder(Space):
        thing = Problem(Opaque)

        @derived(int, thing=thing)
        def value(*, thing: object) -> int:
            return 1

    compiled = compile_space_model(Holder, "root", problem_namespace="problem.root")
    with pytest.raises(AuthoringError, match="has no canonical text"):
        compiled.start({"problem.root.thing": Opaque()})


def test_a_space_model_compiled_without_its_tree_cannot_start() -> None:
    compiled = model()
    detached = SpaceModel(compiled.specification, compiled.branches)
    with pytest.raises(AuthoringError, match="cannot start an occurrence"):
        detached.start({"problem.root.width": 4, "problem.root.supplied": True})


# -- diagnostics --------------------------------------------------------------


def refusal(occurrence: Occurrence) -> tuple[Diagnostic, ...]:
    assessment = occurrence.assess(Stage.model_view)
    return occurrence.diagnostics(assessment)


def test_a_diagnostic_names_the_occurrence_the_member_and_the_projection() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    diagnostics = refusal(stage)
    assert len(diagnostics) == 1
    only = diagnostics[0]
    assert only.projection == "root.pair.first.model_view"
    assert only.scope == ("root", "pair", "first")
    assert only.space == "Stage"
    assert only.member == "fits"


def test_a_diagnostic_retains_the_raw_finding_and_its_trace() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    only = refusal(stage)[0]
    assert only.finding.kind is FindingKind.REJECTION
    assert only.finding.code == "stage-too-wide"
    assert only.finding.path == QualifiedPath("constraint.root.pair.first.fits")
    assert dict(only.finding.values)["got"] == 20


def test_a_rendering_reads_in_declaration_vocabulary() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    rendered = refusal(stage)[0].render()
    assert "projection root.pair.first.model_view" in rendered
    assert "occurrence root / pair / first" in rendered
    assert "Stage.fits" in rendered
    assert "rejection stage-too-wide" in rendered


def test_two_placements_of_one_class_produce_differently_scoped_diagnostics() -> None:
    root = start(width=5)
    left = first_stage(root).assign(Stage.lanes, 4)
    right = left.root.child(Root.pair).child(Pair.second).assign(Stage.lanes, 4)
    assert refusal(left)[0].scope == ("root", "pair", "first")
    assert refusal(right)[0].scope == ("root", "pair", "second")


def test_diagnostics_interpret_a_bare_answer_too() -> None:
    stage = first_stage(start())
    diagnostics = stage.diagnostics(stage.answer(Stage.throughput))
    assert diagnostics
    assert {item.projection for item in diagnostics} == {None}
    assert {item.member for item in diagnostics} == {"lanes"}
    assert {item.scope for item in diagnostics} == {("root", "pair", "first")}


def test_a_flat_refusal_is_explained_even_though_it_carries_no_finding() -> None:
    """``Decided(False)`` says no without saying why; the projection says where."""

    class Blunt(Space):
        width = Problem(int)

        @derived(int, width=width)
        def value(*, width: int) -> int:
            return width

        @constraint(width=width)
        def never(*, width: int) -> object:
            return False

        checks = ConstraintGroup(never)
        view = Projection(value, constraints=checks)

    root = compile_space_model(Blunt, "root", problem_namespace="problem.root").start(
        {"problem.root.width": 3}
    )
    answer = root.project(Blunt.view)
    assert isinstance(answer, Absent)
    assert codes(answer) == ("projection-constraint-refused",)
    generated = [
        item
        for item in root.diagnostics(root.assess(Blunt.view))
        if item.finding.code == "projection-constraint-refused"
    ]
    assert len(generated) == 1
    # The path is the projection's own name, which no declaration owns; that is
    # reported as unowned rather than attributed to whatever compiled nearby.
    assert generated[0].space is None and generated[0].member is None
    assert generated[0].finding.trace == (QualifiedPath("constraint.root.never"),)


def test_a_finding_that_appears_twice_is_reported_once() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    assessment = stage.assess(Stage.build_view)
    findings = [item.finding for item in stage.diagnostics(assessment)]
    assert len(findings) == len(set(findings))


# -- capability boundary ------------------------------------------------------

_FORBIDDEN = (Engine, DesignPoint, DesignSpace)


def test_no_public_occurrence_attribute_is_a_runtime_object() -> None:
    root = start()
    stage = first_stage(root).assign(Stage.lanes, 2)
    for occurrence in (root, stage, stage.root):
        for name in dir(occurrence):
            if name.startswith("_"):
                continue
            value = getattr(occurrence, name)
            assert not isinstance(value, _FORBIDDEN), name


def test_the_occurrence_surface_is_exactly_the_supported_operations() -> None:
    public = {name for name in dir(Occurrence) if not name.startswith("_")}
    assert public == {
        "answer",
        "assess",
        "assign",
        "branch",
        "child",
        "diagnostics",
        "fingerprint",
        "is_stale_for",
        "namespace",
        "project",
        "root",
        "scope",
        "space_type",
    }
    assert not {"point", "engine", "query", "path", "commit"} & public


def test_no_returned_value_carries_a_runtime_object() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)
    returned: list[object] = [
        stage.answer(Stage.throughput),
        stage.assess(Stage.ready),
        stage.assess(Stage.model_checks),
        stage.assess(Stage.model_view),
        stage.project(Stage.model_view),
        stage.root.branch(Root.choice),
        *stage.diagnostics(stage.assess(Stage.model_view)),
    ]
    for value in returned:
        for name in dir(value):
            if name.startswith("_"):
                continue
            assert not isinstance(getattr(value, name, None), _FORBIDDEN), name


_AUDIT = '''
"""Audit the public surface in a clean interpreter and report as JSON."""

import json

from finn.dataflow._engine import DesignPoint, DesignSpace, Engine
from finn.dataflow.model.compiler import _CompiledSpace, _Ref, compile_space_model
from finn.dataflow.model.declarations import Problem, Space, constraint, derived

FORBIDDEN = (Engine, DesignPoint, DesignSpace, _CompiledSpace, _Ref)

seen = []


class Audited(Space):
    width = Problem(int)

    @derived(int, width=width)
    def value(*, width):
        seen.append(type(width).__name__)
        return width * 2

    @constraint(width=width)
    def positive(*, width):
        seen.append(type(width).__name__)
        return width > 0


model = compile_space_model(Audited, "root", problem_namespace="problem.root")
occurrence = model.start({"problem.root.width": 3})
answer = occurrence.answer(Audited.value)
occurrence.diagnostics(answer)


def leaks(value, depth):
    """Every public attribute reachable from a value that is a runtime object."""

    if depth == 0 or isinstance(value, (str, bytes, int, float, bool, type(None))):
        return []
    found = []
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            member = getattr(value, name)
        except Exception:
            continue
        if isinstance(member, FORBIDDEN):
            found.append(type(value).__name__ + "." + name)
        elif not callable(member):
            found.extend(leaks(member, depth - 1))
    return sorted(set(found))


print(
    json.dumps(
        {
            "callback_saw": sorted(set(seen)),
            "answer": repr(answer),
            "leaks": leaks(occurrence, 3),
            "model_leaks": [
                name
                for name in dir(model)
                if not name.startswith("_") and isinstance(getattr(model, name), FORBIDDEN)
            ],
        }
    )
)
'''


def test_a_contributor_callback_never_receives_a_runtime_object() -> None:
    """Audited in a clean interpreter, so no fixture can have handed it in."""

    finished = subprocess.run(
        [sys.executable, "-c", _AUDIT],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(finished.stdout)
    assert report["callback_saw"] == ["int"]
    assert report["answer"] == "Decided(value=6)"
    assert report["leaks"] == []
    assert report["model_leaks"] == []


# -- concurrency floor --------------------------------------------------------


@dataclass(frozen=True)
class _Read:
    throughput: object
    accepted: object


def test_concurrent_reads_of_one_occurrence_are_safe_and_identical() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 4)

    def read(_index: int) -> _Read:
        return _Read(stage.answer(Stage.throughput), stage.project(Stage.model_view))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read, range(64)))
    assert len(set(results)) == 1
    assert results[0].throughput == Decided(20)
    assert isinstance(results[0].accepted, Absent)


def test_concurrent_successor_creation_never_mutates_the_shared_occurrence() -> None:
    stage = first_stage(start())

    def commit(lanes: int) -> object:
        return stage.assign(Stage.lanes, lanes).answer(Stage.throughput)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(commit, [1, 2, 4] * 12))
    assert set(results) == {Decided(4), Decided(8), Decided(16)}
    assert isinstance(stage.answer(Stage.lanes), Unresolved)


def test_a_successor_point_never_contaminates_its_predecessor_s_answers() -> None:
    stage = first_stage(start())
    successors = [stage.assign(Stage.lanes, lanes) for lanes in (1, 2, 4)]
    assert [item.answer(Stage.throughput) for item in successors] == [
        Decided(4),
        Decided(8),
        Decided(16),
    ]
    assert isinstance(stage.answer(Stage.throughput), Unresolved)


def test_concurrent_compilation_under_several_namespaces_agrees_with_serial() -> None:
    namespaces = [f"root{index}" for index in range(16)]

    def compile_one(namespace: str) -> tuple[str, ...]:
        compiled = compile_space_model(Root, namespace, problem_namespace=f"problem.{namespace}")
        return tuple(str(item.path) for item in compiled.specification.decisions)

    serial = {namespace: compile_one(namespace) for namespace in namespaces}
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = dict(zip(namespaces, pool.map(compile_one, namespaces)))
    assert concurrent == serial
    assert all(paths[0].startswith(f"{namespace}.") for namespace, paths in serial.items())


def test_findings_are_deterministic_across_threads() -> None:
    stage = first_stage(start(width=5)).assign(Stage.lanes, 3)

    def render(_index: int) -> str:
        return "\n".join(
            item.render() for item in stage.diagnostics(stage.assess(Stage.build_view))
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rendered = set(pool.map(render, range(32)))
    assert len(rendered) == 1
