# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD1b: the branch catalog is sufficient for external specialization policy.

Every algorithm below is test-only and reaches the design space through exactly
two things: the immutable :class:`BranchCatalog` and the public ``Engine``.  No
algorithm reconstructs a selector path from a naming convention, touches a
private compiler record, or is stored on a reusable declaration.  That is the
whole claim of the seam, so proving several unrelated policies work against one
unchanged declaration is the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from finn.dataflow._engine import (
    Absent,
    Decided,
    DesignPoint,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.model.branching import BranchCatalog, BranchInfo, CaseInfo
from finn.dataflow.model.compiler import compile_space_model
from finn.dataflow.model.declarations import (
    Case,
    Decision,
    Input,
    OneOf,
    Problem,
    Readiness,
    Space,
    constraint,
    derived,
    reject,
)

# -- spaces -------------------------------------------------------------------


class Cheap(Space):
    size = Input(int)
    stages = Decision(int, values=(1, 2))

    @derived(int, size=size, stages=stages)
    def result(*, size: int, stages: int) -> int:
        return size + stages

    @derived(int, stages=stages)
    def cost(*, stages: int) -> int:
        return 10 * stages

    @constraint(size=size)
    def small_enough(*, size: int) -> object:
        if size > 8:
            return reject("cheap-too-large", "this case only covers small problems")
        return True

    ready = Readiness(decisions=(stages,), name="ready")
    exports = (result, cost)


class Costly(Space):
    size = Input(int)

    @derived(int, size=size)
    def result(*, size: int) -> int:
        return size * 2

    @derived(int, size=size)
    def cost(*, size: int) -> int:
        return 1

    exports = (result, cost)


class Inner(Space):
    size = Input(int)
    nested = OneOf(
        Case(Cheap, name="cheap", size=size),
        Case(Costly, name="costly", size=size),
        outputs=("result", "cost"),
    )
    result = nested.result
    cost = nested.cost
    exports = (result, cost)


class Root(Space):
    size = Problem(int)
    top = OneOf(
        Case(Inner, name="inner", size=size),
        Case(Costly, name="costly", size=size),
        outputs=("result", "cost"),
    )


class Only(Space):
    size = Problem(int)
    solo = OneOf(Case(Costly, name="costly", size=size), outputs=("result",))


def _model(
    space_type: type[Space] = Root, size: int = 4
) -> tuple[Engine, DesignPoint, BranchCatalog]:
    engine = Engine()
    model = compile_space_model(space_type, "root", problem_namespace="problem.root")
    point = engine.start(engine.validate(model.specification), {"problem.root.size": size})
    return engine, point, model.branches


# -- test-only specialization algorithms --------------------------------------


def assign_case(
    engine: Engine, point: DesignPoint, branch: BranchInfo, case_id: str
) -> DesignPoint:
    """Explicit selection by case id."""

    branch.case(case_id)
    if branch.selector is None:
        return point
    return engine.commit_assignments(point, {branch.selector: case_id}).point


def _case_is_refused(engine: Engine, point: DesignPoint, case: CaseInfo) -> bool:
    if case.constraint_paths:
        assessment = engine.evaluate_constraints(point, case.constraint_paths)
        if assessment.verdict is False:
            return True
    return False


def first_feasible_case(
    engine: Engine, point: DesignPoint, branch: BranchInfo
) -> tuple[str, DesignPoint] | None:
    """Trial each case in declaration order and keep the first that survives."""

    for case in branch.cases:
        trial = assign_case(engine, point, branch, case.id)
        if not _case_is_refused(engine, trial, case):
            return case.id, trial
    return None


@dataclass(frozen=True)
class Assessment:
    case_id: str
    refused: bool
    ready: bool


def exhaustive_trial(
    engine: Engine, point: DesignPoint, branch: BranchInfo
) -> tuple[Assessment, ...]:
    """Assess every case without committing to any of them."""

    results = []
    for case in branch.cases:
        trial = assign_case(engine, point, branch, case.id)
        ready = all(
            engine.check_readiness(trial, profile).ready is True
            for profile in case.readiness_profiles
        )
        results.append(Assessment(case.id, _case_is_refused(engine, trial, case), ready))
    return tuple(results)


def first_globally_feasible_case(
    engine: Engine, point: DesignPoint, branch: BranchInfo
) -> tuple[str, DesignPoint] | None:
    """Trial each case and keep the first the *whole space* accepts.

    Weaker than `first_feasible_case` in what it assumes and stronger in what it
    proves: a case can be perfectly satisfiable on its own constraints and still
    make the enclosing composition invalid.  `evaluate_constraints` with no path
    list assesses everything the space declares, which needs no knowledge of
    where the enclosing feasibility set lives or what it is called.

    ``verdict is True`` and not ``is not False``.  An unresolved verdict means
    the space has not yet been told enough to judge the case, which is not the
    same as accepting it; treating it as acceptance would let this helper report
    a case it has no evidence for.
    """

    for case in branch.cases:
        trial = assign_case(engine, point, branch, case.id)
        if engine.evaluate_constraints(trial).verdict is True:
            return case.id, trial
    return None


def cheapest_case_by_property(
    engine: Engine, point: DesignPoint, branch: BranchInfo, name: str
) -> tuple[str, DesignPoint] | None:
    """Score each case on a caller-named property the case itself publishes.

    The path comes from `CaseInfo.property_paths`; nothing here builds one.
    """

    scored: list[tuple[int, str, DesignPoint]] = []
    for case in branch.cases:
        try:
            path = case.property_named(name)
        except KeyError:
            continue
        trial = assign_case(engine, point, branch, case.id)
        answer = engine.query_property(trial, path)
        if isinstance(answer, Decided):
            scored.append((int(cast(int, answer.value)), case.id, trial))
    if not scored:
        return None
    best = min(scored, key=lambda item: (item[0], item[1]))
    return best[1], best[2]


def resolve_recursively(
    engine: Engine, point: DesignPoint, catalog: BranchCatalog, branch: BranchInfo
) -> DesignPoint:
    """Pick the first feasible case, then descend into whatever it owns."""

    chosen = first_feasible_case(engine, point, branch)
    assert chosen is not None
    case_id, point = chosen
    for namespace in branch.case(case_id).child_branches:
        point = resolve_recursively(engine, point, catalog, catalog.branch(namespace))
    return point


def cheapest_case(
    engine: Engine,
    point: DesignPoint,
    branch: BranchInfo,
    cost_output: str,
) -> tuple[str, DesignPoint]:
    """A mock cost-guided policy reading one caller-designated derived property."""

    path = next(item.path for item in branch.outputs if item.name == cost_output)
    scored: list[tuple[int, str, DesignPoint]] = []
    for case in branch.cases:
        trial = assign_case(engine, point, branch, case.id)
        answer = engine.query_property(trial, path)
        if isinstance(answer, Decided):
            scored.append((int(answer.value), case.id, trial))
    best = min(scored, key=lambda item: (item[0], item[1]))
    return best[1], best[2]


# -- what the catalog reports -------------------------------------------------


def test_the_catalog_lists_every_branch_outermost_first() -> None:
    _engine, _point, catalog = _model()
    assert catalog.namespaces == ("root.top", "root.top.inner.nested")


def test_a_branch_reports_its_selector_cases_and_selected_outputs() -> None:
    _engine, _point, catalog = _model()
    top = catalog.branch("root.top")
    assert top.selector == QualifiedPath("root.top.case")
    assert tuple(case.id for case in top.cases) == ("inner", "costly")
    assert tuple(case.namespace for case in top.cases) == (
        "root.top.inner",
        "root.top.costly",
    )
    assert tuple(item.name for item in top.outputs) == ("result", "cost")
    assert tuple(str(item.path) for item in top.outputs) == (
        "semantic.root.top.result",
        "semantic.root.top.cost",
    )


def test_a_case_reports_the_declarations_it_owns() -> None:
    _engine, _point, catalog = _model()
    inner = catalog.branch("root.top.inner.nested")
    cheap = inner.case("cheap")
    assert cheap.decision_paths == (QualifiedPath("root.top.inner.nested.cheap.stages"),)
    assert cheap.constraint_paths == (
        QualifiedPath("constraint.root.top.inner.nested.cheap.small_enough"),
    )
    assert cheap.readiness_profiles == ("root.top.inner.nested.cheap.ready",)


def test_nested_ownership_is_reported_by_the_owning_case() -> None:
    _engine, _point, catalog = _model()
    top = catalog.branch("root.top")
    assert top.case("inner").child_branches == ("root.top.inner.nested",)
    assert top.case("costly").child_branches == ()


def test_a_singleton_is_inspectable_without_inventing_a_selector() -> None:
    _engine, _point, catalog = _model(Only)
    solo = catalog.branch("root.solo")
    assert solo.selector is None
    assert tuple(case.id for case in solo.cases) == ("costly",)


def test_an_ambiguous_case_property_name_is_refused() -> None:
    """Scoring the first `.cost` that happens to compile is a silent wrong answer."""

    class Priced(Space):
        size = Input(int)

        @derived(int, size=size)
        def cost(*, size: int) -> int:
            return size

        exports = (cost,)

    class TwoPrices(Space):
        size = Input(int)
        left = OneOf(Case(Priced, name="only", size=size), outputs=("cost",))
        right = OneOf(Case(Priced, name="only", size=size), outputs=("cost",))
        exports = ()

    class Root_(Space):
        size = Problem(int)
        choice = OneOf(Case(TwoPrices, name="both", size=size), name="branch")

    catalog = compile_space_model(Root_, "root", problem_namespace="problem.root").branches
    case = catalog.branch("root.branch").case("both")
    with pytest.raises(KeyError, match="properties named 'cost'"):
        case.property_named("cost")
    # Naming one exactly still works, off a path the catalog published.
    assert QualifiedPath("semantic.root.branch.both.left.cost") in case.property_paths


def test_a_case_property_name_that_matches_nothing_is_refused() -> None:
    _engine, _point, catalog = _model()
    with pytest.raises(KeyError, match="owns no property named"):
        catalog.branch("root.top").case("costly").property_named("absent")


def test_an_unknown_branch_or_case_is_a_deterministic_lookup_error() -> None:
    _engine, _point, catalog = _model()
    with pytest.raises(KeyError):
        catalog.branch("root.absent")
    with pytest.raises(KeyError):
        catalog.branch("root.top").case("absent")


# -- what external algorithms can do with it ----------------------------------


def test_an_algorithm_can_assign_one_case_explicitly() -> None:
    engine, point, catalog = _model()
    point = assign_case(engine, point, catalog.branch("root.top"), "costly")
    assert engine.query_property(point, "semantic.root.top.result") == Decided(8)


def test_an_algorithm_can_take_the_first_feasible_case() -> None:
    engine, point, catalog = _model(size=16)
    inner = catalog.branch("root.top.inner.nested")
    point = assign_case(engine, point, catalog.branch("root.top"), "inner")
    chosen = first_feasible_case(engine, point, inner)
    assert chosen is not None
    assert chosen[0] == "costly"


def test_a_rejected_trial_does_not_mutate_the_original_point() -> None:
    engine, point, catalog = _model(size=16)
    top = catalog.branch("root.top")
    rejected = assign_case(engine, point, top, "inner")
    assert dict(point.assignments) == {}
    assert dict(rejected.assignments) == {QualifiedPath("root.top.case"): "inner"}
    assert rejected is not point


def test_an_algorithm_can_assess_every_case_without_committing() -> None:
    engine, point, catalog = _model(size=16)
    point = assign_case(engine, point, catalog.branch("root.top"), "inner")
    assessments = exhaustive_trial(engine, point, catalog.branch("root.top.inner.nested"))
    assert assessments == (
        Assessment("cheap", refused=True, ready=False),
        Assessment("costly", refused=False, ready=True),
    )


def test_an_algorithm_can_traverse_a_nested_branch() -> None:
    engine, point, catalog = _model()
    point = resolve_recursively(engine, point, catalog, catalog.branch("root.top"))
    assert dict(point.assignments) == {
        QualifiedPath("root.top.case"): "inner",
        QualifiedPath("root.top.inner.nested.case"): "cheap",
    }
    assert isinstance(engine.query_property(point, "semantic.root.top.result"), Unresolved)
    point = engine.commit_assignments(point, {"root.top.inner.nested.cheap.stages": 2}).point
    assert engine.query_property(point, "semantic.root.top.result") == Decided(6)


def test_an_algorithm_may_be_cost_guided_without_the_declaration_knowing() -> None:
    engine, point, catalog = _model()
    point = assign_case(engine, point, catalog.branch("root.top"), "inner")
    point = engine.commit_assignments(point, {"root.top.inner.nested.cheap.stages": 1}).point
    case_id, point = cheapest_case(engine, point, catalog.branch("root.top.inner.nested"), "cost")
    assert case_id == "costly"


def test_a_branch_may_be_left_unresolved_without_choosing() -> None:
    engine, point, catalog = _model()
    top = catalog.branch("root.top")
    assert engine.decision_state(point, top.selector) is not None
    assert isinstance(engine.query_property(point, "semantic.root.top.result"), Unresolved)
    assert dict(point.assignments) == {}


def test_an_unresolved_nested_decision_stays_visible_rather_than_refusing() -> None:
    engine, point, catalog = _model()
    point = assign_case(engine, point, catalog.branch("root.top"), "inner")
    point = assign_case(engine, point, catalog.branch("root.top.inner.nested"), "cheap")
    answer = engine.query_property(point, "semantic.root.top.result")
    assert isinstance(answer, Unresolved)
    assert not isinstance(answer, Absent)


def test_committing_a_selector_leaves_the_sibling_case_non_rejecting() -> None:
    engine, point, catalog = _model(size=16)
    point = assign_case(engine, point, catalog.branch("root.top"), "costly")
    inner = catalog.branch("root.top.inner.nested")
    assessment = engine.evaluate_constraints(point, inner.case("cheap").constraint_paths)
    assert assessment.verdict is not False
    assert engine.query_property(point, "semantic.root.top.result") == Decided(32)


def test_inspection_is_domain_neutral() -> None:
    """Nothing in the catalog names a Kernel, a Region, or a Design."""

    _engine, _point, catalog = _model()
    text = repr(catalog)
    for word in ("Kernel", "Region", "Network", "Design"):
        assert word not in text
