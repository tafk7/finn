# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Phase 1 gate: a reusable scope placed twice, validated by the engine.

The authoring surface must produce ordinary engine declarations, allocate
non-colliding paths when the same authoring function is placed more than once,
and let a caller query results through declaration handles without writing a
raw path string.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import pytest

from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Ref,
    Scope,
    divisors_of,
    domain,
    enum_semantics,
    evaluator,
    finite,
    predicate,
    reject,
    semantics_for,
    unresolved,
)
from finn.dataflow.design import (
    ABSENT,
    DependencyView,
    Absent,
    AbsenceMode,
    Constraint,
    Decided,
    Decision,
    DependencyKind,
    DerivedProperty,
    Engine,
    FindingKind,
    ProblemField,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.spec_algebra import assemble_specs

EXTENT_PATH = QualifiedPath("problem.example.extent")
BUDGET_PATH = QualifiedPath("problem.example.budget")


class Style(str, Enum):
    NARROW = "narrow"
    WIDE = "wide"


@dataclass(frozen=True)
class Placement:
    """What one placement of the reusable scope exposes to its caller."""

    scope: Scope
    lanes: Ref[int]
    style: Ref[Style]
    folds: Ref[int]
    width: Ref[int]
    fits_budget: ConstraintRef
    even_lanes: ConstraintRef


def build_scope(namespace: str, *, extent: Ref[int], budget: Ref[int]) -> Placement:
    """One reusable authoring function, placeable under any namespace."""

    scope = Scope(namespace)

    lanes = scope.decision("lanes", int, domain=divisors_of(extent))
    style = scope.decision("style", Style, domain=finite(Style))

    def fold(extent: int, lanes: int) -> int:
        return extent // lanes

    folds = scope.derived(
        "folds", int, dependencies={"extent": extent, "lanes": lanes}, evaluate=fold
    )

    def width(folds: int, style: Style) -> int:
        return folds * (1 if style is Style.NARROW else 2)

    width_ref = scope.derived(
        "width", int, dependencies={"folds": folds, "style": style}, evaluate=width
    )

    budget_handle = budget

    def fits(folds: int, budget: object) -> object:
        if budget is ABSENT:
            # Traced to the handle, not to the absent runtime value.
            return unresolved(
                "example-budget-missing", "a budget is required", trace=(budget_handle,)
            )
        return folds <= cast(int, budget)

    fits_budget = scope.constraint(
        "fits_budget",
        dependencies={"folds": folds, "budget": budget_handle.allow_absent()},
        evaluate=fits,
        sets=(f"{namespace}.feasibility",),
    )

    def even_lanes(lanes: int) -> object:
        return True if lanes % 2 == 0 else reject("example-odd-lanes", "lanes must be even")

    even = scope.constraint(
        "even_lanes",
        dependencies={"lanes": lanes},
        evaluate=even_lanes,
        sets=(f"{namespace}.feasibility",),
    )
    return Placement(scope, lanes, style, folds, width_ref, fits_budget, even)


def _problem_scope() -> tuple[Scope, Ref[int], Ref[int]]:
    scope = Scope("example.problem")
    extent = scope.problem_field(EXTENT_PATH, int)
    budget = scope.problem_field(BUDGET_PATH, int, required=False)
    return scope, extent, budget


def _placed_twice() -> tuple[Scope, Placement, Placement]:
    problem, extent, budget = _problem_scope()
    left = build_scope("example.left", extent=extent, budget=budget)
    right = build_scope("example.right", extent=extent, budget=budget)
    return problem, left, right


# -- the gate ----------------------------------------------------------------


def test_a_reusable_scope_placed_twice_validates_and_answers_through_handles() -> None:
    problem, left, right = _placed_twice()
    spec = assemble_specs((problem.spec(), left.scope.spec(), right.scope.spec()))

    engine = Engine()
    space = engine.validate(spec)
    point = engine.start(space, {EXTENT_PATH: 8, BUDGET_PATH: 4})

    # Assignments are addressed by handle, not by a written-out path.
    point = engine.commit_assignments(
        point,
        {
            left.lanes.path: 2,
            left.style.path: Style.NARROW,
            right.lanes.path: 4,
            right.style.path: Style.WIDE,
        },
    ).point

    assert engine.query_property(point, left.folds.path) == Decided(4)
    assert engine.query_property(point, left.width.path) == Decided(4)
    assert engine.query_property(point, right.folds.path) == Decided(2)
    assert engine.query_property(point, right.width.path) == Decided(4)

    # Both placements evaluate their own feasibility set independently.
    assert engine.evaluate_constraint_set(point, "example.left.feasibility").verdict is True
    assert engine.evaluate_constraint_set(point, "example.right.feasibility").verdict is True


def test_each_placement_constrains_only_its_own_decisions() -> None:
    problem, left, right = _placed_twice()
    engine = Engine()
    space = engine.validate(assemble_specs((problem.spec(), left.scope.spec(), right.scope.spec())))
    point = engine.start(space, {EXTENT_PATH: 8, BUDGET_PATH: 4})
    point = engine.commit_assignments(
        point,
        {
            left.lanes.path: 1,  # odd: rejected by this placement only
            left.style.path: Style.NARROW,
            right.lanes.path: 4,
            right.style.path: Style.WIDE,
        },
    ).point
    assert engine.evaluate_constraint_set(point, "example.left.feasibility").verdict is not True
    assert engine.evaluate_constraint_set(point, "example.right.feasibility").verdict is True


def test_placing_the_same_scope_twice_produces_disjoint_paths() -> None:
    _problem, left, right = _placed_twice()
    left_paths = {item.path for item in left.scope.spec().decisions}
    right_paths = {item.path for item in right.scope.spec().decisions}
    assert left_paths and right_paths
    assert left_paths.isdisjoint(right_paths)


# -- path allocation matches the existing FINN convention --------------------


def test_generated_paths_match_the_existing_convention() -> None:
    _problem, left, _right = _placed_twice()
    spec = left.scope.spec()
    assert {item.path.value for item in spec.decisions} == {
        "example.left.lanes",
        "example.left.style",
    }
    assert {item.path.value for item in spec.properties} == {
        "semantic.example.left.folds",
        "semantic.example.left.width",
    }
    assert {item.path.value for item in spec.constraints} == {
        "constraint.example.left.fits_budget",
        "constraint.example.left.even_lanes",
    }


# -- declarations are ordinary engine objects --------------------------------


def test_every_declaration_is_an_ordinary_engine_object() -> None:
    problem, left, _right = _placed_twice()
    assert all(isinstance(item, ProblemField) for item in problem.spec().problem_schema.fields)
    assert all(isinstance(item, Decision) for item in left.scope.spec().decisions)
    assert all(isinstance(item, DerivedProperty) for item in left.scope.spec().properties)
    assert all(isinstance(item, Constraint) for item in left.scope.spec().constraints)


def test_a_handle_supplies_its_own_kind_semantics_and_path() -> None:
    _problem, extent, _budget = _problem_scope()
    reference = extent.dependency("extent")
    assert reference.kind is DependencyKind.PROBLEM
    assert reference.path == EXTENT_PATH
    assert reference.value_semantics.accepts(3)
    assert not reference.value_semantics.accepts("3")


# -- constraint sets are registered where the constraint is declared ---------


def test_constraint_sets_are_registered_at_the_declaration_site() -> None:
    _problem, left, _right = _placed_twice()
    sets = {item.name: set(item.constraints) for item in left.scope.spec().constraint_sets}
    assert sets["example.left.feasibility"] == {
        QualifiedPath("constraint.example.left.fits_budget"),
        QualifiedPath("constraint.example.left.even_lanes"),
    }
    assert {item.path for item in left.scope.constraints_in("example.left.feasibility")} == sets[
        "example.left.feasibility"
    ]


def test_include_in_adds_an_existing_constraint_to_another_set() -> None:
    _problem, left, _right = _placed_twice()
    (first,) = [item for item in left.scope.constraints_in("example.left.feasibility")][:1]
    left.scope.include_in("example.left.admission", first)
    sets = {item.name: set(item.constraints) for item in left.scope.spec().constraint_sets}
    assert sets["example.left.admission"] == {first.path}


# -- result adaptation -------------------------------------------------------


def test_a_bare_return_becomes_decided() -> None:
    owner = QualifiedPath("constraint.example.bare")
    spec = evaluator(owner, {}, lambda: 7)
    assert spec.evaluator(_view({})) == Decided(7)


def test_reject_and_unresolved_take_the_owning_path() -> None:
    owner = QualifiedPath("constraint.example.owned")
    rejected = evaluator(owner, {}, lambda: reject("code-a", "nope")).evaluator(_view({}))
    assert isinstance(rejected, Absent)
    assert rejected.findings[0].path == owner
    assert rejected.findings[0].kind is FindingKind.REJECTION

    missing = evaluator(owner, {}, lambda: unresolved("code-b", "need it")).evaluator(_view({}))
    assert isinstance(missing, Unresolved)
    assert missing.findings[0].path == owner
    assert missing.findings[0].kind is FindingKind.LIMITATION


def test_an_explicit_answer_passes_through() -> None:
    owner = QualifiedPath("constraint.example.explicit")
    spec = evaluator(owner, {}, lambda: Decided("kept"))
    assert spec.evaluator(_view({})) == Decided("kept")


# -- absence is a use-site property ------------------------------------------


def test_absence_is_expressed_at_the_use_site_not_the_declaration() -> None:
    _problem, _extent, budget = _problem_scope()
    assert budget.absence is AbsenceMode.REQUIRES_APPLICABLE
    optional = budget.allow_absent()
    assert optional.absence is AbsenceMode.ALLOWS_ABSENT
    # The same field is still required at another use site.
    assert budget.absence is AbsenceMode.REQUIRES_APPLICABLE
    assert optional.required().absence is AbsenceMode.REQUIRES_APPLICABLE


# -- authoring mistakes are rejected at declaration time ---------------------


def test_a_signature_that_does_not_match_its_mapping_is_rejected() -> None:
    _problem, extent, _budget = _problem_scope()
    with pytest.raises(AuthoringError, match="unbound parameters"):
        evaluator(QualifiedPath("constraint.example.bad"), {"extent": extent}, lambda other: other)


def test_an_unused_dependency_is_rejected() -> None:
    _problem, extent, budget = _problem_scope()
    with pytest.raises(AuthoringError, match="unused dependencies"):
        evaluator(
            QualifiedPath("constraint.example.bad"),
            {"extent": extent, "budget": budget},
            lambda extent: extent,
        )


def test_varargs_evaluators_are_rejected() -> None:
    with pytest.raises(AuthoringError, match="named parameters"):
        evaluator(QualifiedPath("constraint.example.bad"), {}, lambda *args: args)


# -- semantics derivation ----------------------------------------------------


def test_enum_semantics_compare_by_identity() -> None:
    semantics = semantics_for(Style)
    assert semantics.accepts(Style.NARROW)
    assert not semantics.accepts("narrow")
    assert semantics.values_equal(Style.NARROW, Style.NARROW)
    assert not semantics.values_equal(Style.NARROW, Style.WIDE)
    assert enum_semantics(Style).type_token is Style


def test_explicit_semantics_pass_through() -> None:
    explicit = semantics_for(int)
    assert semantics_for(explicit) is explicit


# -- domains -----------------------------------------------------------------


def test_an_explicit_domain_binds_candidate_alongside_its_dependencies() -> None:
    _problem, extent, _budget = _problem_scope()

    def accepts(candidate: object, extent: int) -> bool:
        return type(candidate) is int and candidate < extent

    def candidates(extent: int) -> tuple[object, ...]:
        return tuple(range(extent))

    built = domain({"extent": extent}, accepts=accepts, candidates=candidates)(
        QualifiedPath("example.dom")
    )
    assert built.accepts(2, _view({"extent": 4})) == Decided(True)
    assert built.accepts(9, _view({"extent": 4})) == Decided(False)
    assert built.candidates is not None
    assert built.candidates.evaluator(_view({"extent": 3})) == Decided((0, 1, 2))


# -- applicability -----------------------------------------------------------


def test_one_predicate_gates_several_declarations() -> None:
    """Applicability stays explicit and is reusable across declarations."""

    problem, extent, budget = _problem_scope()
    scope = Scope("example.gated")

    def wide_enough(extent: int) -> bool:
        return extent >= 4

    gate = predicate(QualifiedPath("example.gated.applies"), {"extent": extent}, wide_enough)

    scope.derived(
        "doubled",
        int,
        dependencies={"extent": extent},
        evaluate=lambda extent: extent * 2,
        applies_if=gate,
    )
    scope.constraint(
        "within_budget",
        dependencies={"extent": extent, "budget": budget},
        evaluate=lambda extent, budget: extent <= budget,
        applies_if=gate,
    )

    spec = scope.spec()
    assert spec.properties[0].applies_if is gate
    assert spec.constraints[0].applies_if is gate

    engine = Engine()
    space = engine.validate(assemble_specs((problem.spec(), spec)))

    narrow = engine.start(space, {EXTENT_PATH: 2, BUDGET_PATH: 1})
    assert isinstance(engine.query_property(narrow, spec.properties[0].path), Absent)

    wide = engine.start(space, {EXTENT_PATH: 4, BUDGET_PATH: 8})
    assert engine.query_property(wide, spec.properties[0].path) == Decided(8)


def _view(values: dict[str, object]) -> DependencyView:
    return DependencyView(values)
