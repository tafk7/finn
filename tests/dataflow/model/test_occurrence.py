# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import pytest

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.model import (
    Case,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceError,
    OneOf,
    Problem,
    Projection,
    ProjectionAssessment,
    Readiness,
    Space,
    Use,
    constraint,
    derived,
)


class Leaf(Space):
    supplied = Input(int)
    factor = Decision(int, values=(1, 2, 4))

    @derived(int, supplied=supplied, factor=factor)
    def result(*, supplied: int, factor: int) -> int:
        return supplied * factor

    @constraint(result=result)
    def below_limit(*, result: int) -> bool:
        return result < 20

    legal = ConstraintGroup(below_limit)
    ready = Readiness(decisions=(factor,), properties=(result,), constraints=legal)
    data = Projection(result, readiness=ready, constraints=(legal,))
    report = Projection(result, readiness=ready, constraints=(legal,))
    exports = (result,)


class Nested(Space):
    supplied = Input(int)
    leaf = Use(Leaf, supplied=supplied)
    result = leaf.result
    exports = (result,)


class Root(Space):
    size = Problem(int)
    mode = Decision(str, values=("small", "large"))
    left = Use(Leaf, supplied=size)
    right = Use(Leaf, supplied=size)
    choice = OneOf(
        Case(Nested, name="nested", supplied=size),
        Case(Leaf, name="direct", supplied=size),
        outputs=("result",),
    )

    @derived(int, selected=choice.result)
    def selected_result(*, selected: int) -> int:
        return selected


def test_authored_classes_are_the_root_and_child_occurrence_types() -> None:
    root = Root.start({Root.size: 4})
    assert type(root) is Root
    assert root.root is root
    assert root.size == 4

    left = root.child(Root.left)
    right = root.child(Root.right)
    assert type(left) is Leaf
    assert type(right) is Leaf
    assert left.root is root
    assert right.root is root
    assert left.supplied == 4


def test_assignment_returns_immutable_successors_of_the_same_authored_class() -> None:
    root = Root.start({Root.size: 4})
    partial = root.child(Root.left)
    assert isinstance(partial.answer(Leaf.result), Unresolved)

    complete = partial.assign(Leaf.factor, 2)
    assert type(complete) is Leaf
    assert complete is not partial
    assert complete.root is not root
    assert isinstance(partial.answer(Leaf.result), Unresolved)
    assert complete.answer(Leaf.result) == Decided(8)
    assert complete.result == 8

    successor_root = complete.root.assign(Root.mode, "small")
    assert type(successor_root) is Root
    assert successor_root.answer(Root.mode) == Decided("small")
    assert isinstance(complete.root.answer(Root.mode), Unresolved)


def test_repeated_child_classes_require_an_exact_use_site() -> None:
    root = Root.start({Root.size: 4})
    with pytest.raises(OccurrenceError, match="3 direct occurrences of Leaf"):
        root.child(Leaf)
    with pytest.raises(OccurrenceError, match="assign it through the exact child occurrence"):
        root.assign(Leaf.factor, 2)

    left = root.child(Root.left).assign(Leaf.factor, 2)
    right = left.root.child(Root.right)
    assert left.answer(Leaf.result) == Decided(8)
    assert isinstance(right.answer(Leaf.result), Unresolved)


def test_branch_views_select_and_navigate_cases_without_paths() -> None:
    root = Root.start({Root.size: 3})
    branch = root.branch(Root.choice)
    assert branch.name == "choice"
    assert branch.cases == ("nested", "direct")
    assert isinstance(branch.selected(), Unresolved)

    selected = branch.select("nested")
    assert selected.selected() == Decided("nested")
    nested = selected.case("nested")
    leaf = nested.child(Nested.leaf).assign(Leaf.factor, 4)
    final_root = leaf.root
    assert final_root.answer(Root.selected_result) == Decided(12)
    assert final_root.branch(Root.choice).selected() == Decided("nested")


def test_assessment_is_declaration_oriented_and_scope_checked() -> None:
    root = Root.start({Root.size: 4})
    left = root.child(Root.left)
    assert left.assess(Leaf.ready).ready is None
    assert left.assign(Leaf.factor, 2).assess(Leaf.ready).ready is True
    with pytest.raises(OccurrenceError, match="does not own that Readiness"):
        root.assess(Leaf.ready)


def test_invalid_assignments_do_not_create_a_successor() -> None:
    root = Root.start({Root.size: 4})
    with pytest.raises(OccurrenceError, match="not accepted") as error:
        root.assign(Root.mode, "medium")
    assert error.value.findings
    assert isinstance(root.answer(Root.mode), Unresolved)


def test_projection_assessment_keeps_readiness_constraints_and_raw_output_separate() -> None:
    root = Root.start({Root.size: 6})
    partial = root.child(Root.left).project(Leaf.data)
    assert isinstance(partial, ProjectionAssessment)
    assert tuple(field.name for field in fields(partial)) == (
        "readiness",
        "constraints",
        "output",
        "accepted_answer",
    )
    assert partial.readiness.ready is None
    assert partial.constraints[0].verdict is None
    assert isinstance(partial.output, Unresolved)
    assert isinstance(partial.accepted_answer, Unresolved)
    interpreted = root.child(Root.left).diagnostics(partial, projection=Leaf.data)
    assert any(item.finding.trace for item in interpreted)
    assert all(item.finding in partial.accepted_answer.findings for item in interpreted)

    valid = root.child(Root.left).assign(Leaf.factor, 2).project(Leaf.data)
    assert valid.readiness.ready is True
    assert valid.constraints[0].verdict is True
    assert valid.output == Decided(12)
    assert valid.accepted_answer == Decided(12)


def test_a_false_but_final_constraint_is_ready_and_rejects_the_projection() -> None:
    root = Root.start({Root.size: 6})
    leaf = root.child(Root.left).assign(Leaf.factor, 4)
    assessment = leaf.project(Leaf.data)
    assert assessment.readiness.ready is True
    assert assessment.constraints[0].verdict is False
    assert assessment.output == Decided(24)
    assert isinstance(assessment.accepted_answer, Absent)
    assert assessment.accepted_answer.is_rejection


def test_one_constraint_group_can_validate_several_projections() -> None:
    leaf = Root.start({Root.size: 4}).child(Root.left).assign(Leaf.factor, 2)
    assert leaf.project(Leaf.data).accepted_answer == Decided(8)
    assert leaf.project(Leaf.report).accepted_answer == Decided(8)
    assert leaf.data.accepted_answer == Decided(8)


def test_a_finally_inapplicable_output_is_a_non_rejecting_absence() -> None:
    selected = Root.start({Root.size: 4}).branch(Root.choice).select("nested")
    direct = selected.case("direct")
    assessment = direct.project(Leaf.data)
    assert assessment.readiness.ready is True
    assert isinstance(assessment.output, Absent)
    assert isinstance(assessment.accepted_answer, Absent)
    assert not assessment.accepted_answer.is_rejection


def test_problem_context_is_frozen_and_staleness_is_explicit() -> None:
    source: dict[str, object] = {"size": 4, "note": "first"}

    def project_problem() -> dict[object, object]:
        return {Root.size: source["size"]}

    root = Root.start(project_problem)
    fingerprint = root.problem_fingerprint
    assert root.problem_snapshot == {Root.size: 4}
    with pytest.raises(TypeError):
        root.problem_snapshot[Root.size] = 5  # type: ignore[index]

    source["note"] = "changed but irrelevant"
    assert not root.is_stale()
    assert root.problem_fingerprint == fingerprint

    source["size"] = 5
    assert root.is_stale()
    assert root.answer(Root.size) == Decided(4)
    fresh = root.assign(Root.mode, "small").reconstruct()
    assert type(fresh) is Root
    assert fresh.answer(Root.size) == Decided(5)
    assert isinstance(fresh.answer(Root.mode), Unresolved)
    assert fresh.problem_fingerprint != fingerprint


def test_an_incompatible_persisted_problem_fingerprint_is_rejected() -> None:
    original = Root.start({Root.size: 4})
    with pytest.raises(OccurrenceError, match="incompatible problem fingerprint"):
        Root.start(
            {Root.size: 5},
            expected_problem_fingerprint=original.problem_fingerprint,
        )


def test_a_child_reconstructs_as_the_same_authored_class_in_a_fresh_lineage() -> None:
    source = {Root.size: 4}
    child = Root.start(source).child(Root.left).assign(Leaf.factor, 2)
    source[Root.size] = 5
    fresh = child.reconstruct()
    assert type(fresh) is Leaf
    assert fresh.root.answer(Root.size) == Decided(5)
    assert isinstance(fresh.answer(Leaf.factor), Unresolved)


def test_diagnostics_retain_raw_findings_but_render_occurrence_vocabulary() -> None:
    leaf = Root.start({Root.size: 6}).child(Root.left).assign(Leaf.factor, 4)
    assessment = leaf.project(Leaf.data)
    diagnostics = leaf.diagnostics(assessment, projection=Leaf.data)
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.finding.path.value == "constraint.root.left.below_limit"
    assert diagnostic.scope == ("Root root", "left: Leaf")
    assert diagnostic.declaration == "Leaf.below_limit"
    assert diagnostic.projection == "data"
    assert diagnostic.render() == (
        "Root root / left: Leaf / Leaf.below_limit [data]: "
        "constraint rejected projection 'root.left.data' "
        "(projection-constraint-rejected)"
    )
    assert leaf.diagnostics(assessment, projection=Leaf.data) == diagnostics


def test_concurrent_reads_and_successors_are_deterministic_and_point_isolated() -> None:
    root = Root.start({Root.size: 4})
    complete = root.child(Root.left).assign(Leaf.factor, 2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        reads = tuple(pool.map(lambda _index: complete.project(Leaf.data), range(64)))
    assert all(item.accepted_answer == Decided(8) for item in reads)

    partial = root.child(Root.left)
    with ThreadPoolExecutor(max_workers=3) as pool:
        successors = tuple(pool.map(lambda value: partial.assign(Leaf.factor, value), (1, 2, 4)))
    assert tuple(item.answer(Leaf.result) for item in successors) == (
        Decided(4),
        Decided(8),
        Decided(16),
    )
    assert isinstance(partial.answer(Leaf.result), Unresolved)


def test_concurrent_compilation_under_several_namespaces_is_independent() -> None:
    def build(index: int) -> tuple[str, object]:
        occurrence = Root.start({Root.size: index + 1}, namespace=f"root_{index}")
        child = occurrence.child(Root.left).assign(Leaf.factor, 1)
        return type(occurrence).__name__, child.answer(Leaf.result)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(build, range(12)))
    assert results == tuple(("Root", Decided(index + 1)) for index in range(12))


def test_callbacks_and_public_occurrences_cannot_reach_raw_runtime_objects() -> None:
    root = Path(__file__).parents[3]
    script = r"""
from finn.dataflow._engine import DesignPoint, Engine, QualifiedPath
from finn.dataflow.model import Decision, Problem, Projection, Readiness, Space, derived
from finn.dataflow.model.compiler import _Ref

seen = []

class Audit(Space):
    supplied = Problem(int)
    choice = Decision(int, values=(1, 2))

    @derived(int, supplied=supplied, choice=choice)
    def result(*, supplied, choice):
        seen.append((type(supplied), type(choice)))
        forbidden = (Engine, DesignPoint, QualifiedPath, _Ref)
        assert not isinstance(supplied, forbidden)
        assert not isinstance(choice, forbidden)
        return supplied * choice

    ready = Readiness(decisions=(choice,), properties=(result,))
    output = Projection(result, readiness=ready)

root = Audit.start({Audit.supplied: 3}).assign(Audit.choice, 2)
assessment = root.project(Audit.output)
assert assessment.accepted_answer.value == 6
assert seen == [(int, int)]
for name in ("engine", "point", "ref", "path", "runtime"):
    assert not hasattr(root, name)
for value in (root, root.root, root.problem_snapshot, root.branch if False else None):
    assert not isinstance(value, (Engine, DesignPoint, QualifiedPath, _Ref))
print("CAPABILITY_AUDIT_PASS")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CAPABILITY_AUDIT_PASS"
