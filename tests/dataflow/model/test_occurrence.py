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
    AuthoringError,
    Case,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceContext,
    OccurrenceError,
    OneOf,
    Problem,
    Projection,
    ProjectionAssessment,
    Readiness,
    Space,
    Use,
    RESERVED_LIFECYCLE_NAMES,
    constraint,
    derived,
)
from finn.dataflow.model.compiler import compile_space, compile_space_model, compiled_model_for
from finn.dataflow.model.occurrence import _Lineage, _occurrence_state, is_attached_occurrence


def _lineage(instance: Space) -> _Lineage:
    """Reach the private lineage record; a test may, a contributor may not."""

    return _occurrence_state(instance).runtime.lineage


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


# -- C2: construction, compiled-model reuse, dispatch, reserved names ----------


def test_a_declaration_may_not_shadow_a_lifecycle_operation() -> None:
    with pytest.raises(AuthoringError) as raised:

        class Shadowing(Space):
            size = Problem(int)
            project = Decision(int, values=(1, 2))  # type: ignore[assignment]

    message = str(raised.value)
    assert "Shadowing.project" in message
    assert "Decision" in message
    assert "Space.project" in message


@pytest.mark.parametrize("member", sorted(RESERVED_LIFECYCLE_NAMES))
def test_every_reserved_lifecycle_name_is_refused(member: str) -> None:
    with pytest.raises(AuthoringError, match=f"Space.{member}"):
        type(
            "Reserved",
            (Space,),
            {"size": Problem(int), member: Decision(int, values=(1, 2))},
        )


def test_a_declaration_attached_after_class_creation_is_refused_at_compilation() -> None:
    """``__init_subclass__`` cannot see this one; the compiler backstop must."""

    class Late(Space):
        size = Problem(int)

    Late.assign = Decision(int, values=(1, 2))  # type: ignore[method-assign, assignment]

    with pytest.raises(AuthoringError, match="Space.assign"):
        compile_space(Late, "root", problem_namespace="problem.root")


def test_a_reserved_name_is_still_available_as_a_stable_compiled_name() -> None:
    """The check is on the Python member, not on the engine path."""

    class Renamed(Space):
        size = Problem(int)
        choice = Decision(int, values=(1, 2), name="project")

    specification = compile_space(Renamed, "root", problem_namespace="problem.root")
    assert tuple(item.path.value for item in specification.decisions) == ("root.project",)


class _Contextual(Space):
    """An authored class whose ordinary constructor requires external context."""

    size = Problem(int)
    lanes = Decision(int, values=(1, 2))
    inner = Use(Leaf, supplied=size)

    def __init__(self, label: str) -> None:
        self.label = label

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        instance = object.__new__(cls)
        parent = context.root
        instance.__init__(  # type: ignore[misc]
            context.namespace if parent is None else getattr(parent, "label")
        )
        return instance


def test_a_context_bearing_class_is_initialized_by_the_construction_hook() -> None:
    root = _Contextual.start({_Contextual.size: 3})
    assert type(root) is _Contextual
    assert root.label == "root"

    # A child view is constructed through the same hook and reads the root's
    # context rather than having it copied in by the occurrence layer.
    child = root.child(_Contextual.inner)
    assert type(child) is Leaf
    assert child.root is root

    # A successor is allocated through the hook as well, so its context is
    # constructed rather than carried over by attribute copying.
    successor = root.assign(_Contextual.lanes, 2)
    assert type(successor) is _Contextual
    assert successor.label == "root"
    assert successor is not root


def test_the_construction_hook_must_return_an_instance_of_the_authored_class() -> None:
    class Wrong(Space):
        size = Problem(int)

        @classmethod
        def _new_occurrence(cls, context: OccurrenceContext) -> Space:
            del context
            return Leaf.__new__(Leaf)

    with pytest.raises(AuthoringError, match="not an instance of Wrong"):
        Wrong.start({Wrong.size: 1})


def test_repeated_starts_reuse_one_compiled_model() -> None:
    first = compiled_model_for(Root, "root", problem_namespace="problem.root")
    second = compiled_model_for(Root, "root", problem_namespace="problem.root")
    assert first is second
    assert first._compiled_tree() is second._compiled_tree()
    assert first._design_space() is second._design_space()


def test_a_mutated_declaration_structure_forces_recompilation() -> None:
    class Mutable(Space):
        size = Problem(int)
        lanes = Decision(int, values=(1, 2))

    before = compiled_model_for(Mutable, "root", problem_namespace="problem.root")
    Mutable.lanes = Decision(int, values=(1, 2, 4))
    after = compiled_model_for(Mutable, "root", problem_namespace="problem.root")
    assert after is not before
    assert compiled_model_for(Mutable, "root", problem_namespace="problem.root") is after


def test_independent_roots_share_compilation_but_not_engine_or_lock() -> None:
    first = Root.start({Root.size: 4})
    second = Root.start({Root.size: 8})
    left = _lineage(first)
    right = _lineage(second)
    assert left.model is right.model
    assert left.tree is right.tree
    assert left.engine is not right.engine
    assert left.lock is not right.lock

    # A successor stays inside its own lineage.
    successor = first.assign(Root.mode, "small")
    assert _lineage(successor).engine is left.engine
    assert _lineage(successor).lock is left.lock


def test_the_compiled_model_is_a_public_compiler_service_entry() -> None:
    model = compile_space_model(Root, "root", problem_namespace="problem.root")
    first = model.start({Root.size: 4})
    second = model.start({Root.size: 8})
    assert type(first) is Root
    assert type(second) is Root
    assert model.space_type is Root
    assert first.size == 4
    assert second.size == 8
    assert _lineage(first).engine is not _lineage(second).engine


def test_a_legacy_configured_instance_still_resolves_through_its_own_hook() -> None:
    """Attached and configured are two protocols; the dispatcher must not guess."""

    class Legacy(Space):
        size = Problem(int)

        def __init__(self, value: int) -> None:
            self._value = value

        def _space_value(self, declaration: object) -> object:
            del declaration
            return self._value

    configured = Legacy(11)
    assert not is_attached_occurrence(configured)
    assert configured.size == 11

    attached = Legacy.start({Legacy.size: 4})
    assert is_attached_occurrence(attached)
    assert attached.size == 4


def test_an_instance_that_is_neither_attached_nor_configured_has_no_values() -> None:
    class Plain(Space):
        size = Problem(int)

    with pytest.raises(AttributeError, match="configured instances"):
        _ = Plain().size
