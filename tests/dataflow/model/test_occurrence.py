# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import cast

import pytest

from finn.dataflow._engine import (
    Absent,
    Decided,
    DesignPoint,
    DesignSpace,
    Engine,
    QualifiedPath,
    RequestError,
    Unresolved,
)
from finn.dataflow.model import (
    AuthoringError,
    CanonicalValueCodec,
    RESERVED_PROTOCOL_NAMES,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceContext,
    Variant,
    Problem,
    Projection,
    ProjectionAssessment,
    Readiness,
    Space,
    Subspace,
    RESERVED_LIFECYCLE_NAMES,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.declarations import enum_semantics
from finn.dataflow.model.compiler import (
    SpaceModel,
    _CompiledSpace,
    _Ref,
    compile_space,
    compile_space_model,
)
from finn.dataflow.model.declarations import declared_members
from finn.dataflow.model.occurrence import (
    VariantView,
    _Lineage,
    _occurrence_state,
    _Runtime,
    _State,
    is_attached_occurrence,
)


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
    leaf = Subspace(Leaf, supplied=supplied)
    result = leaf.result
    exports = (result,)


class Root(Space):
    size = Problem(int)
    mode = Decision(str, values=("small", "large"))
    left = Subspace(Leaf, supplied=size)
    right = Subspace(Leaf, supplied=size)
    choice = Variant(
        {"nested": Subspace(Nested, supplied=size), "direct": Subspace(Leaf, supplied=size)},
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

    left = root.left
    right = root.right
    assert type(left) is Leaf
    assert type(right) is Leaf
    assert left.root is root
    assert right.root is root
    assert left.supplied == 4


def test_assignment_returns_immutable_successors_of_the_same_authored_class() -> None:
    root = Root.start({Root.size: 4})
    partial = root.left
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
    # There is no class-keyed navigation verb left to misuse: ``Leaf`` names a
    # family, and the only way to an occurrence is the member that places it.
    assert not hasattr(root, "child")
    assert not hasattr(root, "branch")
    with pytest.raises(AuthoringError) as refusal:
        root.assign(Leaf.factor, 2)
    # The refusal names every placement, so the caller can reach the one meant.
    assert "root.left" in str(refusal.value)
    assert "root.right" in str(refusal.value)
    assert "root.choice.direct" in str(refusal.value)

    left = root.left.assign(Leaf.factor, 2)
    right = cast("Root", left.root).right
    assert left.answer(Leaf.result) == Decided(8)
    assert isinstance(right.answer(Leaf.result), Unresolved)


def test_variant_views_select_and_navigate_alternatives_without_paths() -> None:
    root = Root.start({Root.size: 3})
    branch = root.choice
    assert isinstance(branch, VariantView)
    assert branch.name == "choice"
    assert branch.alternatives == ("nested", "direct")
    assert isinstance(branch.selected(), Unresolved)

    selected = branch.select("nested")
    assert selected.selected() == Decided("nested")
    nested = cast("Nested", selected.alternative("nested"))
    leaf = nested.leaf.assign(Leaf.factor, 4)
    final_root = cast("Root", leaf.root)
    assert final_root.answer(Root.selected_result) == Decided(12)
    assert final_root.choice.selected() == Decided("nested")


def test_assessment_is_declaration_oriented_and_scope_checked() -> None:
    root = Root.start({Root.size: 4})
    left = root.left
    assert left.assess(Leaf.ready).ready is None
    assert left.assign(Leaf.factor, 2).assess(Leaf.ready).ready is True
    with pytest.raises(AuthoringError, match="not owned by Root at root"):
        root.assess(Leaf.ready)


def test_invalid_assignments_do_not_create_a_successor() -> None:
    root = Root.start({Root.size: 4})
    with pytest.raises(RequestError) as error:
        root.assign(Root.mode, "medium")
    assert error.value.findings
    assert isinstance(root.answer(Root.mode), Unresolved)


def test_projection_assessment_keeps_readiness_constraints_and_raw_output_separate() -> None:
    root = Root.start({Root.size: 6})
    partial = root.left.project(Leaf.data)
    assert isinstance(partial, ProjectionAssessment)
    assert partial.projection == "root.left.data"
    assert tuple(field.name for field in fields(partial)) == (
        "projection",
        "readiness",
        "constraints",
        "output",
        "accepted_answer",
    )
    assert partial.readiness.ready is None
    assert partial.constraints[0].verdict is None
    assert isinstance(partial.output, Unresolved)
    assert isinstance(partial.accepted_answer, Unresolved)
    interpreted = root.left.diagnostics(partial)
    assert any(item.finding.trace for item in interpreted)
    assert all(item.finding in partial.accepted_answer.findings for item in interpreted)

    valid = root.left.assign(Leaf.factor, 2).project(Leaf.data)
    assert valid.readiness.ready is True
    assert valid.constraints[0].verdict is True
    assert valid.output == Decided(12)
    assert valid.accepted_answer == Decided(12)


def test_a_false_but_final_constraint_is_ready_and_rejects_the_projection() -> None:
    root = Root.start({Root.size: 6})
    leaf = root.left.assign(Leaf.factor, 4)
    assessment = leaf.project(Leaf.data)
    assert assessment.readiness.ready is True
    assert assessment.constraints[0].verdict is False
    assert assessment.output == Decided(24)
    assert isinstance(assessment.accepted_answer, Absent)
    assert assessment.accepted_answer.is_rejection


def test_one_constraint_group_can_validate_several_projections() -> None:
    leaf = Root.start({Root.size: 4}).left.assign(Leaf.factor, 2)
    assert leaf.project(Leaf.data).accepted_answer == Decided(8)
    assert leaf.project(Leaf.report).accepted_answer == Decided(8)
    assert leaf.data.accepted_answer == Decided(8)


def test_a_finally_inapplicable_output_is_a_non_rejecting_absence() -> None:
    selected = Root.start({Root.size: 4}).choice.select("nested")
    direct = selected.alternative("direct")
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
    with pytest.raises(RequestError, match="engine request failed"):
        Root.start(
            {Root.size: 5},
            expected_problem_fingerprint=original.problem_fingerprint,
        )


def test_a_child_reconstructs_as_the_same_authored_class_in_a_fresh_lineage() -> None:
    source = {Root.size: 4}
    child = Root.start(source).left.assign(Leaf.factor, 2)
    source[Root.size] = 5
    fresh = child.reconstruct()
    assert type(fresh) is Leaf
    assert fresh.root.answer(Root.size) == Decided(5)
    assert isinstance(fresh.answer(Leaf.factor), Unresolved)


class _Watched(Space):
    """A Space whose one constraint refuses with a reason of its own."""

    supplied = Input(int)

    @constraint(supplied=supplied)
    def within_range(*, supplied: int) -> object:
        return True if supplied <= 4 else reject("out-of-range", "supplied is too large")

    checks = ConstraintGroup(within_range)
    ready = Readiness(constraints=checks)
    view = Projection(supplied, readiness=ready, constraints=checks)


class _WatchedNest(Space):
    supplied = Input(int)
    inner = Subspace(_Watched, supplied=supplied)


class _DiagRoot(Space):
    size = Problem(int)
    left = Subspace(_Watched, supplied=size)
    right = Subspace(_Watched, supplied=size)
    pick = Variant(
        {
            "nested": Subspace(_WatchedNest, supplied=size),
            "direct": Subspace(_Watched, supplied=size),
        }
    )


def test_a_declared_finding_is_attributed_to_the_member_that_declared_it() -> None:
    left = _DiagRoot.start({_DiagRoot.size: 9}).left
    diagnostics = left.diagnostics(left.assess(_Watched.checks), projection=_Watched.view)
    assert [(item.space, item.member) for item in diagnostics] == [("_Watched", "within_range")]
    diagnostic = diagnostics[0]
    assert diagnostic.scope == ("root", "left")
    assert diagnostic.case is None
    assert diagnostic.projection == "root.left.view"
    # The raw finding survives whole -- kind, code, path, values and trace.
    assert diagnostic.finding.path == QualifiedPath("constraint.root.left.within_range")
    assert diagnostic.finding.code == "out-of-range"
    assert diagnostic.finding.message == "supplied is too large"
    raw = left.assess(_Watched.checks).answers[QualifiedPath("constraint.root.left.within_range")]
    assert isinstance(raw, Absent)
    assert diagnostic.finding is raw.findings[0]
    assert "_Watched.within_range" in diagnostic.render()


def test_a_generated_path_is_left_unattributed_rather_than_guessed_at() -> None:
    leaf = Root.start({Root.size: 6}).left.assign(Leaf.factor, 4)
    assessment = leaf.project(Leaf.data)
    diagnostics = leaf.diagnostics(assessment)
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    # The reduction synthesized this finding against the projection's own path,
    # which no declaration owns.  Naming ``Leaf.data`` from the last path
    # segment would be an identifier nobody wrote.
    assert diagnostic.finding.code == "projection-constraint-refused"
    assert diagnostic.finding.trace == (QualifiedPath("constraint.root.left.below_limit"),)
    assert (diagnostic.space, diagnostic.member) == (None, None)
    assert diagnostic.scope == ("root", "left")
    assert "declaration <generated>" in diagnostic.render()
    assert leaf.diagnostics(assessment, projection=Leaf.data) == diagnostics


def test_two_placements_of_one_class_receive_distinct_scopes() -> None:
    root = _DiagRoot.start({_DiagRoot.size: 9})
    scopes = {
        view.diagnostics(view.assess(_Watched.checks))[0].scope for view in (root.left, root.right)
    }
    assert scopes == {("root", "left"), ("root", "right")}


def test_a_finding_inside_a_branch_case_names_that_case() -> None:
    root = _DiagRoot.start({_DiagRoot.size: 9})
    nested = cast("_WatchedNest", root.pick.select("nested").alternative("nested"))
    inner = nested.inner
    diagnostics = inner.diagnostics(inner.assess(_Watched.checks))
    assert len(diagnostics) == 1
    assert diagnostics[0].case == "nested"
    assert diagnostics[0].scope == ("root", "pick", "nested", "inner")
    assert (diagnostics[0].space, diagnostics[0].member) == ("_Watched", "within_range")
    assert "case nested" in diagnostics[0].render()


def test_a_supplied_input_is_attributed_to_its_supplier() -> None:
    """A child ``Input`` compiles to the supplier's path; the supplier owns it."""

    root = _DiagRoot.start({_DiagRoot.size: 9})
    owners = _lineage(root).model._owner_index()
    supplier = owners[QualifiedPath("problem.root.size")]
    assert (supplier.space, supplier.member) == ("_DiagRoot", "size")
    # ``_Watched.supplied`` is bound to it and never claims it.
    assert all(owner.member != "supplied" for owner in owners.values())


def test_identical_findings_are_reported_once_and_deterministically() -> None:
    left = _DiagRoot.start({_DiagRoot.size: 9}).left
    assessment = left.project(_Watched.view)
    diagnostics = left.diagnostics(assessment)
    # ``within_range`` reaches the assessment through the readiness profile and
    # again through the projection's constraint group; that is one problem.
    assert len(diagnostics) == len({item.finding for item in diagnostics})
    assert diagnostics == left.diagnostics(assessment)


def test_concurrent_reads_and_successors_are_deterministic_and_point_isolated() -> None:
    root = Root.start({Root.size: 4})
    complete = root.left.assign(Leaf.factor, 2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        reads = tuple(pool.map(lambda _index: complete.project(Leaf.data), range(64)))
    assert all(item.accepted_answer == Decided(8) for item in reads)

    partial = root.left
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
        child = occurrence.left.assign(Leaf.factor, 1)
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
for value in (root, root.root, root.problem_snapshot):
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
    inner = Subspace(Leaf, supplied=size)

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
    child = root.inner
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


def test_one_held_model_is_the_explicit_reuse_path() -> None:
    model = compile_space_model(Root, "root", problem_namespace="problem.root")
    first = model.start({Root.size: 4})
    second = model.start({Root.size: 8})
    assert type(first) is Root
    assert type(second) is Root
    assert model.space_type is Root
    assert first.size == 4
    assert second.size == 8

    # One compiled model and one validated DesignSpace, shared by both roots.
    assert _lineage(first).model is model
    assert _lineage(second).model is model
    assert _lineage(first).tree is _lineage(second).tree
    assert model._design_space() is model._design_space()

    # Nothing runtime is shared: separate Engines, separate locks.
    assert _lineage(first).engine is not _lineage(second).engine
    assert _lineage(first).lock is not _lineage(second).lock

    # A successor stays inside its own lineage.
    successor = first.assign(Root.mode, "small")
    assert _lineage(successor).engine is _lineage(first).engine
    assert _lineage(successor).lock is _lineage(first).lock


def test_there_is_no_hidden_compiled_model_cache() -> None:
    """Reuse is something a caller holds, never something the class does."""

    class Cacheable(Space):
        size = Problem(int)
        lanes = Decision(int, values=(1, 2))

    first = compile_space_model(Cacheable, "root", problem_namespace="problem.root")
    second = compile_space_model(Cacheable, "root", problem_namespace="problem.root")
    assert first is not second
    assert first._compiled_tree() is not second._compiled_tree()
    assert not any(name.startswith("_space_compiled") for name in vars(Cacheable))

    # Two ergonomic one-shot starts likewise share no compilation, so a class
    # body edited between them cannot be answered from a stale model.
    left = _lineage(Cacheable.start({Cacheable.size: 1}))
    right = _lineage(Cacheable.start({Cacheable.size: 1}))
    assert left.model is not right.model


def test_an_export_only_edit_is_visible_to_the_next_compilation() -> None:
    """One of the compilation inputs the retired cache key could not see."""

    class Exporting(Space):
        supplied = Input(int)

        @derived(int, supplied=supplied)
        def doubled(*, supplied: int) -> int:
            return supplied * 2

    class Holder(Space):
        size = Problem(int)
        inner = Subspace(Exporting, supplied=size)

    before = compile_space_model(Holder, "root", problem_namespace="problem.root")
    assert not dict(before._compiled_tree().child("inner").exports)
    Exporting.exports = (Exporting.doubled,)
    after = compile_space_model(Holder, "root", problem_namespace="problem.root")
    assert "doubled" in dict(after._compiled_tree().child("inner").exports)


def test_declared_values_exist_only_on_an_attached_occurrence() -> None:
    """One protocol.

    U2 and U3 retired the detached *configured* Kernel and Design objects that
    answered through a second ``_space_value`` hook, so the dispatcher no longer
    has two protocols to choose between and an unattached instance is simply an
    instance with no values.  ``_space_value`` stays reserved: the failure the
    reservation prevents -- two protocols deciding by inheritance order -- is a
    property of having a hook at all, not of anyone currently using it.
    """

    class Plain(Space):
        size = Problem(int)

    bare = object.__new__(Plain)
    assert not is_attached_occurrence(bare)
    with pytest.raises(AttributeError, match="only on an attached Space occurrence"):
        _ = bare.size

    attached = Plain.start({Plain.size: 4})
    assert is_attached_occurrence(attached)
    assert attached.size == 4
    assert "_space_value" in RESERVED_PROTOCOL_NAMES


# -- C3: projection, navigation, errors, fingerprints -------------------------


class _Payload(Space):
    size = Input(int)

    @derived(int, size=size)
    def value(*, size: int) -> int:
        return size

    exports = (value,)


class _Refusing(Space):
    """A Space whose output can be absent, refused, or both, on demand."""

    size = Input(int)
    present = Decision(bool, values=(False, True))
    inner = Subspace(_Payload, size=size, when=present)
    payload = inner.value

    @constraint(size=size)
    def small_enough(*, size: int) -> object:
        return True if size <= 10 else reject("too-large", "this size is refused")

    limits = ConstraintGroup(small_enough)
    ready = Readiness(decisions=(present,), properties=(payload,))
    view = Projection(payload, readiness=ready, constraints=limits)


class _RefusingRoot(Space):
    size = Problem(int)
    inner = Subspace(_Refusing, size=size)


def test_unresolved_dominates_every_other_verdict() -> None:
    root = _RefusingRoot.start({_RefusingRoot.size: 50})
    assessment = root.inner.project(_Refusing.view)
    # The constraint has already refused, but the presence decision is
    # uncommitted, so the honest answer is "not yet", not "no".
    assert assessment.constraints[0].verdict is False
    assert isinstance(assessment.accepted_answer, Unresolved)


def test_final_absence_precedes_constraint_refusal() -> None:
    root = _RefusingRoot.start({_RefusingRoot.size: 50})
    inner = root.inner.assign(_Refusing.present, False)
    assessment = inner.project(_Refusing.view)

    assert assessment.readiness.ready is True
    assert assessment.constraints[0].verdict is False
    assert isinstance(assessment.output, Absent)
    accepted = assessment.accepted_answer
    # The value does not arise here.  Saying "a constraint refused it" would be
    # a different and misleading sentence, so absence propagates unchanged and
    # carries the output's own findings rather than the refusal's.
    assert isinstance(accepted, Absent)
    assert accepted is assessment.output
    assert all(finding.code != "projection-constraint-refused" for finding in accepted.findings)


def test_constraint_refusal_governs_an_available_output() -> None:
    root = _RefusingRoot.start({_RefusingRoot.size: 50})
    inner = root.inner.assign(_Refusing.present, True)
    assessment = inner.project(_Refusing.view)

    assert assessment.readiness.ready is True
    assert assessment.output == Decided(50)
    assert isinstance(assessment.accepted_answer, Absent)
    # The raw output stays visible even though validation rejected it.
    assert any(finding.code == "too-large" for finding in assessment.accepted_answer.findings)


def test_a_projection_requires_an_explicit_readiness_declaration() -> None:
    with pytest.raises(TypeError, match="readiness"):
        Projection(Leaf.result)  # type: ignore[call-arg]
    with pytest.raises(AuthoringError, match="declare an empty profile"):
        Projection(Leaf.result, readiness=None)  # type: ignore[arg-type]


def test_a_projection_refuses_a_bare_constraint_where_a_group_belongs() -> None:
    with pytest.raises(AuthoringError, match="belongs to a group"):
        Projection(Leaf.result, readiness=Leaf.ready, constraints=(Leaf.below_limit,))  # type: ignore[arg-type]


def test_a_projection_may_not_name_the_same_group_twice() -> None:
    class Duplicated(Space):
        size = Problem(int)

        @constraint(size=size)
        def positive(*, size: int) -> bool:
            return size > 0

        checks = ConstraintGroup(positive)
        ready = Readiness(constraints=checks)
        view = Projection(size, readiness=ready, constraints=(checks, checks))

    with pytest.raises(AuthoringError, match="names constraint group 'checks' twice"):
        compile_space(Duplicated, "root", problem_namespace="problem.root")


def test_a_projection_may_not_name_a_group_another_class_declares() -> None:
    class Borrowed(Space):
        size = Problem(int)
        ready = Readiness()
        view = Projection(size, readiness=ready, constraints=(Leaf.legal,))

    with pytest.raises(AuthoringError, match="does not declare"):
        compile_space(Borrowed, "root", problem_namespace="problem.root")


def test_a_projection_may_not_borrow_another_class_readiness() -> None:
    class Borrowed(Space):
        size = Problem(int)
        view = Projection(size, readiness=Leaf.ready)

    with pytest.raises(AuthoringError, match="Readiness declaration"):
        compile_space(Borrowed, "root", problem_namespace="problem.root")


def test_one_constraint_group_serves_several_projections() -> None:
    left = Root.start({Root.size: 4}).left.assign(Leaf.factor, 2)
    data = left.project(Leaf.data)
    report = left.project(Leaf.report)
    assert data.projection == "root.left.data"
    assert report.projection == "root.left.report"
    assert data.constraints[0].verdict is True
    assert report.constraints[0].verdict is True
    assert data.accepted_answer == report.accepted_answer == Decided(8)


def test_adding_a_projection_changes_no_engine_declaration() -> None:
    """A Projection is a stored question over paths the lowering already made."""

    def build(with_projection: bool) -> object:
        body: dict[str, object] = {
            "size": Problem(int),
            "lanes": Decision(int, values=(1, 2)),
        }
        body["ready"] = Readiness(decisions=(body["lanes"],))  # type: ignore[arg-type]
        if with_projection:
            body["view"] = Projection(
                body["lanes"],  # type: ignore[arg-type]
                readiness=body["ready"],  # type: ignore[arg-type]
            )
        return compile_space(type("Twin", (Space,), body), "root", problem_namespace="problem.root")

    plain = build(False)
    projected = build(True)
    for attribute in (
        "decisions",
        "properties",
        "constraints",
        "constraint_sets",
        "readiness_profiles",
    ):
        assert [
            item.path if hasattr(item, "path") else item.name for item in getattr(plain, attribute)
        ] == [
            item.path if hasattr(item, "path") else item.name
            for item in getattr(projected, attribute)
        ]


def test_a_query_never_commits_anything() -> None:
    root = Root.start({Root.size: 4})
    left = root.left
    for _index in range(3):
        left.project(Leaf.data)
        left.assess(Leaf.ready)
        left.answer(Leaf.result)
        root.choice.selected()
    assert isinstance(left.answer(Leaf.factor), Unresolved)
    assert isinstance(root.choice.selected(), Unresolved)


def test_a_singleton_branch_is_selected_without_a_committed_decision() -> None:
    class Only(Space):
        size = Problem(int)
        pick = Variant({"one": Subspace(Leaf, supplied=size)}, outputs=("result",))

    root = Only.start({Only.size: 2})
    branch = root.pick
    assert branch.alternatives == ("one",)
    assert branch.selected() == Decided("one")
    assert type(branch.alternative("one")) is Leaf


def test_an_unknown_branch_case_is_an_authoring_error() -> None:
    root = Root.start({Root.size: 2})
    with pytest.raises(AuthoringError, match="has no alternative 'missing'"):
        root.choice.select("missing")
    with pytest.raises(AuthoringError, match="has no alternative 'missing'"):
        root.choice.alternative("missing")


def test_the_occurrence_surface_never_accepts_a_path() -> None:
    root = Root.start({Root.size: 4})
    calls: tuple[Callable[[], object], ...] = (
        lambda: root.answer("semantic.root.left.result"),  # type: ignore[arg-type]
        lambda: root.project("root.left.data"),  # type: ignore[arg-type]
        lambda: root.assess("root.left.ready"),  # type: ignore[call-overload]
    )
    for call in calls:
        with pytest.raises((AuthoringError, TypeError, AttributeError)):
            call()


# -- fingerprints --------------------------------------------------------------


class _Colour(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True)
class _Shape:
    rows: int
    cols: int


class _Opaque:
    """A value with no canonical encoding of its own, like a QONNX datatype."""

    def __init__(self, name: str) -> None:
        self.name = name


_OPAQUE_CODEC: CanonicalValueCodec[_Opaque] = CanonicalValueCodec(
    "test.opaque", 1, lambda value: value.name
)


class _Fingerprinted(Space):
    count = Problem(int)
    ratio = Problem(float)
    colour = Problem(enum_semantics(_Colour))
    shape = Problem(_Shape)
    tags = Problem(tuple, required=False)
    opaque = Problem(_Opaque, canonical=_OPAQUE_CODEC)


def _fingerprint(**values: object) -> str:
    supplied = {getattr(_Fingerprinted, name): value for name, value in values.items()}
    return _Fingerprinted.start(supplied).problem_fingerprint


_BASE: dict[str, object] = {
    "count": 4,
    "ratio": 0.5,
    "colour": _Colour.RED,
    "shape": _Shape(2, 3),
    "tags": ("a", "b"),
    "opaque": _Opaque("INT8"),
}


def test_identical_problems_fingerprint_identically() -> None:
    assert _fingerprint(**_BASE) == _fingerprint(**{**_BASE, "opaque": _Opaque("INT8")})


@pytest.mark.parametrize(
    "changed",
    [
        {"count": 5},
        {"ratio": 0.5000001},
        {"colour": _Colour.BLUE},
        {"shape": _Shape(3, 2)},
        {"tags": ("b", "a")},
        {"opaque": _Opaque("INT16")},
    ],
)
def test_any_changed_declared_fact_changes_the_fingerprint(changed: dict[str, object]) -> None:
    assert _fingerprint(**{**_BASE, **changed}) != _fingerprint(**_BASE)


def test_explicit_absence_is_distinguished_from_any_value() -> None:
    without = {name: value for name, value in _BASE.items() if name != "tags"}
    assert _fingerprint(**without) != _fingerprint(**_BASE)
    assert _fingerprint(**without) != _fingerprint(**{**_BASE, "tags": ()})


def test_two_space_families_sharing_a_problem_shape_do_not_collide() -> None:
    class First(Space):
        size = Problem(int)

    class Second(Space):
        size = Problem(int)

    assert (
        First.start({First.size: 4}).problem_fingerprint
        != Second.start({Second.size: 4}).problem_fingerprint
    )


def test_a_value_with_no_canonical_encoding_is_refused_rather_than_guessed() -> None:
    class Unencodable(Space):
        thing = Problem(_Opaque)

    with pytest.raises(AuthoringError, match="has no canonical encoding"):
        Unencodable.start({Unencodable.thing: _Opaque("INT8")}).problem_fingerprint


def test_a_declared_codec_identity_travels_with_the_value() -> None:
    """Re-encoding the same value under a new codec version is a new problem."""

    def family(codec: CanonicalValueCodec[_Opaque]) -> str:
        thing: Problem[_Opaque] = Problem(_Opaque, canonical=codec)
        space: type[Space] = type("Coded", (Space,), {"thing": thing})
        return space.start({thing: _Opaque("INT8")}).problem_fingerprint

    first = family(CanonicalValueCodec("test.opaque", 1, lambda value: value.name))
    second = family(CanonicalValueCodec("test.opaque", 2, lambda value: value.name))
    assert first != second


def test_a_non_finite_float_is_refused() -> None:
    class Floaty(Space):
        ratio = Problem(float)

    with pytest.raises(AuthoringError, match="finite float"):
        Floaty.start({Floaty.ratio: float("inf")}).problem_fingerprint


# -- C4: the public capability boundary ---------------------------------------

#: What no public return value may be, or transitively reach.
#:
#: ``QualifiedPath`` is deliberately absent.  It remains the engine's stable
#: identity and appears inside every immutable ``Finding``, which is exactly
#: where a diagnostic needs it; what is withheld is the ability to *construct*
#: one and reach a declaration nobody offered, and that is pinned separately by
#: ``test_the_occurrence_surface_never_accepts_a_path``.
_FORBIDDEN: tuple[type, ...] = (
    Engine,
    DesignPoint,
    DesignSpace,
    _Ref,
    _CompiledSpace,
    SpaceModel,
    _Lineage,
    _Runtime,
    _State,
)


def _reachable(value: object, depth: int = 3) -> list[str]:
    """Every forbidden object reachable from a public attribute, to ``depth``."""

    if depth < 0:
        return []
    leaks: list[str] = []
    if isinstance(value, _FORBIDDEN):
        return [type(value).__name__]
    if isinstance(value, (str, bytes, int, float, bool, type(None), type)):
        return []
    if isinstance(value, Mapping):
        for key, item in value.items():
            leaks.extend(_reachable(key, depth - 1))
            leaks.extend(_reachable(item, depth - 1))
        return leaks
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            leaks.extend(_reachable(item, depth - 1))
        return leaks
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attribute = getattr(value, name)
        except Exception:  # noqa: BLE001 - an unreadable attribute leaks nothing
            continue
        if callable(attribute):
            continue
        leaks.extend(f"{name}.{leak}" for leak in _reachable(attribute, depth - 1))
    return leaks


def test_no_public_return_value_reaches_the_runtime() -> None:
    root = _DiagRoot.start({_DiagRoot.size: 9})
    left = root.left
    branch = root.pick
    assessment = left.project(_Watched.view)
    returns: tuple[object, ...] = (
        root,
        left,
        left.root,
        root.problem_snapshot,
        root.problem_fingerprint,
        branch,
        branch.alternatives,
        branch.selected(),
        branch.alternative("direct"),
        assessment,
        left.assess(_Watched.checks),
        left.assess(_Watched.ready),
        left.answer(_Watched.supplied),
        left.diagnostics(assessment, projection=_Watched.view),
    )
    assert [leak for value in returns for leak in _reachable(value)] == []
    # Paths do survive inside findings, and that is the intent, not a leak.
    assert any(
        isinstance(item.finding.path, QualifiedPath)
        for item in left.diagnostics(assessment, projection=_Watched.view)
    )


def test_the_public_occurrence_surface_is_exactly_the_lifecycle() -> None:
    root = Root.start({Root.size: 4})
    declared = {name for name, _declaration in declared_members(Root)}
    public = {
        name
        for name in dir(root)
        if not name.startswith("_") and name not in declared and name != "exports"
    }
    assert public == set(RESERVED_LIFECYCLE_NAMES)


def test_private_reflection_is_outside_the_supported_guarantee() -> None:
    """Stated rather than pretended: this is Python privacy, not a sandbox."""

    root = Root.start({Root.size: 4})
    assert isinstance(_occurrence_state(root), _State)
    # Deliberately hostile code reaches it.  The supported guarantee is that no
    # *public* operation hands it over, which the scan above proves; adding weak
    # maps to hide it from reflection would buy complexity and no safety.
    assert isinstance(getattr(root, "_occurrence_state").runtime.lineage.engine, Engine)


# -- C5: consolidated forcing and the concurrency contract ---------------------


class _Placed(Space):
    supplied = Input(int)
    scale = Decision(int, values=(1, 2, 3))

    @derived(int, supplied=supplied, scale=scale)
    def scaled(*, supplied: int, scale: int) -> int:
        return supplied * scale

    exports = (scaled,)


class _PlacedPair(Space):
    supplied = Input(int)
    first = Subspace(_Placed, supplied=supplied)
    second = Subspace(_Placed, supplied=supplied)


class _FivePlacements(Space):
    size = Problem(int)
    pair = Subspace(_PlacedPair, supplied=size)
    solo = Subspace(_Placed, supplied=size)
    pick = Variant(
        {"one": Subspace(_Placed, supplied=size), "two": Subspace(_Placed, supplied=size)},
        outputs=("scaled",),
    )


def test_five_placements_of_one_class_are_five_distinct_occurrences() -> None:
    root = _FivePlacements.start({_FivePlacements.size: 2})
    pair = root.pair
    views = (
        pair.first,
        pair.second,
        root.solo,
        root.pick.alternative("one"),
        root.pick.alternative("two"),
    )
    assert all(type(view) is _Placed for view in views)
    assert len({_occurrence_state(view).compiled.namespace for view in views}) == 5

    # Assigning through one view moves that one and no other.
    first = views[0].assign(_Placed.scale, 3)
    assert first.answer(_Placed.scaled) == Decided(6)
    assert isinstance(
        cast("_FivePlacements", first.root).pair.second.answer(_Placed.scaled),
        Unresolved,
    )


def test_a_subspace_inside_a_variant_alternative_is_reached_by_naming_both() -> None:
    root = Root.start({Root.size: 3})
    nested = root.choice.select("nested").alternative("nested")
    assert type(nested) is Nested
    leaf = nested.leaf
    assert type(leaf) is Leaf
    assert _occurrence_state(leaf).compiled.namespace == "root.choice.nested.leaf"
    assert leaf.assign(Leaf.factor, 4).root.answer(Root.selected_result) == Decided(12)


def test_the_declared_namespaces_of_the_existing_stack_are_unchanged() -> None:
    """U1 renames, reparents and removes no ``QualifiedPath``."""

    specification = compile_space(Root, "root", problem_namespace="problem.root")
    assert tuple(item.path.value for item in specification.decisions) == (
        "root.mode",
        "root.choice.case",
        "root.left.factor",
        "root.right.factor",
        "root.choice.nested.leaf.factor",
        "root.choice.direct.factor",
    )
    assert tuple(item.name for item in specification.readiness_profiles) == (
        "root.left.ready",
        "root.right.ready",
        "root.choice.nested.leaf.ready",
        "root.choice.direct.ready",
    )
    assert tuple(item.name for item in specification.constraint_sets) == (
        "root.left.legal",
        "root.right.legal",
        "root.choice.nested.leaf.legal",
        "root.choice.direct.legal",
    )


def test_no_evaluation_result_crosses_between_two_points() -> None:
    root = Root.start({Root.size: 4})
    partial = root.left
    doubled = partial.assign(Leaf.factor, 2)
    quadrupled = partial.assign(Leaf.factor, 4)

    # Interleave deliberately: a cache keyed by anything but the point would
    # hand the second reader the first one's answer.
    for _round in range(4):
        assert doubled.answer(Leaf.result) == Decided(8)
        assert quadrupled.answer(Leaf.result) == Decided(16)
        assert isinstance(partial.answer(Leaf.result), Unresolved)


def test_diagnostics_are_identical_across_threads() -> None:
    left = _DiagRoot.start({_DiagRoot.size: 9}).left
    assessment = left.project(_Watched.view)

    def render(_index: int) -> tuple[str, ...]:
        return tuple(
            item.render() for item in left.diagnostics(assessment, projection=_Watched.view)
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rendered = set(pool.map(render, range(32)))
    assert len(rendered) == 1


def test_independent_roots_do_not_serialize_on_one_another() -> None:
    """Two lineages, two locks: holding one must not block the other."""

    first = Root.start({Root.size: 4})
    second = Root.start({Root.size: 8})
    with _lineage(first).lock:
        # A different lineage answers while this one's lock is held.
        assert second.answer(Root.size) == Decided(8)
    assert _lineage(first).lock is not _lineage(second).lock


def test_the_lineage_lock_is_reentrant_for_same_thread_callbacks() -> None:
    """Evaluator callbacks run under the lineage lock; re-entry must be safe.

    U1 serializes engine calls at the occurrence boundary because the engine's
    own caches are unsynchronized.  A contributor callback that queries back into
    the same occurrence on the same thread therefore re-enters, and the lock is
    re-entrant so that it does not deadlock.  A callback that hands work to
    another thread and waits for it *would* deadlock; that is the documented U7
    limitation, not something this slice changes.
    """

    root = Root.start({Root.size: 4})
    lineage = _lineage(root)
    with lineage.lock, lineage.lock:
        assert root.answer(Root.size) == Decided(4)


# -- C7/C8: the Human Gate 1 corrective round ---------------------------------


def test_the_construction_hook_must_return_a_fresh_instance() -> None:
    """A hook that recycles an occurrence would mutate the old one into the new."""

    class Recycled(Space):
        size = Problem(int)
        lanes = Decision(int, values=(1, 2))
        _cached: Space | None = None

        @classmethod
        def _new_occurrence(cls, context: OccurrenceContext) -> Space:
            del context
            if cls._cached is None:
                cls._cached = object.__new__(cls)
            return cls._cached

    root = Recycled.start({Recycled.size: 1})
    assert type(root) is Recycled
    assert root.size == 1

    with pytest.raises(AuthoringError, match="already attached"):
        root.assign(Recycled.lanes, 2)
    # The refusal happened before anything was written, so the original stands.
    assert root.answer(Recycled.size) == Decided(1)
    assert isinstance(root.answer(Recycled.lanes), Unresolved)


class _ReturnsParent(Space):
    """A child whose hook hands back the parent it was given."""

    supplied = Input(int)

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        return object.__new__(cls) if context.root is None else context.root


def test_the_construction_hook_may_not_hand_back_the_root_it_was_given() -> None:
    class ReturnsRoot(Space):
        size = Problem(int)
        inner = Subspace(_ReturnsParent, supplied=size)

    root = ReturnsRoot.start({ReturnsRoot.size: 4})
    # Refused, and refused before the root's own state could be overwritten:
    # the returned object is both the wrong class and already attached.
    with pytest.raises(AuthoringError):
        _ = root.inner
    assert root.answer(ReturnsRoot.size) == Decided(4)
    assert type(root) is ReturnsRoot


class _Falsey(Space):
    """A root that is falsey, which is an ordinary thing for a class to be."""

    size = Problem(int)
    inner = Subspace(Leaf, supplied=size)
    pick = Variant({"one": Subspace(Leaf, supplied=size)}, outputs=("result",))

    def __bool__(self) -> bool:
        return False


def test_a_falsey_root_is_still_the_root_of_its_children() -> None:
    root = _Falsey.start({_Falsey.size: 4})
    assert not root
    assert root.inner.root is root
    assert root.pick.alternative("one").root is root
    assert root.pick.root is root


class _ContextualChild(Space):
    """A context-bearing authored class placed as a child rather than a root."""

    supplied = Input(int)

    def __init__(self, label: str) -> None:
        self.label = label

    @classmethod
    def _new_occurrence(cls, context: OccurrenceContext) -> Space:
        instance = object.__new__(cls)
        parent = context.root
        instance.__init__(  # type: ignore[misc]
            ".".join(context.scope)
            if parent is None
            else f"{type(parent).__name__}/{context.namespace}"
        )
        return instance


def test_a_contextual_class_is_constructed_as_a_child_too() -> None:
    """Root and successor were already forced; the child placement was not.

    The allocation seam is what U1 proves.  The concrete channel a
    ``DataflowOp`` will need -- a ``NodeProto``, a ``ModelWrapper`` -- is
    deliberately not in ``OccurrenceContext`` yet and is U4 work.
    """

    class ContextualParent(Space):
        size = Problem(int)
        held = Subspace(_ContextualChild, supplied=size)

    root = ContextualParent.start({ContextualParent.size: 3})
    child = root.held
    assert type(child) is _ContextualChild
    # Constructed through the hook, which was told its real root and scope.
    assert child.label == "ContextualParent/root.held"
    assert child.root is root
    assert _occurrence_state(child).scope == ("root", "held")
    assert _occurrence_state(child).compiled.namespace == "root.held"
    assert child.answer(_ContextualChild.supplied) == Decided(3)


def test_a_declaration_may_not_shadow_a_private_protocol_name() -> None:
    with pytest.raises(AuthoringError) as raised:

        class Shadowing(Space):
            size = Problem(int)
            _space_value = Decision(int, values=(1, 2))

    message = str(raised.value)
    assert "Shadowing._space_value" in message
    assert "private Space protocol member" in message


@pytest.mark.parametrize("member", sorted(RESERVED_PROTOCOL_NAMES))
def test_every_reserved_protocol_name_is_refused(member: str) -> None:
    with pytest.raises(AuthoringError, match="private Space protocol member"):
        type(
            "ReservedProtocol",
            (Space,),
            {"size": Problem(int), member: Decision(int, values=(1, 2))},
        )


def test_overriding_a_protocol_member_itself_stays_legal() -> None:
    """The check prohibits a declaration under the name, not the override."""

    class Overriding(Space):
        size = Problem(int)

        @derived(int, size=size)
        def doubled(*, size: int) -> int:
            return size * 2

        @classmethod
        def _new_occurrence(cls, context: OccurrenceContext) -> Space:
            return object.__new__(cls)

    Overriding.exports = (Overriding.doubled,)
    assert Overriding.start({Overriding.size: 2}).doubled == 4


def test_a_codec_definition_is_validated_where_it_is_written() -> None:
    with pytest.raises(AuthoringError, match="non-empty stable string"):
        CanonicalValueCodec("", 1, str)
    with pytest.raises(AuthoringError, match="positive integer version"):
        CanonicalValueCodec("test.codec", 0, str)
    with pytest.raises(AuthoringError, match="positive integer version"):
        CanonicalValueCodec("test.codec", "1", str)  # type: ignore[arg-type]
    with pytest.raises(AuthoringError, match="callable encode"):
        CanonicalValueCodec("test.codec", 1, "not-callable")  # type: ignore[arg-type]


def test_a_codec_that_produces_an_unencodable_value_is_refused_by_name() -> None:
    """Not a bare ``TypeError`` from the JSON encoder naming neither party."""

    leaky: CanonicalValueCodec[_Opaque] = CanonicalValueCodec(
        "test.leaky",
        1,
        lambda value: object(),  # type: ignore[arg-type, return-value]
    )

    class Leaky(Space):
        thing = Problem(_Opaque, canonical=leaky)

    with pytest.raises(AuthoringError) as raised:
        Leaky.start({Leaky.thing: _Opaque("INT8")}).problem_fingerprint
    message = str(raised.value)
    assert "test.leaky" in message
    assert "Leaky.thing" in message
    assert "not a canonical value" in message


def test_a_codec_that_produces_a_non_finite_float_is_refused() -> None:
    infinite: CanonicalValueCodec[_Opaque] = CanonicalValueCodec(
        "test.infinite",
        1,
        lambda value: float("nan"),  # type: ignore[arg-type, return-value]
    )

    class Infinite(Space):
        thing = Problem(_Opaque, canonical=infinite)

    with pytest.raises(AuthoringError, match="non-finite float"):
        Infinite.start({Infinite.thing: _Opaque("INT8")}).problem_fingerprint


def test_a_codec_may_produce_ordinary_nested_canonical_values() -> None:
    structured: CanonicalValueCodec[_Opaque] = CanonicalValueCodec(
        "test.structured", 1, lambda value: {"name": value.name, "parts": [1, 2.5, None, True]}
    )

    class Structured(Space):
        thing = Problem(_Opaque, canonical=structured)

    first = Structured.start({Structured.thing: _Opaque("INT8")}).problem_fingerprint
    again = Structured.start({Structured.thing: _Opaque("INT8")}).problem_fingerprint
    other = Structured.start({Structured.thing: _Opaque("INT16")}).problem_fingerprint
    assert first == again != other


def test_diagnostics_take_their_projection_identity_from_the_assessment() -> None:
    left = _DiagRoot.start({_DiagRoot.size: 9}).left
    assessment = left.project(_Watched.view)
    derived_identity = left.diagnostics(assessment)
    assert derived_identity
    assert all(item.projection == assessment.projection for item in derived_identity)
    # Restating it is allowed and must agree.
    assert left.diagnostics(assessment, projection=_Watched.view) == derived_identity


def test_a_disagreeing_projection_argument_is_refused_rather_than_preferred() -> None:
    left = Root.start({Root.size: 6}).left.assign(Leaf.factor, 4)
    assessment = left.project(Leaf.data)
    with pytest.raises(AuthoringError, match="for an assessment of"):
        left.diagnostics(assessment, projection=Leaf.report)


def test_assess_takes_a_group_and_not_a_bare_constraint() -> None:
    """A named group is the shareable, nameable, diagnosable authoring unit."""

    left = Root.start({Root.size: 4}).left
    assert left.assess(Leaf.legal).verdict is None
    assert left.assess(Leaf.ready).ready is None
    with pytest.raises(AuthoringError, match="named ConstraintGroup"):
        left.assess(Leaf.below_limit)  # type: ignore[call-overload]


class _Wide(Space):
    supplied = Input(int)

    @derived(int, supplied=supplied)
    def result(*, supplied: int) -> int:
        return supplied + 1

    exports = (result,)


class _Narrow(Space):
    supplied = Input(int)
    offset = Decision(int, values=(0, 10))

    @derived(int, supplied=supplied, offset=offset)
    def result(*, supplied: int, offset: int) -> int:
        return supplied - offset

    exports = (result,)


class _Heterogeneous(Space):
    """One fixed Subspace beside a Variant over two unrelated Space classes."""

    size = Problem(int)
    fixed = Subspace(_Wide, supplied=size)
    implementation = Variant(
        {"wide": Subspace(_Wide, supplied=size), "narrow": Subspace(_Narrow, supplied=size)},
        outputs=("result",),
    )


def test_a_variant_holds_heterogeneous_subspaces_and_stays_inferrable() -> None:
    root = _Heterogeneous.start({_Heterogeneous.size: 5})

    # Class access is the declaration; instance access is the bound view.
    assert isinstance(_Heterogeneous.fixed, Subspace)
    assert isinstance(_Heterogeneous.implementation, Variant)
    assert isinstance(root.implementation, VariantView)

    fixed = root.fixed
    assert type(fixed) is _Wide
    assert fixed.answer(_Wide.result) == Decided(6)

    variant = root.implementation
    assert variant.alternatives == ("wide", "narrow")
    assert type(variant.alternative("wide")) is _Wide
    assert type(variant.alternative("narrow")) is _Narrow

    narrow = variant.select("narrow").alternative("narrow").assign(_Narrow.offset, 10)
    assert narrow.answer(_Narrow.result) == Decided(-5)
    assert narrow.root.answer(_Heterogeneous.implementation.result) == Decided(-5)


def test_a_conditional_subspace_keeps_its_own_when() -> None:
    """``when=`` survives on a direct Subspace; only Variant members refuse it."""

    class Conditional(Space):
        size = Problem(int)
        present = Decision(bool, values=(False, True))
        inner = Subspace(_Wide, supplied=size, when=present)

    root = Conditional.start({Conditional.size: 4})
    assert isinstance(root.inner.answer(_Wide.result), Unresolved)
    assert root.assign(Conditional.present, True).inner.answer(_Wide.result) == Decided(5)
    absent = root.assign(Conditional.present, False).inner.answer(_Wide.result)
    assert isinstance(absent, Absent)


def test_a_descriptor_needs_an_attached_occurrence() -> None:
    class Plain(Space):
        size = Problem(int)
        inner = Subspace(Leaf, supplied=size)
        pick = Variant({"one": Subspace(Leaf, supplied=size)}, outputs=("result",))

    plain = Plain()
    with pytest.raises(AttributeError, match="attached Space occurrence"):
        _ = plain.inner
    with pytest.raises(AttributeError, match="attached Space occurrence"):
        _ = plain.pick


def test_a_stable_compiled_name_is_independent_of_its_python_member() -> None:
    class Aliased(Space):
        size = Problem(int)
        choice = Variant({"one": Subspace(Leaf, supplied=size)}, outputs=("result",), name="branch")
        held = Subspace(Leaf, supplied=size, name="child")

    specification = compile_space(Aliased, "root", problem_namespace="problem.root")
    paths = tuple(item.path.value for item in specification.decisions)
    assert "root.branch.one.factor" in paths
    assert "root.child.factor" in paths

    root = Aliased.start({Aliased.size: 2})
    assert type(root.held) is Leaf
    assert root.choice.alternatives == ("one",)


# -- C9: the descriptor API's static contract ---------------------------------


def test_the_descriptor_api_types_are_pinned_statically() -> None:
    """The claims a runtime test cannot make.

    `pipeline.fixed` returns the right object whatever its annotations say, so
    the thing at risk is what a contributor's type checker sees: a descriptor
    overload that degrades to `Any`, or to the declaration, breaks nothing a
    runtime assertion would notice.  The fixture is a positive one -- strict
    mypy must accept it unchanged -- and it is checked in a subprocess for the
    same reason the existing negative fixture is: mypy is the oracle, not the
    interpreter running the suite.
    """

    mypy = shutil.which("mypy")
    if mypy is None:
        pytest.skip("mypy is not installed in this test environment")
    root = Path(__file__).parents[3]
    fixture = root / "tests" / "dataflow" / "typing" / "space_descriptor_types.py"
    environment = dict(os.environ)
    environment["MYPYPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests")))
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [mypy, "--no-incremental", "--strict", "--explicit-package-bases", str(fixture)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout or completed.stderr
    assert "Success" in completed.stdout
