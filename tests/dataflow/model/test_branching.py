# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""KD1: exclusive branching is a generic ``Space`` composition capability."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pytest

from finn.dataflow._engine import (
    Absent,
    Decided,
    Engine,
    QualifiedPath,
    Unresolved,
)
from finn.dataflow.model.compiler import compile_space, compile_space_model
from finn.dataflow.model.declarations import (
    AuthoringError,
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


class Scaled(Space):
    """A case that owns one local decision."""

    scale = Input(int)
    pipeline = Decision(int, values=(1, 2))

    @derived(int, scale=scale, pipeline=pipeline)
    def result(*, scale: int, pipeline: int) -> int:
        return scale * pipeline

    @constraint(pipeline=pipeline, scale=scale)
    def deep_enough(*, pipeline: int, scale: int) -> bool:
        return pipeline * scale >= 2

    ready = Readiness(decisions=(pipeline,), name="ready")
    exports = (result,)


class Offset(Space):
    """A case with a different Input vocabulary and no local decision."""

    base = Input(int)

    @derived(int, base=base)
    def result(*, base: int) -> int:
        return base + 100

    @constraint(base=base)
    def never(*, base: int) -> object:
        return reject("offset-refuses", "this case always refuses", values={"base": base})

    exports = (result,)


class Mismatched(Space):
    value = Input(int)

    @derived(bool, value=value)
    def result(*, value: int) -> bool:
        return value > 0

    exports = (result,)


class Silent(Space):
    value = Input(int)

    @derived(int, value=value)
    def other(*, value: int) -> int:
        return value

    exports = (other,)


class Alternatives(Space):
    size = Problem(int)
    implementation = OneOf(
        Case(Scaled, name="scaled", scale=size),
        Case(Offset, name="offset", base=size),
        outputs=("result",),
    )

    @derived(int, selected=implementation.result)
    def observed(*, selected: int) -> int:
        return selected


class Singleton(Space):
    size = Problem(int)
    implementation = OneOf(
        Case(Scaled, name="scaled", scale=size),
        outputs=("result",),
    )

    @derived(int, selected=implementation.result)
    def observed(*, selected: int) -> int:
        return selected


def _started(space_type: type[Space], size: int = 4) -> tuple[Engine, object]:
    engine = Engine()
    specification = compile_space(space_type, "root", problem_namespace="problem.root")
    point = engine.start(engine.validate(specification), {"problem.root.size": size})
    return engine, point


# -- lowering -----------------------------------------------------------------


def test_multi_case_branch_lowers_to_one_selector_and_stable_namespaces() -> None:
    specification = compile_space(Alternatives, "root", problem_namespace="problem.root")
    assert [str(item.path) for item in specification.decisions] == [
        "root.implementation.case",
        "root.implementation.scaled.pipeline",
    ]
    assert [str(item.path) for item in specification.properties] == [
        "semantic.root.implementation.result",
        "semantic.root.observed",
        "semantic.root.implementation.scaled.result",
        "semantic.root.implementation.offset.result",
    ]
    assert [str(item.path) for item in specification.constraints] == [
        "constraint.root.implementation.scaled.deep_enough",
        "constraint.root.implementation.offset.never",
    ]


def test_selector_domain_is_the_finite_ordered_case_id_list() -> None:
    engine, point = _started(Alternatives)
    candidates = engine.enumerate_candidates(point, "root.implementation.case")
    assert isinstance(candidates, Decided)
    assert candidates.value == ("scaled", "offset")


def test_a_singleton_adds_no_selector_but_keeps_the_selected_output_path() -> None:
    specification = compile_space(Singleton, "root", problem_namespace="problem.root")
    assert [str(item.path) for item in specification.decisions] == [
        "root.implementation.scaled.pipeline"
    ]
    assert "semantic.root.implementation.result" in {
        str(item.path) for item in specification.properties
    }
    engine, point = _started(Singleton)
    point = engine.commit_assignments(point, {"root.implementation.scaled.pipeline": 2}).point
    assert engine.query_property(point, "semantic.root.observed") == Decided(8)


def test_the_selected_output_forwards_the_exact_case_value() -> None:
    engine, point = _started(Alternatives)
    chosen = engine.commit_assignments(
        point,
        {"root.implementation.case": "scaled", "root.implementation.scaled.pipeline": 2},
    ).point
    assert engine.query_property(chosen, "semantic.root.observed") == Decided(8)
    other = engine.commit_assignments(point, {"root.implementation.case": "offset"}).point
    assert engine.query_property(other, "semantic.root.observed") == Decided(104)


def test_bindings_come_from_problem_decision_property_and_parent_input() -> None:
    class Supplier(Space):
        size = Problem(int)
        doubled = Decision(int, values=(1, 2))

        @derived(int, size=size, doubled=doubled)
        def scaled(*, size: int, doubled: int) -> int:
            return size * doubled

    class Inner(Space):
        supplied = Input(int)
        branch = OneOf(Case(Scaled, name="scaled", scale=supplied), outputs=("result",))

    class Outer(Space):
        size = Problem(int)
        doubled = Decision(int, values=(1, 2))

        @derived(int, size=size, doubled=doubled)
        def scaled(*, size: int, doubled: int) -> int:
            return size * doubled

        from_problem = OneOf(Case(Scaled, name="a", scale=size), outputs=("result",))
        from_decision = OneOf(Case(Scaled, name="b", scale=doubled), outputs=("result",))
        from_property = OneOf(Case(Scaled, name="c", scale=scaled), outputs=("result",))

    del Supplier, Inner
    specification = compile_space(Outer, "root", problem_namespace="problem.root")
    engine = Engine()
    point = engine.start(engine.validate(specification), {"problem.root.size": 3})
    point = engine.commit_assignments(
        point,
        {
            "root.doubled": 2,
            "root.from_problem.a.pipeline": 1,
            "root.from_decision.b.pipeline": 1,
            "root.from_property.c.pipeline": 1,
        },
    ).point
    assert engine.query_property(point, "semantic.root.from_problem.result") == Decided(3)
    assert engine.query_property(point, "semantic.root.from_decision.result") == Decided(2)
    assert engine.query_property(point, "semantic.root.from_property.result") == Decided(6)


def test_a_parent_input_may_be_bound_into_a_nested_case() -> None:
    class Middle(Space):
        supplied = Input(int)
        branch = OneOf(Case(Scaled, name="scaled", scale=supplied), outputs=("result",))
        result = branch.result
        exports = (result,)

    class Root(Space):
        size = Problem(int)
        middle = OneOf(Case(Middle, name="middle", supplied=size), outputs=("result",))

    specification = compile_space(Root, "root", problem_namespace="problem.root")
    engine = Engine()
    point = engine.start(engine.validate(specification), {"problem.root.size": 5})
    point = engine.commit_assignments(point, {"root.middle.middle.branch.scaled.pipeline": 2}).point
    assert engine.query_property(point, "semantic.root.middle.result") == Decided(10)


# -- inactive cases -----------------------------------------------------------


def test_an_inactive_case_reduces_to_non_rejecting_absence() -> None:
    engine, point = _started(Alternatives)
    point = engine.commit_assignments(point, {"root.implementation.case": "offset"}).point
    absent = engine.query_property(point, "semantic.root.implementation.scaled.result")
    assert isinstance(absent, Absent)
    assert not absent.is_rejection
    assessment = engine.evaluate_constraints(
        point, ("constraint.root.implementation.scaled.deep_enough",)
    )
    assert assessment.verdict is not False


def test_an_inactive_case_decision_does_not_block_a_readiness_profile() -> None:
    engine, point = _started(Alternatives)
    point = engine.commit_assignments(point, {"root.implementation.case": "offset"}).point
    readiness = engine.check_readiness(point, "root.implementation.scaled.ready")
    assert readiness.ready is True


def test_a_rejecting_absence_in_the_selected_case_refuses_feasibility() -> None:
    engine, point = _started(Alternatives)
    point = engine.commit_assignments(point, {"root.implementation.case": "offset"}).point
    assessment = engine.evaluate_constraints(
        point, ("constraint.root.implementation.offset.never",)
    )
    assert assessment.verdict is False


def test_an_unresolved_selected_dependency_keeps_the_branch_unresolved() -> None:
    engine, point = _started(Alternatives)
    point = engine.commit_assignments(point, {"root.implementation.case": "scaled"}).point
    assert isinstance(engine.query_property(point, "semantic.root.observed"), Unresolved)


def test_an_uncommitted_selector_keeps_the_selected_output_unresolved() -> None:
    engine, point = _started(Alternatives)
    assert isinstance(
        engine.query_property(point, "semantic.root.implementation.result"), Unresolved
    )


def test_a_branch_condition_gates_the_whole_branch() -> None:
    class Conditional(Space):
        size = Problem(int)

        @derived(bool, size=size)
        def large(*, size: int) -> bool:
            return size > 10

        implementation = OneOf(
            Case(Scaled, name="scaled", scale=size),
            Case(Offset, name="offset", base=size),
            outputs=("result",),
            when=large,
        )

    specification = compile_space(Conditional, "root", problem_namespace="problem.root")
    engine = Engine()
    point = engine.start(engine.validate(specification), {"problem.root.size": 4})
    selected = engine.query_property(point, "semantic.root.implementation.result")
    assert isinstance(selected, Absent) and not selected.is_rejection
    state = engine.decision_state(point, "root.implementation.case")
    assert isinstance(state, Absent)


# -- nesting and reuse --------------------------------------------------------


class Nested(Space):
    supplied = Input(int)
    inner = OneOf(
        Case(Scaled, name="scaled", scale=supplied),
        Case(Offset, name="offset", base=supplied),
        outputs=("result",),
    )

    @derived(int, inner=inner.result)
    def result(*, inner: int) -> int:
        return inner

    exports = (result,)


class Outermost(Space):
    size = Problem(int)
    outer = OneOf(
        Case(Nested, name="nested", supplied=size),
        Case(Offset, name="offset", base=size),
        outputs=("result",),
    )


def test_a_nested_branch_gets_its_own_selector_beneath_its_case() -> None:
    specification = compile_space(Outermost, "root", problem_namespace="problem.root")
    assert [str(item.path) for item in specification.decisions] == [
        "root.outer.case",
        "root.outer.nested.inner.case",
        "root.outer.nested.inner.scaled.pipeline",
    ]
    engine = Engine()
    point = engine.start(engine.validate(specification), {"problem.root.size": 4})
    point = engine.commit_assignments(
        point,
        {
            "root.outer.case": "nested",
            "root.outer.nested.inner.case": "scaled",
            "root.outer.nested.inner.scaled.pipeline": 2,
        },
    ).point
    assert engine.query_property(point, "semantic.root.outer.result") == Decided(8)


def test_one_case_class_may_be_used_by_several_branches_without_mutation() -> None:
    class Twice(Space):
        size = Problem(int)
        left = OneOf(Case(Scaled, name="scaled", scale=size), outputs=("result",))
        right = OneOf(Case(Scaled, name="scaled", scale=size), outputs=("result",))

    specification = compile_space(Twice, "root", problem_namespace="problem.root")
    assert [str(item.path) for item in specification.decisions] == [
        "root.left.scaled.pipeline",
        "root.right.scaled.pipeline",
    ]
    engine = Engine()
    point = engine.start(engine.validate(specification), {"problem.root.size": 3})
    point = engine.commit_assignments(
        point, {"root.left.scaled.pipeline": 1, "root.right.scaled.pipeline": 2}
    ).point
    assert engine.query_property(point, "semantic.root.left.result") == Decided(3)
    assert engine.query_property(point, "semantic.root.right.result") == Decided(6)


def test_concurrent_compilation_of_one_class_is_deterministic() -> None:
    def compile_once(_index: int) -> tuple[str, ...]:
        specification = compile_space(Alternatives, "root", problem_namespace="problem.root")
        return tuple(str(item.path) for item in specification.decisions)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(compile_once, range(16)))
    assert len(set(results)) == 1


# -- authoring refusals -------------------------------------------------------


def test_a_generic_case_needs_an_explicit_stable_name() -> None:
    class Unnamed(Space):
        size = Problem(int)
        implementation = OneOf(Case(Scaled, scale=size), outputs=("result",))

    with pytest.raises(AuthoringError, match="explicit name="):
        compile_space(Unnamed, "root", problem_namespace="problem.root")


def test_case_names_are_non_empty_and_unique() -> None:
    with pytest.raises(AuthoringError, match="name must be non-empty"):
        Case(Scaled, name="", scale=Input(int))

    class Duplicated(Space):
        size = Problem(int)
        implementation = OneOf(
            Case(Scaled, name="same", scale=size),
            Case(Offset, name="same", base=size),
            outputs=("result",),
        )

    with pytest.raises(AuthoringError, match="case id 'same' twice"):
        compile_space(Duplicated, "root", problem_namespace="problem.root")


def test_a_branch_needs_at_least_one_case() -> None:
    with pytest.raises(AuthoringError, match="at least one Case"):
        OneOf(outputs=("result",))


def test_a_case_binding_must_be_exact_and_semantics_compatible() -> None:
    class Missing(Space):
        size = Problem(int)
        implementation = OneOf(Case(Scaled, name="scaled"), outputs=("result",))

    with pytest.raises(AuthoringError, match="Input binding is not exact"):
        compile_space(Missing, "root", problem_namespace="problem.root")

    class WrongType(Space):
        size = Problem(int)

        @derived(bool, size=size)
        def flag(*, size: int) -> bool:
            return size > 0

        implementation = OneOf(Case(Scaled, name="scaled", scale=flag), outputs=("result",))

    with pytest.raises(AuthoringError, match="expects int"):
        compile_space(WrongType, "root", problem_namespace="problem.root")


def test_every_case_must_expose_each_selected_output_compatibly() -> None:
    class Absent_(Space):
        size = Problem(int)
        implementation = OneOf(
            Case(Scaled, name="scaled", scale=size),
            Case(Silent, name="silent", value=size),
            outputs=("result",),
        )

    with pytest.raises(AuthoringError, match="does not export"):
        compile_space(Absent_, "root", problem_namespace="problem.root")

    class Incompatible(Space):
        size = Problem(int)
        implementation = OneOf(
            Case(Scaled, name="scaled", scale=size),
            Case(Mismatched, name="mismatched", value=size),
            outputs=("result",),
        )

    with pytest.raises(AuthoringError, match="changes value semantics"):
        compile_space(Incompatible, "root", problem_namespace="problem.root")


def test_an_undeclared_selected_output_is_not_reachable() -> None:
    branch = OneOf(Case(Scaled, name="scaled", scale=Input(int)), outputs=("result",))
    with pytest.raises(AttributeError, match="does not select an output"):
        branch.other


def test_a_case_must_be_a_space_subclass() -> None:
    with pytest.raises(AuthoringError, match="requires a Space subclass"):
        Case(int)  # type: ignore[type-var]


def test_a_case_may_not_declare_a_problem() -> None:
    class OwnsProblem(Space):
        local = Problem(int)

        @derived(int, local=local)
        def result(*, local: int) -> int:
            return local

        exports = (result,)

    class Root(Space):
        size = Problem(int)
        implementation = OneOf(Case(OwnsProblem, name="owns"), outputs=("result",))

    with pytest.raises(AuthoringError, match="declares a Problem inside a reusable child Space"):
        compile_space(Root, "root", problem_namespace="problem.root")


def test_a_recursive_case_cycle_is_rejected_deterministically() -> None:
    class Recursive(Space):
        supplied = Input(int)

    Recursive.branch = OneOf(  # type: ignore[attr-defined]
        Case(Recursive, name="again", supplied=Recursive.supplied), outputs=()
    )

    class Root(Space):
        size = Problem(int)
        implementation = OneOf(Case(Recursive, name="recursive", supplied=size))

    with pytest.raises(AuthoringError, match="cycle"):
        compile_space(Root, "root", problem_namespace="problem.root")


def test_branching_introduces_no_nested_engine_or_point() -> None:
    model = compile_space_model(Outermost, "root", problem_namespace="problem.root")
    carried = [
        getattr(item, field.name)
        for item in (
            *model.branches.branches,
            *(case for branch in model.branches.branches for case in branch.cases),
        )
        for field in fields(item)
    ]
    assert carried
    for value in carried:
        assert not isinstance(value, Engine)
        assert not hasattr(value, "design_space")
        assert not callable(value)
    for branch in model.branches.branches:
        assert isinstance(branch.namespace, str)
        assert branch.selector is None or isinstance(branch.selector, QualifiedPath)
