# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U1: the occurrence lifecycle and the validated projection over one point.

Every Space below is synthetic.  That is the point of the phase: the claim is
about a generic lifecycle, and justifying it against MVAU would prove only that
the layer fits the one shape it was written beside.  Nothing here names a
Kernel, a Region, a Design, or a Network.

This module covers the occurrence lifecycle itself:

```text
one root, two occurrences of one child class      Stage under `pair`
nested Use and OneOf                              Pair.inner inside Root.pair
child assignment ambiguity rejection              §ambiguity
partial then complete successor occurrence        §specialization
no raw runtime through public methods             §capability
callback capability audit in a subprocess         §capability
```
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import cast

import pytest

from finn.dataflow._engine import (
    Decided,
    DesignPoint,
    DesignSpace,
    Engine,
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
    Readiness,
    Space,
    Use,
    constraint,
    derived,
    reject,
)
from finn.dataflow.model.occurrence import Occurrence

# -- synthetic spaces ---------------------------------------------------------


class Stage(Space):
    """A leaf placed five times over, so no view can guess which one is meant."""

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

    #: A conditional placement, so a value can be *finally inapplicable* rather
    #: than merely unresolved.
    optional = Use(Stage, when=supplied, width=width)
    optional_value = optional.throughput

    chosen_ready = Readiness(properties=(selected,), name="chosen_ready")


def model() -> SpaceModel:
    return compile_space_model(Root, "root", problem_namespace="problem.root")


def start(*, width: int = 4, supplied: bool = True) -> Occurrence:
    return model().start({"problem.root.width": width, "problem.root.supplied": supplied})


def first_stage(occurrence: Occurrence) -> Occurrence:
    return occurrence.child(Root.pair).child(Pair.first)


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
    root.answer(Root.selected)
    first_stage(root).assess(Stage.ready)
    first_stage(root).assess(Stage.model_checks)
    assert isinstance(first_stage(root).answer(Stage.lanes), Unresolved)


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
        "namespace",
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
        stage.root.branch(Root.choice),
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
