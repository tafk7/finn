# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.model import (
    Case,
    ConstraintGroup,
    Decision,
    Input,
    OccurrenceError,
    OneOf,
    Problem,
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
