# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import pytest

from finn.core.space import (
    Available,
    Const,
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    constraint,
    derived,
    design_space,
    view,
)
from finn.core.space.errors import EvaluationError


@dataclass
class Bag:
    values: list[int]


BAG = ValueSemantics(
    Bag,
    "bag",
    lambda value: type(value) is Bag,
    lambda left, right: left.values == right.values,
    lambda value: Bag(list(value.values)),
)


class Mutable(Space):
    source: Bag = Param(semantics=BAG)
    choice: Bag = Decision(values=(Bag([3]),), semantics=BAG)

    @derived(semantics=BAG)
    def raw(*, source: Bag) -> Bag:
        return Bag(source.values)

    physical = View(raw)

    @view(semantics=BAG)
    def computed(*, source: Bag) -> Bag:
        return Bag(source.values)


def test_value_and_function_views_detach_all_public_assessment_payloads() -> None:
    original = Bag([1])
    point = design_space(Mutable(source=original))
    original.values.append(9)
    # Both view forms read as their accepted value; each read is a detached copy.
    point.physical.values.append(10)
    point.computed.values.append(10)
    assert (point.physical, point.computed) == (Bag([1]), Bag([1]))
    for declaration in (Mutable.physical, Mutable.computed):
        bound = point.field(declaration)
        bound.get().values.append(10)
        assert bound.get() == Bag([1])
        queried = point.query(declaration)
        assert isinstance(queried, Available)
        queried.value.values.append(11)
        assert point.query(declaration) == Available(Bag([1]))
        assessment = point.inspect(declaration)
        assert isinstance(assessment.output_result, Available)
        assert isinstance(assessment.accepted_result, Available)
        assessment.output_result.value.values.append(2)
        assessment.accepted_result.value.values.append(3)
        for answer in assessment.readiness.results.values():
            if isinstance(answer, Available) and isinstance(answer.value, Bag):
                answer.value.values.append(4)
        again = point.inspect(declaration)
        assert isinstance(again.output_result, Available)
        assert isinstance(again.accepted_result, Available)
        assert again.output_result.value == Bag([1])
        assert again.accepted_result.value == Bag([1])
        for answer in again.readiness.results.values():
            if isinstance(answer, Available) and isinstance(answer.value, Bag):
                assert answer.value == Bag([1])
    assert point.source == Bag([1])
    assert point.raw == Bag([1])


def test_decision_reads_and_candidates_cannot_mutate_frozen_commitments() -> None:
    base = design_space(Mutable(source=Bag([1])))
    candidates = base.field(Mutable.choice).candidates()
    assert isinstance(candidates, Available)
    candidates.value[0].values.append(9)
    point = base.with_choices(choice=Bag([3]))
    state = point.field(Mutable.choice).state
    assert isinstance(state, Available)
    assert state.value.value is not None
    state.value.value.values.append(8)
    answer = point.query(Mutable.choice)
    assert isinstance(answer, Available)
    answer.value.values.append(7)
    assert point.choice == Bag([3])
    assert point.with_choices(choice=Bag([3])) is point


def test_public_snapshot_failure_retains_declaration_role_and_cause() -> None:
    fail = False

    def snapshot(value: Bag) -> Bag:
        if fail:
            raise RuntimeError("snapshot unavailable")
        return Bag(list(value.values))

    semantics = ValueSemantics(Bag, "bag", lambda value: type(value) is Bag, BAG.equal, snapshot)

    class Failing(Space):
        source: Bag = Param(semantics=semantics)
        physical = View(source)

    point = design_space(Failing(source=Bag([1])))
    _ = point.physical  # a view read: warms the answer
    fail = True
    with pytest.raises(EvaluationError) as answer_error:
        point.query(Failing.source)
    assert answer_error.value.owner == "source"
    assert answer_error.value.role == "public value snapshot"
    assert isinstance(answer_error.value.__cause__, RuntimeError)
    with pytest.raises(EvaluationError) as assessment_error:
        point.inspect(Failing.physical)
    assert assessment_error.value.owner == "source"
    assert isinstance(assessment_error.value.__cause__, RuntimeError)


def test_grouped_view_obligations_keep_refusals_visible_while_waiting() -> None:
    class Grouped(Space):
        output = Const(4)
        lanes: int = Decision(values=(1, 2))

        @constraint
        def refused() -> bool:
            return False

        @constraint
        def pending(*, lanes: int) -> bool:
            return lanes > 0

        support = ConstraintGroup(refused, pending)
        physical = View(output, requires=(support,))

    point = design_space(Grouped())
    grouped = point.inspect(Grouped.support)
    for _ in range(2):
        assessment = point.inspect(Grouped.physical)
        assert isinstance(assessment.accepted_result, Unresolved)
        assert assessment.constraints.refused == ("refused",)
        assert assessment.constraints.results == grouped.results
        refused = assessment.constraints.results["refused"]
        assert isinstance(refused, Rejected)
        assert refused.findings[0].owner == "refused"
        assert isinstance(assessment.constraints.results["pending"], Unresolved)
        readiness = assessment.readiness
        assert readiness.ready is None
        assert isinstance(readiness.results["refused"], Rejected)
    committed = point.with_choices(lanes=1)
    assert isinstance(committed.inspect(Grouped.physical).accepted_result, Rejected)
    assert committed.inspect(Grouped.physical).readiness.ready is True


def test_empty_named_obligations_can_be_assessed_without_value_semantics() -> None:
    class Empty(Space):
        output = Const(4)
        group = ConstraintGroup()
        physical = View(output, requires=(group,))

    point = design_space(Empty())
    assert point.physical == 4
    assert point.inspect(Empty.physical).readiness.ready is True
