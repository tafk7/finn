# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import pytest

from finn.core.space import (
    Const,
    ConstraintGroup,
    Available,
    Decision,
    Param,
    Readiness,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    View,
    compile_space,
    constraint,
    derived,
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
    source = Param(BAG)
    choice = Decision(BAG, values=(Bag([3]),))

    @derived(semantics=BAG)
    def raw(*, source: Bag) -> Bag:
        return Bag(source.values)

    physical = View(raw, requires=(source,))
    ready = Readiness(raw)

    @view(semantics=BAG, requires=(source,))
    def computed(*, source: Bag) -> Bag:
        return Bag(source.values)


def test_value_and_function_views_detach_all_public_assessment_payloads() -> None:
    original = Bag([1])
    point = compile_space(Mutable).bind({Mutable.source: original})
    original.values.append(9)
    for declaration in (Mutable.physical, Mutable.computed):
        assessment = point.assess(declaration)
        assert isinstance(assessment.output_result, Available)
        assert isinstance(assessment.accepted_result, Available)
        assessment.output_result.value.values.append(2)
        assessment.accepted_result.value.values.append(3)
        for answer in assessment.readiness.results.values():
            if isinstance(answer, Available) and isinstance(answer.value, Bag):
                answer.value.values.append(4)
        again = point.assess(declaration)
        assert isinstance(again.output_result, Available)
        assert isinstance(again.accepted_result, Available)
        assert again.output_result.value == Bag([1])
        assert again.accepted_result.value == Bag([1])
        for answer in again.readiness.results.values():
            if isinstance(answer, Available) and isinstance(answer.value, Bag):
                assert answer.value == Bag([1])
    assert point.source == Bag([1])
    assert point.raw == Bag([1])


def test_standalone_readiness_returns_detached_required_values() -> None:
    point = Mutable({Mutable.source: Bag([1])})
    readiness = point.assess(Mutable.ready)
    raw = readiness.results["raw"]
    assert isinstance(raw, Available)
    assert isinstance(raw.value, Bag)
    raw.value.values.append(2)
    again = point.assess(Mutable.ready).results["raw"]
    assert isinstance(again, Available)
    assert again.value == Bag([1])


def test_decision_reads_and_candidates_cannot_mutate_frozen_commitments() -> None:
    base = Mutable({Mutable.source: Bag([1])})
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
        source = Param(semantics)
        physical = View(source)

    point = Failing({Failing.source: Bag([1])})
    point.physical()
    fail = True
    with pytest.raises(EvaluationError) as answer_error:
        point.query(Failing.source)
    assert answer_error.value.owner == "source"
    assert answer_error.value.role == "public value snapshot"
    assert isinstance(answer_error.value.__cause__, RuntimeError)
    with pytest.raises(EvaluationError) as assessment_error:
        point.physical()
    assert assessment_error.value.owner == "source"
    assert isinstance(assessment_error.value.__cause__, RuntimeError)


def test_grouped_view_obligations_keep_refusals_visible_while_waiting() -> None:
    class Grouped(Space):
        output = Const(4)
        lanes = Decision(int, values=(1, 2))

        @constraint
        def refused() -> bool:
            return False

        @constraint
        def pending(*, lanes: int) -> bool:
            return lanes > 0

        support = ConstraintGroup(refused, pending)
        ready = Readiness(support)
        physical = View(output, constraints=(support,))
        ready_view = View(output, requires=(ready,))

    point = Grouped()
    grouped = point.assess(Grouped.support)
    for _ in range(2):
        assessment = point.physical()
        assert isinstance(assessment.accepted_result, Unresolved)
        assert assessment.constraints.refused == ("refused",)
        assert assessment.constraints.results == grouped.results
        refused = assessment.constraints.results["refused"]
        assert isinstance(refused, Rejected)
        assert refused.findings[0].owner == "refused"
        assert isinstance(assessment.constraints.results["pending"], Unresolved)
        readiness = point.assess(Grouped.ready)
        assert readiness.ready is None
        assert isinstance(readiness.results["refused"], Rejected)
        via_readiness = point.ready_view()
        assert isinstance(via_readiness.accepted_result, Unresolved)
        assert isinstance(via_readiness.readiness.results["refused"], Rejected)
    committed = point.with_choices(lanes=1)
    assert isinstance(committed.physical().accepted_result, Rejected)
    assert committed.assess(Grouped.ready).ready is True


def test_empty_named_obligations_can_be_assessed_without_value_semantics() -> None:
    class Empty(Space):
        output = Const(4)
        group = ConstraintGroup()
        ready = Readiness()
        physical = View(output, requires=(group, ready))

    point = Empty()
    assert point.physical().accepted_result == Available(4)
    assert point.assess(Empty.ready).ready is True
