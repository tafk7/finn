# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import pytest

from finn.kernels.space._next import (
    Const,
    ConstraintGroup,
    Decided,
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
from finn.kernels.space._next.errors import EvaluationError


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
    point = compile_space(Mutable).start({Mutable.source: original})
    original.values.append(9)
    for declaration in (Mutable.physical, Mutable.computed):
        assessment = point.assess(declaration)
        assert isinstance(assessment.output_answer, Decided)
        assert isinstance(assessment.accepted_answer, Decided)
        assessment.output_answer.value.values.append(2)
        assessment.accepted_answer.value.values.append(3)
        for answer in assessment.readiness.answers.values():
            if isinstance(answer, Decided) and isinstance(answer.value, Bag):
                answer.value.values.append(4)
        again = point.assess(declaration)
        assert isinstance(again.output_answer, Decided)
        assert isinstance(again.accepted_answer, Decided)
        assert again.output_answer.value == Bag([1])
        assert again.accepted_answer.value == Bag([1])
        for answer in again.readiness.answers.values():
            if isinstance(answer, Decided) and isinstance(answer.value, Bag):
                assert answer.value == Bag([1])
    assert point.source == Bag([1])
    assert point.raw == Bag([1])


def test_standalone_readiness_returns_detached_required_values() -> None:
    point = Mutable.start({Mutable.source: Bag([1])})
    readiness = point.assess(Mutable.ready)
    raw = readiness.answers["raw"]
    assert isinstance(raw, Decided)
    assert isinstance(raw.value, Bag)
    raw.value.values.append(2)
    again = point.assess(Mutable.ready).answers["raw"]
    assert isinstance(again, Decided)
    assert again.value == Bag([1])


def test_decision_reads_and_candidates_cannot_mutate_frozen_commitments() -> None:
    base = Mutable.start({Mutable.source: Bag([1])})
    candidates = base.candidates(Mutable.choice)
    assert isinstance(candidates, Decided)
    candidates.value[0].values.append(9)
    point = base.assign(Mutable.choice, Bag([3]))
    state = point.decision_state(Mutable.choice)
    assert isinstance(state, Decided)
    assert state.value.value is not None
    state.value.value.values.append(8)
    answer = point.answer(Mutable.choice)
    assert isinstance(answer, Decided)
    answer.value.values.append(7)
    assert point.choice == Bag([3])
    assert point.assign(Mutable.choice, Bag([3])) is point


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

    point = Failing.start({Failing.source: Bag([1])})
    point.physical()
    fail = True
    with pytest.raises(EvaluationError) as answer_error:
        point.answer(Failing.source)
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

    point = Grouped.start()
    grouped = point.assess(Grouped.support)
    for _ in range(2):
        assessment = point.physical()
        assert isinstance(assessment.accepted_answer, Unresolved)
        assert assessment.constraints.refused == ("refused",)
        assert assessment.constraints.answers == grouped.answers
        refused = assessment.constraints.answers["refused"]
        assert isinstance(refused, Rejected)
        assert refused.findings[0].owner == "refused"
        assert isinstance(assessment.constraints.answers["pending"], Unresolved)
        readiness = point.assess(Grouped.ready)
        assert readiness.ready is None
        assert isinstance(readiness.answers["refused"], Rejected)
        via_readiness = point.ready_view()
        assert isinstance(via_readiness.accepted_answer, Unresolved)
        assert isinstance(via_readiness.readiness.answers["refused"], Rejected)
    committed = point.assign(Grouped.lanes, 1)
    assert isinstance(committed.physical().accepted_answer, Rejected)
    assert committed.assess(Grouped.ready).ready is True
