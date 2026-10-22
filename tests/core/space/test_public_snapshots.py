# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import pytest

from finn.core.space import (
    Available,
    Const,
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
    inspection,
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


def test_choice_states_and_possible_values_hand_out_detached_values() -> None:
    """``inspection.states`` hands out each stored value, and ``inspection.viable`` each
    listed value, detached by its choice's semantics: a reader changing one changes no
    stored choice, and no listed value another design space of the same facts reads."""
    base = design_space(Mutable(source=Bag([1])))
    (possible,) = inspection.viable(base, [Mutable.choice])
    assert possible.values is not None
    possible.values[0].value.values.append(9)
    point = base.with_choices(choice=Bag([3]))
    held = dict(inspection.states(point))["choice"]
    assert isinstance(held, inspection.Set)
    held.value.values.append(8)
    assert dict(inspection.states(point))["choice"] == inspection.Set(Bag([3]))
    assert point.choice == Bag([3])
    (again,) = inspection.viable(base, [Mutable.choice])
    assert again.values is not None and again.values[0].value == Bag([3])
    assert base.field(Mutable.choice).candidates() == Available((Bag([3]),))


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


def test_a_view_of_a_refused_space_refuses_and_keeps_what_waits_visible() -> None:
    class Grouped(Space):
        output = Const(4)
        lanes: int = Decision(values=(1, 2))

        @constraint
        def refused() -> bool:
            return False

        @constraint
        def pending(*, lanes: int) -> bool:
            return lanes > 0

        physical = View(output)

    point = design_space(Grouped())
    for _ in range(2):
        assessment = point.inspect(Grouped.physical)
        assert assessment.output_result == Available(4)
        assert isinstance(assessment.accepted_result, Rejected)
        assert isinstance(assessment.validity, Rejected)
        assert [finding.owner for finding in assessment.validity.findings] == ["refused"]
        assert assessment.accepted_result == assessment.validity
        # Readiness, a diagnostic, names what the refused Space still waits on.
        readiness = assessment.readiness
        assert readiness.ready is None
        assert isinstance(readiness.results["$validity"], Rejected)
        assert isinstance(readiness.results["pending"], Unresolved)
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["refused"]
    committed = point.with_choices(lanes=1)
    assert isinstance(committed.inspect(Grouped.physical).accepted_result, Rejected)
    assert committed.inspect(Grouped.physical).readiness.ready is True


def test_a_view_of_a_space_with_no_pieces_reads_its_output() -> None:
    class Empty(Space):
        output = Const(4)
        physical = View(output)

    point = design_space(Empty())
    assert point.physical == 4
    assert point.inspect(Empty.physical).readiness.ready is True
    assert inspection.validity(point) == inspection.Validity("valid", (), Available(True))
