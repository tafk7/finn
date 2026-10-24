# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The running type's evaluation of one design space: how each kind of node reads,
validity's pieces, and answers compared by type and ``==``.

Each test opens an evaluation of a root declaration's facts and stored choices, in a
session of its own (``core.space._evaluation_support``).
A choice reads its stored value through its type check over facts, or unassigned, or
(under a scope not in play) inapplicable; nothing is forced. Validity's pieces are the
constraints, present children, a present choice whose listed domain lists no value,
and a guard refused through a refused input.
"""

from __future__ import annotations

import pytest

from core.space._evaluation_support import answer, entry, node, open_evaluation, validity
from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    divisors_of,
    reject,
)
from finn.core.space._running.evaluate import _same_answer
from finn.core.space.results import ConstraintAssessment


def _codes(result: object) -> list[tuple[str, str]]:
    assert isinstance(result, (Inapplicable, Rejected, Unresolved))
    return [(finding.code, finding.owner) for finding in result.findings]


class Choices(Space):
    extent: int = Param()
    enabled: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=enabled)
    width: int = Decision(values=(8, 16))


def test_a_choice_reads_its_stored_value_unassigned_waiting_or_dormant() -> None:
    world = open_evaluation(Choices(extent=4))
    assert _codes(answer(world, "enabled")) == [("decision-unassigned", "enabled")]
    # Its presence waits on the open selector of its scope: unknown, never forced.
    assert _codes(answer(world, "lanes")) == [("decision-unassigned", "enabled")]

    dormant = open_evaluation(Choices(extent=4), {"enabled": False, "lanes": 2})
    assert isinstance(answer(dormant, "lanes"), Inapplicable)
    assert dormant.ds.stored[node(dormant, "lanes")] == 2  # kept, out of play

    present = open_evaluation(Choices(extent=4), {"enabled": True})
    assert _codes(answer(present, "lanes")) == [("decision-unassigned", "lanes")]

    stored = open_evaluation(Choices(extent=4), {"enabled": True, "lanes": 2})
    lanes = answer(stored, "lanes")
    assert lanes == Available(2)


class Facts(Space):
    extent: int = Param()

    @derived
    def checked(*, extent: int) -> int | Rejected:
        if extent < 2:
            return reject("extent-small", "the extent is below two")
        return extent

    lanes: int = Decision(domain=divisors_of(extent))
    banks: int = Decision(domain=divisors_of(checked))


def test_a_stored_value_reads_through_its_type_check_over_facts() -> None:
    outside = open_evaluation(Facts(extent=4), {"lanes": 3})
    assert _codes(answer(outside, "lanes")) == [("domain-membership", "lanes")]

    refused = open_evaluation(Facts(extent=1), {"banks": 1})
    assert _codes(answer(refused, "banks")) == [("extent-small", "checked")]


class Holder(Space):
    extent: int = Param()
    on: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=on)
    other: int = Decision(values=(1, 2))


@pytest.mark.parametrize(
    ("extent", "choices", "expected"),
    (
        (0, {"on": True}, [("domain-empty", "lanes")]),
        # Presence unknown: an empty list is an unknown piece, waiting on the selector.
        (0, {}, [("decision-unassigned", "on")]),
        (0, {"on": False}, None),
        # A non-empty list adds nothing, present or not; an open choice is no piece.
        (4, {}, None),
        (4, {"on": True}, None),
        # A stored value's type check is no piece.
        (4, {"on": True, "lanes": 3}, None),
    ),
)
def test_validity_counts_a_present_domain_that_lists_no_value(
    extent: int, choices: dict[str, object], expected: list[tuple[str, str]] | None
) -> None:
    found = validity(open_evaluation(Holder(extent=extent), choices))
    if expected is None:
        assert found.result == Available(True)
    else:
        assert _codes(found.result) == expected
    assert isinstance(found.assessment, ConstraintAssessment)
    assert "other" not in found.assessment.results


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))


class Guarded(Space):
    size: int = Param()

    @derived
    def big(*, size: int) -> bool | Rejected:
        if size < 0:
            return reject("size-negative", "a size is never negative")
        return size > 4

    @constraint(when=big)
    def limit(*, size: int) -> bool:
        return size < 100

    banks: int = Decision(values=(1, 2), when=big)
    leaf = Leaf(width=size, when=big)


def test_a_guard_refused_through_its_input_is_one_piece_per_guard() -> None:
    world = open_evaluation(Guarded(size=-1))
    assessment = validity(world).assessment
    assert isinstance(assessment, ConstraintAssessment)
    assert {key: sorted(_codes(result)) for key, result in assessment.results.items()} == {
        "banks": [("guard-refused", "banks"), ("size-negative", "big")],
        "leaf.$validity": [("guard-refused", "leaf"), ("size-negative", "big")],
        "limit": [("guard-refused", "limit"), ("size-negative", "big")],
    }
    # The scope is not in play (§ 16.8): its constraint does not apply, the guard's piece
    # standing for it; a value under it reads refused through the guard, no origin.
    assert isinstance(answer(world, "limit"), Inapplicable)
    assert _codes(answer(world, "banks")) == [("size-negative", "big")]
    assert not entry(world, "banks").own_refusal
    assert entry(world, "big").own_refusal

    assert validity(open_evaluation(Guarded(size=8))).result == Available(True)


class Parity(Space):
    count: int = Decision(values=(1, 2, 3, 4))
    other: int = Decision(values=(1, 2))

    @derived
    def odd(*, count: int) -> bool:
        return count % 2 == 1

    @derived
    def label(*, odd: bool) -> str:
        return "odd" if odd else "even"

    @derived
    def doubled(*, other: int) -> int:
        return 2 * other


def test_same_compares_by_type_and_eq_never_bare_equality() -> None:
    world = open_evaluation(Parity(), {"count": 1, "other": 1})
    odd = node(world, "odd")
    # ``True == 1`` in Python; the two are of different types.
    assert not _same_answer(odd, Available(True), Available(1))
    assert _same_answer(odd, Available(True), Available(True))
    refused = reject("x", "y", owner="z")
    assert _same_answer(odd, refused, reject("x", "y", owner="z"))
    assert not _same_answer(odd, refused, Unresolved(refused.findings))
