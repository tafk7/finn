# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``allow_absent()``: absence tolerance as a property of the reader.

``Input(allow_absent=True)`` already says "this child tolerates an absent
value".  What had no spelling was the same statement made by a *dependency*: a
Kernel constraint that must report "an inactive role contributes no node" while
everything inside the active role still requires that node.  Without it such a
constraint is unwritable -- the absent dependency propagates and the constraint
answers ``Absent`` instead of ``True``.
"""

from __future__ import annotations

import pytest

from finn.kernels._engine import ABSENT, Absent, Decided
from finn.kernels.space import (
    AuthoringError,
    ConstraintGroup,
    Input,
    Problem,
    Space,
    Subspace,
    allow_absent,
    allow_inapplicable,
    constraint,
    derived,
)
from finn.kernels.space.declarations import reject


class Conditional(Space):
    """One value that may legitimately not arise, and two readers of it."""

    present = Problem(bool)

    @derived(bool, present=present)
    def maybe(*, present: bool) -> object:
        if not present:
            return reject("not-applicable", "this value does not arise here")
        return True

    @derived(str, value=maybe)
    def strict(*, value: bool) -> str:
        return f"strict:{value}"

    @derived(str, value=allow_absent(maybe))
    def tolerant(*, value: object) -> str:
        return "tolerant:absent" if value is ABSENT else f"tolerant:{value}"

    @constraint(value=allow_absent(maybe))
    def tolerated(*, value: object) -> bool:
        return True

    checks = ConstraintGroup(tolerated)


def test_a_strict_reader_propagates_the_absence():
    space = Conditional.start({Conditional.present: False})

    assert isinstance(space.answer(Conditional.strict), Absent)


def test_a_tolerant_reader_is_handed_the_absent_sentinel():
    space = Conditional.start({Conditional.present: False})

    assert space.answer(Conditional.tolerant) == Decided("tolerant:absent")


def test_tolerance_changes_nothing_when_the_value_is_present():
    space = Conditional.start({Conditional.present: True})

    assert space.answer(Conditional.strict) == Decided("strict:True")
    assert space.answer(Conditional.tolerant) == Decided("tolerant:True")


def test_a_tolerant_constraint_still_answers_over_an_absent_dependency():
    """The case the marker exists for: a refusal must be a *verdict*, not silence."""

    space = Conditional.start({Conditional.present: False})

    assessment = space.assess(Conditional.checks)

    assert assessment.verdict is True
    assert assessment.refused == ()


def test_the_marker_is_not_a_declaration():
    # Python wraps whatever ``__set_name__`` raises, so the refusal is checked
    # where it is actually delivered rather than where it was raised.
    with pytest.raises(RuntimeError) as raised:

        class Wrong(Space):
            present = Problem(bool)
            leaked = allow_absent(present)

    cause = raised.value.__cause__
    assert isinstance(cause, AuthoringError)
    assert "declares nothing" in str(cause)


def test_the_marker_refuses_a_non_declaration():
    with pytest.raises(AuthoringError, match="one value declaration"):
        allow_absent(object())  # type: ignore[arg-type]


def test_the_marker_refuses_to_stack():
    with pytest.raises(AuthoringError, match="already applied"):
        allow_absent(allow_absent(Conditional.present))


class InapplicabilityOnly(Space):
    present = Problem(bool)

    @derived(bool, present=present)
    def maybe(*, present: bool) -> object:
        return True if present else reject("rejected-value", "the value was rejected")

    @derived(str, value=allow_inapplicable(maybe))
    def reader(*, value: object) -> str:
        return "absent" if value is ABSENT else str(value)


def test_allow_inapplicable_does_not_swallow_a_rejection() -> None:
    answer = InapplicabilityOnly.start({InapplicabilityOnly.present: False}).answer(
        InapplicabilityOnly.reader
    )
    assert isinstance(answer, Absent)
    assert {finding.code for finding in answer.findings} == {"rejected-value"}


class OptionalValue(Space):
    value = Input(int)

    @derived(int, value=value)
    def forwarded(*, value: int) -> int:
        return value

    exports = (forwarded,)


class InactiveReader(Space):
    enabled = Problem(bool)
    value = Problem(int)
    child = Subspace(OptionalValue, when=enabled, value=value)

    @derived(str, value=allow_inapplicable(child.forwarded))
    def reader(*, value: object) -> str:
        return "absent" if value is ABSENT else str(value)


def test_allow_inapplicable_tolerates_nonrejecting_absence() -> None:
    root = InactiveReader.start({InactiveReader.enabled: False, InactiveReader.value: 7})
    assert root.answer(InactiveReader.reader) == Decided("absent")
