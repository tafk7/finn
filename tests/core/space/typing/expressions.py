"""Expressions as mypy sees them.

Formals are annotated with their value type and a reference to a node's
member is typed as its value (option A), so arithmetic on either is statically
an ``int``: a member defined that way is an ``int`` at class level and at
instance level (at runtime it is an ``Expr``).
"""

from __future__ import annotations

from typing import assert_type

from finn.core.space import Decision, Param, Space, derived


class Arithmetic(Space):
    extent: int = Param()
    lanes: int = Decision(values=(1, 2))

    @derived
    def doubled(*, extent: int) -> int:
        return extent * 2

    width = (doubled + 3) * lanes
    reflected = 1 + 12 // lanes - extent % 2
    negated = -extent


class Parent(Space):
    extent: int = Param()
    child = Arithmetic(extent=3)
    assert_type(child.width, int)
    result = child.width + 1
    # Formals are annotated with their value type, so an expression over them
    # is typed as its value too (it is an Expr at runtime).
    mixed = child.width + extent
    assert_type(mixed, int)
    other = Arithmetic(extent=child.width * 2)


assert_type(Arithmetic.width, int)
assert_type(Arithmetic.reflected, int)
assert_type(Arithmetic.negated, int)
assert_type(Parent.child.width, int)
assert_type(Parent.result, int)
assert_type(Parent.mixed, int)


def reads(point: Arithmetic, parent: Parent) -> None:
    assert_type(point.width, int)
    assert_type(point.reflected, int)
    assert_type(point.negated, int)
    assert_type(parent.result, int)
    assert_type(parent.mixed, int)
    assert_type(parent.other.width, int)
