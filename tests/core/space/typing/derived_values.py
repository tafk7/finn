"""Derived values over references as mypy sees them.

Formals are annotated with their value type and a reference to a node's member is typed as
its value, so a derived value that reads either computes on plain values: a member declared
that way is an ``int`` at class level and at instance level.
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

    @derived
    def width(*, doubled: int, lanes: int) -> int:
        return (doubled + 3) * lanes

    @derived
    def reflected(*, extent: int, lanes: int) -> int:
        return 1 + 12 // lanes - extent % 2

    @derived
    def negated(*, extent: int) -> int:
        return -extent


class Parent(Space):
    extent: int = Param()
    child = Arithmetic(extent=3)
    assert_type(child.width, int)

    @derived(width=child.width)
    def result(*, width: int) -> int:
        return width + 1

    # Formals are annotated with their value type and a reference is typed as its value,
    # so a derived value over both reads plain integers.
    @derived(width=child.width, extent=extent)
    def mixed(*, width: int, extent: int) -> int:
        return width + extent

    @derived(width=child.width)
    def twice(*, width: int) -> int:
        return width * 2

    other = Arithmetic(extent=twice)


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
