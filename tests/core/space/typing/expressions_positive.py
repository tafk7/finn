from __future__ import annotations

from typing_extensions import assert_type

from finn.core.space import Decision, Param, Space, Subspace, derived
from finn.core.space.expressions import Expr


class Arithmetic(Space):
    extent = Param(int)
    lanes = Decision(int, values=(1, 2))

    @derived
    def doubled(*, extent: int) -> int:
        return extent * 2

    width = (doubled + 3) * lanes
    reflected = 1 + 12 // lanes - extent % 2
    negated = -extent


class Parent(Space):
    child = Subspace(Arithmetic, extent=3)
    result = child.ref(Arithmetic.width) + 1


assert_type(Arithmetic.width, Expr)
assert_type(Arithmetic.reflected, Expr)
assert_type(Arithmetic.negated, Expr)
assert_type(Parent.result, Expr)


def reads(point: Arithmetic, parent: Parent) -> None:
    assert_type(point.width, int)
    assert_type(point.reflected, int)
    assert_type(point.negated, int)
    assert_type(parent.result, int)
