# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Run with mypy --strict; invalid examples live in the separate negative fixture."""

from typing_extensions import assert_type
from finn.kernels.space._self_prototype import (
    BoundView,
    Change,
    Decision,
    Derived,
    Domain,
    Param,
    Space,
    Subspace,
    View,
    derived,
    view,
)


class Child(Space):
    fact = Param(int)
    choice = Decision(int, domain=Domain(lambda self, candidate: candidate > 0))

    @derived
    def output(self) -> int:
        return self.fact + self.choice

    @view()
    def product(self) -> tuple[int, int]:
        return self.fact, self.output


class Family(Space):
    fact = Param(int)
    child = Subspace(Child, fact=fact)

    @derived
    def output(self) -> int:
        return self.child.output


point = Family(fact=1)
assert_type(Family.fact, Param[int])
assert_type(Family.child, Subspace[Child])
assert_type(Family.output, Derived[int])
assert_type(Child.product, View[tuple[int, int]])
assert_type(point.fact, int)
assert_type(point.child, Child)
assert_type(point.child.choice, int)
assert_type(point.output, int)
assert_type(point.child.product, BoundView[tuple[int, int]])
assert_type(point.child.product(), tuple[int, int])
change = point.child.field(Child.choice).change(2)
assert_type(change, Change[int])
assert_type(point.with_choices(change), Family)
assert_type(point.child.with_choices(choice=2), Child)
