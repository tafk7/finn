# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The inclusion spelling keeps primitive and namespace types in each context."""

from typing import assert_type

from finn.core.space import (
    BoundDecision,
    Decision,
    Group,
    Param,
    QueryResult,
    Space,
    ViewAssessment,
    derived,
    design_space,
    view,
)


class Arithmetic(Group):
    width: int = Param()
    factor: int = Decision(values=(1, 2))

    @derived
    def product(self) -> int:
        return self.width * self.factor

    @view
    def shown(self) -> int:
        return self.product


class Owner(Space):
    size: int = Param(default=4)
    first = Arithmetic(width=size)
    second = Arithmetic(width=first.product)


def typed(point: Owner) -> None:
    assert_type(Owner.first, Arithmetic)
    assert_type(Owner.first.width, int)
    assert_type(point.first, Arithmetic)
    assert_type(point.first.product, int)
    assert_type(point.first.shown, int)
    assert_type(point.first.query(Arithmetic.product), QueryResult[int])
    assert_type(point.first.inspect(Arithmetic.shown), ViewAssessment[int])
    assert_type(point.field(Owner.first.factor), BoundDecision[int])
    assert_type(point.with_choices({Owner.first.factor: 2}), Owner)
    assert_type(design_space(Owner()), Owner)
