# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Builder output types and wrapper handles remain useful without generated stubs."""

from typing import assert_type

from finn.core.space import (
    Available,
    Derived,
    Param,
    QueryResult,
    ScopeBuilder,
    Space,
    Subspace,
    ValueKey,
    ValueRef,
    ValueSemantics,
    View,
    ViewKey,
    derived,
)


class Shape(Space):
    lanes = Param(int)


class Placement(Subspace[Shape]):
    @property
    def lanes(self) -> ValueRef[int]:
        return self.ref(Shape.lanes)


def doubled(*, lanes: int) -> int:
    return lanes * 2


def answer(*, lanes: int) -> QueryResult[int]:
    return Available(lanes)


builder = ScopeBuilder(Shape)
computed = builder.add("doubled", derived(doubled))
answered = builder.add("answered", derived(semantics=ValueSemantics.immutable_nominal(int))(answer))
view = builder.add("complete", View(computed))
builder.export(ValueKey("bits", int)).value(computed)
builder.export(ViewKey("complete", int)).view(view)
builder.bind(Shape.lanes, 4)
assert_type(computed, Derived[int])
assert_type(answered, Derived[int])
assert_type(view, View[int])
assert_type(builder.finish(), type[Shape])
assert_type(builder.place(), Subspace[Shape])


class Parent(Space):
    child = Placement(builder.finish(), lanes=4)


def check(point: Parent) -> None:
    assert_type(Parent.child.lanes, ValueRef[int])
    assert_type(point.child, Shape)
    assert_type(point.child.lanes, int)
