# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Composing nodes built in plain Python stays typed without generated stubs.

A node keeps its Space class wherever it travels (lists, dicts, function
results); a reference through it is typed as its value; ``composite(...,
base=B)`` is a ``type[B]``, so the base's formals type the new Space class's calls.
"""

from typing import assert_type

from finn.core.space import (
    Available,
    Param,
    QueryResult,
    Space,
    ValueSemantics,
    View,
    ViewKey,
    composite,
    derived,
    design_space,
)


class Shape(Space):
    lanes: int = Param()
    physical = View(lanes)


def doubled(*, lanes: int) -> int:
    return lanes * 2


def answer(*, lanes: int) -> QueryResult[int]:
    return Available(lanes)


computed = derived(doubled)
answered = derived(semantics=ValueSemantics.immutable_nominal(int))(answer)
complete = View(computed)
# A derived member is typed as its value, like every reference, so it may supply a formal.
assert_type(computed, int)
assert_type(answered, int)
assert_type(complete, View[int])

Doubled = composite(
    "Doubled",
    {"doubled": computed, "answered": answered, "complete": complete},
    base=Shape,
    exports={ViewKey("complete", int): complete},
)
assert_type(Doubled, type[Shape])
assert_type(composite("Plain", {}), type[Space])


def stage(lanes: int) -> Shape:
    """A node factory: the node keeps its Space class."""
    return Doubled(lanes=lanes)


def chain(count: int) -> type[Space]:
    """Nodes in a list, joined by assignment in a loop: every element stays typed."""
    stages = [Doubled(lanes=4), *(Doubled() for _ in range(1, count))]
    assert_type(stages, list[Shape])
    assert_type(stages[0].lanes, int)
    assert_type(stages[0].physical, int)  # a view reference is typed as its value
    for previous, current in zip(stages, stages[1:]):
        current.lanes = previous.lanes  # typed by Param.__set__
    nodes = {f"s{index}": node for index, node in enumerate(stages)}
    return composite(f"Chain{count}", nodes)


class Parent(Space):
    width: int = Param()
    first = Doubled(lanes=width)
    second = Doubled()
    second.lanes = first.lanes
    assert_type(first, Shape)
    assert_type(first.lanes, int)


def check(point: Parent) -> None:
    assert_type(Parent.first, Shape)
    assert_type(Parent.first.lanes, int)
    assert_type(point.first, Shape)
    assert_type(point.first.lanes, int)
    assert_type(point.first.physical, int)
    assert_type(design_space(Doubled(lanes=2)), Shape)
    assert_type(design_space(Parent(width=2)), Parent)
    assert_type(design_space(chain(3)()), Space)
