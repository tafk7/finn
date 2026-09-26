# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Composing nodes built in plain Python stays typed without generated stubs.

A node keeps its family type wherever it travels (lists, dicts, function
results); a reference through it is typed as its value; ``composite(...,
base=B)`` is a ``type[B]``, so the base's formals type the new family's calls.
"""

from typing_extensions import assert_type

from finn.core.space import (
    OPEN,
    Available,
    Bind,
    BoundView,
    Derived,
    Param,
    QueryResult,
    Space,
    ValueSemantics,
    View,
    ViewKey,
    composite,
    configure,
    derived,
)


class Shape(Space):
    lanes: Param[int] = Param(int)
    physical = View(lanes)


def doubled(*, lanes: int) -> int:
    return lanes * 2


def answer(*, lanes: int) -> QueryResult[int]:
    return Available(lanes)


computed = derived(doubled)
answered = derived(semantics=ValueSemantics.immutable_nominal(int))(answer)
complete = View(computed)
assert_type(computed, Derived[int])
assert_type(answered, Derived[int])
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
    """A node factory: the node keeps its family type."""
    return Doubled(lanes=lanes)


def chain(count: int) -> type[Space]:
    """Nodes in a list and edges in a dict: every element stays typed."""
    stages = [Doubled(lanes=4), *(Doubled(lanes=OPEN) for _ in range(1, count))]
    assert_type(stages, list[Shape])
    assert_type(stages[0].lanes, int)
    assert_type(stages[0].physical, BoundView[int])
    edges = {
        f"e{index}": Bind(stages[index].lanes, stages[index - 1].lanes) for index in range(1, count)
    }
    assert_type(edges, dict[str, Bind[int]])
    nodes = {f"s{index}": node for index, node in enumerate(stages)}
    return composite(f"Chain{count}", {**nodes, **edges})


class Parent(Space):
    width: Param[int] = Param(int)
    first = Doubled(lanes=width)
    second = Doubled(lanes=OPEN)
    edge = Bind(second.lanes, first.lanes)
    assert_type(first, Shape)
    assert_type(first.lanes, int)
    assert_type(edge, Bind[int])


def check(point: Parent) -> None:
    assert_type(Parent.first, Shape)
    assert_type(Parent.first.lanes, int)
    assert_type(point.first, Shape)
    assert_type(point.first.lanes, int)
    assert_type(point.first.physical(), int)
    assert_type(configure(Doubled(lanes=2)), Shape)
    assert_type(configure(Parent(width=2)), Parent)
    assert_type(configure(chain(3)()), Space)
