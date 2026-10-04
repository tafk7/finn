# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``supplied(formal)``: whether a value input is supplied, as a guard the compiler reads.

It answers what ``present(formal)`` answers in a method. As the guard of a
Decision over nodes, where the declaration never supplies the input the
Decision can never apply: its candidates are not compiled, it has no key, and
whatever is read through it is inapplicable.
"""

from __future__ import annotations

from typing import Any

from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Space,
    View,
    derived,
    design_space,
    inspection,
    supplied,
    view,
)


class Memory(Space):
    contents: int = Param()

    @view
    def words(self) -> int:
        return 2 * self.contents


class Fetcher(Space):
    contents: int = Param()

    @view
    def words(self) -> int:
        return self.contents


SOURCES: dict[str, type[Space] | Space] = {"memory": Memory, "fetch": Fetcher}


class Edge(Space):
    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Memory | Fetcher = Decision(SOURCES, when=valued, contents=contents)
    words = View(source.words)


class Owner(Space):
    known: bool = Param()

    @view(when=known)
    def value(self) -> int:
        return 3


class Graph(Space):
    """A valued edge, an edge nothing supplies, and one bound to a guarded view."""

    owner = Owner(known=True)
    stored = Edge(contents=4)
    streamed = Edge()
    bound = Edge()
    bound.contents = owner.value


def keys(point: Any) -> set[str]:
    return {item.key for item in inspection.decisions(point)}


def test_supplied_reads_as_present() -> None:
    point = design_space(Graph())
    assert point.stored.valued is True and point.streamed.valued is False
    # Bound to a guarded view: supplied when the view applies.
    assert point.bound.valued is True


def test_a_choice_guarded_by_an_input_nothing_supplies_is_not_compiled() -> None:
    point = design_space(Graph())
    assert {"stored.source", "bound.source"} <= keys(point)
    assert not any(key.startswith("streamed.source") for key in keys(point))
    scopes = {scope.name for scope in inspection.model(point).linked.scopes}
    assert {"stored.source.memory", "stored.source.fetch"} <= scopes
    assert not any(name.startswith("streamed.source") for name in scopes)
    # What is read through it is inapplicable, typed as the candidates declare it.
    assert isinstance(point.streamed.query(Edge.words), Inapplicable)
    assert isinstance(point.streamed.query(Edge.source), Inapplicable)
    chosen = point.with_choices({Graph.stored.source: "memory"})
    assert chosen.stored.query(Edge.words) == Available(8)


def test_a_roots_own_input_is_an_input_and_its_choice_is_compiled() -> None:
    # The root's formals are bound at design_space(): nothing is known before.
    alone = design_space(Edge())
    assert "source" in keys(alone)
    assert isinstance(alone.query(Edge.source), Inapplicable)
    assert design_space(Edge(contents=2)).with_choices(source="fetch").words == 2


def test_an_unsupplied_input_guard_is_unsupplied_inside_a_method_too() -> None:
    class Reader(Space):
        edge = Edge()

        @derived
        def has_value(self) -> bool:
            return self.edge.valued

    assert design_space(Reader()).has_value is False
