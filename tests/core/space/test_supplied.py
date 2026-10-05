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

import pytest

from finn.core.space import (
    Available,
    Decision,
    EvaluationError,
    Inapplicable,
    Param,
    Rejected,
    Space,
    ValueUnavailableError,
    View,
    derived,
    design_space,
    inspection,
    reject,
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


# Presence before value: present() on a value input answers whether a source
# applies and never evaluates the source; a refusing source is present, and its
# refusal surfaces where the value is read.

CALLS: list[str] = []


class Refusing(Space):
    on: bool = Param()

    @view(when=on)
    def guarded(self) -> int | Rejected:
        return reject("refused", "the guarded source refuses")

    @view
    def unguarded(self) -> int | Rejected:
        return reject("refused", "the unguarded source refuses")

    @derived
    def loud(self) -> int:
        CALLS.append("loud")
        raise RuntimeError("presence evaluated the source")


class Reader(Space):
    x: int = Param(required=False)
    has_x = supplied(x)

    @derived
    def present_x(self) -> bool:
        return self.present(Reader.x)


class Refusals(Space):
    source = Refusing(on=True)
    guarded = Reader(x=source.guarded)
    unguarded = Reader(x=source.unguarded)
    loud = Reader(x=source.loud)
    absent = Reader()


@pytest.mark.parametrize("name", ["guarded", "unguarded"])
def test_a_refusing_source_is_present_and_refuses_where_read(name: str) -> None:
    reader = getattr(design_space(Refusals()), name)
    assert reader.present(Reader.x) is True
    assert reader.has_x is True and reader.present_x is True
    with pytest.raises(ValueUnavailableError) as raised:
        reader.x
    assert isinstance(raised.value.result, Rejected)
    assert [finding.code for finding in raised.value.result.findings] == ["refused"]


def test_presence_does_not_evaluate_the_source() -> None:
    CALLS.clear()
    point = design_space(Refusals())
    assert point.loud.present(Reader.x) is True
    assert point.loud.has_x is True and point.loud.present_x is True
    assert point.absent.present(Reader.x) is False and point.absent.has_x is False
    assert CALLS == []
    with pytest.raises(EvaluationError):
        point.loud.x
    assert CALLS == ["loud"]


class Through(Space):
    """A value input bound through a selection: present when the chosen candidate
    has the member, undecided while the choice is."""

    stored = Edge(contents=4)
    streamed = Edge()
    chosen = Reader(x=stored.words)
    nothing = Reader(x=streamed.words)


def test_presence_through_a_selection_reads_the_choice_not_the_value() -> None:
    point = design_space(Through())
    assert point.nothing.present(Reader.x) is False
    with pytest.raises(ValueUnavailableError):
        point.chosen.present(Reader.x)
    chosen = point.with_choices({Through.stored.source: "memory"})
    assert chosen.chosen.present(Reader.x) is True and chosen.chosen.x == 8


class Echo(Space):
    x: int = Param(required=False)

    @derived
    def doubled(self) -> int:
        return 2 if self.present(Echo.x) else 0


def test_a_source_may_read_whether_it_supplies_its_reader() -> None:
    # Evaluating the source to answer presence would make this a dependency cycle.
    class Outer(Space):
        echo = Echo()
        echo.x = echo.doubled

    point = design_space(Outer())
    assert point.echo.present(Echo.x) is True
    assert point.echo.x == 2
