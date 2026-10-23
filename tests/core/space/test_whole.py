# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The questions about a whole design space, answered from compiled forms (``_whole``): the
flow order and its loops, the presence readers, ``inspection``'s decisions, choices,
members and statistics, the scopes below a scope.

Each equals the line's answer (``6d02612cc``: its algorithms over one table of every node,
``core.space._whole_support``) on ``expand(C)``, by key, for every Space class the space
tests and the engine's support modules define and for the classes below, which take the
ways the corpus does not (a name the placing Space owns read by a method; a dynamic
method's subtree holding formals, a fresh Decision and a named shared Decision; a validity
piece reached by a guard); and answering it evaluates no member. The kernels, the Chain's
members and stage 7's chain are the kernel tree's (``kernel_ops/test_space_templates``).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from core.space._templates_support import defined, links
from core.space._whole_support import differences
from finn.core.space import (
    Decision,
    Param,
    Space,
    derived,
    design_space,
    inspection,
    view,
)
from finn.core.space._template import compiled
from finn.core.space.compiler import model_of

HERE = Path(__file__).resolve().parent

# The methods whose reads, inferred per template, gained a read against the line's per
# scope: two helpers calling each other (``test_reads``), each reading what both read (the
# line kept what it found first while the cycle was open, in the order it met them).
GAINED = {
    "Mutual": {"by_second": ["width"]},
    "PlacesMutual": {"inner.by_first": ["inner.lanes"]},
}


def _corpus() -> list[type[Space]]:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    return [space_type for space_type in defined(modules) if links(space_type)]


def test_the_space_tests_classes_answer_as_the_line() -> None:
    classes = _corpus()
    assert len(classes) > 150
    failures: list[str] = []
    gained: dict[str, dict[str, list[str]]] = {}
    loops = 0
    for space_type in classes:
        compared = differences(space_type)
        failures.extend(compared.differences)
        loops += compared.loops
        if compared.gained:
            gained[space_type.__qualname__] = compared.gained
    assert not failures, "\n".join(failures)
    assert gained == GAINED
    assert loops  # a loop is among them


# -- the ways the corpus does not take ----------------------------------------------------------


class Inner(Space):
    depth: int = Param()
    lanes: int = Decision(values=(1, 2))


#: The name a dynamic method reads by: a module global, so its source does not fix it.
LANES = "lanes"


class Dynamic(Space):
    """A dynamic method: it may read every node of its scope and below: a formal its
    placing Space binds to a Decision of its own, and a placed Space's formal bound to a
    Decision of this class."""

    width: int = Param()
    depth: int = Decision(values=(3, 4))
    inner = Inner(depth=depth)

    @derived
    def wide(self) -> bool:
        return cast(int, getattr(self.inner, LANES)) > 1

    gated: int = Decision(values=(1, 2), when=wide)


class PlacesDynamic(Space):
    pick: bool = Decision(values=(False, True))
    owned: int = Decision(values=(5, 6))
    dynamic = Dynamic(width=owned)
    later: int = Decision(values=(1, 2), when=dynamic.wide)


class Room(Space):
    depth: int = Param()


class Pieces(Space):
    """A guard reading a view, which reads its Space's validity: the walk from the guard
    reaches a validity piece, a Decision of the class bound into a placed Space's formal."""

    depth: int = Decision(values=(1, 2))
    room = Room(depth=depth)

    @view
    def flag(self) -> bool:
        return self.room.depth > 1

    @derived
    def shown(self) -> bool:
        return self.flag

    extra: int = Decision(values=(1, 2), when=shown)


@pytest.mark.parametrize(
    "space_type",
    (PlacesDynamic, Pieces),
    ids=lambda space_type: space_type.__name__,
)
def test_a_way_the_corpus_does_not_take_answers_as_the_line(space_type: type[Space]) -> None:
    compared = differences(space_type)
    assert not compared.differences, "\n".join(compared.differences)
    assert not compared.gained


def test_a_dynamic_method_reads_its_scopes_subtree_and_what_its_placing_space_owns() -> None:
    keys = [item.key for item in inspection.decisions(PlacesDynamic)]
    for read in ("owned", "dynamic.inner.lanes", "dynamic.depth"):
        assert keys.index(read) < keys.index("dynamic.gated")
        assert keys.index(read) < keys.index("later")


# -- answered from compiled forms: nothing evaluated ---------------------------------------------


def test_the_flow_order_the_readers_and_the_lists_evaluate_no_member() -> None:
    model = model_of(compiled(PlacesDynamic))  # a model of its own: nothing asked yet
    inspection.decisions(model)
    inspection.choices(model)
    inspection.members(model)
    inspection.statistics(model)
    scopes = model.scopes
    for source in (*scopes.parameters, *scopes.decisions):
        model.per_class.presence_readers(source)
    assert model.per_class._session is None  # no question ran: nothing evaluated
    assert scopes.counts()["scopes made"] == 3


def test_positions_are_places_in_the_flow_order() -> None:
    point = design_space(PlacesDynamic())
    scopes = inspection.model(point).scopes
    flow = inspection.model(point).per_class.flow()
    assert [scopes.position(index) for index in flow] == list(range(len(flow)))
    assert list(scopes.flow().order) == list(flow)
