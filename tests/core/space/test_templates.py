# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``_template``): each Space class's own scope compiled once, the Spaces it
places stubs; and the eager expansion, equal to ``link_space`` by key.

``expand(C)`` rebuilds ``link_space(C)``'s model from templates alone; its ``NodeId``s
differ, so the two are compared in a canonical form by key (``_templates_support``). Here
the engine's corpus: every Space class the space tests and the engine's support modules
define, and the classes below for the edge cases. The kernels, the rest of the test
support, the Chain's members and stage 7's chain are the kernel tree's
(``kernel_ops/test_space_templates``): this tree imports the engine only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.space._templates_support import assert_expands, defined, links
from core.space._toys_support import Boiler, HeatPump, Room, Thermostat
from finn.core.space import (
    Decision,
    Param,
    Space,
    View,
    _template,
    composite,
    derived,
    supplied,
    view,
)
from finn.core.space._template import Template, expand, link_template

HERE = Path(__file__).resolve().parent


def test_the_space_tests_classes_expand_as_they_link() -> None:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 150
    assert_expands(classes)


# -- the edge cases -------------------------------------------------------------------------------


class Pair(Space):
    first: int = Param()
    second: int = Param()

    @derived
    def total(self) -> int:
        return self.first + self.second


# Named shared Decisions: bound at calls, never class attributes.
_WIDTH: int = Decision(values=(1, 2, 4), name="width")
_DEPTH: int = Decision(values=(2, 3), name="depth")


class OwnedAtChild(Space):
    """A named shared Decision whose uses are one child's (the child owns it), and one
    whose uses are two children's (the parent owns it)."""

    pair = Pair(first=_WIDTH, second=_WIDTH)
    other = Pair(first=1, second=_DEPTH)
    again = Pair(first=_DEPTH, second=4)


class Heater(Space):
    heating: Boiler | HeatPump = Decision({"boiler": Boiler(kw=24), "heat_pump": HeatPump(kw=8)})
    pump: HeatPump | None = Decision({"heat_pump": HeatPump(kw=4)}, optional=True)


class ReadsThroughAChild(Space):
    """``decision.member`` read by the parent through a child's choice: a selection the
    child's template made (``heating.kw``), and one the parent makes (``pump.cop``)."""

    heater = Heater()
    thermostat = Thermostat(kw=heater.heating.kw)
    cop = View(heater.pump.cop)  # type: ignore[union-attr]
    direct = View(heater.heating["heat_pump"].cop)  # type: ignore[index]


class ReadsItself(Space):
    pump: HeatPump | None = Decision({"heat_pump": HeatPump(kw=4)}, optional=True)
    cop = View(pump.cop)  # type: ignore[union-attr]


class ReadsLater(Space):
    """A selection read by the child and by the parent: owned by the first reader as
    ``link_space`` lowers, here the child's member before the parent's expression."""

    first = ReadsItself()
    second = ReadsItself()
    later = View(first.pump.cop + 1)  # type: ignore[union-attr]
    earlier = View(second.pump.cop)  # type: ignore[union-attr]


class Memory(Space):
    contents: int = Param()

    @view
    def words(self) -> int:
        return 2 * self.contents


class Fetcher(Space):
    contents: int = Param()
    lanes: int = Decision(values=(1, 2))

    @view
    def words(self) -> int:
        return self.contents * self.lanes


class Edge(Space):
    """Choices that never apply where ``contents`` is unsupplied: one read through by a
    selection its candidates share, one by a selection of its one candidate."""

    contents: int = Param(required=False)
    valued = supplied(contents)
    source: Memory | Fetcher = Decision(
        {"memory": Memory, "fetch": Fetcher}, when=valued, contents=contents
    )
    words = View(source.words)
    cache: Fetcher | None = Decision(
        {"fetch": Fetcher}, optional=True, when=valued, contents=contents
    )
    cache_lanes = View(cache.lanes)  # type: ignore[union-attr]


class SuppliedOnceAndNot(Space):
    """The same supplied-guarded choice placed once supplied, once not; the parent reads
    through each."""

    stored = Edge(contents=4)
    streamed = Edge()
    stored_words = View(stored.source.words)
    streamed_words = View(streamed.source.words)
    streamed_contents = View(streamed.source.contents)


class Stream(Space):
    contents: int = Param()
    lanes: int = Param()

    @view
    def words(self) -> int:
        return self.contents * self.lanes


_SPLIT: int = Decision(values=(1, 4), name="split")


class Guarded(Space):
    """What exists only where a supplied-guarded choice applies: a case's guard, a
    candidate's formal bound to a fresh Decision with a ``when`` and one bound to an
    expression, and a named shared Decision only its candidates use."""

    contents: int = Param(required=False)
    wide: bool = Decision(values=(False, True))
    valued = supplied(contents)
    source: Stream | None = Decision(
        {
            "fetch": Stream(lanes=Decision(values=(1, 2), when=wide), when=wide),
            "plus": Stream(lanes=contents + 1),
        },
        optional=True,
        when=valued,
        contents=contents,
    )
    words = View(source.words)  # type: ignore[union-attr]
    pair: Stream | None = Decision(
        {"left": Stream(lanes=_SPLIT), "right": Stream(lanes=_SPLIT)},
        optional=True,
        when=valued,
        contents=contents,
    )


class GuardedOnceAndNot(Space):
    on = Guarded(contents=3)
    off = Guarded()


class GuardedDecision(Space):
    """A formal bound to a fresh Decision with a ``when``."""

    big: bool = Decision(values=(False, True))
    room = Room(area=Decision(values=(10, 20), when=big))
    hall = Room(area=12, when=big)


class Inner(Space):
    depth: int = Param()

    @derived
    def doubled(self) -> int:
        return 2 * self.depth


class Middle(Space):
    depth: int = Param()
    inner = Inner(depth=depth)


class Outer(Space):
    """A three-level reference: ``middle.inner.doubled``."""

    middle = Middle(depth=3)
    seen = View(middle.inner.doubled)
    formal = View(middle.inner.depth)


class Packed(Space):
    lanes: int = Param()
    packed: Boiler | None = Decision({"compute": Boiler(kw=lanes)}, optional=True)


class Plain(Space):
    lanes: int = Param()


class Dotted(Space):
    """Cases named so that a scope's name holds a choice's: ``compute.packed.packed.compute``."""

    compute: Packed | Plain = Decision({"packed": Packed, "plain": Plain}, lanes=2)
    kw = View(compute["packed"].packed.kw)  # type: ignore[index]


EDGE_CASES = (
    OwnedAtChild,
    ReadsThroughAChild,
    ReadsLater,
    SuppliedOnceAndNot,
    GuardedOnceAndNot,
    GuardedDecision,
    Outer,
    Dotted,
)


@pytest.mark.parametrize("space_type", EDGE_CASES, ids=lambda space_type: space_type.__name__)
def test_an_edge_case_expands_as_it_links(space_type: type[Space]) -> None:
    assert_expands([space_type])


def test_a_named_decision_used_only_by_one_child_is_its_own() -> None:
    template = link_template(OwnedAtChild)
    assert {template.nodes[index].key for index in template.contributed} == {"pair.width"}
    assert template.nodes[template.keys["depth"]].scope == 0
    linked = expand(OwnedAtChild)
    assert linked.scopes[linked.nodes[linked.keys["pair.width"]].scope].name == "pair"
    assert linked.nodes[linked.keys["depth"]].scope == 0


def test_a_selection_read_by_several_templates_is_one_owned_by_its_first_reader() -> None:
    linked = expand(ReadsLater)
    assert linked.nodes[linked.keys["first.pump.$member.cop"]].owner == "first.cop"
    assert linked.nodes[linked.keys["second.pump.$member.cop"]].owner == "earlier"
    (choice,) = (choice for choice in linked.choices if choice.key == "second.pump")
    assert choice.members == {"cop": linked.keys["second.pump.$member.cop"]}


def test_a_choice_never_applies_where_its_formal_is_unsupplied() -> None:
    linked = expand(SuppliedOnceAndNot)
    assert linked.nodes[linked.keys["stored.source"]].kind == "decision"
    assert linked.nodes[linked.keys["streamed.source"]].kind == "const"
    assert not any(key.startswith("streamed.source.") for key in linked.keys if "$" not in key)
    # Its template compiled the candidates once, and records the formal.
    source, cache = link_template(Edge).choices
    assert source.never_formal == cache.never_formal == "contents"
    assert [case for case, stub in source.cases if stub is not None] == ["memory", "fetch"]
    assert linked.nodes[linked.keys["streamed.cache.$member.lanes"]].alternatives == ()


# -- templates: once per class, and their sizes --------------------------------------------------


# One class placed many times.
Many = composite("Many", {f"room{area}": Room(area=area) for area in range(1, 9)})


def test_a_class_placed_many_times_is_compiled_once(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled: list[type[Space]] = []
    original = _template._compile

    def counted(space_type: type[Space]) -> Template:
        compiled.append(space_type)
        return original(space_type)

    monkeypatch.setattr(_template, "_compile", counted)
    template = link_template(Many)
    assert compiled.count(Many) <= 1 and compiled.count(Room) <= 1
    assert link_template(Many) is template
    stubs = [scope for scope in template.scopes[1:] if scope.parent == 0]
    assert len(stubs) == 8
    assert {id(link_template(stub.space_type)) for stub in stubs} == {id(link_template(Room))}
    assert len(expand(Many).scopes) == 9


def test_a_template_counts_its_own_nodes_by_kind() -> None:
    template = link_template(OwnedAtChild)
    assert template.counts() == {
        "guard": 1,
        "validity": 1,
        "decision": 2,
        # Each formal bound to a named shared Decision: its use, an editable alias.
        "alias": 4,
        "import": 3,
    }
    # The literals 1 and 4 are values on their stubs, not nodes.
    assert link_template(OwnedAtChild).scopes[2].bindings["first"].value == 1
    assert len(link_template(Room).nodes) == len(link_template(Room).keys)
