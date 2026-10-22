# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Templates (``_template``): each Space class's own scope compiled once, the Spaces it
places stubs; a model's structure (``_structure``), its scopes made when addressed and its
nodes resolved when read; and the eager expansion, ``expand(C)``: every node of the design
space in one table, from templates.

A structure fully resolved, in either numbering, equals ``expand(C)`` in a canonical form
by key (``_templates_support``); that both are the line's one table of every placement was
checked, when templates replaced it, against digests taken from that table. Here the
engine's corpus: every
Space class the space tests and the engine's support modules define, and the classes
below for the edge cases. The kernels, the rest of the test support, the Chain's members
and stage 7's chain are the kernel tree's (``kernel_ops/test_space_templates``): this tree
imports the engine only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.space._templates_support import assert_expands, defined, links
from core.space._toys_support import Boiler, HeatPump, House, Room, Thermostat
from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    View,
    _template,
    bind,
    composite,
    derived,
    design_space,
    inspection,
    supplied,
    view,
)
from finn.core.space._structure import EAGER, PERMUTE, Structure, expand
from finn.core.space._template import Template, link_template
from finn.core.space.errors import RequestError
from finn.core.space.references import DecisionHandle, ValueHandle
from finn.core.space.results import Unresolved

HERE = Path(__file__).resolve().parent


def test_the_space_tests_classes_resolve_as_they_expand() -> None:
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


class BoundIntoChildren(Space):
    """Decisions of the class bound into its children's formals: one bound twice into one
    child, one into two children. Each is the class's own, keyed by its name."""

    width: int = Decision(values=(1, 2, 4))
    depth: int = Decision(values=(2, 3))
    pair = Pair(first=width, second=width)
    other = Pair(first=1, second=depth)
    again = Pair(first=depth, second=4)


class Heater(Space):
    heating: Boiler | HeatPump = Select(boiler=Boiler(kw=24), heat_pump=HeatPump(kw=8))
    pump: HeatPump | None = Select(none=None, heat_pump=HeatPump(kw=4))


class ReadsThroughAChild(Space):
    """``decision.member`` read by the parent through a child's choice: a selection the
    child's template made (``heating.kw``), and one the parent makes (``pump.cop``)."""

    heater = Heater()
    thermostat = Thermostat(kw=heater.heating.kw)
    cop = View(heater.pump.cop)  # type: ignore[union-attr]
    direct = View(heater.heating["heat_pump"].cop)  # type: ignore[index]


class ReadsItself(Space):
    pump: HeatPump | None = Select(none=None, heat_pump=HeatPump(kw=4))
    cop = View(pump.cop)  # type: ignore[union-attr]


class ReadsLater(Space):
    """A selection read by the child and by the parent: one node, owned in the child's
    template, where its choice is (the line's one table owned it by its first reader as
    it was lowered: the child's member before the parent's expression, the parent's
    member before the child's)."""

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
    source: Memory | Fetcher = Select(memory=Memory, fetch=Fetcher, when=valued, contents=contents)
    words = View(source.words)
    cache: Fetcher | None = Select(none=None, fetch=Fetcher, when=valued, contents=contents)
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


class WideStream(Space):
    """A stream whose lanes are its own choice, which exists only while ``wide``."""

    contents: int = Param()
    wide: bool = Param()
    lanes: int = Decision(values=(1, 2), when=wide)

    @view
    def words(self) -> int:
        return self.contents * self.lanes


class Guarded(Space):
    """What exists only where a supplied-guarded choice applies: a case's guard, a case's
    own choice with a ``when``, and a case's formal bound to an expression; and a Decision
    of the class its cases share, which exists wherever the class does."""

    contents: int = Param(required=False)
    wide: bool = Decision(values=(False, True))
    split: int = Decision(values=(1, 4))
    valued = supplied(contents)
    source: WideStream | Stream | None = Select(
        none=None,
        fetch=WideStream(wide=wide, when=wide),
        plus=Stream(lanes=contents + 1),
        when=valued,
        contents=contents,
    )
    words = View(source.words)  # type: ignore[union-attr]
    pair: Stream | None = Select(
        none=None,
        left=Stream(lanes=split),
        right=Stream(lanes=split),
        when=valued,
        contents=contents,
    )


class GuardedOnceAndNot(Space):
    on = Guarded(contents=3)
    off = Guarded()


class GuardedDecision(Space):
    """A formal bound to a Decision of the class with a ``when``."""

    big: bool = Decision(values=(False, True))
    area: int = Decision(values=(10, 20), when=big)
    room = Room(area=area)
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
    packed: Boiler | None = Select(none=None, compute=Boiler(kw=lanes))


class Plain(Space):
    lanes: int = Param()


class Dotted(Space):
    """Cases named so that a scope's name holds a choice's: ``compute.packed.packed.compute``."""

    compute: Packed | Plain = Select(packed=Packed, plain=Plain, lanes=2)
    kw = View(compute["packed"].packed.kw)  # type: ignore[index]


EDGE_CASES = (
    BoundIntoChildren,
    ReadsThroughAChild,
    ReadsLater,
    SuppliedOnceAndNot,
    GuardedOnceAndNot,
    GuardedDecision,
    Outer,
    Dotted,
)


@pytest.mark.parametrize("space_type", EDGE_CASES, ids=lambda space_type: space_type.__name__)
def test_an_edge_case_resolves_as_it_expands(space_type: type[Space]) -> None:
    assert_expands([space_type])


def test_a_decision_bound_into_children_is_the_class_own() -> None:
    template = link_template(BoundIntoChildren)
    assert {template.nodes[index].key for index in template.contributed} == set()
    linked = expand(BoundIntoChildren)
    assert linked.nodes[linked.keys["width"]].scope == 0
    assert linked.nodes[linked.keys["depth"]].scope == 0


def test_a_selection_read_by_several_templates_is_one_owned_in_its_choices_template() -> None:
    linked = expand(ReadsLater)
    assert linked.nodes[linked.keys["first.pump.$member.cop"]].owner == "first.cop"
    # The line: "earlier", the parent's member, lowered before the child's.
    assert linked.nodes[linked.keys["second.pump.$member.cop"]].owner == "second.cop"
    (choice,) = (choice for choice in linked.choices if choice.key == "second.pump")
    assert choice.members == {"cop": linked.keys["second.pump.$member.cop"]}
    # Read by the parent alone: owned by its reader there.
    through = expand(ReadsThroughAChild)
    assert through.nodes[through.keys["heater.pump.$member.cop"]].owner == "cop"


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
    template = link_template(BoundIntoChildren)
    assert template.counts() == {
        "guard": 1,
        "validity": 1,
        "decision": 2,
        # Each formal bound to one of the class's Decisions: an alias of it.
        "alias": 4,
        "import": 3,
    }
    # The literals 1 and 4 are values on their stubs, not nodes.
    assert link_template(BoundIntoChildren).scopes[2].bindings["first"].value == 1
    assert len(link_template(Room).nodes) == len(link_template(Room).keys)


# -- the structure: scopes made when addressed, nodes resolved when read -----------------------


def test_keys_are_matched_against_stub_names_never_split_on_dots() -> None:
    structure = Structure(link_template(Dotted))  # nothing addressed but the root
    # The case guard of ``compute.packed.packed.compute`` belongs to the scope that places
    # it: splitting the key at its last dot names the candidate's scope instead.
    guard = structure.keys["compute.packed.packed.compute.$selected"]
    assert structure.scopes[structure.nodes[guard].scope].name == "compute.packed"
    kw = structure.keys["compute.packed.packed.compute.kw"]
    assert structure.scopes[structure.nodes[kw].scope].name == "compute.packed.packed.compute"
    for permuted in (False, True):
        whole = Structure(link_template(Dotted), permuted=permuted)
        assert all(whole.keys[node.key] == node.index for node in whole.nodes)
    with pytest.raises(KeyError):
        structure.keys["compute.packed.packed.compute"]  # a scope, no node


def test_a_design_space_read_for_one_value_makes_part_of_a_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    Street = composite("Street", {f"house{number}": House(budget=100) for number in range(4)})
    point = design_space(Street())
    kitchen = point.house0.kitchen  # type: ignore[attr-defined]
    assert isinstance(kitchen.query(Room.cost), Unresolved)  # its finish is open
    structure = inspection.model(point).structure
    assert structure is not None
    read = structure.counts()
    # The root's Scope places its four houses; the house read places its own; nothing of
    # the other three houses is placed.
    names = {record.name for record in structure.records}
    assert "house0.kitchen" in names and not any(name.startswith("house1.") for name in names)
    structure.resolve_all()
    whole = structure.counts()
    table = expand(Street)
    assert whole == {
        "scopes addressed": len(table.scopes),
        "scopes in play": len(table.scopes),
        "nodes resolved": len(table.nodes),
        "const nodes resolved": sum(1 for node in table.nodes if node.kind == "const"),
    }
    # The kitchen's scope alone has a node resolved: its cost, what that reads, and its
    # validity's pieces are none of the other houses'.
    assert read == {
        "scopes addressed": 12,
        "scopes in play": 1,
        "nodes resolved": 5,
        "const nodes resolved": 1,
    }


def test_an_identity_standing_for_another_is_no_node() -> None:
    structure = Structure(link_template(Room))
    guard = link_template(Room).placement_guard  # the root's own placement: no guard
    with pytest.raises(IndexError):
        structure.nodes[guard]
    with pytest.raises(RequestError):
        ValueHandle(structure, guard)


@pytest.mark.parametrize("permuted", [False, True])
def test_a_change_names_a_node_of_an_addressed_scope(
    monkeypatch: pytest.MonkeyPatch, permuted: bool
) -> None:
    """A change's scope and node are asked of the model (``holds_scope``, ``holds_node``),
    the same in either numbering: an identity in an addressed scope's block that stands
    for another, or outside every block, names no node; a scope not addressed, none."""
    monkeypatch.delenv(EAGER, raising=False)
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")
    else:
        monkeypatch.delenv(PERMUTE, raising=False)

    class Reads(Space):  # a class of its own: its model is made under this mode
        heater = Heater()
        thermostat = Thermostat(kw=heater.heating.kw)
        cop = View(heater.pump.cop)  # type: ignore[union-attr]

    point = design_space(Reads())
    structure = inspection.model(point).structure
    assert structure is not None and structure.permuted is permuted
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    change = point.field(handles["heater.heating"]).change("boiler")
    assert point.try_with_choices(change).accepted
    # The root's block: its imports and its children's formals' bindings stand for nodes
    # of the children's scopes.
    root = structure.records[0]
    block = [structure._id(root, node) for node in range(len(root.template.nodes))]
    standing = [index for index in block if not structure.holds_node(index)]
    assert len(standing) >= 2
    ends = [(record.base, record.base + len(record.template.nodes)) for record in structure.records]
    outside = [min(start for start, _ in ends) - 1, max(end for _, end in ends)]
    assert not any(structure.holds_node(index) for index in outside)
    for index in (*standing, *outside):
        with pytest.raises(RequestError, match="change node does not belong"):
            point.try_with_choices(replace(change, node=index))
    assert not structure.holds_scope(len(structure.records))
    with pytest.raises(RequestError, match="change scope does not belong"):
        point.try_with_choices(replace(change, scope=len(structure.records)))
    # ``len(nodes)``: the identities allocated, one block per addressed scope.
    assert len(structure.nodes) == sum(end - start for start, end in ends)


@pytest.mark.parametrize("permuted", [False, True])
def test_an_identity_standing_for_a_decision_is_refused_beside_its_own(
    monkeypatch: pytest.MonkeyPatch, permuted: bool
) -> None:
    """A choice has one identity. An id that stands for a decision (an import of a child's
    Decision) is refused wherever an identity is
    keyed by: a change naming it, alone or beside the decision's own id, is a
    ``RequestError`` (never followed into the decision, so never two changes of one
    choice, nor one choice stored under two ids), in a bind as in a change; and no handle
    is made from it, so a handle's equality and hash, and ``Model.resolve`` and
    ``decision`` of one, are by the decision's own id."""
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")
    else:
        monkeypatch.delenv(PERMUTE, raising=False)

    class Standing(Space):  # a class of its own: its model is made under this mode
        fetcher = Fetcher(contents=4)
        lanes = View(fetcher.lanes)

    point = design_space(Standing())
    model = inspection.model(point)
    structure = model.structure
    assert structure is not None and structure.permuted is permuted
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    root = structure.records[0]
    standing: dict[str, int] = {}
    for node in range(len(root.template.nodes)):
        index, own = structure._id(root, node), structure.ident(root, node)
        if own is not None and own != index and structure.nodes[own].kind == "decision":
            standing[structure.nodes[own].key] = index
    assert set(standing) == {"fetcher.lanes"}
    for key, value in (("fetcher.lanes", 2),):
        change = point.field(handles[key]).change(value)
        assert point.try_with_choices(change).accepted
        assert model.decision(change.scope, handles[key]) == change.node
        index = standing[key]
        # Named by the scope the decision is in, or by the scope whose block holds the id.
        for other in (replace(change, node=index), replace(change, node=index, scope=0)):
            for batch in ((other,), (change, other), (other, change)):
                with pytest.raises(RequestError, match="change node does not belong"):
                    point.try_with_choices(*batch)
                with pytest.raises(RequestError, match="change node does not belong"):
                    bind(point, *batch)
        for kind in (ValueHandle, DecisionHandle):
            with pytest.raises(RequestError, match="handle does not identify"):
                kind(structure, index)


def test_resolution_takes_the_models_lock() -> None:
    model = inspection.model(House)
    assert model.structure is not None
    assert model.lock is model.evaluator.lock is model.structure.lock
