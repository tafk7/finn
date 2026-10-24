# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Compiled forms (``_template``): each Space class compiled once, the Spaces it places
slots; and a model's scope records (``_scopes``), each made when reached, keyed by path,
its members read by address.

A model's scope records read whole read alike made by demand, in the permuted order
(reversed by path) and eagerly, in a canonical form by key (``_templates_support``). Here
the engine's corpus: every Space class the space tests and the engine's support modules
define, and the classes below for the edge cases. The kernels, the rest of the test
support, the Chain's members and stage 7's chain are the kernel tree's
(``kernel_ops/test_space_templates``): this tree imports the engine only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from core.space._templates_support import assert_reads_alike, defined, links
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
    view,
)
from finn.core.space._scopes import EAGER, PERMUTE, Address, ScopeRecord, Scopes
from finn.core.space._template import Compiled, compiled
from finn.core.space.compiler import compile_model, model_of
from finn.core.space.errors import RequestError
from finn.core.space.occurrence import held
from finn.core.space.references import DecisionHandle, ValueHandle
from finn.core.space.results import Unresolved

HERE = Path(__file__).resolve().parent


def test_the_space_tests_classes_read_alike_in_any_order_their_scopes_are_made_in() -> None:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 150
    assert_reads_alike(classes)


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
    """A selection read by the child and by the parent: one address, in the child's scope,
    where its choice is, owned by the choice's key (the line's one table owned it by its
    first reader as it was lowered: the child's member before the parent's expression, the
    parent's member before the child's)."""

    first = ReadsItself()
    second = ReadsItself()

    @derived(cop=first.pump.cop)  # type: ignore[union-attr]
    def next_cop(*, cop: int) -> int:
        return cop + 1

    later = View(next_cop)
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
    """Choices under a guard over an optional fact, absent where ``contents`` is None: one
    read through by a selection its candidates share, one by a selection of its one
    candidate."""

    contents: int | None = Param(default=None)

    @derived
    def valued(self) -> bool:
        return self.contents is not None

    source: Memory | Fetcher = Select(memory=Memory, fetch=Fetcher, when=valued, contents=contents)
    words = View(source.words)
    cache: Fetcher | None = Select(none=None, fetch=Fetcher, when=valued, contents=contents)
    cache_lanes = View(cache.lanes)  # type: ignore[union-attr]


class GivenOnceAndNot(Space):
    """The same guarded choice placed once with its fact given, once left out; the parent
    reads through each."""

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
    """What exists only where a choice guarded over an optional fact applies: a case's
    guard, a case's own choice with a ``when``, and a case's formal bound to an expression;
    and a Decision of the class its cases share, which exists wherever the class does."""

    contents: int | None = Param(default=None)
    wide: bool = Decision(values=(False, True))
    split: int = Decision(values=(1, 4))

    @derived
    def valued(self) -> bool:
        return self.contents is not None

    @derived(when=valued)
    def widened(self) -> int:
        return cast(int, self.contents) + 1  # read only where valued

    source: WideStream | Stream | None = Select(
        none=None,
        fetch=WideStream(wide=wide, when=wide),
        plus=Stream(lanes=widened),
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
    GivenOnceAndNot,
    GuardedOnceAndNot,
    GuardedDecision,
    Outer,
    Dotted,
)


@pytest.mark.parametrize("space_type", EDGE_CASES, ids=lambda space_type: space_type.__name__)
def test_an_edge_case_reads_alike_in_any_order_its_scopes_are_made_in(
    space_type: type[Space],
) -> None:
    assert_reads_alike([space_type])


def test_a_decision_bound_into_children_is_the_class_own() -> None:
    form = compiled(BoundIntoChildren)
    assert {"width", "depth"} <= set(form.keys)  # members of the class itself
    scopes = compile_model(BoundIntoChildren).scopes
    assert scopes.find("width").scope is scopes.root
    assert scopes.find("depth").scope is scopes.root


@pytest.mark.parametrize("permuted", [False, True])
def test_a_selection_read_by_several_classes_is_one_address_in_its_choices_scope(
    permuted: bool,
) -> None:
    """Read by the choice's own class and by a class above, or by a class above alone: one
    address, a member of the choice's scope, its owner the choice's key (the line
    owned them by their first reader, ``first.cop``, ``second.cop``, ``cop``)."""
    scopes = Scopes(compiled(ReadsLater), permuted=permuted)
    for name in ("first", "second"):
        found = scopes.find(f"{name}.pump.$member.cop")
        assert found.owner == f"{name}.pump"
        assert found.scope is scopes.root.named_children[name]
        (choice,) = (choice for choice in scopes.choices() if choice.key == f"{name}.pump")
        assert choice.members == {"cop": found}
    # Read by the class above alone: a member of the choice's scope numbered after its
    # class's records, reading the record the class above writes.
    through = Scopes(compiled(ReadsThroughAChild), permuted=permuted)
    found = through.find("heater.pump.$member.cop")
    heater = through.root.named_children["heater"]
    assert found.owner == "heater.pump"
    assert found.scope is heater and found.member >= len(heater.compiled.records)
    assert found.held.scope is through.root and found.held != found
    assert through.holds_address(found) and not through.holds_address(found.held)


# -- compiled forms: once per class ---------------------------------------------------------------


# One class placed many times.
Many = composite("Many", {f"room{area}": Room(area=area) for area in range(1, 9)})


def test_a_class_placed_many_times_is_compiled_once(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[type[Space]] = []
    original = _template._compile

    def counted(space_type: type[Space]) -> Compiled:
        made.append(space_type)
        return original(space_type)

    monkeypatch.setattr(_template, "_compile", counted)
    form = compiled(Many)
    assert made.count(Many) <= 1 and made.count(Room) <= 1
    assert compiled(Many) is form
    assert len(form.slots) == 8
    assert {id(slot.compiled) for slot in form.slots} == {id(compiled(Room))}
    assert len(compile_model(Many).scopes.ordered()) == 9


# -- the scope records: made when reached, members read by address ------------------------------


def test_keys_are_matched_against_slot_names_never_split_on_dots() -> None:
    scopes = Scopes(compiled(Dotted))  # nothing made but the root
    # The case guard of ``compute.packed.packed.compute`` belongs to the scope that places
    # it: splitting the key at its last dot names the candidate's scope instead.
    guard = scopes.find("compute.packed.packed.compute.$selected")
    assert guard.scope.name == "compute.packed"
    kw = scopes.find("compute.packed.packed.compute.kw")
    assert kw.scope.name == "compute.packed.packed.compute"
    for permuted in (False, True):
        whole = Scopes(compiled(Dotted), permuted=permuted)
        assert all(whole.find(node.key) is node for node in whole.all_addresses())
    with pytest.raises(KeyError):
        scopes.find("compute.packed.packed.compute")  # a scope, no member


def test_a_design_space_read_for_one_value_makes_part_of_a_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EAGER, raising=False)  # the eager test mode makes everything at once
    Street = composite("Street", {f"house{number}": House(budget=100) for number in range(4)})
    point = design_space(Street())
    kitchen = point.house0.kitchen  # type: ignore[attr-defined]
    assert isinstance(kitchen.query(Room.cost), Unresolved)  # its finish is open
    scopes = inspection.model(point).scopes
    made = scopes.counts()["scopes made"]
    # The root's Scope places its four houses; the house read places its own; nothing of
    # the other three houses is placed.
    names = {record.name for record in scopes.records()}
    assert "house0.kitchen" in names and not any(name.startswith("house1.") for name in names)
    # The kitchen's scope alone has a member evaluated: its cost (a derived value) and what
    # that reads, ``area`` and ``finish``: three. (As a view, its cost also evaluated its
    # output record and the scope's validity: five.)
    ds = held(point).ds
    evaluated = {index for ev in ds.evaluations.values() for index in ev.memo}
    assert {index.scope.name for index in evaluated} == {"house0.kitchen"}
    assert (made, len(evaluated)) == (12, 3)
    scopes.read_all()
    whole = scopes.counts()
    assert {key: whole[key] for key in ("scopes made", "members read")} == {
        "scopes made": len(scopes.ordered()),
        "members read": sum(len(record.compiled.records) for record in scopes.ordered())
        - sum(
            1
            for record in scopes.ordered()
            for member in range(len(record.compiled.records))
            if record.canonical(member) != Address(record, member)
        ),
    }


def _another(space_type: type[Space], key: str) -> Address:
    """``key``'s address in a model of ``space_type`` of its own (another model's)."""
    return model_of(compiled(space_type)).scopes.find(key)


def test_a_handle_names_an_address_of_its_model() -> None:
    """A handle holds an address of its model: one of another model, or a member number
    past its class's count, is refused; a handle of another model resolves in none."""
    model = model_of(compiled(Room))
    root = model.root
    own = model.scopes.parameters[0]
    for address in (Address(root, len(root.compiled.records)), _another(Room, own.key)):
        for kind in (ValueHandle, DecisionHandle):
            with pytest.raises(RequestError, match="handle does not identify"):
                kind(model, address)
    other = model_of(compiled(Room))
    stale = ValueHandle[object](other, other.scopes.find(own.key))
    with pytest.raises(RequestError, match="belongs to a different compiled model"):
        model.resolve(root, stale)
    assert ValueHandle[object](model, own) == ValueHandle[object](model, Address(root, own[1]))


@pytest.mark.parametrize("permuted", [False, True])
def test_a_change_names_an_address_of_a_scope_of_its_model(
    monkeypatch: pytest.MonkeyPatch, permuted: bool
) -> None:
    """A change's scope and address are asked of the model (``holds``, ``holds_address``),
    the same in either order of making scopes: a record a class writes for a Space it
    places (a formal's binding, a selection) stands for an address elsewhere and is none;
    a member number past the count, an address or a scope record of another model, and a
    record at a path the model did not make, name nothing of it."""
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
    scopes = inspection.model(point).scopes
    assert scopes.permuted is permuted
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    change = point.field(handles["heater.heating"]).change("boiler")
    assert point.try_with_choices(change).accepted
    root = scopes.root
    standing = [
        Address(root, record)
        for record in root.compiled.written
        if root.canonical(record) != Address(root, record)
    ]
    assert standing
    elsewhere = _another(Reads, "heater.heating")
    for address in (
        *standing,
        Address(root, len(root.compiled.records)),
        Address(root, -1),
        elsewhere,
        Address(
            change.node.scope,
            len(change.node.scope.compiled.records) + len(change.node.scope.selections),
        ),
    ):
        assert not scopes.holds_address(address)
        with pytest.raises(RequestError, match="change node does not belong"):
            point.try_with_choices(replace(change, node=address))
    stray = ScopeRecord(scopes.registry, root.compiled.slots[0].compiled, root, 0)
    for scope in (elsewhere.scope, stray):
        assert not scopes.holds(scope)
        with pytest.raises(RequestError, match="change scope does not belong"):
            point.try_with_choices(replace(change, scope=scope))


@pytest.mark.parametrize("permuted", [False, True])
def test_a_decision_named_by_another_models_address_is_refused_beside_its_own(
    monkeypatch: pytest.MonkeyPatch, permuted: bool
) -> None:
    """A choice has one address. The same decision's address in another model of the
    class, or an address past its class's count, is refused wherever an address is keyed
    by: a change naming it, alone or beside the decision's own, is a ``RequestError``
    (never two changes of one choice, nor one choice stored under two addresses), in a
    bind as in a change; and no handle is made from it, so a handle's equality and hash,
    and ``Model.resolve`` and ``decision`` of one, are by the decision's own address."""
    if permuted:
        monkeypatch.setenv(PERMUTE, "1")
    else:
        monkeypatch.delenv(PERMUTE, raising=False)

    class Standing(Space):  # a class of its own: its model is made under this mode
        fetcher = Fetcher(contents=4)
        lanes = View(fetcher.lanes)

    point = design_space(Standing())
    model = inspection.model(point)
    assert model.scopes.permuted is permuted
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    for key, value in (("fetcher.lanes", 2),):
        change = point.field(handles[key]).change(value)
        assert point.try_with_choices(change).accepted
        assert model.decision(change.scope, handles[key]) == change.node
        scope = change.node.scope
        for address in (_another(Standing, key), Address(scope, len(scope.compiled.records))):
            # Named by the scope the decision is in, or by the root's.
            for other in (
                replace(change, node=address),
                replace(change, node=address, scope=model.root),
            ):
                for batch in ((other,), (change, other), (other, change)):
                    with pytest.raises(RequestError, match="change node does not belong"):
                        point.try_with_choices(*batch)
                    with pytest.raises(RequestError, match="change node does not belong"):
                        bind(point, *batch)
            for kind in (ValueHandle, DecisionHandle):
                with pytest.raises(RequestError, match="handle does not identify"):
                    kind(model, address)


def test_resolution_takes_the_models_lock() -> None:
    model = inspection.model(House)
    assert model.lock is model.per_class.lock is model.scopes.lock
