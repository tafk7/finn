# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A model's scope records, keyed by path, and the addresses of their members
(``finn.core.space._scopes``).

A scope is a Space class placed at a path, interned by its path and never ordered by when it
was made; an address is (scope, member), equal and hashed by the scope record's identity and
the member's number, ordered by the scope's breadth-first position, then the member's number.
Each address reads its member record through its scope, its edges mapped when read.

The claim pinned here: every key the model's structure resolved is an address, reading the
same fields and edges by key, and every scope and choice reads the same, by demand, made in
the permuted order and made eagerly; a key names its address; any order of making scopes
gives the same records and addresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.space._templates_support import _name, _stable, defined, links
from finn.core.space import Decision, Param, Select, Space, derived, supplied
from finn.core.space._running.evaluate import evaluation
from finn.core.space._scopes import Address, Scopes, existence
from finn.core.space._structure import Structure
from finn.core.space._template import compiled, link_template
from finn.core.space.compiler import model_of
from finn.core.space.occurrence import held, opened

HERE = Path(__file__).resolve().parent


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @derived
    def bits(*, width: int, lanes: int) -> int:
        return width * lanes


class Wide(Space):
    width: int = Param()
    extra: int = Decision(values=(3, 4))


class Narrow(Space):
    width: int = Param()


class Outer(Space):
    w: int = Param()
    pick: Wide | Narrow = Select(wide=Wide(width=w), narrow=Narrow(width=w))
    leaf = Leaf(width=w)
    left = Leaf(width=leaf.bits)
    chosen = pick.width  # a selection the candidates share


class Gated(Space):
    stream: int = Param(required=False)
    given = supplied(stream)
    source: Wide | Narrow = Select(wide=Wide, narrow=Narrow, when=given, width=stream)


class Holds(Space):
    w: int = Param()
    given = Gated(stream=w)
    missing = Gated()


def _old(space_type: type[Space]) -> dict[str, object]:
    """The structure's raw nodes, scopes and choices by key."""
    structure = Structure(link_template(space_type))
    records = structure.records

    def key(index: int | None) -> str | None:
        return None if index is None else structure.raw(index).key

    found: dict[str, object] = {}
    for index in list(structure.all_nodes()):
        node = structure.raw(index)
        found["node " + node.key] = (
            *_fields(node, key),
            records[node.scope].name,
        )
    for index in range(len(records)):
        scope = structure.scope(index)
        found["scope " + scope.name] = (
            None if scope.parent is None else records[scope.parent].name,
            scope.space_type,
            key(scope.guard),
            key(scope.validity),
            sorted((name, key(member)) for name, member in scope.named_members.items()),
            sorted((name, records[child].name) for name, child in scope.named_children.items()),
            {(id(member), key(index)) for member, index in scope.members.items()},
            {(id(child), records[index].name) for child, index in scope.children.items()},
            {(id(choice), structure.choice(index).key) for choice, index in scope.choices.items()},
            id(scope.record),
        )
    for index in range(len(structure._choice_owners)):
        choice = structure.choice(index)
        found["choice " + choice.key] = (
            records[choice.scope].name,
            key(choice.selector),
            key(choice.guard),
            tuple((case, None if c is None else records[c].name) for case, c in choice.cases),
            sorted((member, key(index)) for member, index in choice.members.items()),
        )
    found["decisions"] = [key(index) for index in structure.decisions]
    found["parameters"] = [key(index) for index in structure.parameters]
    found["statistics"] = structure.statistics()
    return found


def _fields(node: Any, key: Any) -> tuple[object, ...]:
    return (
        node.kind,
        node.owner,
        node.origin,
        None if node.semantics is None else node.semantics.name,
        key(node.guard),
        tuple((argument.name, key(argument.node)) for argument in node.arguments),
        tuple((argument.name, key(argument.node)) for argument in node.domain_arguments),
        key(node.output),
        key(node.selector),
        tuple(key(target) for target in node.constraints),
        tuple((label, key(target)) for label, target in node.alternatives),
        _stable(node.value),
        _stable(node.domain),
        node.required,
        _name(node.function),
        node.call_style,
    )


def _new(scopes: Scopes) -> dict[str, object]:
    """The scope records' addresses, scopes and choices by key."""

    def key(address: Address | None) -> str | None:
        return None if address is None else address.key

    found: dict[str, object] = {}
    for address in list(scopes.all_addresses()):
        found["node " + address.key] = (*_fields(address, key), address.standing.name)
    for scope in scopes.ordered():
        found["scope " + scope.name] = (
            None if scope.parent is None else scope.parent.name,
            scope.space_type,
            key(scope.guard),
            key(scope.validity),
            sorted((name, key(member)) for name, member in scope.named_members.items()),
            sorted((name, child.name) for name, child in scope.named_children.items()),
            {(id(member), key(address)) for member, address in scope.members.items()},
            {(id(child), record.name) for child, record in scope.children.items()},
            {(id(choice), record.key) for choice, record in scope.choices.items()},
            id(scope.record),
        )
    for choice in scopes.choices():
        found["choice " + choice.key] = (
            choice.scope.name,
            key(choice.selector),
            key(choice.guard),
            tuple((case, None if c is None else c.name) for case, c in choice.cases),
            sorted((member, key(address)) for member, address in choice.members.items()),
        )
    found["decisions"] = [key(address) for address in scopes.decisions]
    found["parameters"] = [key(address) for address in scopes.parameters]
    found["statistics"] = scopes.statistics()
    return found


def _differences(space_type: type[Space]) -> list[str]:
    held = _old(space_type)
    found: list[str] = []
    for mode in ({}, {"permuted": True}, {"eager": True}):
        scopes = Scopes(
            compiled(space_type),
            permuted=mode.get("permuted", False),
            eager=mode.get("eager", False),
        )
        got = _new(scopes)
        for entry in sorted(held.keys() | got.keys()):
            if held.get(entry) != got.get(entry):
                found.append(
                    f"{space_type.__qualname__} {mode}: {entry}\n"
                    f"  structure: {held.get(entry)}\n  scopes: {got.get(entry)}"
                )
                if len(found) == 3:
                    return found
        for address in scopes.all_addresses():
            if scopes.find(address.key) is not address:
                found.append(f"{space_type.__qualname__} {mode}: {address.key} finds no address")
    return found


def test_the_space_tests_classes_read_by_address_as_their_structure_resolved_them() -> None:
    modules = [f"core.space.{path.stem}" for path in sorted(HERE.glob("*.py"))] + [
        "support.space_cores",
        "support.space_pipeline",
        "support.space_walk",
    ]
    classes = [space_type for space_type in defined(modules) if links(space_type)]
    assert len(classes) > 150
    failures = [entry for space_type in classes for entry in _differences(space_type)]
    assert not failures, "\n".join(failures)


def test_a_scope_is_its_path_whatever_order_scopes_are_made_in() -> None:
    lazy = Scopes(compiled(Outer))
    permuted = Scopes(compiled(Outer), permuted=True)
    # Made in different orders: the leaf first in one, the candidates first in the other.
    assert lazy.find("left.bits").key == "left.bits"
    assert permuted.find("pick.wide.extra").key == "pick.wide.extra"
    assert [record.path for record in lazy.records()] != [
        record.path for record in permuted.records()
    ]
    for scopes in (lazy, permuted):
        scopes.address_all()
        by_path = {record.path: record.name for record in scopes.records()}
        assert by_path == {
            (): "",
            (0,): "pick.wide",
            (1,): "pick.narrow",
            (2,): "leaf",
            (3,): "left",
        }
        # Interned: a path reaches one record, an address is one object.
        root = scopes.root
        assert root.child(2) is root.reach((2,))
        assert root.address(0) is root.address(0)
        assert scopes.find("leaf.width") is scopes.find("leaf.width")
    assert [a.key for a in lazy.all_addresses()] == [a.key for a in permuted.all_addresses()]


def test_addresses_are_ordered_by_scope_position_then_member() -> None:
    scopes = Scopes(compiled(Outer))
    found = list(scopes.all_addresses())
    assert found == sorted(found)
    assert [address.key for address in sorted(reversed(found))] == [a.key for a in found]
    # A scope's breadth-first position: the root's members, then the candidates' (by their
    # Decision's place), then the children's.
    keys = [address.key for address in found]
    assert keys.index("w") < keys.index("pick.wide.width") < keys.index("leaf.width")
    assert keys.index("pick.narrow.width") < keys.index("leaf.width") < keys.index("left.width")


def test_an_address_equals_and_hashes_by_its_scope_record_and_number() -> None:
    scopes = Scopes(compiled(Outer))
    leaf = scopes.find("leaf.lanes")
    made = Address(leaf.scope, leaf.member)
    assert made == leaf and hash(made) == hash(leaf)
    other = Scopes(compiled(Outer)).find("leaf.lanes")
    assert other != leaf and other.key == leaf.key
    assert scopes.holds_address(leaf) and not scopes.holds_address(other)
    assert not scopes.holds_address(Address(leaf.scope, len(leaf.scope.compiled.records)))


def test_a_formal_bound_by_reference_has_its_own_address_and_reads_its_binding() -> None:
    scopes = Scopes(compiled(Outer))
    width = scopes.find("left.width")
    assert width.scope.name == "left" and width.kind == "alias"
    assert width.output is scopes.find("leaf.bits")
    # Bound by a literal: a constant under its scope's guard; left unsupplied: present.
    held = Scopes(compiled(Holds))
    assert held.find("missing.stream").kind == "present"
    # A choice that never applies where its formal is unsupplied: no candidates there.
    assert held.find("missing.source").kind == "const"
    (source,) = (choice for choice in held.choices() if choice.key == "missing.source")
    assert source.scope is held.root.named_children["missing"]
    assert dict(source.cases) == {"wide": None, "narrow": None}
    (placed,) = (choice for choice in held.choices() if choice.key == "given.source")
    assert [case.name for _, case in placed.cases if case is not None] == [
        "given.source.wide",
        "given.source.narrow",
    ]


@pytest.mark.parametrize("permuted", [False, True])
def test_an_address_existence_guards_are_its_own_then_its_placements(permuted: bool) -> None:
    scopes = Scopes(compiled(Outer), permuted=permuted)
    extra = scopes.find("pick.wide.extra")
    guards = existence(extra)
    assert guards and guards[-1].scope is scopes.root
    assert [guard.key for guard in guards] == [
        guard.key for guard in existence(Scopes(compiled(Outer)).find("pick.wide.extra"))
    ]


READ = ("leaf.bits", "left.bits", "chosen", "pick.wide.extra", "$validity")


@pytest.mark.parametrize("permuted", [False, True])
def test_two_design_spaces_of_one_model_share_its_addresses_and_its_cache(permuted: bool) -> None:
    """Two design spaces of one model, equal in value, built through changes in different
    orders and read in different orders (each making scopes as it reaches them), read the
    same address objects: the model's scope records, one per path. So the model's cache,
    keyed by address, hits across them."""
    model = model_of(compiled(Outer), permuted=permuted)  # a model of its own, fresh
    start = opened(model, {model.scopes.parameters[0]: 3})
    one = start.with_choices({Outer.pick: "wide"}).with_choices({Outer.leaf.lanes: 2})
    other = start.with_choices({Outer.leaf.lanes: 2}).with_choices({Outer.pick: "wide"})
    first, second = held(one).ds, held(other).ds
    assert first == second and first is not second
    session = model.per_class.session()
    with model.lock:
        ev = evaluation(first, session)
        for key in READ:
            ev.get(first.node(key))
        hits = session.counters.hits
        again = evaluation(second, session)
        for key in reversed(READ):
            again.get(second.node(key))
    assert session.counters.hits > hits  # the cache hit across them
    mine = {address.key: address for address in ev.memo}
    theirs = {address.key: address for address in again.memo}
    assert set(theirs) <= set(mine)
    assert all(theirs[key] is mine[key] for key in theirs)
    assert all(first.node(key) is second.node(key) for key in READ)
