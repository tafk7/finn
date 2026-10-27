# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A model's scopes, as records keyed by path, and the addresses of their members.

**A scope** (``ScopeRecord``) is a Space class placed at a path: its class's compiled form
(``_template.Compiled``), its parent, its slot there, its key prefix and the bindings its
placement writes (its parent's slot's). **Its identity is its path**: the slots from the
root, each a slot of the class the one before reaches; the root's is empty. A model's scopes
(``Scopes``, held by the model for its life) intern one record per path, made when first
reached (a slot followed, a key looked up, a child's validity read), under the model's lock;
every design space of the model reaches the same records. Nothing is ordered by when a scope
was made: a scope's place is its breadth-first position (``Position``: its depth, then its
slots' declaration positions), which ``rank`` reads.

**An address** (``Address``) is (scope, member): member ``i`` of the scope's class's
compiled form. Addresses are interned per scope; they are equal, and hash, by the scope
record's identity and the member's number, and are ordered by the scope's position, then
the member's number. An address is **canonical**: what a record of a class stands for at a
placement (``ScopeRecord.canonical``) is
- a member of the class, its own;
- the binding by reference a placing class writes for a child's formal: the child's formal
  (a formal bound by reference has an address of its own, and reads through its binding);
- a selection of a choice's member, made by the choice's own class or by a class above it:
  one address whoever makes it, a member of the choice's scope (``ScopeRecord.selection``):
  the selection the candidates share, where the choice applies; else the class's own, where
  it reads it; else, where only a class above reads it, a member numbered after the class's
  records (``ScopeRecord.selections``, fixed by the compiled forms on its path), reading the
  record of the nearest class above that reads it. Its owner is always the choice's key;
- none, for a selection its candidates would share that their disagreeing types dropped.

**Reading an address** resolves its member in its class's compiled form and maps each of
its edges through the scope when it is read (``ScopeRecord.target``): ``Local`` in the
scope, ``Placed`` in the scope its slots reach, ``Formal`` the scope's own formal,
``Placing`` the guard the scope is placed under (its parent's slot guard), ``Selected``
the selection's address. A formal of a placed scope reads as its placement binds it: a
literal (or the formal's default) is a constant under the scope's guard, a reference the
record the placing class writes for it, its edges mapped in the placing scope. No node is
built per placement: an address reads its class's member record (the one a placement's
shape changes, a formal's literal, is made once per shape and model).

An address's **existence guards** are its own guards in its class, then the placement
guards up its path, innermost first: its ``guard``, then each guard's own, to the root.

Two test modes, read when a model is made: ``FINN_SPACE_PERMUTE=1`` makes scopes in a
permuted order, reversed by path (a scope's children all made when its first one is, the
last slot first; every scope made in reverse breadth-first order where all are made);
``FINN_SPACE_EAGER=1`` makes every scope and reads every member when the model is made.
Both give the same keys and answers.
"""

from __future__ import annotations

import os
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from ._table import Binding, member_key
from .errors import DefinitionError, RequestError
from .ir import PLACING, Edge, Formal, Local, Member, NodeKind, Placed, Placing, Selected
from .semantics import ValueSemantics

if TYPE_CHECKING:
    from ._configuration import _Namespace
    from ._nodes import NodeDecl
    from ._template import Compiled
    from ._whole import Flow, Whole
    from .domains import Domain

_OWNER = "space.structure"

# The test modes (read when a model is made).
PERMUTE = "FINN_SPACE_PERMUTE"
EAGER = "FINN_SPACE_EAGER"

# A scope's place in breadth-first order: its depth, then the declaration positions of its
# slots (each its parent's structure item and 0 for a child node, 1 + its case for a
# candidate).
Position = tuple[int, tuple[tuple[int, int], ...]]
# Above the root: before every scope.
_ABOVE: Position = (-1, ())

_UNSET: Any = object()


def model_modes() -> tuple[bool, bool]:
    """The test modes the environment asks for, read when a model is made: (permuted,
    eager). Public as ``finn.core.space.model_modes``: what keeps a model by its class keeps
    one per mode."""
    return os.environ.get(PERMUTE) == "1", os.environ.get(EAGER) == "1"


class Argument(NamedTuple):
    """A named edge of an address, mapped: the address it reads."""

    name: str
    node: Address


# What reading a canonical address resolves (``ScopeRecord.read``): the member record its
# frame reads, the scope its edges are mapped through, its key and its owner's key where it
# has one of its own.
_Read = tuple[Member, "ScopeRecord", str, "str | None"]


class Address(tuple["ScopeRecord", int]):
    """(scope, member): member ``member`` of ``scope``'s class's compiled form (see the
    module docstring). Equal and hashed by the scope record's identity and the number,
    ordered by the scope's position, then the number. What its frame reads of it is its
    member record's, through its scope: its own fields as stated, each edge mapped to the
    address it reads when it is read."""

    __slots__ = ()

    def __new__(cls, scope: ScopeRecord, member: int) -> Address:
        return tuple.__new__(cls, (scope, member))

    def __repr__(self) -> str:
        return f"Address({self[0].path}, {self[1]})"

    # -- identity

    @property
    def scope(self) -> ScopeRecord:
        """The scope it is a member of (its identity's)."""
        return self[0]

    @property
    def member(self) -> int:
        """Its member's number in its scope's class."""
        return self[1]

    @property
    def held(self) -> Address:
        """The record it reads, as the address of that record where it is written: itself,
        but for a selection only a class above makes, that class's record (its edges mapped
        there)."""
        scope, member = self
        if member < len(scope.compiled.records):
            return self
        return cast(Address, scope.merged(*scope.selections[member - len(scope.compiled.records)]))

    def _read(self) -> _Read:
        scope, member = self
        found = scope._reads[member]
        return found if found is not None else scope.read(member)

    @property
    def record(self) -> Member:
        """The member record its frame reads."""
        return self._read()[0]

    # -- its own fields

    @property
    def key(self) -> str:
        return self._read()[2]

    @property
    def kind(self) -> NodeKind:
        return self._read()[0].kind

    @property
    def semantics(self) -> ValueSemantics[object] | None:
        return self._read()[0].semantics

    @property
    def function(self) -> Callable[..., object] | None:
        return self._read()[0].function

    @property
    def call_style(self) -> str:
        return self._read()[0].call_style

    @property
    def value(self) -> object:
        return self._read()[0].value

    @property
    def required(self) -> bool:
        return self._read()[0].required

    @property
    def domain(self) -> Domain[object] | None:
        return self._read()[0].domain

    @property
    def origin(self) -> str | None:
        return self._read()[0].origin

    @property
    def source_owner(self) -> str | None:
        return self._read()[3]

    @property
    def owner(self) -> str:
        """Authored owner for diagnostics from generated computation nodes."""
        read = self._read()
        return read[3] if read[3] is not None else read[2]

    # -- its edges, mapped through its scope when read

    @property
    def guard(self) -> Address | None:
        member, through, key, _ = self._read()
        edge = member.guard
        return None if edge is None else through.guard_at(edge, key)

    @property
    def output(self) -> Address | None:
        member, through, key, _ = self._read()
        edge = member.output
        return None if edge is None else through.mapped(edge, key)

    @property
    def selector(self) -> Address | None:
        member, through, key, _ = self._read()
        edge = member.selector
        return None if edge is None else through.mapped(edge, key)

    @property
    def arguments(self) -> tuple[Argument, ...]:
        member, through, key, _ = self._read()
        return tuple(Argument(name, through.mapped(edge, key)) for name, edge in member.arguments)

    @property
    def domain_arguments(self) -> tuple[Argument, ...]:
        member, through, key, _ = self._read()
        return tuple(
            Argument(name, through.mapped(edge, key)) for name, edge in member.domain_arguments
        )

    @property
    def constraints(self) -> tuple[Address, ...]:
        member, through, key, _ = self._read()
        if member.kind == "view":
            # A template may be included through Groups: its real owner is known here.
            return (self.scope.owning_space.validity,)
        return tuple(through.mapped(edge, key) for edge in member.constraints)

    @property
    def alternatives(self) -> tuple[tuple[str, Address], ...]:
        member, through, key, _ = self._read()
        return tuple((label, through.mapped(edge, key)) for label, edge in member.alternatives)

    @property
    def selection_index(self) -> Mapping[str, Address] | None:
        """A selection's alternatives by case (a case without the member is absent)."""
        if self.kind != "select":
            return None
        return _SelectionIndex(self)

    @property
    def dependencies(self) -> tuple[Address, ...]:
        """Explicit edges mapped once, plus a View's intrinsic real-owner validity edge.

        That owner is resolved at placement, so Group templates never need their own
        validity node or an artificial dependency on an arbitrary enclosing class.
        """
        member, through, key, _ = self._read()
        found = (
            through.guard_at(edge, key) if edge is member.guard else through.mapped(edge, key)
            for edge in member.dependencies
        )
        dependencies = tuple(address for address in found if address is not None)
        if member.kind == "view":
            dependencies += self.constraints
        return tuple(dict.fromkeys(dependencies))


class _SelectionIndex(Mapping[str, Address]):
    """A selection's alternative by case, each mapped when it is read."""

    __slots__ = ("_address",)

    def __init__(self, address: Address) -> None:
        self._address = address

    def __getitem__(self, case: str) -> Address:
        member, through, key, _ = self._address._read()
        for label, edge in member.alternatives:
            if label == case:
                return through.mapped(edge, key)
        raise KeyError(case)

    def __iter__(self) -> Iterator[str]:
        return iter([label for label, _ in self._address.record.alternatives])

    def __len__(self) -> int:
        return len(self._address.record.alternatives)


def implies(required: Address | None, guard: Address | None) -> bool:
    """Whether ``required`` holds whenever ``guard`` does: it is ``guard`` or an outer
    guard."""
    if required is None:
        return True
    current = guard
    while current is not None:
        if current == required:
            return True
        current = current.guard
    return False


def existence(address: Address) -> tuple[Address, ...]:
    """``address``'s existence guards, innermost first: its own guard, and each guard that
    one is under, to the root (its own guards in its class, then the placement guards up its
    path)."""
    found: list[Address] = []
    current = address.guard
    while current is not None:
        found.append(current)
        current = current.guard
    return tuple(found)


class _Registry:
    """What a model's scope records share: the model's lock, the mode, the records by path,
    the selections' addresses, and the member records made per shape. It holds
    nothing of the model's ``Scopes``, so that an address kept anywhere (a cache's answer)
    keeps the model's scope records, never the object the model's tables are keyed by."""

    __slots__ = ("lock", "permuted", "records", "merged", "made", "written_keys", "evaluated")

    def __init__(self, lock: RLock, permuted: bool) -> None:
        self.lock = lock
        self.permuted = permuted
        self.records: dict[tuple[int, ...], ScopeRecord] = {}
        # Every address an evaluation of the model evaluated (``Scopes.counts``).
        self.evaluated: set[Address] = set()
        self.merged: dict[tuple[ScopeRecord, int, str], Address | None] = {}
        self.made: dict[tuple[object, ...], Member] = {}
        self.written_keys: dict[int, Mapping[str, int]] = {}

    def member(self, key: tuple[object, ...], make: Callable[[], Member]) -> Member:
        """The member record of one shape (made once per model)."""
        found = self.made.get(key)
        if found is None:
            found = self.made[key] = make()
        return found


class ScopeRecord:
    """A Space class placed at a path (see the module docstring)."""

    __slots__ = (
        "registry",
        "path",
        "parent",
        "slot",
        "name",
        "compiled",
        "position",
        "bindings",
        "_children",
        "_addresses",
        "_reads",
        "_guard",
        "_view",
        "_choices",
        "selections",
        "_numbers",
    )

    def __init__(
        self,
        registry: _Registry,
        compiled: Compiled,
        parent: ScopeRecord | None,
        slot: int | None,
    ) -> None:
        self.registry = registry
        self.parent = parent
        self.slot = slot
        self.compiled = compiled
        bindings: Mapping[str, Binding]
        if parent is None:
            self.path: tuple[int, ...] = ()
            self.name = ""
            self.position: Position = (0, ())
            bindings = MappingProxyType({})
        else:
            held = parent.compiled.slots[cast(int, slot)]
            self.path = (*parent.path, cast(int, slot))
            self.name = member_key(parent.name, held.name)
            depth, path = parent.position
            self.position = (depth + 1, (*path, held.position))
            bindings = held.bindings
        self.bindings = bindings
        # The selections the classes above may make in its class's choices, (choice, member)
        # in order: each a member of the scope numbered after its class's records, the same
        # whatever has been made or read (fixed by the compiled forms on its path).
        found: set[tuple[int, str]] = set()
        if parent is not None:
            steps: tuple[int, ...] = (cast(int, slot),)
            above: ScopeRecord | None = parent
            while above is not None:
                found.update(
                    (choice, member)
                    for (path, choice, member) in above.compiled.chosen
                    if path == steps
                )
                if above.parent is not None:
                    steps = (cast(int, above.slot), *steps)
                above = above.parent
        self.selections: tuple[tuple[int, str], ...] = tuple(sorted(found))
        self._numbers = {
            pair: len(compiled.records) + number for number, pair in enumerate(self.selections)
        }
        self._children: dict[int, ScopeRecord] = {}
        self._addresses: dict[int, Address] = {}
        self._reads: list[_Read | None] = [None] * (len(compiled.records) + len(found))
        self._guard: Address | None = _UNSET
        self._view: tuple[Any, ...] | None = None
        self._choices: dict[int, ChoiceRecord] = {}

    def __repr__(self) -> str:
        return f"ScopeRecord({self.name or '<root>'!r}, {self.path})"

    def __lt__(self, other: ScopeRecord) -> bool:
        return self.position < other.position

    def __gt__(self, other: ScopeRecord) -> bool:
        return self.position > other.position

    @property
    def space_type(self) -> type[_Namespace]:
        return self.compiled.space_type

    def within(self, other: ScopeRecord) -> bool:
        """Whether this scope is ``other`` or placed below it (``other``'s path a prefix of
        its own)."""
        path = other.path
        return self.path[: len(path)] == path

    def binding(self, formal: str) -> Binding | None:
        """How its placement binds one of its formals (None for the root)."""
        if self.parent is None:
            return None
        return self.bindings[formal]

    # -- the scopes below

    def child(self, slot: int) -> ScopeRecord:
        """The scope at slot ``slot`` of its class, made if needed."""
        found = self._children.get(slot)
        if found is not None:
            return found
        registry = self.registry
        with registry.lock:
            found = self._children.get(slot)
            if found is not None:
                return found
            if registry.permuted:
                # The permuted order: every child made with the first, the last slot first.
                for each in reversed(range(len(self.compiled.slots))):
                    if each not in self._children:
                        self._make(each)
                return self._children[slot]
            return self._make(slot)

    def _make(self, slot: int) -> ScopeRecord:
        path = (*self.path, slot)
        made = self.registry.records.get(path)
        if made is None:
            made = ScopeRecord(self.registry, self.compiled.slots[slot].compiled, self, slot)
            self.registry.records[path] = made
        self._children[slot] = made
        return made

    def reach(self, slots: Sequence[int]) -> ScopeRecord:
        """The scope its slots ``slots`` reach."""
        current = self
        for slot in slots:
            current = current.child(slot)
        return current

    @property
    def guard(self) -> Address | None:
        """The guard it is placed under (its parent's slot guard; None at the root)."""
        if self._guard is _UNSET:
            with self.registry.lock:
                if self._guard is _UNSET:
                    parent = self.parent
                    self._guard = (
                        None
                        if parent is None
                        else parent.guard_at(
                            parent.compiled.slots[cast(int, self.slot)].guard, self.name
                        )
                    )
        return self._guard

    # -- addresses

    def address(self, member: int) -> Address:
        """The address of member ``member`` here (interned; whether it is canonical is
        ``canonical``'s question)."""
        found = self._addresses.get(member)
        if found is None:
            found = self._addresses.setdefault(member, Address(self, member))
        return found

    def canonical(self, member: int) -> Address | None:
        """What record ``member`` of its class stands for here (see the module docstring);
        None for a selection nothing reads."""
        compiled = self.compiled
        if member >= len(compiled.records):  # a selection a class above makes
            return self.selection(*self.selections[member - len(compiled.records)])
        if member in compiled.written:
            bound = compiled.binds.get(member)
            if bound is not None:  # a binding by reference: the child's formal
                child = self.child(bound[0])
                return child.address(child.compiled.formals[bound[1]])
            path, choice, name = compiled.chooses[member]
            return self.reach(path).selection(choice, name)
        selection = compiled.selections.get(member)
        if selection is not None:
            choice = selection[0]
            if member in compiled.choices[choice].shared:
                return self.address(member)  # its candidates' (kept, if dropped, unread)
            return self.selection(*selection)
        return self.address(member)

    def member_address(self, ref: int | Placing) -> Address | None:
        """What a walk's find stands for here: a record of its class, or the guard it is
        placed under."""
        if isinstance(ref, Placing):
            return self.guard
        return self.canonical(ref)

    def selection(self, choice: int, member: str) -> Address | None:
        """The one address of ``decision.member`` through its choice ``choice``, a member of
        this scope (the choice's): the selection its candidates share; else its class's,
        where it reads it; else, where only a class above reads it, the member it is
        numbered here (``selections``), reading the record of the nearest class above that
        reads it. None where nothing reads it."""
        held = self.merged(choice, member)
        if held is None or held[0] is self:
            return held
        return self.address(self._numbers[(choice, member)])

    def merged(self, choice: int, member: str) -> Address | None:
        """The record ``selection``'s address reads, as its address where it is written
        (this class's, or the nearest class above's that reads it), found once."""
        registry = self.registry
        key = (self, choice, member)
        if key in registry.merged:
            return registry.merged[key]
        with registry.lock:
            if key not in registry.merged:
                registry.merged[key] = self._merge(choice, member)
            return registry.merged[key]

    def _merge(self, choice: int, member: str) -> Address | None:
        compiled = self.compiled
        spec = compiled.choices[choice]
        own = compiled.own_selection.get((choice, member))
        if own is not None:
            if own in spec.shared:
                # A selection its candidates share, unless dropped for disagreeing types.
                return self.address(own) if spec.members.get(member) == own else None
            return self.address(own)  # its class reads it
        # The classes above: each that reaches the scope through its slots.
        path: tuple[int, ...] = (cast(int, self.slot),)
        above = self.parent
        while above is not None:
            record = above.compiled.chosen.get((path, choice, member))
            if record is not None:
                return above.address(record)
            if above.parent is None:
                break
            path = (cast(int, above.slot), *path)
            above = above.parent
        return None

    def target(self, edge: Edge) -> Address | None:
        """The address ``edge`` of its class reads here (None for a selection nothing
        reads)."""
        kind = type(edge)
        if kind is Local:
            return self.canonical(cast(Local, edge).member)
        if kind is Placed:
            placed = cast(Placed, edge)
            return self.reach(placed.path).canonical(placed.member)
        if kind is Formal:
            return self.address(self.compiled.formals[cast(Formal, edge).name])
        if kind is Placing:
            return self.guard
        selected = cast(Selected, edge)
        written = self.compiled.chosen.get((selected.path, selected.choice, selected.member))
        if written is not None:  # a selection this class makes in a placed Space's choice
            return self.canonical(written)
        scope = self.reach(selected.path)
        own = scope.compiled.own_selection.get((selected.choice, selected.member))
        if own is None:
            return scope.selection(selected.choice, selected.member)
        return scope.canonical(own)

    def mapped(self, edge: Edge, owner: str) -> Address:
        """``target``, where the edge must read something here: a reference to what the
        placement does not place is refused (``reference-not-placed``)."""
        found = self.target(edge)
        if found is None:
            raise DefinitionError.broken(
                "reference-not-placed",
                _OWNER,
                f"{owner}: {self._key_of(edge)} is not placed",
                declaration=owner,
            )
        return found

    def guard_at(self, edge: Edge, owner: str) -> Address | None:
        """A guard edge here: the guard the scope is placed under (None at the root), or
        what ``mapped`` maps."""
        if type(edge) is Placing:
            return self.guard
        return self.mapped(edge, owner)

    def _key_of(self, edge: Edge) -> str:
        compiled = self.compiled
        if isinstance(edge, Local):
            return member_key(self.name, compiled.records[edge.member].key)
        if isinstance(edge, Placed):
            return member_key(self.name, compiled.key_of(edge))
        if isinstance(edge, Selected):
            reached = compiled.reached(edge.path)
            names = [compiled.slots[edge.path[0]].name] if edge.path else []
            return member_key(
                member_key(self.name, ".".join(names)),
                member_key(reached.choices[edge.choice].key, edge.member),
            )
        return self.name

    # -- keys

    def find(self, key: str) -> Address:
        """The address of a key in its model (from the root, whatever scope asks): matched
        against each class's slots' names (which hold dots: ``compute.packed``), longest
        first, scope by scope; then against the class's own records' keys. Never by
        splitting the key on its dots alone."""
        registry = self.registry
        with registry.lock:
            found = _find(registry, key)
            if found is None:
                raise KeyError(key)
            return found

    # -- reading a member

    def read(self, member: int) -> _Read:
        """What reading the (canonical) address of ``member`` here resolves: its member
        record, the scope its edges are mapped through, its key and its owner's key."""
        found = self._reads[member]
        if found is not None:
            return found
        with self.registry.lock:
            found = self._reads[member]
            if found is None:
                found = self._reads[member] = self._resolve(member)
            return found

    def _resolve(self, index: int) -> _Read:
        compiled = self.compiled
        registry = self.registry
        if index >= len(compiled.records):
            # A selection only a class above makes: its record there, read there.
            held = cast(Address, self.merged(*self.selections[index - len(compiled.records)]))
            return held[0].read(held[1])
        member = compiled.records[index]
        key = member_key(self.name, member.key)
        formal = compiled.formal_names.get(index)
        if formal is not None and self.parent is not None:
            parent = self.parent
            binding = self.bindings[formal]
            if binding.kind == "literal":
                made = registry.member(
                    ("literal", id(parent.compiled), self.slot, formal),
                    lambda: Member(
                        index,
                        member.key,
                        "const",
                        member.semantics,
                        guard=PLACING,
                        value=binding.value,
                        origin=member.origin,
                    ),
                )
                return made, self, key, None
            # A reference: the record the placing class writes for it, read there.
            written = parent.compiled.records[cast(int, binding.node)]
            return written, parent, key, _owned(parent, written)
        owner = _owned(self, member)
        if index == compiled.validity:
            pieces = self._pieces(member)
            if pieces != member.constraints:
                member = registry.member(
                    ("validity", id(compiled), pieces),
                    lambda: replace(member, constraints=pieces),
                )
        return member, self, key, owner

    @staticmethod
    def _pieces(validity: Member) -> tuple[Edge, ...]:
        """Its validity's pieces in today's order: its constraints and decisions in member
        order, then each placed Space's validity in placement order. A formal is never a
        piece (bound to a fresh Decision, it was)."""
        return tuple(edge for edge in validity.constraints if not isinstance(edge, Formal))

    # -- what the scope's class declares, here

    def _declared(self) -> tuple[Any, ...]:
        found = self._view
        if found is None:
            with self.registry.lock:
                found = self._view
                if found is None:
                    found = self._view = self._make_view()
        return found

    def _make_view(self) -> tuple[Any, ...]:
        compiled = self.compiled
        members: dict[object, Address] = {}
        named: dict[str, Address] = {}
        for declaration, index in compiled.members.items():
            address = self.canonical(index)
            if address is not None:
                members[declaration] = address
        for name, index in compiled.named_members.items():
            address = self.canonical(index)
            if address is not None:
                named[name] = address
        children = {
            declaration: self.child(slot) for declaration, slot in compiled.children.items()
        }
        named_children = {name: self.child(slot) for name, slot in compiled.named_children.items()}
        choices = {
            declaration: self.choice(index) for declaration, index in compiled.choice_index.items()
        }
        return (
            MappingProxyType(members),
            MappingProxyType(named),
            MappingProxyType(children),
            MappingProxyType(named_children),
            MappingProxyType(choices),
        )

    @property
    def members(self) -> Mapping[object, Address]:
        """Its class's members by declaration, each's address here."""
        return cast("Mapping[object, Address]", self._declared()[0])

    @property
    def named_members(self) -> Mapping[str, Address]:
        return cast("Mapping[str, Address]", self._declared()[1])

    @property
    def children(self) -> Mapping[object, ScopeRecord]:
        """The scopes of its class's child nodes, by declaration."""
        return cast("Mapping[object, ScopeRecord]", self._declared()[2])

    @property
    def named_children(self) -> Mapping[str, ScopeRecord]:
        return cast("Mapping[str, ScopeRecord]", self._declared()[3])

    @property
    def choices(self) -> Mapping[object, ChoiceRecord]:
        """Its class's Decisions over nodes, by declaration."""
        return cast("Mapping[object, ChoiceRecord]", self._declared()[4])

    @property
    def record(self) -> NodeDecl | None:
        """The node declaration placed here (None for the root)."""
        if self.parent is None:
            return None
        return self.parent.compiled.slots[cast(int, self.slot)].record

    @property
    def owning_space(self) -> ScopeRecord:
        """Nearest real Space; a Group is only a namespace in this same model."""
        current = self
        while current.compiled.validity is None:
            assert current.parent is not None
            current = current.parent
        return current

    @property
    def validity(self) -> Address:
        """The address answering its Space's validity."""
        if self.compiled.validity is None:
            raise RequestError("a Group has no independent validity; query its owning Space")
        return self.address(self.compiled.validity)

    def choice(self, index: int) -> ChoiceRecord:
        """Its class's Decision over nodes ``index``, here."""
        found = self._choices.get(index)
        if found is None:
            with self.registry.lock:
                found = self._choices.get(index)
                if found is None:
                    found = self._choices[index] = ChoiceRecord(self, index)
        return found


def _owned(record: ScopeRecord, member: Member) -> str | None:
    return None if member.source_owner is None else member_key(record.name, member.source_owner)


class ChoiceRecord:
    """A Decision over nodes of a scope's class, at that scope: its key, its selector's
    address, its candidates' scopes (None where a case places nothing), the selections of its
    members made here or above (``decision.member``), and its guard."""

    __slots__ = ("scope", "index", "key", "selector", "cases", "members", "guard")

    def __init__(self, scope: ScopeRecord, index: int) -> None:
        compiled = scope.compiled
        spec = compiled.choices[index]
        names = dict.fromkeys(
            member
            for record, member in spec.selects.items()
            if compiled.selections.get(record) == (index, member)
        )
        path: tuple[int, ...] = (cast(int, scope.slot),)
        above = scope.parent
        while above is not None:
            names.update(
                dict.fromkeys(
                    member
                    for (slots, choice, member) in above.compiled.chosen
                    if slots == path and choice == index
                )
            )
            if above.parent is None:
                break
            path = (cast(int, above.slot), *path)
            above = above.parent
        members: dict[str, Address] = {}
        for member in names:
            found = scope.selection(index, member)
            if found is not None:
                members[member] = found
        cases: list[tuple[str, ScopeRecord | None]] = []
        for case, slot in spec.cases:
            child = None if slot is None else scope.child(slot)
            cases.append((case, child))
        self.scope = scope
        self.index = index
        self.key = member_key(scope.name, spec.key)
        self.selector = scope.address(spec.selector)
        self.cases: tuple[tuple[str, ScopeRecord | None], ...] = tuple(cases)
        self.members: Mapping[str, Address] = MappingProxyType(members)
        self.guard: Address | None = scope.guard_at(spec.guard, self.key)

    def __repr__(self) -> str:
        return f"ChoiceRecord({self.key!r})"


class Scopes:
    """A model's scope records, keyed by path, and the questions about its whole design
    space (see the module docstring). The model holds it for its life, and it holds the
    model's tables: the running type's per-model record (``record``) and each node cache's
    table (``tables``, keyed weakly by the cache). They live exactly as long as the model (a
    dropped model's go at the next full collection: the model, its classes and its tables
    hold one another): what they hold (addresses) reaches the model through its classes, so
    no table keyed weakly by the model elsewhere could ever drop."""

    __slots__ = (
        "compiled",
        "registry",
        "root",
        "parameters",
        "keys",
        "record",
        "tables",
        "_decisions",
        "_whole",
        "__weakref__",
    )

    def __init__(
        self,
        compiled: Compiled,
        lock: RLock | None = None,
        *,
        permuted: bool = False,
        eager: bool = False,
    ) -> None:
        self.compiled = compiled
        self.registry = _Registry(RLock() if lock is None else lock, permuted)
        self._decisions: tuple[Address, ...] | None = None
        self._whole: Whole | None = None
        self.keys = _Keys(self)
        # The running type's per-model record (``_running.structure.Record``), and each
        # node cache's table for this model, keyed weakly by the cache
        # (``_running.cache.NodeCache``): a cache gone takes its tables with it.
        self.record: Any = None
        self.tables: weakref.WeakKeyDictionary[object, Any] = weakref.WeakKeyDictionary()
        with self.lock:
            self.root = ScopeRecord(self.registry, compiled, None, None)
            self.registry.records[()] = self.root
            self.parameters: tuple[Address, ...] = tuple(
                self.root.address(index) for index in compiled.formals.values()
            )
            if eager:
                self.read_all()

    @property
    def lock(self) -> RLock:
        return self.registry.lock

    @property
    def permuted(self) -> bool:
        """Whether its scopes are made in the permuted order (``PERMUTE``)."""
        return self.registry.permuted

    # -- scopes

    def holds(self, scope: object) -> bool:
        """Whether ``scope`` is a scope record of this model."""
        return (
            isinstance(scope, ScopeRecord)
            and scope.registry is self.registry
            and self.registry.records.get(scope.path) is scope
        )

    def holds_address(self, address: object) -> bool:
        """Whether ``address`` is a canonical address of this model: of one of its scopes,
        a member of that scope's class whose address it is (never one standing for another,
        or excluded by its placement)."""
        if not isinstance(address, Address) or not self.holds(address[0]):
            return False
        scope, member = address
        if type(member) is not int or not 0 <= member < len(scope._reads):
            return False
        return scope.canonical(member) == address

    def records(self) -> tuple[ScopeRecord, ...]:
        """The scopes made so far, in the order they were made."""
        with self.lock:
            return tuple(self.registry.records.values())

    def address_all(self) -> None:
        """Make every scope (in the permuted mode, in reverse breadth-first order)."""
        with self.lock:
            pending = deque([self.root])
            while pending:
                record = pending.popleft()
                slots = range(len(record.compiled.slots))
                for slot in reversed(slots) if self.registry.permuted else slots:
                    pending.append(record.child(slot))

    def ordered(self) -> list[ScopeRecord]:
        """Every scope, by position (each made)."""
        with self.lock:
            self.address_all()
            return sorted(self.registry.records.values())

    def all_addresses(self) -> Iterator[Address]:
        """Every address of the model: every scope by position, each member in order."""
        with self.lock:
            found: list[Address] = []
            for record in self.ordered():
                for member in range(len(record._reads)):
                    address = record.canonical(member)
                    if address is not None and address[0] is record and address[1] == member:
                        found.append(address)
        return iter(found)

    def read_all(self) -> None:
        """Make every scope and read every member: each record and its edges, each scope's
        declarations and choices (the eager mode)."""
        with self.lock:
            for address in tuple(self.all_addresses()):
                address.dependencies  # noqa: B018 - read, every edge mapped
            for record in self.ordered():
                record._declared()
                for index in range(len(record.compiled.choices)):
                    record.choice(index)

    def subtree(self, scope: ScopeRecord) -> frozenset[ScopeRecord]:
        """``scope`` and every scope below it (every scope made)."""
        with self.lock:
            self.address_all()
            return frozenset(
                record for record in self.registry.records.values() if record.within(scope)
            )

    def choices(self) -> tuple[ChoiceRecord, ...]:
        """Every Decision over nodes of the model, at its scope (every scope made)."""
        return tuple(
            record.choice(index)
            for record in self.ordered()
            for index in range(len(record.compiled.choices))
        )

    def counts(self) -> dict[str, int]:
        """What is made and evaluated so far: scopes made, scopes with a member evaluated
        (in play), members read (their records resolved: key, kind, edges), members
        evaluated, and how many of those are constants."""
        with self.lock:
            records = tuple(self.registry.records.values())
            evaluated = self.registry.evaluated
            return {
                "scopes made": len(records),
                "scopes in play": len({address.scope for address in evaluated}),
                "members read": sum(
                    1 for record in records for read in record._reads if read is not None
                ),
                "members evaluated": len(evaluated),
                "constants evaluated": sum(1 for address in evaluated if address.kind == "const"),
            }

    def in_play(self) -> frozenset[ScopeRecord]:
        """The scopes with a member evaluated so far."""
        with self.lock:
            return frozenset(address.scope for address in self.registry.evaluated)

    # -- keys

    def find(self, key: str) -> Address:
        """The address of a key (``ScopeRecord.find``)."""
        return self.root.find(key)

    # -- the whole design space

    def selector_choice(self, address: Address) -> ChoiceRecord | None:
        """The Decision over nodes whose selector ``address`` is."""
        scope, member = address
        choice = scope.compiled.selectors.get(member)
        return None if choice is None else scope.choice(choice)

    def case_of(self, choice: ChoiceRecord, guard: Address) -> tuple[Address, str] | None:
        """The selector and case of ``choice`` where ``guard`` holds, if it is the guard a
        case's scope is placed under (directly or under its ``when``) that reads the
        selector alone."""
        with self.lock:
            scope = choice.scope
            spec = scope.compiled.choices[choice.index]
            selector = scope.address(spec.selector)
            selected = (Argument("selected", selector),)
            for case, slot in spec.cases:
                child = None if slot is None else scope.child(slot)
                current = None if child is None else child.guard
                while current is not None:
                    if current.kind == "derived" and current.arguments == selected:
                        if current == guard:
                            return selector, case
                        break
                    current = current.guard
            return None

    def rank(self, address: Address) -> tuple[object, ...]:
        """A decision's place in declaration order, from its scope's path: a scope's own
        decisions in member order when the scope is placed, a selector when its scope's
        members are placed (its candidates after it); scopes breadth first
        (``Position``)."""
        record, member = address
        compiled = record.compiled
        choice = compiled.selectors.get(member)
        if choice is not None:
            return (record.position, compiled.choice_positions[choice], 0, -1)
        position = compiled.value_positions.get(member)
        if position is None:
            raise KeyError(address)
        parent = record.parent
        if parent is None:
            return (_ABOVE, 0, 0, position)
        return (parent.position, *parent.compiled.slots[cast(int, record.slot)].position, position)

    @property
    def decisions(self) -> tuple[Address, ...]:
        """Every decision in declaration order (``rank``): every scope made, no member
        evaluated."""
        with self.lock:
            if self._decisions is None:
                found: list[Address] = []
                for record in self.ordered():
                    for member, spec in enumerate(record.compiled.records):
                        if spec.kind not in ("decision", "param"):
                            continue
                        address = record.canonical(member)
                        if address is None or address[0] is not record or address[1] != member:
                            continue
                        if address.kind == "decision":
                            found.append(address)
                self._decisions = tuple(sorted(found, key=self.rank))
            return self._decisions

    @property
    def whole(self) -> Whole:
        """Its answers to the questions about the whole design space (``_whole``)."""
        if self._whole is None:
            from ._whole import Whole  # noqa: PLC0415 - a cycle

            with self.lock:
                if self._whole is None:
                    self._whole = Whole(self)
        return self._whole

    def flow(self) -> Flow:
        """Its choices in the flow order, with its loops (``_whole.Whole.flow``)."""
        return self.whole.flow()

    def readers(self, source: Address) -> tuple[Address, ...]:
        """The choices whose presence may read ``source`` (``_whole.Whole.readers``)."""
        return self.whole.readers(source)

    def position(self, decision: Address) -> int:
        """A choice's place in the flow order (``KeyError`` for anything else)."""
        return self.whole.position(decision)

    # -- statistics

    def statistics(self) -> tuple[int, int, int, int, int, int, int]:
        """``inspection.statistics``' counts: authored declarations, scopes, members,
        potential edges, Decisions over nodes, owning decisions. Each class is counted at
        every placement: once per **shape** of placement (its class, whether it is placed
        under a guard), on one scope of that shape, times the placements of that shape. A
        formal is counted where its binding is written, by its placing class. A member's
        potential edges are the addresses its edges read there, each once."""
        with self.lock:
            totals: dict[tuple[object, ...], tuple[int, ...]] = {}
            return cast(tuple[int, int, int, int, int, int, int], self._counted(self.root, totals))

    def _counted(
        self, record: ScopeRecord, totals: dict[tuple[object, ...], tuple[int, ...]]
    ) -> tuple[int, ...]:
        shape = (id(record.compiled), record.guard is not None)
        held = totals.get(shape)
        if held is not None:
            return held
        compiled = record.compiled
        authored = sum(
            1 for index in compiled.named_members.values() if record.canonical(index) is not None
        )
        nodes = edges = decisions = 0
        for member in range(len(compiled.records)):
            address = record.canonical(member)
            if address is None or address.held != record.address(member):
                continue  # counted where its record is written (a selection made here too)
            nodes += 1
            if member in compiled.formal_names and record.parent is not None:
                continue  # counted by its placing class, where its binding is written
            edges += _edges(address)
            if address.kind == "decision":
                decisions += 1
        found = [
            authored,
            int(compiled.validity is not None),
            nodes,
            edges,
            len(compiled.choices),
            decisions,
            int(compiled.validity is None),
        ]
        for slot in range(len(compiled.slots)):
            child = record.child(slot)
            found[0] += 1  # the placement
            for index in child.compiled.formals.values():
                address = child.address(index)
                found[3] += _edges(address)
                found[5] += address.kind == "decision"
            for position, count in enumerate(self._counted(child, totals)):
                found[position] += count
        result = totals[shape] = tuple(found)
        return result


def _find(registry: _Registry, key: str) -> Address | None:
    # A depth-first search over the slots a key's prefix may name (longest first), by hand:
    # a chain of placements may be deep. Each entry: a scope, the rest of the key there, and
    # the prefixes left to try (longest first).
    root = registry.records[()]
    stack = [(root, key, _prefixes(root, key))]
    while stack:
        at, rest, prefixes = stack[-1]
        prefix = next(prefixes, None)
        if prefix is not None:
            slot, tail = prefix
            child = at.child(slot)
            stack.append((child, tail, _prefixes(child, tail)))
            continue
        stack.pop()
        member = at.compiled.keys.get(rest)
        if member is None:
            member = _written_keys(registry, at.compiled).get(rest)
        found = None if member is None else at.canonical(member)
        if found is not None:
            return found
    return None


def _written_keys(registry: _Registry, compiled: Compiled) -> Mapping[str, int]:
    """A class's written records by key (a selection it makes in a placed Space's choice, a
    formal's binding)."""
    found = registry.written_keys.get(id(compiled))
    if found is None:
        found = registry.written_keys[id(compiled)] = {
            compiled.records[record].key: record for record in compiled.written
        }
    return found


def _edges(address: Address) -> int:
    """How many addresses a member's edges read (each once)."""
    member, through, _, _ = address._read()
    targets = {found for edge in member.dependencies if (found := through.target(edge))}
    if member.kind == "view":
        targets.add(address.scope.owning_space.validity)
    return len(targets)


def _prefixes(record: ScopeRecord, rest: str) -> Iterator[tuple[int, str]]:
    """The slots of ``record``'s class whose name a prefix of ``rest`` is (ending at a dot),
    longest first, with the rest of the key after each."""
    names = record.compiled.slot_names
    end = len(rest)
    while True:
        end = rest.rfind(".", 0, end)
        if end < 0:
            return
        slot = names.get(rest[:end])
        if slot is not None:
            yield slot, rest[end + 1 :]


class _Keys(Mapping[str, Address]):
    """``keys[key]``: the address of a key (``Scopes.find``); iterating reads every
    address."""

    __slots__ = ("_scopes",)

    def __init__(self, scopes: Scopes) -> None:
        self._scopes = scopes

    def __getitem__(self, key: object) -> Address:
        if not isinstance(key, str):
            raise KeyError(key)
        return self._scopes.find(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._scopes.lock:
            return _find(self._scopes.registry, key) is not None

    def __iter__(self) -> Iterator[str]:
        return iter([address.key for address in self._scopes.all_addresses()])

    def __len__(self) -> int:
        return sum(1 for _ in self._scopes.all_addresses())


__all__ = [
    "EAGER",
    "PERMUTE",
    "Address",
    "Argument",
    "ChoiceRecord",
    "ScopeRecord",
    "Scopes",
    "existence",
    "implies",
    "model_modes",
]
