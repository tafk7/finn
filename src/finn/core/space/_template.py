# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Each Space class's compiled form, compiled once and cached on the class.

The front end compiles a Space class into its compiled form (``Compiled``, ``compiled(C)``),
and nothing else: there is no table of a whole design space. ``link_template(C)`` runs the
linker's phases on one table (``_table``): C's own scope is allocated, lowered and checked,
and each Space C places (a child node, a candidate of a Decision over nodes) is a **slot**:
the placed class's compiled form (compiled first) and what C's body writes for it, nothing
of the child's own. A formal binding by reference is a record of C's, lowered in C; a
literal or a default is a value on the slot. C's own formals are its ``param`` records:
where C is placed, the placing class's slot binds each, so a formal has one identity, its
scope and its formal record, whatever binds it.

C's members are numbered within the class, each reference an edge (``ir.Edge``: a member of
C; a member of a class C places, through its slots; C's own formal; the guard C is placed
under; a selection of a choice's member, in the choice's own scope). No record stands for a
member of a placed class, for a scope C places, or for the guard C is placed under; what C
writes for a Space it places (a formal's binding by reference, a selection made in that
Space's choice) is a record of C's, addressed by its slot. Each Space C places is a slot
(``CompiledSlot``): its class's compiled form, its bindings and guard.

What crosses a slot is checked per class and composed (``_compose``), on the compiled
forms: cycles through the Spaces it places and the reads its placements do not place, as
it is compiled; the domain rule, from the root's compiled form, when a model is made
(``compiler.compile_model``). Read inference (``_reads``) and the questions about the whole
design space (``_whole``) walk the compiled forms too.

A model places compiled forms as scopes (``_scopes``: scope records keyed by path, made when
reached), each member read by address through its scope.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from . import _compose
from ._allocate import allocate
from ._check import check
from ._configuration import _Namespace
from ._lower import lower
from ._nodes import NodeDecision, NodeDecl, slot_declaration
from ._table import Binding, Table, member_key
from .errors import DefinitionError
from .ir import PLACING, Edge, Formal, Local, Member, NodeRef, Placed, Placing, Selected

_OWNER = "space.template"


def _frozen(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


# -- the compiled form ----------------------------------------------------------------------------

# What a walk over a compiled form reads an edge as (``Compiled.ref``): a record of the class
# (its number), the guard it is placed under, or a member of a class it places (``Placed``),
# walked there.
Ref: TypeAlias = "int | Placing | Placed"


@dataclass(frozen=True, eq=False, slots=True)
class CompiledSlot:
    """A Space a class places (a child node, or a candidate of a Decision over nodes), slot
    ``index`` of its compiled form: its local name, its class's compiled form, the node
    declaration placed, the guard it is placed under (an edge of the placing class), its
    candidacy (choice, case), its formals' bindings (``Binding``: a literal or a default as a
    value; a reference by the record the placing class writes for it, ``node``), and its
    declaration position (``position``: its structure item, 0 for a child node or 1 + its
    case for a candidate)."""

    index: int
    name: str
    compiled: Compiled = field(repr=False)
    record: NodeDecl | None
    guard: Edge
    candidate: tuple[int, str] | None
    bindings: Mapping[str, Binding]
    position: tuple[int, int]

    @property
    def space_type(self) -> type[_Namespace]:
        return self.compiled.space_type


@dataclass(frozen=True, slots=True)
class CompiledChoice:
    """An own Decision over nodes: its key, its selector (a record), its guard, its cases
    (each a slot, or None for a case that places nothing), its selections of a member
    (records), every selection made for it (record -> member) and those its candidates
    share, and the shared members dropped for their candidates' disagreeing types."""

    index: int
    key: str
    selector: int
    guard: Edge
    cases: tuple[tuple[str, int | None], ...]
    members: Mapping[str, int]
    selects: Mapping[int, str]
    shared: frozenset[int]
    dropped: frozenset[str]


@dataclass(frozen=True, eq=False, slots=True)
class Compiled:
    """A Space class's compiled form: computed once, and cached on the class (``compiled``).

    ``records``: its members, numbered within the class, and between them, in the order they
    were made, the records it writes for the Spaces it places: a formal's binding
    by reference (``bound``, by slot and formal) and a selection it makes in a placed Space's
    choice (``chosen``, by the selection's slots, choice and member); each written record's
    slot is ``written``'s. Every reference is an edge (``ir.Edge``): no record stands for a
    member of another class, for a scope it places, or for the guard it is placed under. A
    walk reads each edge as ``ref`` says. ``order``: its records in dependency order.

    ``binds`` and ``chooses`` read each written record back: what it is written for.

    ``slots``: the Spaces it places, in placement order (structure order, candidates in case
    order). Its own maps, by record and slot: ``formals``, ``members``,
    ``children``, ``choices``, ``selections`` (each own selection's (choice, member))."""

    space_type: type[_Namespace]
    records: tuple[Member, ...]
    slots: tuple[CompiledSlot, ...]
    choices: tuple[CompiledChoice, ...]
    validity: int | None
    pieces: tuple[Edge, ...]
    value_members: tuple[int, ...]
    formals: Mapping[str, int]
    formal_names: Mapping[int, str]
    selections: Mapping[int, tuple[int, str]]
    own_selection: Mapping[tuple[int, str], int]
    selectors: Mapping[int, int]
    bound: Mapping[tuple[int, str], int]
    chosen: Mapping[tuple[tuple[int, ...], int, str], int]
    written: Mapping[int, int]
    # Each written record back to what it is written for: a binding's (slot, formal), a
    # selection's (slots, choice, member).
    binds: Mapping[int, tuple[int, str]]
    chooses: Mapping[int, tuple[tuple[int, ...], int, str]]
    members: Mapping[object, int]
    named_members: Mapping[str, int]
    children: Mapping[object, int]
    named_children: Mapping[str, int]
    choice_index: Mapping[object, int]
    slot_names: Mapping[str, int]
    choice_positions: Mapping[int, int]
    value_positions: Mapping[int, int]
    structure: tuple[tuple[str, int], ...]
    keys: Mapping[str, int]
    refs: Mapping[Edge, Ref]
    order: tuple[int, ...]
    # What the checks composed across slots found of it, per placement (``_compose``,
    # ``_whole``).
    composed: dict[str, object] = field(default_factory=dict, repr=False)

    def ref(self, edge: Edge) -> Ref:
        """What a walk over this class reads ``edge`` as: a record of the class (a member,
        an own formal, a selection it reads, a formal of a placed Space it binds by
        reference), the guard it is placed under, or a member of a placed Space, walked
        there."""
        if type(edge) is Local:
            return edge.member
        return self.refs[edge]

    def reached(self, path: Sequence[int]) -> Compiled:
        """The compiled form the slots ``path`` reach."""
        current = self
        for slot in path:
            current = current.slots[slot].compiled
        return current

    def key_of(self, ref: int | Placed) -> str:
        """A record's key, or a placed Space's member's under its slots' names."""
        if isinstance(ref, int):
            return self.records[ref].key
        names: list[str] = []
        current = self
        for slot in ref.path:
            names.append(current.slots[slot].name)
            current = current.slots[slot].compiled
        return member_key(".".join(names), current.records[ref.member].key)


def compiled(space_type: type[_Namespace]) -> Compiled:
    """The compiled form of a Space class: computed once (``link_template``), and cached on
    the class."""
    return link_template(space_type)


def cached_compiled(space_type: type[_Namespace]) -> Compiled:
    """The compiled form of a Space class already compiled (``compiled``), as cached."""
    found = _cached(space_type)
    if found is None:
        raise DefinitionError.broken(
            "template-cache-invalid", _OWNER, "invalid compiled form cache on Space class"
        )
    return found


def link_template(space_type: type[_Namespace]) -> Compiled:
    """The compiled form of a Space class: computed once, cached on the class. The classes it
    places are compiled first, deepest first, so no chain of placements recurses."""

    from .compiler import _PREPARATION_LOCK  # noqa: PLC0415 - a cycle

    if not isinstance(space_type, type) or not issubclass(space_type, _Namespace):
        raise DefinitionError.broken(
            "not-a-space", _OWNER, "link_template requires a Space subclass"
        )
    with _PREPARATION_LOCK:
        cached = _cached(space_type)
        if cached is not None:
            return cached
        for placed in _placed_first(space_type):
            if _cached(placed) is None:
                _link(placed)
        return _link(space_type)


def _cached(space_type: type[_Namespace]) -> Compiled | None:
    cached = space_type.__dict__.get("_space_compiled")
    if cached is not None and (
        not isinstance(cached, Compiled) or cached.space_type is not space_type
    ):
        raise DefinitionError.broken(
            "template-cache-invalid", _OWNER, "invalid compiled form cache on Space class"
        )
    return cached


def _placed_first(space_type: type[_Namespace]) -> list[type[_Namespace]]:
    """The Space classes ``space_type`` places, through any depth, each after those it
    places; none where it places itself (its compiling refuses it)."""
    order: list[type[_Namespace]] = []
    done: set[type[_Namespace]] = set()
    stack: list[tuple[type[_Namespace], Iterator[type[_Namespace]]]] = [
        (space_type, iter(_places(space_type)))
    ]
    open_ = {space_type}
    while stack:
        current, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            open_.discard(current)
            done.add(current)
            if current is not space_type:
                order.append(current)
            continue
        if child in open_:
            return []  # recursive placement: refused where the class is compiled
        if child not in done and _cached(child) is None:
            open_.add(child)
            stack.append((child, iter(_places(child))))
    return order


def _places(space_type: type[_Namespace]) -> list[type[_Namespace]]:
    """The Space classes a class's own members place (child nodes, candidates), read from
    its class attributes, the nearest definition of each name (what it places is compiled
    first; allocating it compiles anything else it places)."""
    found: list[type[_Namespace]] = []
    seen: set[str] = set()
    for base in space_type.__mro__:
        if base is _Namespace:
            break
        for name, value in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            declaration = slot_declaration(value)
            if isinstance(declaration, NodeDecl):
                found.append(declaration.space_type)
            elif isinstance(declaration, NodeDecision):
                found.extend(
                    record.space_type
                    for record in declaration.candidates.values()
                    if record is not None
                )
    return found


def _link(space_type: type[_Namespace]) -> Compiled:
    from .compiler import _validate_constructors  # noqa: PLC0415 - a cycle

    # What the class is compiled from no longer changes: it was closed when it was created
    # (``SpaceMeta``), its declarations frozen with it.
    form = _compile(space_type)
    _validate_constructors((space_type,))
    type.__setattr__(space_type, "_space_compiled", form)
    return form


def _compile(space_type: type[_Namespace]) -> Compiled:
    table = allocate(space_type, link_template)
    lower(table)
    # What crosses a slot: cycles through the Spaces it places (``_compose``).
    order = _compose.order(table)
    check(table, order)
    names = [node.key for node in table.nodes]
    names.extend(table.edge_name(edge) for edge in table.edges)
    if len(set(names)) != len(names):
        raise DefinitionError.broken("node-names-collide", _OWNER, "generated node names collide")
    return _Freezing(table, order).compiled()


class _Freezing:
    """A class's compiled form, from its table: each record's references as edges."""

    def __init__(self, table: Table, order: tuple[int, ...]) -> None:
        self.table = table
        self.order = order
        space_type = table.space_type
        self.own = own = table.drafts[0]
        kinds = table.kinds[space_type]
        self.formals = {
            name: own.named_members[name] for name, kind in kinds.items() if kind == "param"
        }
        self.formal_names = {record: name for name, record in self.formals.items()}
        self.selections = {
            record: (choice.index, member)
            for choice in table.choice_drafts
            for record, member in choice.selects.items()
        }
        # Each binding record this class writes: its slot and formal.
        self.binds: dict[int, tuple[int, str]] = {}
        for draft in table.slot_drafts:
            placed = table.placing(draft)
            for name, binding in placed.bindings.items():
                if binding.node is not None:
                    self.binds[binding.node] = (placed.slot, name)
        self.slot_of = {draft: slot for slot, draft in enumerate(table.slot_drafts)}

    def edge(self, ref: NodeRef) -> Edge:
        """What a reference of the table is, as an edge of the class."""
        if not isinstance(ref, int):
            return ref
        formal = self.formal_names.get(ref)
        if formal is not None:
            return Formal(formal)
        selection = self.selections.get(ref)
        if selection is not None:
            return Selected((), *selection)
        bound = self.binds.get(ref)
        if bound is not None:
            slot, name = bound
            placed = self.table.placing(self.table.slot_drafts[slot])
            return Placed((slot,), placed.compiled.formals[name])
        chosen = self.table.chooses.get(ref)
        if chosen is not None:
            return Selected(*chosen)
        return Local(ref)

    def member(self, record: int) -> Member:
        node = self.table.nodes[record]
        edge = self.edge
        return Member(
            record,
            node.key,
            node.kind,
            node.semantics,
            guard=None if node.guard is None else edge(node.guard),
            arguments=tuple((argument.name, edge(argument.node)) for argument in node.arguments),
            function=node.function,
            call_style=node.call_style,
            value=node.value,
            required=node.required,
            domain=node.domain,
            domain_arguments=tuple(
                (argument.name, edge(argument.node)) for argument in node.domain_arguments
            ),
            output=None if node.output is None else edge(node.output),
            constraints=tuple(edge(piece) for piece in node.constraints),
            alternatives=tuple((label, edge(target)) for label, target in node.alternatives),
            selector=None if node.selector is None else edge(node.selector),
            source_owner=node.source_owner,
            origin=node.origin,
        )

    def compiled(self) -> Compiled:
        table, own, space_type = self.table, self.own, self.table.space_type
        nodes = table.nodes
        records = tuple(self.member(record) for record in range(len(nodes)))
        bound: dict[tuple[int, str], int] = {}
        chosen: dict[tuple[tuple[int, ...], int, str], int] = {}
        written: dict[int, int] = {}
        for record, node in enumerate(nodes):
            if node.scope == 0:
                continue
            binds = self.binds.get(record)
            if binds is not None:
                bound[binds] = record
                written[record] = binds[0]
                continue
            path, choice, member = table.chooses[record]
            chosen[(path, choice, member)] = record
            written[record] = path[0]
        structure: list[tuple[str, int]] = []
        for name, declaration in own.effective.members.items():
            kind = table.kinds[space_type][name]
            if kind == "node":
                structure.append(("child", self.slot_of[own.named_children[name]]))
            elif kind == "choice":
                structure.append(("choice", own.choices[declaration]))
        positions: dict[int, tuple[int, int]] = {}
        choice_positions: dict[int, int] = {}
        for position, (item, index) in enumerate(structure):
            if item == "child":
                positions[index] = (position, 0)
                continue
            choice_positions[index] = position
            for case, (_, child) in enumerate(table.choice_drafts[index].cases):
                if child is not None:
                    positions[self.slot_of[child]] = (position, 1 + case)
        # Slots are allocated in placement order: structure order, candidates in case order.
        assert sorted(positions, key=positions.__getitem__) == list(range(len(positions)))
        slots = tuple(
            self.slot(slot, draft, positions[slot]) for slot, draft in enumerate(table.slot_drafts)
        )
        formals = self.formals
        own_selection = {selection: record for record, selection in self.selections.items()}
        # What a walk reads each edge as: the records and slots are known now.
        refs: dict[Edge, Ref] = {}
        edges = [edge for member in records for edge in member.dependencies]
        edges.extend(slot.guard for slot in slots)
        edges.extend(self.edge(piece) for piece in own.pieces)
        for edge in edges:
            if type(edge) is not Local and edge not in refs:
                refs[edge] = _ref(edge, slots, formals, own_selection, chosen)
        value_members = tuple(
            own.named_members[name]
            for name, kind in table.kinds[space_type].items()
            if kind not in ("node", "choice")
        )
        return Compiled(
            space_type,
            records,
            slots,
            tuple(self.choice(index) for index in range(len(table.choice_drafts))),
            own.validity,
            tuple(self.edge(piece) for piece in own.pieces),
            value_members,
            _frozen(formals),
            _frozen(self.formal_names),
            _frozen(self.selections),
            _frozen(own_selection),
            _frozen({choice.selector: choice.index for choice in table.choice_drafts}),
            _frozen(bound),
            _frozen(chosen),
            _frozen(written),
            _frozen({record: written_for for written_for, record in bound.items()}),
            _frozen({record: written_for for written_for, record in chosen.items()}),
            _frozen(own.members),
            _frozen(own.named_members),
            _frozen({key: self.slot_of[draft] for key, draft in own.children.items()}),
            _frozen({name: self.slot_of[draft] for name, draft in own.named_children.items()}),
            _frozen(own.choices),
            _frozen({table.drafts[draft].name: slot for draft, slot in self.slot_of.items()}),
            _frozen(choice_positions),
            _frozen({record: position for position, record in enumerate(value_members)}),
            tuple(structure),
            _frozen({node.key: node.index for node in nodes if node.scope == 0}),
            _frozen(refs),
            self.order,
        )

    def slot(self, index: int, draft: int, position: tuple[int, int]) -> CompiledSlot:
        scope = self.table.drafts[draft]
        placed = self.table.placing(draft)
        return CompiledSlot(
            index,
            scope.name,
            placed.compiled,
            scope.record,
            self.edge(cast(NodeRef, scope.guard)),
            scope.candidate,
            _frozen(placed.bindings),
            position,
        )

    def choice(self, index: int) -> CompiledChoice:
        draft = self.table.choice_drafts[index]
        return CompiledChoice(
            draft.index,
            draft.key,
            draft.selector,
            self.edge(cast(NodeRef, draft.guard)),
            tuple(
                (case, None if child is None else self.slot_of[child])
                for case, child in draft.cases
            ),
            _frozen(draft.members),
            _frozen(draft.selects),
            frozenset(draft.shared),
            frozenset(draft.dropped),
        )


def _ref(
    edge: Edge,
    slots: tuple[CompiledSlot, ...],
    formals: Mapping[str, int],
    own_selection: Mapping[tuple[int, str], int],
    chosen: Mapping[tuple[tuple[int, ...], int, str], int],
) -> Ref:
    """What a walk reads ``edge`` (not a local one) as (``Compiled.ref``)."""
    if isinstance(edge, Formal):
        return formals[edge.name]
    if isinstance(edge, Placing):
        return PLACING
    if isinstance(edge, Selected):
        found = chosen.get((edge.path, edge.choice, edge.member))
        if found is not None:
            return found
        if not edge.path:
            return own_selection[(edge.choice, edge.member)]
        reached = slots[edge.path[0]].compiled.reached(edge.path[1:])
        return Placed(edge.path, reached.own_selection[(edge.choice, edge.member)])
    placed = cast(Placed, edge)
    if len(placed.path) == 1:
        slot = slots[placed.path[0]]
        formal = slot.compiled.formal_names.get(placed.member)
        if formal is not None and slot.bindings[formal].node is not None:
            # A formal this class binds by reference: the record it writes for it.
            return cast(int, slot.bindings[formal].node)
    return placed


__all__ = [
    "Compiled",
    "CompiledChoice",
    "CompiledSlot",
    "Ref",
    "cached_compiled",
    "compiled",
    "link_template",
]
