# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Templates: each Space class's own scope, compiled once, and its compiled form.

The front end compiles a Space class into its template, and nothing else: there is no
table of a whole design space. ``link_template(C)`` runs the linker's phases on one
table (``_table``): C's own scope is allocated, lowered and checked, and each Space C
places (a child node, a candidate of a Decision over nodes) is a **stub**: what C's body
writes for it and nothing of the child's own. A formal binding is a node (an alias, a
fresh Decision), lowered in C; a literal or a default is a value on the stub; a reference
into a child is an ``import`` node, whose target is a node of the child class's template,
linked first and cached on the class. C's own formals are its template's ``param`` nodes:
where C is placed, the placing template's stub binds each, so a formal has one identity,
its scope and its formal node, whatever binds it.

A Decision over nodes guarded by ``supplied(formal)`` never applies where that formal is
unsupplied. A template is compiled once, with the choice's candidates; it records the
formal, and what exists only where the choice applies (``Template.requires``); where the
template is placed, the rule is applied per placement.

**The compiled form** (``Compiled``, ``compiled(C)``) is made from the template when it is
linked, and cached on the class with it: C's members numbered within the class, each
reference an edge (``ir.Edge``: a member of C; a member of a class C places, through its
slots; C's own formal; the guard C is placed under; a selection of a choice's member, in
the choice's own scope). No record stands for a member of a placed class (an import), for
a stub, or for the guard C is placed under; what C writes for a Space it places (a formal's
binding by reference, a selection made in that Space's choice) is a record of C's,
addressed by its slot. Each Space C places is a slot (``CompiledSlot``): its class's
compiled form, its bindings and guard.

What crosses a slot is checked per class and composed (``_compose``), on the compiled
forms: cycles through the Spaces it places and the reads its placements do not place, as
it is linked; the domain rule, from the root's compiled form, when a model is made
(``compiler.compile_model``). Read inference (``_reads``) and the questions about the whole
design space (``_whole``) walk the compiled forms too.

A model's structure (``_structure``) places templates as scopes, made when addressed, and
resolves each node of a scope from its template when read; ``_structure.expand`` is the
eager form, every scope placed and every node resolved, into one table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from . import _compose
from ._allocate import allocate
from ._check import check
from ._collapse import collapse
from ._configuration import Space
from ._lower import lower
from ._nodes import NodeDecision, NodeDecl, slot_declaration
from ._table import Binding, Read, ScopeDraft, Table, member_key
from .errors import DefinitionError
from .ir import PLACING, Edge, Formal, Local, Member, Node, Placed, Placing, Selected

_OWNER = "space.template"


def _frozen(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class TemplateScope:
    """A local scope of a template: its own (0) or a stub.

    A placed stub (``parent == 0``) is a Space the template places, under ``guard`` (a
    node of the template); a nested stub stands for a scope below a stub, reached by a
    reference (its ``slot``: its scope in its parent stub's template; its bindings that
    template's). ``members`` and ``named_members`` are the own scope's members, or the
    nodes the template reserved in a stub (named shared Decisions it owns); ``children``,
    ``named_children`` and ``choices`` are the own scope's only.
    """

    index: int
    parent: int | None
    name: str
    space_type: type[Space]
    record: NodeDecl | None
    guard: int | None
    slot: int | None
    candidate: tuple[int, str] | None
    requires: int | None
    bindings: Mapping[str, Binding]
    members: Mapping[object, int]
    named_members: Mapping[str, int]
    children: Mapping[object, int]
    named_children: Mapping[str, int]
    choices: Mapping[object, int]


@dataclass(frozen=True, slots=True)
class TemplateChoice:
    """A Decision over nodes of the own scope: as ``ir.Choice``, with what a placement
    where it never applies needs (``never_formal`` unsupplied there): the selector's
    value there, every selection made for it (node -> member) and those its candidates
    share, and each read of a member through it. A selection there has no alternatives
    and keeps the type the template checked its readers against (``_structure``)."""

    index: int
    key: str
    selector: int
    guard: int
    cases: tuple[tuple[str, int | None], ...]
    members: Mapping[str, int]
    space_types: tuple[type[Space], ...]
    never_formal: str | None
    never_value: str
    selects: Mapping[int, str]
    shared: frozenset[int]
    readers: Mapping[str, tuple[Read, ...]]
    dropped: frozenset[str]


@dataclass(frozen=True, slots=True)
class TemplateIndex:
    """What placing a template as scopes looks up, found once from it (``_structure``).

    ``selections``: each own selection (``decision.member`` through an own choice) ->
    (the choice, the member), and back (``own_selection``); ``stub_selections``: each
    selection made in a stub's choice by (stub, the choice in its template, member), and
    the members made per (stub, choice) (``stub_members``); ``nested``: each nested stub
    by (its parent stub, its scope in that stub's template). Positions, for declaration
    order: each placed stub's (its structure item, 0 for a child node or 1 + its case for a
    candidate), each own choice's structure item, each own value member's position, and
    each named shared Decision's. ``stubs``: the placed stubs by local name
    (``heating.gas``), for keys."""

    selections: Mapping[int, tuple[int, str]]
    own_selection: Mapping[tuple[int, str], int]
    stub_selections: Mapping[tuple[int, int, str], int]
    stub_members: Mapping[tuple[int, int], tuple[str, ...]]
    nested: Mapping[tuple[int, int], int]
    slot_positions: Mapping[int, tuple[int, int]]
    choice_positions: Mapping[int, int]
    value_positions: Mapping[int, int]
    named_positions: Mapping[int, int]
    stubs: Mapping[str, int]
    # The placed stubs, in placement order (structure order, candidates in case order).
    placed: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Template:
    """One Space class's own scope, compiled once; see the module docstring.

    Node keys are local (``area``, ``kitchen.$guard``, ``kitchen.area`` for a formal's
    binding in stub ``kitchen``); ``placement_guard`` stands for the guard the own scope
    is placed under. The nodes are lowered and checked, not collapsed: ``forward`` is
    collapse's forwarding inside the template only.
    """

    space_type: type[Space]
    nodes: tuple[Node, ...]
    scopes: tuple[TemplateScope, ...]
    choices: tuple[TemplateChoice, ...]
    placement_guard: int
    validity: int
    # The own scope's value members (member order), its formals and their nodes.
    value_members: tuple[int, ...]
    formals: Mapping[str, int]
    formal_names: Mapping[int, str]
    # A placed stub's binding node -> the formal it binds; each import -> its target in its
    # stub's template; a node reserved in a stub on this body's behalf (contributed).
    binding_names: Mapping[int, str]
    imports: Mapping[int, int]
    contributed: frozenset[int]
    # Each node that exists only where an own choice applies -> that choice.
    requires: Mapping[int, int]
    # Named shared Decisions (owned here or at a stub), in declaration order.
    named: tuple[int, ...]
    # A selection made in a stub's choice -> (stub, the choice in its template, member),
    # and each read of it.
    stub_selects: Mapping[int, tuple[int, int, str]]
    stub_reads: Mapping[int, tuple[Read, ...]]
    # The own scope's structure in member order: ("child", stub) or ("choice", choice);
    # each own choice's selector -> the choice.
    structure: tuple[tuple[str, int], ...]
    selectors: Mapping[int, int]
    order: tuple[int, ...]
    forward: tuple[int, ...]
    keys: Mapping[str, int]
    # What placing it as scopes looks up (``TemplateIndex``).
    index: TemplateIndex

    @property
    def members(self) -> Mapping[object, int]:
        return self.scopes[0].members

    @property
    def named_members(self) -> Mapping[str, int]:
        return self.scopes[0].named_members

    @property
    def children(self) -> Mapping[object, int]:
        return self.scopes[0].children

    @property
    def named_children(self) -> Mapping[str, int]:
        return self.scopes[0].named_children

    @property
    def choice_index(self) -> Mapping[object, int]:
        return self.scopes[0].choices

    def counts(self) -> dict[str, int]:
        """Its nodes by kind."""
        return dict(Counter(node.kind for node in self.nodes))


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
    candidacy (choice, case), the Decision over nodes it exists only where it applies
    (``requires``), its formals' bindings (``Binding``: a literal or a default as a value; a
    reference by the record the placing class writes for it, ``node``; or unsupplied), and
    its declaration position (``position``: its structure item, 0 for a child node or 1 + its
    case for a candidate)."""

    index: int
    name: str
    compiled: Compiled = field(repr=False)
    record: NodeDecl | None
    guard: Edge
    candidate: tuple[int, str] | None
    requires: int | None
    bindings: Mapping[str, Binding]
    position: tuple[int, int]

    @property
    def space_type(self) -> type[Space]:
        return self.compiled.space_type


@dataclass(frozen=True, slots=True)
class CompiledChoice:
    """An own Decision over nodes, as ``TemplateChoice``: its selector and selections are
    records of the class, its cases slots."""

    index: int
    key: str
    selector: int
    guard: Edge
    cases: tuple[tuple[str, int | None], ...]
    members: Mapping[str, int]
    space_types: tuple[type[Space], ...]
    never_formal: str | None
    never_value: str
    selects: Mapping[int, str]
    shared: frozenset[int]
    readers: Mapping[str, tuple[Read, ...]]
    dropped: frozenset[str]


@dataclass(frozen=True, eq=False, slots=True)
class Compiled:
    """A Space class's compiled form: computed once, from its template, and cached on the
    class (``compiled``).

    ``records``: its members, numbered within the class, and between them, in the order its
    template states them, the records it writes for the Spaces it places: a formal's binding
    by reference (``bound``, by slot and formal) and a selection it makes in a placed Space's
    choice (``chosen``, by the selection's slots, choice and member); each written record's
    slot is ``written``'s. Every reference is an edge (``ir.Edge``): no record stands for a
    member of another class (an import), for a scope it places (a stub), or for the guard it
    is placed under. A walk reads each edge as ``ref`` says.

    ``slots``: the Spaces it places, in placement order (structure order, candidates in case
    order). Its own maps as its template's, by record and slot: ``formals``, ``members``,
    ``children``, ``choices``, ``selections`` (each own selection's (choice, member)),
    ``requires`` (what exists only where an own Decision over nodes applies, a record or a
    member of a placed Space it reads: carried for ``supplied()``'s rule).

    ``nodes``, ``record_of``, ``stubs`` and ``slot_of_stub`` relate it to its template (each
    record's node, each slot's stub), which the model's structure still places."""

    space_type: type[Space]
    records: tuple[Member, ...]
    slots: tuple[CompiledSlot, ...]
    choices: tuple[CompiledChoice, ...]
    validity: int
    value_members: tuple[int, ...]
    formals: Mapping[str, int]
    formal_names: Mapping[int, str]
    selections: Mapping[int, tuple[int, str]]
    own_selection: Mapping[tuple[int, str], int]
    selectors: Mapping[int, int]
    requires: Mapping[object, int]
    bound: Mapping[tuple[int, str], int]
    chosen: Mapping[tuple[tuple[int, ...], int, str], int]
    written: Mapping[int, int]
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
    nodes: tuple[int, ...]
    record_of: Mapping[int, int]
    stubs: tuple[int, ...]
    slot_of_stub: Mapping[int, int]
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


def compiled(space_type: type[Space]) -> Compiled:
    """The compiled form of a Space class: computed once, with its template (``link_template``),
    and cached on the class."""
    link_template(space_type)
    return cached_compiled(space_type)


def cached_compiled(space_type: type[Space]) -> Compiled:
    """The compiled form of a Space class already linked (``compiled``), as cached."""
    found = space_type.__dict__.get("_space_compiled")
    if not isinstance(found, Compiled) or found.space_type is not space_type:
        raise DefinitionError.broken(
            "template-cache-invalid", _OWNER, "invalid compiled form cache on Space class"
        )
    return found


class _Compiling:
    """A template's compiled form (``Compiled``), the classes it places compiled first."""

    def __init__(self, template: Template) -> None:
        self.template = template
        nodes = template.nodes
        self.kept = tuple(
            node.index
            for node in nodes
            if node.index != template.placement_guard and node.kind != "import"
        )
        self.record_of = {node: record for record, node in enumerate(self.kept)}
        self.slot_of_stub = {stub: slot for slot, stub in enumerate(template.index.placed)}
        self.placed = tuple(
            cached_compiled(template.scopes[stub].space_type) for stub in template.index.placed
        )

    def slot_path(self, stub: int) -> tuple[tuple[int, ...], Compiled]:
        """A stub's slots from the class (a placed stub's, then each nested stub's in the
        class the one before reaches), and the compiled form they reach."""
        scopes = self.template.scopes
        stubs: list[int] = []
        while True:
            scope = scopes[stub]
            if scope.parent == 0:
                stubs.append(stub)
                break
            stubs.append(cast(int, scope.slot))
            stub = cast(int, scope.parent)
        stubs.reverse()
        path = [self.slot_of_stub[stubs[0]]]
        current = self.placed[path[0]]
        for nested in stubs[1:]:
            slot = current.slot_of_stub[nested]
            path.append(slot)
            current = current.slots[slot].compiled
        return tuple(path), current

    def edge(self, node: int) -> Edge:
        """What a node of the template stands for, as an edge of the class."""
        template = self.template
        if node == template.placement_guard:
            return PLACING
        spec = template.nodes[node]
        if spec.scope == 0:
            formal = template.formal_names.get(node)
            if formal is not None:
                return Formal(formal)
            selection = template.index.selections.get(node)
            if selection is not None:
                return Selected((), *selection)
            return Local(self.record_of[node])
        if spec.kind == "import":
            path, reached = self.slot_path(spec.scope)
            target = reached.record_of[template.imports[node]]
            selection = reached.selections.get(target)
            if selection is not None:
                return Selected(path, *selection)
            return Placed(path, target)
        formal = template.binding_names.get(node)
        if formal is not None:
            slot = self.slot_of_stub[spec.scope]
            return Placed((slot,), self.placed[slot].formals[formal])
        stub, choice, member = template.stub_selects[node]
        return Selected(self.slot_path(stub)[0], choice, member)

    def member(self, record: int, node: Node) -> Member:
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
        template, record_of = self.template, self.record_of
        records = tuple(
            self.member(record, template.nodes[node]) for record, node in enumerate(self.kept)
        )
        bound: dict[tuple[int, str], int] = {}
        chosen: dict[tuple[tuple[int, ...], int, str], int] = {}
        written: dict[int, int] = {}
        for record, node in enumerate(self.kept):
            scope = template.nodes[node].scope
            if scope == 0:
                continue
            formal = template.binding_names.get(node)
            if formal is not None:
                slot = self.slot_of_stub[scope]
                bound[(slot, formal)] = record
                written[record] = slot
                continue
            stub, choice, member = template.stub_selects[node]
            path = self.slot_path(stub)[0]
            chosen[(path, choice, member)] = record
            written[record] = path[0]
        slots = tuple(self.slot(index, stub) for index, stub in enumerate(template.index.placed))
        formals = {name: record_of[node] for name, node in template.formals.items()}
        selections = {
            record_of[node]: selection for node, selection in template.index.selections.items()
        }
        own_selection = {selection: record for record, selection in selections.items()}
        # What a walk reads each edge as: the records and slots are known now.
        refs: dict[Edge, Ref] = {}
        edges = [edge for member in records for edge in member.dependencies]
        edges.extend(slot.guard for slot in slots)
        for edge in edges:
            if type(edge) is not Local and edge not in refs:
                refs[edge] = self.ref(edge, slots, formals, own_selection, chosen)
        requires: dict[object, int] = {}
        for node, choice in template.requires.items():
            found = record_of.get(node)
            if found is not None:
                requires[found] = choice
                continue
            if node == template.placement_guard:
                continue
            read = refs.get(self.edge(node))
            if read is None:  # an import nothing reads (made once, for its first reader)
                read = self.ref(self.edge(node), slots, formals, own_selection, chosen)
            requires.setdefault(read, choice)
        return Compiled(
            template.space_type,
            records,
            slots,
            tuple(self.choice(choice) for choice in template.choices),
            record_of[template.validity],
            tuple(record_of[node] for node in template.value_members),
            _frozen(formals),
            _frozen({record: name for name, record in formals.items()}),
            _frozen(selections),
            _frozen(own_selection),
            _frozen({record_of[node]: choice for node, choice in template.selectors.items()}),
            _frozen(requires),
            _frozen(bound),
            _frozen(chosen),
            _frozen(written),
            _frozen({key: record_of[node] for key, node in template.members.items()}),
            _frozen({name: record_of[node] for name, node in template.named_members.items()}),
            _frozen({key: self.slot_of_stub[stub] for key, stub in template.children.items()}),
            _frozen(
                {name: self.slot_of_stub[stub] for name, stub in template.named_children.items()}
            ),
            template.choice_index,
            _frozen({name: self.slot_of_stub[stub] for name, stub in template.index.stubs.items()}),
            template.index.choice_positions,
            _frozen(
                {
                    record_of[node]: position
                    for node, position in template.index.value_positions.items()
                }
            ),
            tuple(
                (kind, self.slot_of_stub[index] if kind == "child" else index)
                for kind, index in template.structure
            ),
            _frozen(
                {
                    record.key: record.index
                    for record in records
                    if template.nodes[self.kept[record.index]].scope == 0
                }
            ),
            _frozen(refs),
            self.kept,
            _frozen(record_of),
            template.index.placed,
            _frozen(self.slot_of_stub),
        )

    def ref(
        self,
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

    def slot(self, index: int, stub: int) -> CompiledSlot:
        template = self.template
        scope = template.scopes[stub]
        bindings = {
            name: binding
            if binding.node is None
            else Binding(binding.kind, binding.value, self.record_of[binding.node])
            for name, binding in scope.bindings.items()
        }
        candidate = scope.candidate
        return CompiledSlot(
            index,
            scope.name,
            self.placed[index],
            scope.record,
            self.edge(cast(int, scope.guard)),
            candidate,
            scope.requires,
            _frozen(bindings),
            template.index.slot_positions[stub],
        )

    def choice(self, choice: TemplateChoice) -> CompiledChoice:
        record_of = self.record_of
        return CompiledChoice(
            choice.index,
            choice.key,
            record_of[choice.selector],
            self.edge(choice.guard),
            tuple(
                (case, None if stub is None else self.slot_of_stub[stub])
                for case, stub in choice.cases
            ),
            _frozen({member: record_of[node] for member, node in choice.members.items()}),
            choice.space_types,
            choice.never_formal,
            choice.never_value,
            _frozen({record_of[node]: member for node, member in choice.selects.items()}),
            frozenset(record_of[node] for node in choice.shared),
            choice.readers,
            choice.dropped,
        )


def link_template(space_type: type[Space]) -> Template:
    """The template of a Space class: computed once, cached on the class. The classes it
    places are linked first, deepest first, so no chain of placements recurses."""

    from .compiler import _PREPARATION_LOCK  # noqa: PLC0415 - a cycle

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
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


def _cached(space_type: type[Space]) -> Template | None:
    cached = space_type.__dict__.get("_space_template")
    if cached is not None and (
        not isinstance(cached, Template) or cached.space_type is not space_type
    ):
        raise DefinitionError.broken(
            "template-cache-invalid", _OWNER, "invalid template cache on Space class"
        )
    return cached


def _placed_first(space_type: type[Space]) -> list[type[Space]]:
    """The Space classes ``space_type`` places, through any depth, each after those it
    places; none where it places itself (its linking refuses it)."""
    order: list[type[Space]] = []
    done: set[type[Space]] = set()
    stack: list[tuple[type[Space], Iterator[type[Space]]]] = [
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
            return []  # recursive placement: refused where the class is linked
        if child not in done and _cached(child) is None:
            open_.add(child)
            stack.append((child, iter(_places(child))))
    return order


def _places(space_type: type[Space]) -> list[type[Space]]:
    """The Space classes a class's own members place (child nodes, candidates), read from
    its class attributes, the nearest definition of each name (what it places is linked
    first; allocating it links anything else it places)."""
    found: list[type[Space]] = []
    seen: set[str] = set()
    for base in space_type.__mro__:
        if base is Space:
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


def _link(space_type: type[Space]) -> Template:
    from .compiler import _definition_classes, _validate_constructors  # noqa: PLC0415 - a cycle

    template = _compile(space_type)
    _validate_constructors((space_type,))
    # What the template was compiled from no longer changes: the class is prepared.
    reason = f"{space_type.__name__} was prepared"
    for scope in template.scopes[1:]:
        if scope.parent == 0 and scope.record is not None:
            scope.record.freeze(reason)
    for definition in _definition_classes((space_type,)):
        type.__setattr__(definition, "_space_definition_finalized", True)
    type.__setattr__(space_type, "_space_compiled", _Compiling(template).compiled())
    type.__setattr__(space_type, "_space_template", template)
    return template


def _compile(space_type: type[Space]) -> Template:
    table = allocate(space_type, link_template)
    lower(table)
    # What crosses a slot: a placed Space's nodes that read what its placement does not
    # place, and cycles through the Spaces it places (``_compose``).
    _compose.check_placements(table)
    order = _compose.order(table)
    check(table, order)
    forward = collapse(list(table.nodes), order)
    nodes = tuple(table.nodes)
    keys = {node.key: node.index for node in nodes}
    if len(keys) != len(nodes):
        raise DefinitionError.broken("node-names-collide", _OWNER, "generated node names collide")
    own = table.drafts[0]
    value_members = tuple(
        own.named_members[name]
        for name, kind in table.kinds[space_type].items()
        if kind not in ("node", "choice")
    )
    formals = {
        name: own.named_members[name]
        for name, kind in table.kinds[space_type].items()
        if kind == "param"
    }
    binding_names = {
        binding.node: name
        for draft in table.drafts
        if draft.stub is not None and draft.stub.slot is None
        for name, binding in draft.stub.bindings.items()
        if binding.node is not None
    }
    structure: list[tuple[str, int]] = []
    for name, declaration in own.effective.members.items():
        kind = table.kinds[space_type][name]
        if kind == "node":
            structure.append(("child", own.named_children[name]))
        elif kind == "choice":
            structure.append(("choice", own.choices[declaration]))
    contributed = frozenset(
        node.index
        for node in nodes
        if node.scope != 0 and node.kind != "import" and node.index not in binding_names
    )
    scopes = tuple(_scope(draft, table) for draft in table.drafts)
    choices = tuple(_choice(table, index) for index in range(len(table.choice_drafts)))
    return Template(
        space_type,
        nodes,
        scopes,
        choices,
        cast(int, table.placement_guard),
        cast(int, own.validity),
        value_members,
        _frozen(formals),
        _frozen({node: name for name, node in formals.items()}),
        _frozen(binding_names),
        _frozen(table.imports),
        contributed,
        _frozen(table.requires),
        tuple(table.named),
        _frozen(table.stub_selects),
        _frozen({node: _declared(table, reads) for node, reads in table.stub_reads.items()}),
        tuple(structure),
        _frozen({choice.selector: choice.index for choice in table.choice_drafts}),
        order,
        forward,
        _frozen(keys),
        _index(scopes, choices, tuple(structure), value_members, tuple(table.named), table),
    )


def _index(
    scopes: tuple[TemplateScope, ...],
    choices: tuple[TemplateChoice, ...],
    structure: tuple[tuple[str, int], ...],
    value_members: tuple[int, ...],
    named: tuple[int, ...],
    table: Table,
) -> TemplateIndex:
    selections = {
        node: (choice.index, member)
        for choice in choices
        for node, member in choice.selects.items()
    }
    stub_members: dict[tuple[int, int], list[str]] = {}
    for stub, choice, member in table.stub_selects.values():
        stub_members.setdefault((stub, choice), []).append(member)
    slot_positions: dict[int, tuple[int, int]] = {}
    choice_positions: dict[int, int] = {}
    placed: list[int] = []
    for position, (kind, index) in enumerate(structure):
        if kind == "child":
            slot_positions[index] = (position, 0)
            placed.append(index)
            continue
        choice_positions[index] = position
        for case, (_, candidate) in enumerate(choices[index].cases):
            if candidate is not None:
                slot_positions[candidate] = (position, 1 + case)
                placed.append(candidate)
    return TemplateIndex(
        _frozen(selections),
        _frozen({selection: node for node, selection in selections.items()}),
        _frozen({selection: node for node, selection in table.stub_selects.items()}),
        _frozen({key: tuple(members) for key, members in stub_members.items()}),
        _frozen(
            {
                (cast(int, scope.parent), scope.slot): scope.index
                for scope in scopes
                if scope.slot is not None
            }
        ),
        _frozen(slot_positions),
        _frozen(choice_positions),
        _frozen({node: position for position, node in enumerate(value_members)}),
        _frozen({node: position for position, node in enumerate(named)}),
        _frozen({scopes[stub].name: stub for stub in placed}),
        tuple(placed),
    )


def _scope(draft: ScopeDraft, table: Table) -> TemplateScope:
    stub = draft.stub
    return TemplateScope(
        draft.index,
        draft.parent,
        draft.name if stub is None else table.local_name(cast(int, draft.parent), draft.index),
        draft.effective.space_type,
        draft.record,
        draft.guard,
        None if stub is None else stub.slot,
        draft.candidate,
        None if stub is None else stub.requires,
        _frozen({} if stub is None else stub.bindings),
        _frozen(draft.members),
        _frozen(draft.named_members),
        _frozen(draft.children),
        _frozen(draft.named_children),
        _frozen(draft.choices),
    )


def _declared(table: Table, reads: list[Read]) -> tuple[Read, ...]:
    """Reads of a selection in the template's declaration order: by the member of its own
    scope each reader belongs to (a template-local key's first segment: a member's name
    has none), in member order, and those of no member (a named shared Decision) last;
    reads of one member in the order they were lowered."""
    members = {name: position for position, name in enumerate(table.kinds[table.space_type])}
    return tuple(
        sorted(reads, key=lambda read: members.get(read.owner.split(".", 1)[0], len(members)))
    )


def _choice(table: Table, index: int) -> TemplateChoice:
    draft = table.choice_drafts[index]
    owner = table.drafts[draft.scope].effective
    decision = cast(NodeDecision, owner.members[draft.key])
    return TemplateChoice(
        draft.index,
        draft.key,
        draft.selector,
        cast(int, draft.guard),
        tuple(draft.cases),
        _frozen(draft.members),
        tuple(record.space_type for record in decision.candidates.values() if record is not None),
        draft.never_formal,
        next(iter(decision.candidates)),
        _frozen(draft.selects),
        frozenset(draft.shared),
        _frozen({member: _declared(table, reads) for member, reads in draft.readers.items()}),
        frozenset(draft.dropped),
    )


__all__ = [
    "Compiled",
    "CompiledChoice",
    "CompiledSlot",
    "Ref",
    "Template",
    "TemplateChoice",
    "TemplateIndex",
    "TemplateScope",
    "cached_compiled",
    "compiled",
    "link_template",
]
