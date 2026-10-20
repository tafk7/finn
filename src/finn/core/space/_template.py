# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Templates: each Space class's own scope, compiled once.

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

What crosses a slot is checked per template and composed (``_compose``): cycles through
the Spaces it places and the reads its placements do not place, as it is linked; the
domain rule, from the root's template, when a model is made (``compiler.compile_model``).

A model's structure (``_structure``) places templates as scopes, made when addressed, and
resolves each node of a scope from its template when read; ``_structure.expand`` is the
eager form, every scope placed and every node resolved, into one table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from . import _compose
from ._allocate import allocate
from ._check import check
from ._collapse import collapse
from ._configuration import Space
from ._lower import lower
from ._nodes import NodeDecision, NodeDecl, slot_declaration
from ._table import Binding, Read, ScopeDraft, Table
from .errors import DefinitionError
from .ir import Node

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
    # What the checks composed across slots found of it, per placement (``_compose``).
    composed: dict[str, object] = field(default_factory=dict, compare=False, repr=False)

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


__all__ = ["Template", "TemplateChoice", "TemplateIndex", "TemplateScope", "link_template"]
