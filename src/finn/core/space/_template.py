# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Templates: each Space class's own scope, compiled once; and their eager expansion.

``link_template(C)`` runs the linker's phases in template mode (``_table``): C's own
scope is allocated, lowered and checked as ``link_space`` does it for a root, and each
Space C places (a child node, a candidate of a Decision over nodes) is a **stub**: what
C's body writes for it and nothing of the child's own. A formal binding is the node
today's linker gives the child's formal (an alias, a fresh Decision), lowered in C; a
literal or a default is a value on the stub; a reference into a child is an ``import``
node, whose target is a node of the child class's template, linked first and cached on
the class. C's own formals are its template's ``param`` nodes: where C is placed, the
placing template's stub binds each, so a formal has one identity, its scope and its
formal node, whatever binds it.

A Decision over nodes guarded by ``supplied(formal)`` never applies where that formal is
unsupplied. A template is compiled once, with the choice's candidates; it records the
formal, and what exists only where the choice applies (``Template.requires``); the
expansion applies the rule per placement.

``expand(C)`` rebuilds ``link_space(C)``'s ``LinkedModel`` from templates alone: the
scopes breadth-first from C's template, each node resolved into the node today's linker
builds for the same key, collapse's bypass applied across slots by ``_collapse``'s rule.
A ``decision.member`` selection several templates make for one scope (the child's own
read, a parent's through it) is one node, owned by its first reader in the order
``link_space`` lowers its one table (each read's ``_table.Time``). Its ``NodeId``s may
differ from ``link_space``'s; nothing may depend on their numbering. Cross-template
cycles and the domain rule are ``link_space``'s (and ``compile_model``'s) still.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

from ._allocate import allocate
from ._check import check
from ._collapse import collapse, forwards, guard_implies
from ._configuration import Space
from ._lower import lower
from ._nodes import NodeDecision, NodeDecl
from ._ordering import dependency_order
from ._table import Binding, Read, ScopeDraft, Table, Time, member_key
from .compiler import _PREPARATION_LOCK, _definition_classes
from .errors import DefinitionError
from .ir import Argument, Choice, LinkedModel, Node, NodeId, Scope
from .semantics import STRING, ValueSemantics

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
    share, each read of a member through it, and the type its selection takes there."""

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
    never_semantics: Mapping[str, ValueSemantics[object] | None]
    never_declared: Mapping[str, bool]
    dropped: frozenset[str]


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
    def choice_index(self) -> Mapping[object, int]:
        return self.scopes[0].choices

    def counts(self) -> dict[str, int]:
        """Its nodes by kind."""
        return dict(Counter(node.kind for node in self.nodes))


def link_template(space_type: type[Space]) -> Template:
    """The template of a Space class: computed once, cached on the class."""

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise DefinitionError.broken(
            "not-a-space", _OWNER, "link_template requires a Space subclass"
        )
    with _PREPARATION_LOCK:
        cached = space_type.__dict__.get("_space_template")
        if cached is not None:
            if not isinstance(cached, Template) or cached.space_type is not space_type:
                raise DefinitionError.broken(
                    "template-cache-invalid", _OWNER, "invalid template cache on Space class"
                )
            return cached
        template = _compile(space_type)
        # What the template was compiled from no longer changes (as for a prepared model).
        reason = f"{space_type.__name__}'s template was linked"
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
    order = dependency_order(
        tuple(node.dependencies for node in table.nodes),
        tuple(node.key for node in table.nodes),
    )
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
    return Template(
        space_type,
        nodes,
        tuple(_scope(draft, table) for draft in table.drafts),
        tuple(_choice(table, index) for index in range(len(table.choice_drafts))),
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
        _frozen({node: tuple(reads) for node, reads in table.stub_reads.items()}),
        tuple(structure),
        _frozen({choice.selector: choice.index for choice in table.choice_drafts}),
        order,
        forward,
        _frozen(keys),
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
        _frozen({member: tuple(reads) for member, reads in draft.readers.items()}),
        _frozen(draft.never_semantics),
        _frozen(draft.never_declared),
        frozenset(draft.dropped),
    )


# -- the eager expansion ----------------------------------------------------------------------


class _Absent(Exception):
    """A node that does not exist at this placement (a candidate of a choice that never
    applies here)."""


@dataclass(eq=False)
class _Placed:
    """One scope of the expansion: a template placed at a path."""

    index: int
    parent: _Placed | None
    # Its stub in its parent's template.
    stub: int | None
    name: str
    template: Template
    guard: NodeId | None
    # The own choices that never apply here.
    never: frozenset[int]
    children: dict[int, _Placed] = field(default_factory=dict)
    # The template's nodes placed here (own ones, and those it reserved in its stubs).
    ids: dict[int, NodeId] = field(default_factory=dict)

    def binding(self, formal: str) -> Binding | None:
        """How its placing template binds one of its formals (None for the root)."""
        if self.parent is None:
            return None
        return self.parent.template.scopes[cast(int, self.stub)].bindings[formal]


def expand(space_type: type[Space]) -> LinkedModel:
    """``link_space(space_type)``'s model, rebuilt from templates alone."""
    return _Expansion(link_template(space_type)).run()


class _Expansion:
    def __init__(self, root: Template) -> None:
        self.root = root
        self.scopes: list[_Placed] = []
        # Each NodeId's template node, and the scope whose template it belongs to.
        self.origins: list[tuple[_Placed, int]] = []
        self.decisions: list[NodeId] = []
        self.selects: dict[int, dict[int, tuple[int, str]]] = {}
        # One node per member read through a choice, by (scope, choice, member): the
        # templates' selections for it are one; and its reads (when, reader), where its
        # owner is its first reader (not a selection its candidates share).
        self.merged: dict[tuple[int, int, str], NodeId] = {}
        self.reads: dict[NodeId, list[tuple[object, str]]] = {}

    # -- scopes and identities --------------------------------------------------------------

    def place(self, template: Template, parent: _Placed | None, stub: int | None) -> _Placed:
        if parent is None:
            name, guard = "", None
        else:
            local = parent.template.scopes[cast(int, stub)]
            name = member_key(parent.name, local.name)
            guard = self.guard(parent, local.guard)
        placed = _Placed(len(self.scopes), parent, stub, name, template, guard, frozenset())
        placed.never = frozenset(
            choice.index
            for choice in template.choices
            if choice.never_formal is not None
            and (binding := placed.binding(choice.never_formal)) is not None
            and binding.kind == "unsupplied"
        )
        self.scopes.append(placed)
        if parent is not None:
            parent.children[cast(int, stub)] = placed
        for index, node in enumerate(template.nodes):
            if node.scope != 0 or index == template.placement_guard:
                continue
            # A selection is placed once every scope is (``run``): its reads may be in them.
            if (
                self.selection(placed, index) is None
                and template.requires.get(index) not in placed.never
            ):
                self.identify(placed, index)
        # A scope's own decisions, in member order, when it is placed.
        for index in template.value_members:
            if self.kind(placed, index) == "decision":
                self.decisions.append(placed.ids[index])
        return placed

    def identify(self, placed: _Placed, index: int) -> None:
        placed.ids[index] = len(self.origins)
        self.origins.append((placed, index))

    def kind(self, placed: _Placed, index: int) -> str:
        """An own value member's kind where it is placed: a formal's by its binding."""
        node = placed.template.nodes[index]
        if node.kind != "param":
            return node.kind
        binding = placed.binding(placed.template.formal_names[index])
        if binding is None:
            return "param"
        return {
            "literal": "const",
            "reference": "alias",
            "named": "alias",
            "local-decision": "decision",
            "unsupplied": "present",
        }[binding.kind]

    def selection(self, placed: _Placed, index: int) -> tuple[int, str] | None:
        selects = self.selects.get(id(placed.template))
        if selects is None:
            selects = self.selects[id(placed.template)] = {
                node: (choice.index, member)
                for choice in placed.template.choices
                for node, member in choice.selects.items()
            }
        return selects.get(index)

    def select(
        self, placed: _Placed, index: int, scope: _Placed, choice_index: int, member: str
    ) -> None:
        """A selection of ``placed``'s template through ``scope``'s choice: one node per
        member and scope, wherever made. One its candidates share is there where the choice
        applies; any other where it is read, owned by its first reader."""
        template = placed.template
        choice = scope.template.choices[choice_index]
        if placed is scope and index in choice.shared and choice_index not in placed.never:
            self.identify(placed, index)
            if choice.members.get(member) == index:  # not dropped for disagreeing types
                self.merged[(scope.index, choice_index, member)] = placed.ids[index]
            return
        reads = choice.readers.get(member, ()) if placed is scope else template.stub_reads[index]
        found = [
            (self.event(placed, read.time), member_key(placed.name, read.owner))
            for read in reads
            if read.requires not in placed.never
        ]
        if not found:
            return  # nothing reads it here
        key = (scope.index, choice_index, member)
        merged = self.merged.get(key)
        if merged is None:
            self.identify(placed, index)
            merged = self.merged[key] = placed.ids[index]
            self.reads[merged] = []
        placed.ids[index] = merged
        self.reads[merged].extend(found)

    def when(self, placed: _Placed, time: Time) -> object:
        """A template's lowering time (``_table.Time``) as ``link_space`` orders its one
        table: member tasks (by scope, then member; named shared Decisions last), guards
        (allocation's by scope, then those lowering made), integer expressions."""
        if time[0] == "A":
            if time[1] == "own":
                return (0, 0, placed.index, time[2])
            if time[1] == "stub":
                return (0, 0, self.scope_of(placed, cast(int, time[2])).index, time[3])
            return (0, 1, placed.index, time[2])
        if time[0] == "B":
            if time[1] == "alloc":
                return (1, 0, placed.index, time[2])
            return (1, 1, self.event(placed, cast(Time, time[2])))
        return (2, self.event(placed, cast(Time, time[1])))

    def event(self, placed: _Placed, event: Time) -> object:
        task, position = event
        return (self.when(placed, cast(Time, task)), position)

    def scope_of(self, placed: _Placed, local: int) -> _Placed:
        """The scope standing at a local scope of ``placed``'s template."""
        if local == 0:
            return placed
        scope = placed.template.scopes[local]
        if scope.parent == 0:
            found = placed.children.get(local)
        else:
            above = self.scope_of(placed, cast(int, scope.parent))
            found = above.children.get(cast(int, scope.slot))
        if found is None:
            raise _Absent(f"{member_key(placed.name, scope.name)} is not placed")
        return found

    def resolve(self, placed: _Placed, index: int) -> NodeId | None:
        """The NodeId a node of ``placed``'s template stands for; the placement guard is
        the scope's guard (None where it is unguarded)."""
        template = placed.template
        if index == template.placement_guard:
            return placed.guard
        node = template.nodes[index]
        if node.scope == 0 or index in template.contributed:
            found = placed.ids.get(index)
            if found is None:
                raise _Absent(f"{member_key(placed.name, node.key)} is not placed")
            return found
        child = self.scope_of(placed, node.scope)
        if node.kind == "import":
            return self.resolve(child, template.imports[index])
        # A formal's binding: the child's formal.
        return self.resolve(child, child.template.formals[template.binding_names[index]])

    def reference(self, placed: _Placed, index: int, *, owner: str) -> NodeId:
        try:
            found = self.resolve(placed, index)
        except _Absent as cause:
            raise DefinitionError.broken(
                "reference-not-placed",
                _OWNER,
                f"{owner}: {cause}",
                declaration=owner,
            ) from cause
        assert found is not None, "only a guard reads the placement guard"
        return found

    def guard(self, placed: _Placed, index: int | None) -> NodeId | None:
        return None if index is None else self.resolve(placed, index)

    def dropped_here(self, placed: _Placed, index: int) -> bool:
        """Whether a node of a stub is below a candidate of a choice that never applies."""
        node = placed.template.nodes[index]
        return node.scope != 0 and placed.template.scopes[node.scope].requires in placed.never

    # -- the walk -----------------------------------------------------------------------------

    def run(self) -> LinkedModel:
        root = self.place(self.root, None, None)
        cursor = 0
        while cursor < len(self.scopes):
            placed = self.scopes[cursor]
            cursor += 1
            template = placed.template
            for kind, index in template.structure:
                if kind == "child":
                    self.child(placed, index)
                    continue
                choice = template.choices[index]
                if index in placed.never:
                    continue
                self.decisions.append(placed.ids[choice.selector])
                for _, stub in choice.cases:
                    if stub is not None:
                        self.child(placed, stub)
        for placed in self.scopes:
            template = placed.template
            for index in range(len(template.nodes)):
                selection = self.selection(placed, index)
                if selection is not None:
                    self.select(placed, index, placed, *selection)
            for index in sorted(template.contributed):
                if self.dropped_here(placed, index):
                    continue
                try:
                    scope = self.scope_of(placed, template.nodes[index].scope)
                except _Absent:
                    continue
                if index in template.stub_selects:
                    _, choice_index, member = template.stub_selects[index]
                    self.select(placed, index, scope, choice_index, member)
                elif template.requires.get(index) not in placed.never:
                    self.identify(placed, index)
        for placed in self.scopes:
            for index in placed.template.named:
                if index in placed.ids:
                    self.decisions.append(placed.ids[index])
        raw = [
            self.owned(self.build(placed, index, nid))
            for nid, (placed, index) in enumerate(self.origins)
        ]
        return self.model(root, raw)

    def owned(self, node: Node) -> Node:
        """A selection made where it is read: owned by its first reader."""
        reads = self.reads.get(node.index)
        if not reads:
            return node
        return replace(node, source_owner=min(reads, key=lambda read: cast(Any, read[0]))[1])

    def child(self, placed: _Placed, stub: int) -> None:
        local = placed.template.scopes[stub]
        template = link_template(local.space_type)
        self.place(template, placed, stub)

    # -- nodes --------------------------------------------------------------------------------

    def build(self, placed: _Placed, index: int, nid: NodeId) -> Node:
        template = placed.template
        node = template.nodes[index]
        scope = self.scope_of(placed, node.scope)
        key = member_key(placed.name, node.key)
        if node.kind == "param" and placed.parent is not None:
            return self.formal(placed, index, nid, key)
        selection = self.selection(placed, index)
        if selection is not None:
            choice = template.choices[selection[0]]
            if choice.index in placed.never:
                return self.never_select(placed, choice, selection[1], nid, key)
        never = template.selectors.get(index)
        if never is not None and never in placed.never:
            # A selector that never applies here: inapplicable, no key.
            return Node(
                nid,
                scope.index,
                key,
                "const",
                STRING,
                guard=self.guard(placed, node.guard),
                value=template.choices[never].never_value,
                origin=node.origin,
            )
        if index == template.validity:
            return self.mapped(placed, node, nid, scope.index, key, constraints=self.pieces(placed))
        if node.kind == "members":
            entries = [
                (label, target, value)
                for (label, target), value in zip(
                    node.alternatives, cast(tuple[Any, ...], node.value)
                )
                if not self.dropped_here(placed, target)
            ]
            return self.mapped(
                placed,
                replace(
                    node,
                    alternatives=tuple((label, target) for label, target, _ in entries),
                    value=tuple(value for _, _, value in entries),
                ),
                nid,
                scope.index,
                key,
            )
        return self.mapped(placed, node, nid, scope.index, key)

    def mapped(
        self,
        placed: _Placed,
        node: Node,
        nid: NodeId,
        scope: int,
        key: str,
        *,
        constraints: tuple[NodeId, ...] | None = None,
    ) -> Node:
        """A node of ``placed``'s template, its references resolved there."""

        def ref(index: int) -> NodeId:
            return self.reference(placed, index, owner=key)

        return replace(
            node,
            index=nid,
            scope=scope,
            key=key,
            source_owner=(
                None if node.source_owner is None else member_key(placed.name, node.source_owner)
            ),
            guard=self.guard(placed, node.guard),
            arguments=tuple(Argument(arg.name, ref(arg.node)) for arg in node.arguments),
            domain_arguments=tuple(
                Argument(arg.name, ref(arg.node)) for arg in node.domain_arguments
            ),
            output=None if node.output is None else ref(node.output),
            selector=None if node.selector is None else ref(node.selector),
            constraints=(
                tuple(ref(target) for target in node.constraints)
                if constraints is None
                else constraints
            ),
            alternatives=tuple((label, ref(target)) for label, target in node.alternatives),
        )

    def formal(self, placed: _Placed, index: int, nid: NodeId, key: str) -> Node:
        """A child's formal, as its placing template binds it."""
        node = placed.template.nodes[index]
        binding = cast(Binding, placed.binding(placed.template.formal_names[index]))
        if binding.kind == "literal":
            return Node(
                nid,
                placed.index,
                key,
                "const",
                node.semantics,
                guard=placed.guard,
                value=binding.value,
                origin=node.origin,
            )
        if binding.kind == "unsupplied":
            return Node(
                nid,
                placed.index,
                key,
                "present",
                node.semantics,
                guard=self.guard(placed, node.guard),
                required=node.required,
                origin=node.origin,
            )
        parent = cast(_Placed, placed.parent)
        written = parent.template.nodes[cast(int, binding.node)]
        built = self.mapped(
            parent, written, nid, placed.index, member_key(parent.name, written.key)
        )
        assert built.key == key, (built.key, key)
        return built

    def never_select(
        self, placed: _Placed, choice: TemplateChoice, member: str, nid: NodeId, key: str
    ) -> Node:
        """``decision.member`` through a choice that never applies here: no alternatives,
        typed as the first candidate class declaring the member, owned by its first reader."""
        if not choice.never_declared[member]:
            reader = member_key(placed.name, choice.readers[member][0].owner)
            raise DefinitionError.broken(
                "candidate-member-absent",
                _OWNER,
                f"{reader}: no candidate of {member_key(placed.name, choice.key)} has a "
                f"member {member}",
                declaration=reader,
            )
        return Node(
            nid,
            placed.index,
            key,
            "select",
            choice.never_semantics[member],
            guard=self.guard(placed, choice.guard),
            selector=self.reference(placed, choice.selector, owner=key),
        )

    def pieces(self, placed: _Placed) -> tuple[NodeId, ...]:
        """A scope's validity pieces in today's order: its constraints and decisions in
        node order (a formal where it is bound to a fresh Decision), the named shared
        Decisions its placing template owns here, its own, then each placed Space's
        validity in placement order."""
        template = placed.template
        named = set(template.named)
        own: list[NodeId] = []
        placed_validity: list[NodeId] = []
        for piece in template.nodes[template.validity].constraints:
            node = template.nodes[piece]
            if node.scope != 0:
                if not self.dropped_here(placed, piece):
                    placed_validity.append(self.reference(placed, piece, owner=placed.name))
            elif piece in named or piece not in placed.ids:
                continue
            elif node.kind == "param":
                if self.kind(placed, piece) == "decision":
                    own.append(placed.ids[piece])
            elif self.kind_at(placed, piece) in ("constraint", "decision"):
                own.append(placed.ids[piece])
        contributed: list[NodeId] = []
        parent = placed.parent
        if parent is not None:
            contributed = [
                parent.ids[index]
                for index in parent.template.named
                if parent.template.nodes[index].scope == placed.stub and index in parent.ids
            ]
        owned = [
            placed.ids[index]
            for index in template.named
            if template.nodes[index].scope == 0 and index in placed.ids
        ]
        return (*own, *contributed, *owned, *placed_validity)

    def kind_at(self, placed: _Placed, index: int) -> str:
        """An own node's kind here: a selector's ``const`` where its choice never applies."""
        if placed.template.selectors.get(index) in placed.never:
            return "const"
        return placed.template.nodes[index].kind

    # -- the model ------------------------------------------------------------------------------

    def model(self, root: _Placed, raw: list[Node]) -> LinkedModel:
        order = dependency_order(
            tuple(node.dependencies for node in raw), tuple(node.key for node in raw)
        )
        nodes, forward = _collapsed(raw)
        keys = {node.key: node.index for node in nodes}
        if len(keys) != len(nodes):
            raise DefinitionError.broken(
                "node-names-collide", _OWNER, "generated node names collide"
            )
        choices: list[Choice] = []
        indices: dict[tuple[int, int], int] = {}
        for placed in self.scopes:
            for choice in placed.template.choices:
                indices[(placed.index, choice.index)] = len(choices)
                choices.append(self.choice(placed, choice, len(choices)))
        scopes = tuple(self.scope(placed, indices) for placed in self.scopes)
        editable = frozenset(
            placed.ids[index]
            for placed in self.scopes
            for name, index in placed.template.formals.items()
            if (binding := placed.binding(name)) is not None and binding.kind == "named"
        )
        return LinkedModel(
            nodes,
            scopes,
            order,
            tuple(node.index for node in nodes if node.kind == "param"),
            tuple(self.decisions),
            keys,
            tuple(choices),
            editable,
            forward,
        )

    def choice(self, placed: _Placed, choice: TemplateChoice, index: int) -> Choice:
        never = choice.index in placed.never
        members = {
            member: node
            for (scope, choice_index, member), node in self.merged.items()
            if scope == placed.index and choice_index == choice.index
        }
        return Choice(
            index,
            placed.index,
            member_key(placed.name, choice.key),
            placed.ids[choice.selector],
            tuple(
                (case, None if stub is None or never else placed.children[stub].index)
                for case, stub in choice.cases
            ),
            members,
            self.guard(placed, choice.guard),
        )

    def scope(self, placed: _Placed, indices: Mapping[tuple[int, int], int]) -> Scope:
        own = placed.template.scopes[0]
        members = {key: placed.ids[node] for key, node in own.members.items() if node in placed.ids}
        named = {
            name: placed.ids[node] for name, node in own.named_members.items() if node in placed.ids
        }
        record: object = None
        parent = placed.parent
        if parent is not None:
            stub = parent.template.scopes[cast(int, placed.stub)]
            record = stub.record
            members.update(
                (key, parent.ids[node]) for key, node in stub.members.items() if node in parent.ids
            )
            named.update(
                (name, parent.ids[node])
                for name, node in stub.named_members.items()
                if node in parent.ids
            )
        return Scope(
            placed.index,
            None if parent is None else parent.index,
            placed.name,
            placed.template.space_type,
            members,
            {
                key: placed.children[stub].index
                for key, stub in own.children.items()
                if stub in placed.children
            },
            named,
            {name: placed.children[stub].index for name, stub in own.named_children.items()},
            placed.guard,
            {key: indices[(placed.index, choice)] for key, choice in own.choices.items()},
            record,
            placed.ids[placed.template.validity],
        )


def _collapsed(raw: list[Node]) -> tuple[tuple[Node, ...], tuple[NodeId, ...]]:
    """Collapse's bypass (``_collapse``), resolved node by node: an alias's output jumps to
    the end of its forwarding chain, and a reader's edge to a forwarding alias goes to its
    source where the alias's guard holds whenever the reader's does."""
    forward: list[NodeId | None] = [None] * len(raw)

    def source(index: NodeId) -> NodeId:
        chain: list[NodeId] = []
        current = index
        while forward[current] is None:
            node = raw[current]
            if not forwards(node):
                forward[current] = current
                break
            chain.append(current)
            output = cast(NodeId, node.output)
            if not (forwards(raw[output]) and guard_implies(raw, raw[output].guard, node.guard)):
                forward[current] = output
                break
            current = output
        found = cast(NodeId, forward[current])
        for index_ in reversed(chain):
            if forward[index_] is None:
                forward[index_] = found
        return cast(NodeId, forward[index])

    def bypass(reader: Node, target: NodeId, via: list[tuple[NodeId, NodeId]]) -> NodeId:
        end = source(target)
        if (
            end != target
            and forwards(raw[target])
            and guard_implies(raw, raw[target].guard, reader.guard)
        ):
            via.append((target, end))
            return end
        return target

    nodes: list[Node] = []
    for node in raw:
        if forwards(node):
            end = source(node.index)
            if end != node.output:
                node = replace(node, output=end, via=((cast(NodeId, node.output), end),))
            nodes.append(node)
            continue
        if node.kind == "alias":
            nodes.append(node)
            continue
        via: list[tuple[NodeId, NodeId]] = []
        changes: dict[str, Any] = {}
        for name in ("arguments", "domain_arguments"):
            arguments = cast(tuple[Argument, ...], getattr(node, name))
            rewritten = tuple(
                Argument(argument.name, bypass(node, argument.node, via)) for argument in arguments
            )
            if rewritten != arguments:
                changes[name] = rewritten
        if node.output is not None:
            output = bypass(node, node.output, via)
            if output != node.output:
                changes["output"] = output
        if node.alternatives:
            alternatives = tuple(
                (label, bypass(node, target, via)) for label, target in node.alternatives
            )
            if alternatives != node.alternatives:
                changes["alternatives"] = alternatives
        nodes.append(replace(node, via=tuple(via), **changes) if changes else node)
    return tuple(nodes), tuple(source(index) for index in range(len(raw)))


__all__ = ["Template", "TemplateChoice", "TemplateScope", "expand", "link_template"]
