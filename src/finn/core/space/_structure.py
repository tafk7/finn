# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A model's structure: its root class's template placed as scopes, each made when it is
addressed, each node resolved from its template when it is read.

**A scope** (``ScopeRecord``) is a template placed at a path: its index, its parent, its
stub in the parent's template, its key prefix, and its **base**, the first ``NodeId`` of a
block of one id per node of its template. The root is scope 0 at base 0. A scope is made
the first time a ``NodeId`` inside it, or its ``Scope``, is needed (an import followed, a
child's validity piece listed, a key looked up); making one allocates its block and
nothing else.

**A node is resolved when it is read**: its ``NodeId`` gives its scope (by its base) and
its template node, and that is resolved into the ``Node`` of its key (as the line's one
table of every placement held it), every reference mapped to a ``NodeId`` (a local node
in the block; an import, the child scope's node; a child's formal, as its placing
template binds it), collapse's bypass applied by ``_collapse``'s rule; it is kept. A
``NodeId`` that stands for another (an import, a formal's binding, the placement guard, a
selection made in another template) or for a node a placement excludes (a choice that
never applies there) is no node: reading it is an ``IndexError``.

A ``decision.member`` selection several templates make for one scope (the scope's own
template, a parent reading through it) is one node, owned in the choice's own template
where that template reads it (its first reader there), else in the nearest template
above that reads it. A template's first reader is the first in its declaration order
(the member each reader belongs to, in member order: ``_template._declared``), never the
order it was lowered in; placed where the choice never applies, the first that exists
there. A selection the candidates share is owned by the choice: a finding's owner is the
choice's key (``_allocate.shared_members``); where the choice never applies it has no
candidates, and its first reader owns it, as any other.

The structure is the model's linked view (``ir.Linked``): what the engine reads of a
linked model, with the same names and meanings (``nodes``, ``scopes``, ``choices``,
``keys``, ``selector_choices``, ``editable_aliases``, ``forward``, ``parameters``,
``decisions``, ``ranks``; ``header`` and ``whole``). The questions about the whole design
space are answered from templates and scope records, resolving no node: ``decisions``
(every choice, in declaration order), ``whole`` (the flow order, which ``ranks`` gives
positions in, and the presence readers: ``_whole``), ``header`` (a node as its template
states it), ``subtree``. A ``for`` over ``nodes``, ``scopes`` or ``choices`` addresses the
whole design space, and over ``nodes`` resolves it. Addressing and resolution write shared
state, under the model's lock.

``NodeId``s follow the order scopes are addressed, so nothing may order by them; two test
modes, read when a model is made, show it: ``FINN_SPACE_PERMUTE=1`` allocates the blocks
downward from a large base, each block's nodes numbered in reverse; ``FINN_SPACE_EAGER=1``
addresses every scope and resolves every node when the model is made. ``expand`` is the
eager form: every node of a Space class's design space in one table (``ir.LinkedModel``),
from its templates.
"""

from __future__ import annotations

import os
from bisect import bisect_right, insort
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import TYPE_CHECKING, Any, cast, overload

from ._collapse import forwards, guard_implies
from ._configuration import Space
from ._ordering import dependency_order
from ._table import Binding, member_key
from ._template import Template, TemplateChoice, link_template
from .errors import DefinitionError
from .ir import Argument, Choice, LinkedModel, Node, NodeId, NodeKind, Scope
from .semantics import STRING, ValueSemantics

if TYPE_CHECKING:
    from ._whole import Flow, Whole
    from .domains import Domain

_OWNER = "space.structure"

# The test modes (read when a model is made).
PERMUTE = "FINN_SPACE_PERMUTE"
EAGER = "FINN_SPACE_EAGER"

# Permuted: the first block ends here, and blocks are allocated downward from it.
_TOP = 1 << 40

# A scope's place in breadth-first order: its depth, then the path of its slot positions
# (each its parent's structure item and 0 for a child node, 1 + its case for a candidate).
Position = tuple[int, tuple[tuple[int, int], ...]]
# Above the root: before every scope.
_ABOVE: Position = (-1, ())

_UNSET: Any = object()


def modes() -> tuple[bool, bool]:
    """The test modes the environment asks for: (permuted, eager)."""
    return os.environ.get(PERMUTE) == "1", os.environ.get(EAGER) == "1"


@dataclass(eq=False)
class ScopeRecord:
    """A template placed at a path (see the module docstring). ``never``: its own choices
    that never apply here (guarded by the supply of a formal its placing template leaves
    unsupplied); ``children``: the scopes made for its placed stubs."""

    index: int
    parent: ScopeRecord | None
    stub: int | None
    name: str
    template: Template
    base: int
    position: Position
    never: frozenset[int]
    choice_base: int
    children: dict[int, ScopeRecord] = field(default_factory=dict)
    guard: Any = _UNSET

    def binding(self, formal: str) -> Binding | None:
        """How its placing template binds one of its formals (None for the root)."""
        if self.parent is None:
            return None
        return self.parent.template.scopes[cast(int, self.stub)].bindings[formal]

    def slot(self) -> tuple[int, int]:
        """Its position among its parent's placed scopes."""
        parent = cast(ScopeRecord, self.parent)
        return parent.template.index.slot_positions[cast(int, self.stub)]


@dataclass(frozen=True, slots=True)
class Header:
    """A node as its template and its scope's binding state it, its references not
    resolved (but its guard's identity): what the questions about the whole design space
    (``_whole``), inspection's lists and a handle read of a node. The node resolved holds
    the same values. ``record`` and ``node``: where its content is written (a formal bound
    by a node: the binding, a node of the placing template; else its own template's)."""

    index: NodeId
    scope: int
    key: str
    kind: NodeKind
    semantics: ValueSemantics[object] | None
    source_owner: str | None
    origin: str | None
    domain: Domain[object] | None
    required_choice: bool
    guard: NodeId | None
    record: ScopeRecord
    node: int

    @property
    def owner(self) -> str:
        return self.source_owner if self.source_owner is not None else self.key


class Structure:
    """A model's scopes and nodes, made and resolved on demand; its linked view
    (``ir.Linked``). See the module docstring."""

    def __init__(
        self,
        root: Template,
        lock: RLock | None = None,
        *,
        permuted: bool = False,
        eager: bool = False,
    ) -> None:
        self.template = root
        self.lock = RLock() if lock is None else lock
        self.permuted = permuted
        self.records: list[ScopeRecord] = []
        # The blocks by base (sorted), for a NodeId's scope.
        self._bases: list[int] = []
        self._by_base: dict[int, ScopeRecord] = {}
        self._next = _TOP if permuted else 0
        self.allocated = 0  # NodeIds allocated: one block per addressed scope
        self._choice_owners: list[tuple[ScopeRecord, int]] = []
        # A selection's one node by (scope, choice, member), and its owner.
        self._merged: dict[tuple[int, int, str], tuple[NodeId, str | None] | None] = {}
        self._owners: dict[NodeId, str] = {}
        self._raw: dict[NodeId, Node] = {}
        self._nodes: dict[NodeId, Node] = {}
        self._forward: dict[NodeId, NodeId] = {}
        self._scopes: dict[int, Scope] = {}
        self._choices: dict[int, Choice] = {}
        self._in_play: set[int] = set()
        self._headers: dict[NodeId, Header] = {}
        self._decisions: tuple[NodeId, ...] | None = None
        self._whole: Whole | None = None
        self.nodes = _Nodes(self)
        self.raw_nodes = _RawNodes(self)
        self.scopes = _Scopes(self)
        self.choices = _Choices(self)
        self.keys = _Keys(self)
        self.selector_choices = _SelectorChoices(self)
        self.editable_aliases = _EditableAliases(self)
        self.forward = _Forward(self)
        self.ranks = _Ranks(self)
        with self.lock:
            root_scope = self._make(root, None, None)
            self.parameters: tuple[NodeId, ...] = tuple(
                self._id(root_scope, node) for node in root.formals.values()
            )
            if eager:
                self.resolve_all()

    # -- scopes ------------------------------------------------------------------------------

    def _make(
        self, template: Template, parent: ScopeRecord | None, stub: int | None
    ) -> ScopeRecord:
        size = len(template.nodes)
        if self.permuted:
            self._next -= size
            base = self._next
        else:
            base, self._next = self._next, self._next + size
        self.allocated += size
        position: Position
        if parent is None:
            name, position = "", (0, ())
        else:
            local = parent.template.scopes[cast(int, stub)]
            name = member_key(parent.name, local.name)
            depth, path = parent.position
            position = (depth + 1, (*path, parent.template.index.slot_positions[cast(int, stub)]))
        record = ScopeRecord(
            len(self.records),
            parent,
            stub,
            name,
            template,
            base,
            position,
            frozenset(),
            len(self._choice_owners),
        )
        record.never = frozenset(
            choice.index
            for choice in template.choices
            if choice.never_formal is not None
            and (binding := record.binding(choice.never_formal)) is not None
            and binding.kind == "unsupplied"
        )
        self.records.append(record)
        insort(self._bases, base)
        self._by_base[base] = record
        self._choice_owners.extend((record, choice.index) for choice in template.choices)
        if parent is not None:
            parent.children[cast(int, stub)] = record
        return record

    def child(self, record: ScopeRecord, stub: int) -> ScopeRecord | None:
        """The scope placed at a placed stub of ``record``'s template, made if needed;
        None where it is a candidate of a choice that never applies there."""
        found = record.children.get(stub)
        if found is not None:
            return found
        with self.lock:
            found = record.children.get(stub)
            if found is not None:
                return found
            local = record.template.scopes[stub]
            if local.candidate is not None and local.candidate[0] in record.never:
                return None
            return self._make(link_template(local.space_type), record, stub)

    def scope_of(self, record: ScopeRecord, local: int) -> ScopeRecord | None:
        """The scope standing at a local scope of ``record``'s template (a nested stub
        through its parent stubs), or None where one on the way is not placed."""
        if local == 0:
            return record
        scope = record.template.scopes[local]
        if scope.parent == 0:
            return self.child(record, local)
        above = self.scope_of(record, cast(int, scope.parent))
        return None if above is None else self.child(above, cast(int, scope.slot))

    def guard_of(self, record: ScopeRecord) -> NodeId | None:
        """The guard a scope is placed under (None where it is unguarded)."""
        if record.guard is _UNSET:
            with self.lock:
                parent = record.parent
                if parent is None:
                    record.guard = None
                else:
                    local = parent.template.scopes[cast(int, record.stub)]
                    record.guard = None if local.guard is None else self.ident(parent, local.guard)
        return cast("NodeId | None", record.guard)

    def walk(self, record: ScopeRecord, steps: Sequence[int]) -> ScopeRecord | None:
        """The scope reached from ``record`` through placed stubs ``steps`` (None where one
        is not placed)."""
        current: ScopeRecord | None = record
        for stub in steps:
            if current is None:
                return None
            current = self.child(current, stub)
        return current

    def address_all(self) -> None:
        """Make every scope (the whole design space's records)."""
        with self.lock:
            cursor = 0
            while cursor < len(self.records):
                record = self.records[cursor]
                cursor += 1
                for stub in record.template.index.placed:
                    self.child(record, stub)

    # -- identities ----------------------------------------------------------------------------

    def _id(self, record: ScopeRecord, node: int) -> NodeId:
        if self.permuted:
            return record.base + len(record.template.nodes) - 1 - node
        return record.base + node

    def locate(self, index: NodeId) -> tuple[ScopeRecord, int]:
        """The scope whose block holds ``index``, and its template node there."""
        if type(index) is not int:
            raise IndexError(index)
        with self.lock:
            position = bisect_right(self._bases, index) - 1
            if position < 0:
                raise IndexError(index)
            record = self._by_base[self._bases[position]]
        size = len(record.template.nodes)
        offset = index - record.base
        if offset >= size:
            raise IndexError(index)
        return record, (size - 1 - offset if self.permuted else offset)

    def ident(self, record: ScopeRecord, node: int) -> NodeId | None:
        """The ``NodeId`` a node of ``record``'s template stands for there; None where the
        placement excludes it. The placement guard is the scope's guard."""
        with self.lock:
            template = record.template
            if node == template.placement_guard:
                return self.guard_of(record)
            spec = template.nodes[node]
            if spec.scope == 0:
                return self._own(record, node)
            if node in template.contributed:
                return self._contributed(record, node)
            child = self.scope_of(record, spec.scope)
            if child is None:
                return None
            if spec.kind == "import":
                return self.ident(child, template.imports[node])
            # A formal's binding: the child's formal.
            return self.ident(child, child.template.formals[template.binding_names[node]])

    def _own(self, record: ScopeRecord, node: int) -> NodeId | None:
        template = record.template
        selection = template.index.selections.get(node)
        if selection is not None:
            choice = template.choices[selection[0]]
            if node in choice.shared and choice.index not in record.never:
                return self._id(record, node)  # its candidates' (kept, if dropped, unread)
            found = self.merged(record, *selection)
            return None if found is None else found[0]
        if template.requires.get(node) in record.never:
            return None
        return self._id(record, node)

    def _dropped(self, record: ScopeRecord, node: int) -> bool:
        """Whether a node of a stub is below a candidate of a choice that never applies."""
        spec = record.template.nodes[node]
        return spec.scope != 0 and record.template.scopes[spec.scope].requires in record.never

    def _contributed(self, record: ScopeRecord, node: int) -> NodeId | None:
        template = record.template
        if self._dropped(record, node):
            return None
        scope = self.scope_of(record, template.nodes[node].scope)
        if scope is None:
            return None
        selection = template.stub_selects.get(node)
        if selection is not None:
            found = self.merged(scope, selection[1], selection[2])
            return None if found is None else found[0]
        if template.requires.get(node) in record.never:
            return None
        return self._id(record, node)

    def merged(
        self, scope: ScopeRecord, choice_index: int, member: str
    ) -> tuple[NodeId, str | None] | None:
        """The one node of ``decision.member`` through ``scope``'s choice, and its owner:
        the selection its candidates share, where the choice applies; else the choice's own
        template's, where it reads it there (its first reader there); else the nearest
        template above's that reads it. None where nothing reads it."""
        key = (scope.index, choice_index, member)
        if key in self._merged:
            return self._merged[key]
        with self.lock:
            found = self._merge(scope, choice_index, member)
            self._merged[key] = found
            if found is not None and found[1] is not None:
                self._owners[found[0]] = found[1]
            return found

    def _merge(
        self, scope: ScopeRecord, choice_index: int, member: str
    ) -> tuple[NodeId, str | None] | None:
        template = scope.template
        choice = template.choices[choice_index]
        own = template.index.own_selection.get((choice_index, member))
        if own is not None:
            if own in choice.shared and choice_index not in scope.never:
                # A selection its candidates share, unless dropped for disagreeing types.
                return (self._id(scope, own), None) if choice.members.get(member) == own else None
            owner = self._reader(scope, choice.readers.get(member, ()))
            if owner is not None:
                return self._id(scope, own), owner
        # The templates above: each that reaches the scope through its stubs.
        steps = [cast(int, scope.stub)]
        above = scope.parent
        while above is not None:
            local = self._local(above.template, steps)
            if local is not None:
                node = above.template.index.stub_selections.get((local, choice_index, member))
                if node is not None and not self._dropped(above, node):
                    owner = self._reader(above, above.template.stub_reads[node])
                    if owner is not None:
                        return self._id(above, node), owner
            if above.stub is None:
                break
            steps.insert(0, above.stub)
            above = above.parent
        return None

    @staticmethod
    def _local(template: Template, steps: Sequence[int]) -> int | None:
        """The stub of ``template`` standing for the scope ``steps`` reach (a placed stub,
        then nested stubs), if the template has one."""
        local = steps[0]
        nested = template.index.nested
        for slot in steps[1:]:
            found = nested.get((local, slot))
            if found is None:
                return None
            local = found
        return local

    @staticmethod
    def _reader(record: ScopeRecord, reads: Sequence[Any]) -> str | None:
        """The first read of ``reads`` (``_table.Read``) that exists here: its owner's key."""
        for read in reads:
            if read.requires not in record.never:
                return member_key(record.name, read.owner)
        return None

    def _reference(self, record: ScopeRecord, node: int, owner: str) -> NodeId:
        found = self.ident(record, node)
        if found is None:
            name = member_key(record.name, record.template.nodes[node].key)
            raise DefinitionError.broken(
                "reference-not-placed",
                _OWNER,
                f"{owner}: {name} is not placed",
                declaration=owner,
            )
        return found

    def canonical(self, index: NodeId) -> tuple[ScopeRecord, int]:
        """``index``'s scope and template node, where it is a node (else ``IndexError``)."""
        record, node = self.locate(index)
        if self.ident(record, node) != index:
            raise IndexError(index)
        return record, node

    def holds_scope(self, index: object) -> bool:
        """Whether ``index`` is an addressed scope of this model (scopes are numbered from
        0 in the order they are addressed)."""
        return type(index) is int and 0 <= index < len(self.records)

    def holds_node(self, index: object) -> bool:
        """Whether ``index`` is a node of this model: in the block of an addressed scope, and
        a node there (``canonical``), not an identity standing for another node (an import,
        a formal's binding, the placement guard, a selection made in another template) or
        for one its placement excludes. The same question in every mode."""
        if type(index) is not int:
            return False
        try:
            self.canonical(index)
        except IndexError:
            return False
        return True

    # -- resolution ----------------------------------------------------------------------------

    def node(self, index: NodeId) -> Node:
        """``index``'s node, resolved and collapsed (kept)."""
        found = self._nodes.get(index)
        if found is not None:
            return found
        with self.lock:
            found = self._nodes.get(index)
            if found is None:
                found = self._nodes[index] = self._collapsed(self.raw(index))
            return found

    def raw(self, index: NodeId) -> Node:
        """``index``'s node as its template resolves it here, before collapse (kept)."""
        found = self._raw.get(index)
        if found is not None:
            return found
        with self.lock:
            found = self._raw.get(index)
            if found is None:
                record, node = self.canonical(index)
                found = self._build(record, node, index)
                owner = self._owners.get(index)
                if owner is not None:
                    found = replace(found, source_owner=owner)
                self._raw[index] = found
                self._in_play.add(found.scope)
            return found

    def _build(self, record: ScopeRecord, index: int, nid: NodeId) -> Node:
        template = record.template
        node = template.nodes[index]
        key = member_key(record.name, node.key)
        if node.kind == "param" and record.parent is not None:
            return self._formal(record, index, nid, key)
        scope = cast(ScopeRecord, self.scope_of(record, node.scope))
        selection = template.index.selections.get(index)
        if selection is not None and selection[0] in record.never:
            return self._never_select(record, template.choices[selection[0]], index, nid, key)
        never = template.selectors.get(index)
        if never is not None and never in record.never:
            # A selector that never applies here: inapplicable, no key.
            return Node(
                nid,
                scope.index,
                key,
                "const",
                STRING,
                guard=self._guard(record, node.guard, key),
                value=template.choices[never].never_value,
                origin=node.origin,
            )
        if index == template.validity:
            return self._mapped(
                record, node, nid, scope.index, key, constraints=self._pieces(record)
            )
        if node.kind == "members":
            entries = [
                (label, target, value)
                for (label, target), value in zip(
                    node.alternatives, cast(tuple[Any, ...], node.value)
                )
                if not self._dropped(record, target)
            ]
            node = replace(
                node,
                alternatives=tuple((label, target) for label, target, _ in entries),
                value=tuple(value for _, _, value in entries),
            )
        return self._mapped(record, node, nid, scope.index, key)

    def _guard(self, record: ScopeRecord, node: int | None, owner: str) -> NodeId | None:
        if node is None:
            return None
        if node == record.template.placement_guard:
            return self.guard_of(record)
        return self._reference(record, node, owner)

    def _mapped(
        self,
        record: ScopeRecord,
        node: Node,
        nid: NodeId,
        scope: int,
        key: str,
        *,
        constraints: tuple[NodeId, ...] | None = None,
    ) -> Node:
        """A node of ``record``'s template, its references resolved there."""

        def ref(index: int) -> NodeId:
            return self._reference(record, index, key)

        return replace(
            node,
            index=nid,
            scope=scope,
            key=key,
            source_owner=(
                None if node.source_owner is None else member_key(record.name, node.source_owner)
            ),
            guard=self._guard(record, node.guard, key),
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

    def _formal(self, record: ScopeRecord, index: int, nid: NodeId, key: str) -> Node:
        """A child's formal, as its placing template binds it."""
        node = record.template.nodes[index]
        binding = cast(Binding, record.binding(record.template.formal_names[index]))
        if binding.kind == "literal":
            return Node(
                nid,
                record.index,
                key,
                "const",
                node.semantics,
                guard=self.guard_of(record),
                value=binding.value,
                origin=node.origin,
            )
        if binding.kind == "unsupplied":
            return Node(
                nid,
                record.index,
                key,
                "present",
                node.semantics,
                guard=self._guard(record, node.guard, key),
                required=node.required,
                origin=node.origin,
            )
        parent = cast(ScopeRecord, record.parent)
        written = parent.template.nodes[cast(int, binding.node)]
        return self._mapped(parent, written, nid, record.index, key)

    def _never_select(
        self, record: ScopeRecord, choice: TemplateChoice, node: int, nid: NodeId, key: str
    ) -> Node:
        """``decision.member`` through a choice that never applies here: no alternatives, and
        the type the template checked its readers against. Its guard is the choice's,
        ``supplied`` of the formal left unsupplied here, which never holds: the selection,
        and whatever reads it, is inapplicable, so its value is never read here, and its
        type is the same at every placement (its readers are checked once, in the
        template)."""
        return Node(
            nid,
            record.index,
            key,
            "select",
            record.template.nodes[node].semantics,
            guard=self._guard(record, choice.guard, key),
            selector=self._reference(record, choice.selector, key),
        )

    def _kind(self, record: ScopeRecord, node: int) -> str:
        """An own node's kind here: a formal's by its binding; a selector's ``const`` where
        its choice never applies."""
        template = record.template
        spec = template.nodes[node]
        if spec.kind == "param":
            binding = record.binding(template.formal_names[node])
            if binding is None:
                return "param"
            return {
                "literal": "const",
                "reference": "alias",
                "named": "alias",
                "local-decision": "decision",
                "unsupplied": "present",
            }[binding.kind]
        if template.selectors.get(node) in record.never:
            return "const"
        return spec.kind

    def _pieces(self, record: ScopeRecord) -> tuple[NodeId, ...]:
        """A scope's validity pieces in today's order: its constraints and decisions in
        node order (a formal where it is bound to a fresh Decision), the named shared
        Decisions its placing template owns here, its own, then each placed Space's
        validity in placement order."""
        template = record.template
        named = set(template.named)
        own: list[NodeId] = []
        placed: list[NodeId] = []
        for piece in template.nodes[template.validity].constraints:
            node = template.nodes[piece]
            if node.scope != 0:
                if not self._dropped(record, piece):
                    placed.append(self._reference(record, piece, record.name))
                continue
            if piece in named or self.ident(record, piece) is None:
                continue
            if self._kind(record, piece) in ("constraint", "decision"):
                own.append(self._id(record, piece))
        contributed: list[NodeId] = []
        parent = record.parent
        if parent is not None:
            for index in parent.template.named:
                if parent.template.nodes[index].scope == record.stub:
                    found = self.ident(parent, index)
                    if found is not None:
                        contributed.append(found)
        owned = [
            found
            for index in template.named
            if template.nodes[index].scope == 0 and (found := self.ident(record, index)) is not None
        ]
        return (*own, *contributed, *owned, *placed)

    def _collapsed(self, node: Node) -> Node:
        """Collapse's bypass (``_collapse``), for one node: an alias's output jumps to the
        end of its forwarding chain, and a reader's edge to a forwarding alias goes to its
        source where the alias's guard holds whenever the reader's does."""
        raw = self.raw_nodes
        if forwards(node):
            end = self.source(node.index)
            if end != node.output:
                return replace(node, output=end, via=((cast(NodeId, node.output), end),))
            return node
        if node.kind == "alias":
            return node
        via: list[tuple[NodeId, NodeId]] = []

        def bypass(target: NodeId) -> NodeId:
            end = self.source(target)
            if (
                end != target
                and forwards(raw[target])
                and guard_implies(raw, raw[target].guard, node.guard)
            ):
                via.append((target, end))
                return end
            return target

        changes: dict[str, Any] = {}
        for name in ("arguments", "domain_arguments"):
            arguments = cast(tuple[Argument, ...], getattr(node, name))
            rewritten = tuple(
                Argument(argument.name, bypass(argument.node)) for argument in arguments
            )
            if rewritten != arguments:
                changes[name] = rewritten
        if node.output is not None:
            output = bypass(node.output)
            if output != node.output:
                changes["output"] = output
        if node.alternatives:
            alternatives = tuple((label, bypass(target)) for label, target in node.alternatives)
            if alternatives != node.alternatives:
                changes["alternatives"] = alternatives
        return replace(node, via=tuple(via), **changes) if changes else node

    def source(self, index: NodeId) -> NodeId:
        """``index``'s forwarding source: an alias chain collapsed to its source (itself for
        anything but a forwarding alias)."""
        found = self._forward.get(index)
        if found is not None:
            return found
        with self.lock:
            raw = self.raw_nodes
            chain: list[NodeId] = []
            current = index
            while True:
                held = self._forward.get(current)
                if held is not None:
                    found = held
                    break
                node = raw[current]
                if not forwards(node):
                    found = self._forward[current] = current
                    break
                chain.append(current)
                output = cast(NodeId, node.output)
                if not (
                    forwards(raw[output]) and guard_implies(raw, raw[output].guard, node.guard)
                ):
                    found = self._forward[current] = output
                    break
                current = output
            for item in chain:
                self._forward.setdefault(item, found)
            return self._forward[index]

    # -- the whole design space ------------------------------------------------------------------

    def all_nodes(self) -> Iterator[NodeId]:
        """Every node: every scope addressed, every node of each block that is one."""
        with self.lock:
            self.address_all()
            for record in tuple(self.records):
                for node in range(len(record.template.nodes)):
                    index = self._id(record, node)
                    if self.ident(record, node) == index:
                        yield index

    def resolve_all(self) -> None:
        """Address every scope and resolve every node (the eager mode)."""
        with self.lock:
            for index in tuple(self.all_nodes()):
                self.node(index)
            for index in range(len(self.records)):
                self.scopes[index]
            for index in range(len(self._choice_owners)):
                self.choices[index]

    def header(self, index: NodeId) -> Header:
        """``index``'s node as its template and its scope's binding state it, its references
        not resolved (``Header``; kept). ``IndexError`` where it is no node."""
        found = self._headers.get(index)
        if found is not None:
            return found
        with self.lock:
            found = self._headers.get(index)
            if found is None:
                found = self._headers[index] = self._header(index)
            return found

    def _header(self, index: NodeId) -> Header:
        # As ``_build`` resolves the node, branch by branch, but for its references.
        record, at = self.canonical(index)
        template = record.template
        node = template.nodes[at]
        key = member_key(record.name, node.key)
        owner = self._owners.get(index)
        if node.kind == "param" and record.parent is not None:
            binding = cast(Binding, record.binding(template.formal_names[at]))
            if binding.kind in ("literal", "unsupplied"):
                kind: NodeKind = "const" if binding.kind == "literal" else "present"
                return Header(
                    index,
                    record.index,
                    key,
                    kind,
                    node.semantics,
                    owner,
                    node.origin,
                    None,
                    False,
                    self._guard(record, node.guard, key),
                    record,
                    at,
                )
            parent = record.parent
            written = parent.template.nodes[cast(int, binding.node)]
            return Header(
                index,
                record.index,
                key,
                written.kind,
                written.semantics,
                owner if owner is not None else self._owned(parent, written),
                written.origin,
                written.domain,
                written.required_choice,
                self._guard(parent, written.guard, key),
                parent,
                cast(int, binding.node),
            )
        selection = template.index.selections.get(at)
        if selection is not None and selection[0] in record.never:
            choice = template.choices[selection[0]]
            return Header(
                index,
                record.index,
                key,
                "select",
                node.semantics,
                owner,
                None,
                None,
                False,
                self._guard(record, choice.guard, key),
                record,
                at,
            )
        scope = cast(ScopeRecord, self.scope_of(record, node.scope)).index
        never = template.selectors.get(at)
        if never is not None and never in record.never:
            return Header(
                index,
                scope,
                key,
                "const",
                STRING,
                owner,
                node.origin,
                None,
                False,
                self._guard(record, node.guard, key),
                record,
                at,
            )
        return Header(
            index,
            scope,
            key,
            node.kind,
            node.semantics,
            owner if owner is not None else self._owned(record, node),
            node.origin,
            node.domain,
            node.required_choice,
            self._guard(record, node.guard, key),
            record,
            at,
        )

    @staticmethod
    def _owned(record: ScopeRecord, node: Node) -> str | None:
        return None if node.source_owner is None else member_key(record.name, node.source_owner)

    def subtree(self, scope: int) -> frozenset[int]:
        """The scope ``scope`` and every scope below it, by index (every scope addressed,
        records only)."""
        with self.lock:
            self.address_all()
            inside = {scope}
            for record in self.records:  # a scope is made after its parent
                if record.parent is not None and record.parent.index in inside:
                    inside.add(record.index)
            return frozenset(inside)

    @property
    def whole(self) -> Whole:
        """Its answers to the questions about the whole design space (``_whole``), from its
        templates."""
        if self._whole is None:
            from ._whole import Whole  # noqa: PLC0415 - a cycle

            with self.lock:
                if self._whole is None:
                    self._whole = Whole(self)
        return self._whole

    def flow(self) -> Flow:
        """Its choices in the flow order, with its loops (``_whole.Whole.flow``)."""
        return self.whole.flow()

    def readers(self, source: NodeId) -> tuple[NodeId, ...]:
        """The choices whose presence may read ``source`` (``_whole.Whole.readers``)."""
        return self.whole.readers(source)

    @property
    def decisions(self) -> tuple[NodeId, ...]:
        """Every decision in declaration order (``rank``), from its templates: every scope
        addressed, no node resolved."""
        with self.lock:
            if self._decisions is None:
                self.address_all()
                found: list[NodeId] = []
                for record in self.records:
                    template = record.template
                    for node in range(len(template.nodes)):
                        kind = template.nodes[node].kind
                        if kind not in ("decision", "param"):
                            continue
                        index = self.ident(record, node)
                        if (
                            index == self._id(record, node)
                            and self.header(index).kind == "decision"
                        ):
                            found.append(index)
                self._decisions = tuple(sorted(found, key=self.rank))
            return self._decisions

    def case_of(self, choice: int, guard: NodeId) -> tuple[NodeId, str] | None:
        """The selector and case of choice ``choice`` where ``guard`` holds, if it is the
        guard a case's scope is placed under (directly or under its ``when``) that reads the
        selector alone (``_reads.case_guards``)."""
        with self.lock:
            record, index = self._choice_owners[choice]
            spec = record.template.choices[index]
            if index in record.never:
                return None
            selector = self._id(record, spec.selector)
            selected = (Argument("selected", selector),)
            for case, stub in spec.cases:
                child = None if stub is None else self.child(record, stub)
                current = None if child is None else self.guard_of(child)
                while current is not None:
                    node = self.nodes[current]
                    if node.kind == "derived" and node.arguments == selected:
                        if current == guard:
                            return selector, case
                        break
                    current = node.guard
            return None

    def rank(self, index: NodeId) -> tuple[object, ...]:
        """A decision's place in declaration order, from its scope's path (never from its
        ``NodeId``): a scope's own decisions in member order when the scope is placed, a
        selector when its scope's members are placed (its candidates after it), the named
        shared Decisions last, each in its declaring template's order; scopes breadth
        first (``Position``)."""
        record, node = self.canonical(index)
        index_ = record.template.index
        named = index_.named_positions.get(node)
        if named is not None:
            return (1, record.position, named)
        choice = record.template.selectors.get(node)
        if choice is not None:
            return (0, record.position, index_.choice_positions[choice], 0, -1)
        member = index_.value_positions.get(node)
        if member is None:
            raise KeyError(index)
        if record.parent is None:
            return (0, _ABOVE, 0, 0, member)
        return (0, record.parent.position, *record.slot(), member)

    def counts(self) -> dict[str, int]:
        """What is made so far: scopes addressed, scopes with a node resolved, nodes
        resolved, and how many of those are constants."""
        with self.lock:
            return {
                "scopes addressed": len(self.records),
                "scopes in play": len(self._in_play),
                "nodes resolved": len(self._raw),
                "const nodes resolved": sum(
                    1 for node in self._raw.values() if node.kind == "const"
                ),
            }

    # -- statistics ----------------------------------------------------------------------------

    def statistics(self) -> tuple[int, int, int, int, int, int]:
        """``inspection.statistics``' counts: authored declarations, scopes, nodes, potential
        edges, Decisions over nodes, owning decisions. Each template is counted at every
        placement: once per **shape** of placement (its template, the Decisions over nodes
        that never apply there, whether it is placed under a guard, its formals bound to
        fresh Decisions, the named shared Decisions its placing template owns in it), on
        one scope of that shape (its record: no node resolved), times the number of
        placements of that shape. A formal's node is counted where its binding is written,
        by its placing template. A node's potential edges are its references as its
        template states them, each to its own node (two references of one node that
        forward to one source count twice: the one table of every placement counted them
        after collapse)."""
        with self.lock:
            totals: dict[tuple[object, ...], tuple[int, ...]] = {}
            return cast(tuple[int, int, int, int, int, int], self._counted(self.records[0], totals))

    def _shape(self, record: ScopeRecord) -> tuple[object, ...]:
        """What ``record``'s scope and the scopes below it count by: its template, its
        Decisions over nodes that never apply, whether it is placed under a guard, its
        formals bound to fresh Decisions, the named shared Decisions owned in it."""
        template = record.template
        decided = frozenset(
            name
            for name in template.formals
            if (binding := record.binding(name)) is not None and binding.kind == "local-decision"
        )
        return (
            id(template),
            record.never,
            self.guard_of(record) is not None,
            decided,
            self._owned_here(record),
        )

    def _owned_here(self, record: ScopeRecord) -> int:
        """How many named shared Decisions its placing template owns in ``record``'s
        scope."""
        parent = record.parent
        if parent is None:
            return 0
        return sum(
            1
            for node in parent.template.scopes[cast(int, record.stub)].named_members.values()
            if self.ident(parent, node) is not None
        )

    def _counted(
        self, record: ScopeRecord, totals: dict[tuple[object, ...], tuple[int, ...]]
    ) -> tuple[int, ...]:
        """The counts of ``record``'s scope and of the scopes below it (by shape, once)."""
        shape = self._shape(record)
        held = totals.get(shape)
        if held is not None:
            return held
        template = record.template
        own = template.scopes[0]
        authored = sum(
            1 for node in own.named_members.values() if self.ident(record, node) is not None
        )
        nodes = edges = decisions = 0
        for node in range(len(template.nodes)):
            index = self.ident(record, node)
            if index is None or index != self._id(record, node):
                continue
            nodes += 1
            if node in template.formal_names and record.parent is not None:
                continue  # counted by its placing template, where its binding is written
            edges += self._edges(record, node)
            if self.header(index).kind == "decision":
                decisions += 1
        found = [authored, 1, nodes, edges, len(template.choices), decisions]
        for stub in template.index.placed:
            child = self.child(record, stub)
            if child is None:
                continue
            found[0] += 1 + self._owned_here(child)  # the placement, the names owned there
            for name in child.template.formals:
                index = cast(NodeId, self.ident(child, child.template.formals[name]))
                found[3] += self._edges(child, child.template.formals[name])
                found[5] += self.header(index).kind == "decision"
            for position, count in enumerate(self._counted(child, totals)):
                found[position] += count
        result = totals[shape] = tuple(found)
        return result

    def _edges(self, record: ScopeRecord, node: int) -> int:
        """How many nodes a node of ``record``'s scope references (``Node.dependencies``,
        before collapse), from its template: its references mapped to their identities,
        no node resolved (as ``_build`` maps them)."""
        template = record.template
        spec = template.nodes[node]
        if spec.kind == "param" and record.parent is not None:
            binding = cast(Binding, record.binding(template.formal_names[node]))
            if binding.kind in ("literal", "unsupplied"):  # a constant, or not supplied
                return 0 if self._guard(record, spec.guard, spec.key) is None else 1
            parent = record.parent
            return self._mapped_edges(parent, parent.template.nodes[cast(int, binding.node)])
        selection = template.index.selections.get(node)
        if selection is not None and selection[0] in record.never:
            choice = template.choices[selection[0]]
            guard = self._guard(record, choice.guard, spec.key)
            return 1 + (guard is not None)  # its selector and its guard
        never = template.selectors.get(node)
        if never is not None and never in record.never:  # a constant
            return 0 if self._guard(record, spec.guard, spec.key) is None else 1
        if node == template.validity:
            return self._mapped_edges(record, spec, self._pieces(record))
        if spec.kind == "members":
            spec = replace(
                spec,
                alternatives=tuple(
                    (label, target)
                    for label, target in spec.alternatives
                    if not self._dropped(record, target)
                ),
            )
        return self._mapped_edges(record, spec)

    def _mapped_edges(
        self, record: ScopeRecord, node: Node, constraints: tuple[NodeId, ...] | None = None
    ) -> int:
        def ref(index: int) -> NodeId:
            return self._reference(record, index, node.key)

        found: list[NodeId | None] = [
            *(ref(argument.node) for argument in node.arguments),
            *(ref(argument.node) for argument in node.domain_arguments),
            self._guard(record, node.guard, node.key),
            None if node.output is None else ref(node.output),
            None if node.selector is None else ref(node.selector),
            *((ref(target) for target in node.constraints) if constraints is None else constraints),
            *(ref(target) for _, target in node.alternatives),
        ]
        return len({index for index in found if index is not None})

    # -- the linked view's pieces ------------------------------------------------------------------

    def scope(self, index: int) -> Scope:
        found = self._scopes.get(index)
        if found is not None:
            return found
        with self.lock:
            found = self._scopes.get(index)
            if found is None:
                found = self._scopes[index] = self._scope(self.records[index])
            return found

    def _scope(self, record: ScopeRecord) -> Scope:
        template = record.template
        own = template.scopes[0]
        members: dict[object, NodeId] = {}
        named: dict[str, NodeId] = {}
        for member, node in own.members.items():
            found = self.ident(record, node)
            if found is not None:
                members[member] = found
        for name, node in own.named_members.items():
            found = self.ident(record, node)
            if found is not None:
                named[name] = found
        record_: object = None
        parent = record.parent
        if parent is not None:
            stub = parent.template.scopes[cast(int, record.stub)]
            record_ = stub.record
            for member, node in stub.members.items():
                found = self.ident(parent, node)
                if found is not None:
                    members[member] = found
            for name, node in stub.named_members.items():
                found = self.ident(parent, node)
                if found is not None:
                    named[name] = found
        children: dict[object, int] = {}
        for key, stub_ in own.children.items():
            child = self.child(record, stub_)
            if child is not None:
                children[key] = child.index
        named_children: dict[str, int] = {}
        for name, stub_ in own.named_children.items():
            child = self.child(record, stub_)
            if child is not None:
                named_children[name] = child.index
        return Scope(
            record.index,
            None if parent is None else parent.index,
            record.name,
            template.space_type,
            members,
            children,
            named,
            named_children,
            self.guard_of(record),
            {key: record.choice_base + choice for key, choice in own.choices.items()},
            record_,
            self._id(record, template.validity),
        )

    def choice(self, index: int) -> Choice:
        found = self._choices.get(index)
        if found is not None:
            return found
        with self.lock:
            found = self._choices.get(index)
            if found is None:
                record, choice = self._choice_owners[index]
                found = self._choices[index] = self._choice(
                    record, record.template.choices[choice], index
                )
            return found

    def _choice(self, record: ScopeRecord, choice: TemplateChoice, index: int) -> Choice:
        never = choice.index in record.never
        names = dict.fromkeys(
            member
            for node, member in choice.selects.items()
            if record.template.index.selections.get(node) == (choice.index, member)
        )
        steps = [cast(int, record.stub)]
        above = record.parent
        while above is not None:
            local = self._local(above.template, steps)
            if local is not None:
                names.update(
                    dict.fromkeys(above.template.index.stub_members.get((local, choice.index), ()))
                )
            if above.stub is None:
                break
            steps.insert(0, above.stub)
            above = above.parent
        members: dict[str, NodeId] = {}
        for member in names:
            found = self.merged(record, choice.index, member)
            if found is not None:
                members[member] = found[0]
        cases: list[tuple[str, int | None]] = []
        for case, stub in choice.cases:
            child = None if stub is None or never else self.child(record, stub)
            cases.append((case, None if child is None else child.index))
        return Choice(
            index,
            record.index,
            member_key(record.name, choice.key),
            self._id(record, choice.selector),
            tuple(cases),
            members,
            self._guard(record, choice.guard, member_key(record.name, choice.key)),
        )

    def find(self, key: str) -> NodeId:
        """The node of a key: matched against each template's placed stubs' names (which
        hold dots: ``compute.packed``), longest first, scope by scope; then against the
        template's own keys. Never by splitting the key on its dots alone."""
        with self.lock:
            found = self._find(self.records[0], key)
            if found is None:
                raise KeyError(key)
            return found

    def _find(self, record: ScopeRecord, key: str) -> NodeId | None:
        # A depth-first search over the stubs a key's prefix may name (longest first), by
        # hand: a chain of placements may be deep. Each entry: a scope, the rest of the key
        # there, and the prefixes left to try (longest first).
        stack = [(record, key, self._prefixes(record, key))]
        while stack:
            at, rest, prefixes = stack[-1]
            prefix = next(prefixes, None)
            if prefix is not None:
                stub, tail = prefix
                child = self.child(at, stub)
                if child is not None:
                    stack.append((child, tail, self._prefixes(child, tail)))
                continue
            stack.pop()
            node = at.template.keys.get(rest)
            found = None if node is None else self.ident(at, node)
            if found is not None:
                return found
        return None

    @staticmethod
    def _prefixes(record: ScopeRecord, rest: str) -> Iterator[tuple[int, str]]:
        """The placed stubs of ``record``'s template whose name a prefix of ``rest`` is
        (ending at a dot), longest first, with the rest of the key after each."""
        stubs = record.template.index.stubs
        end = len(rest)
        while True:
            end = rest.rfind(".", 0, end)
            if end < 0:
                return
            stub = stubs.get(rest[:end])
            if stub is not None:
                yield stub, rest[end + 1 :]

    def selector_choice(self, index: NodeId) -> int | None:
        try:
            record, node = self.canonical(index)
        except IndexError:
            return None
        choice = record.template.selectors.get(node)
        return None if choice is None else record.choice_base + choice

    def editable(self, index: NodeId) -> bool:
        """Whether ``index`` is a formal bound to a named shared Decision (edits pass)."""
        try:
            record, node = self.canonical(index)
        except IndexError:
            return False
        name = record.template.formal_names.get(node)
        if name is None:
            return False
        binding = record.binding(name)
        return binding is not None and binding.kind == "named"


# -- the views -----------------------------------------------------------------------------------


class _Nodes(Sequence[Node]):
    """``nodes[i]`` resolves (``IndexError`` where ``i`` is no node: ``holds_node``);
    ``len`` is the number of ``NodeId``s allocated so far, one block per addressed scope,
    holes included, the same in every mode: neither a count of nodes (count by iterating)
    nor a bound on their ids; iterating addresses and resolves the whole design space."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    def _one(self, index: int) -> Node:
        return self._structure.node(index)

    @overload
    def __getitem__(self, index: int) -> Node: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Node]: ...

    def __getitem__(self, index: int | slice) -> Node | Sequence[Node]:
        if isinstance(index, slice):
            raise TypeError("a structure's nodes are read one by one")
        return self._one(index)

    def __len__(self) -> int:
        return self._structure.allocated

    def __iter__(self) -> Iterator[Node]:
        for index in tuple(self._structure.all_nodes()):
            yield self._one(index)

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, Node):
            return False
        try:
            return self._one(value.index) == value
        except IndexError:
            return False


class _RawNodes(_Nodes):
    """The nodes before collapse (for guard implication and forwarding chains)."""

    def _one(self, index: int) -> Node:
        return self._structure.raw(index)


class _Scopes(Sequence[Scope]):
    """``scopes[i]`` builds the ``Scope`` of an addressed scope (addressing its children);
    ``len`` counts the scopes addressed; iterating addresses every scope."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    @overload
    def __getitem__(self, index: int) -> Scope: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Scope]: ...

    def __getitem__(self, index: int | slice) -> Scope | Sequence[Scope]:
        if isinstance(index, slice):
            raise TypeError("a structure's scopes are read one by one")
        if index < 0:
            raise IndexError(index)
        return self._structure.scope(index)

    def __len__(self) -> int:
        return len(self._structure.records)

    def __iter__(self) -> Iterator[Scope]:
        structure = self._structure
        structure.address_all()
        for index in range(len(structure.records)):
            yield structure.scope(index)


class _Choices(Sequence[Choice]):
    """``choices[i]`` builds a scope's ``Choice`` (addressing its candidates); iterating
    addresses every scope."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    @overload
    def __getitem__(self, index: int) -> Choice: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Choice]: ...

    def __getitem__(self, index: int | slice) -> Choice | Sequence[Choice]:
        if isinstance(index, slice):
            raise TypeError("a structure's choices are read one by one")
        if index < 0:
            raise IndexError(index)
        return self._structure.choice(index)

    def __len__(self) -> int:
        return len(self._structure._choice_owners)

    def __iter__(self) -> Iterator[Choice]:
        structure = self._structure
        structure.address_all()
        for index in range(len(structure._choice_owners)):
            yield structure.choice(index)


class _Lookup(Mapping[Any, Any]):
    """A mapping answered on demand; iterating it reads the whole design space."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Keys(_Lookup):
    """``keys[key]``: the node of a key (``Structure.find``)."""

    def __getitem__(self, key: object) -> NodeId:
        if not isinstance(key, str):
            raise KeyError(key)
        return self._structure.find(key)

    def __iter__(self) -> Iterator[str]:
        structure = self._structure
        for index in tuple(structure.all_nodes()):
            yield structure.raw(index).key


class _SelectorChoices(_Lookup):
    """A selector's Choice, by the selector node."""

    def __getitem__(self, index: object) -> int:
        found = None if type(index) is not int else self._structure.selector_choice(index)
        if found is None:
            raise KeyError(index)
        return found

    def __iter__(self) -> Iterator[NodeId]:
        structure = self._structure
        for index in tuple(structure.all_nodes()):
            if structure.selector_choice(index) is not None:
                yield index


class _Ranks(_Lookup):
    """A decision's place in the flow order (``Structure.flow``): what ``_assign`` orders
    the choices it reports by. Asking it answers the flow order (once per model)."""

    def __getitem__(self, index: object) -> int:
        if type(index) is not int:
            raise KeyError(index)
        return self._structure.whole.position(index)

    def __iter__(self) -> Iterator[NodeId]:
        return iter(self._structure.flow().order)


class _EditableAliases(Collection[NodeId]):
    """Formals supplied by a named shared Decision; iterating reads the whole space."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    def __contains__(self, index: object) -> bool:
        return type(index) is int and self._structure.editable(index)

    def __iter__(self) -> Iterator[NodeId]:
        structure = self._structure
        return iter([index for index in tuple(structure.all_nodes()) if structure.editable(index)])

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Forward(Sequence[NodeId]):
    """Each node's forwarding source."""

    def __init__(self, structure: Structure) -> None:
        self._structure = structure

    @overload
    def __getitem__(self, index: int) -> NodeId: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[NodeId]: ...

    def __getitem__(self, index: int | slice) -> NodeId | Sequence[NodeId]:
        if isinstance(index, slice):
            raise TypeError("a structure's forwarding is read node by node")
        structure = self._structure
        structure.canonical(index)
        return structure.source(index)

    def __len__(self) -> int:
        return len(self._structure.nodes)


# -- the eager expansion -------------------------------------------------------------------------


def expand(space_type: type[Space]) -> LinkedModel:
    """Every node of ``space_type``'s design space in one table: a structure with every
    scope addressed and every node resolved, its ``NodeId``s renumbered in order (the eager
    form, for the tests and their comparisons)."""
    structure = Structure(link_template(space_type), eager=True)
    return materialize(structure)


def materialize(structure: Structure, *, collapsed: bool = True) -> LinkedModel:
    """A structure, fully resolved, as one table (``ir.LinkedModel``); with ``collapsed``
    false, its nodes as resolved before collapse, each its own forwarding source."""
    with structure.lock:
        structure.resolve_all()
        ids = sorted(structure.all_nodes())
        number = {index: position for position, index in enumerate(ids)}

        def at(index: NodeId | None) -> NodeId | None:
            return None if index is None else number[index]

        nodes = tuple(
            replace(
                node,
                index=number[node.index],
                guard=at(node.guard),
                arguments=tuple(Argument(a.name, number[a.node]) for a in node.arguments),
                domain_arguments=tuple(
                    Argument(a.name, number[a.node]) for a in node.domain_arguments
                ),
                output=at(node.output),
                selector=at(node.selector),
                constraints=tuple(number[target] for target in node.constraints),
                alternatives=tuple((label, number[target]) for label, target in node.alternatives),
                via=tuple((number[alias], number[source]) for alias, source in node.via),
            )
            for node in (
                structure.node(index) if collapsed else structure.raw(index) for index in ids
            )
        )
        keys = {node.key: node.index for node in nodes}
        if len(keys) != len(nodes):
            raise DefinitionError.broken(
                "node-names-collide", _OWNER, "generated node names collide"
            )
        scopes = tuple(
            replace(
                scope,
                members={member: number[index] for member, index in scope.members.items()},
                named_members={name: number[index] for name, index in scope.named_members.items()},
                guard=at(scope.guard),
                validity=at(scope.validity),
            )
            for scope in (structure.scope(index) for index in range(len(structure.records)))
        )
        choices = tuple(
            replace(
                choice,
                selector=number[choice.selector],
                members={member: number[index] for member, index in choice.members.items()},
                guard=at(choice.guard),
            )
            for choice in (structure.choice(index) for index in range(len(structure.choices)))
        )
        raw = [structure.raw(index) for index in ids]
        order = dependency_order(
            tuple(tuple(number[target] for target in node.dependencies) for node in raw),
            tuple(node.key for node in raw),
        )
        return LinkedModel(
            nodes,
            scopes,
            order,
            tuple(number[index] for index in structure.parameters),
            tuple(number[index] for index in structure.decisions),
            keys,
            choices,
            frozenset(number[index] for index in ids if structure.editable(index)),
            tuple(number[structure.source(index)] if collapsed else number[index] for index in ids),
            _Renumbered(structure, number),
        )


class _Renumbered:
    """A structure's answers to the questions about its whole design space, in a table's
    numbering (``materialize``)."""

    def __init__(self, structure: Structure, number: Mapping[NodeId, int]) -> None:
        self._structure = structure
        self._number = number
        self._back = {position: index for index, position in number.items()}

    def flow(self) -> Flow:
        from ._whole import Flow  # noqa: PLC0415 - a cycle

        found, number = self._structure.flow(), self._number
        return Flow(
            tuple(number[index] for index in found.order),
            tuple(tuple(number[index] for index in loop) for loop in found.loops),
        )

    def readers(self, source: NodeId) -> tuple[NodeId, ...]:
        held = self._back.get(source)
        if held is None:
            return ()
        return tuple(self._number[index] for index in self._structure.readers(held))


__all__ = ["EAGER", "PERMUTE", "ScopeRecord", "Structure", "expand", "materialize", "modes"]
