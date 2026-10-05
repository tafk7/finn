# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The linker's one node table, which its phases fill in turn.

Allocation (``_allocate``) creates every scope, a node for every member and
the tasks that lower them; lowering (``_lower``, resolving references through
``_names``) gives each node its payload, reserving the nodes a reference
needs; ``_check`` validates the linked semantics and ``_collapse`` rewrites
evaluation edges. ``_linker.link_space`` runs them in that order and freezes
the table into a ``LinkedModel``. The table holds what crosses a phase; what a
phase keeps for itself stays in that phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ._bindings import PlacementBinding, Slot
from ._configuration import Space
from ._nodes import NodeDecl, SlotKind, slot_kind
from ._signatures import BoundArgument
from .collection import EffectiveSpace, collect_space
from .declarations import Declaration, ValueRef, ViewKey
from .expressions import IntOperator
from .ir import Choice, Node, NodeKind, Provenance, Scope
from .semantics import ValueSemantics, default_semantics

BOOL = cast(ValueSemantics[object], default_semantics(bool))
STRING = cast(ValueSemantics[object], default_semantics(str))


def member_key(scope: str, member: str) -> str:
    """A member's key: its scope's name and its own, dotted."""
    return f"{scope}.{member}" if scope else member


@dataclass
class ScopeDraft:
    index: int
    parent: int | None
    name: str
    effective: EffectiveSpace
    guard: int | None
    source_scope: int
    record: NodeDecl | None
    # Who wrote the record's own settings (ROOT_WRITER for the root declaration).
    writer: int
    # The nearest strict ancestor whose record assigns through a path.
    carrier: int | None
    slots: dict[str, Slot] = field(default_factory=dict)
    members: dict[object, int] = field(default_factory=dict)
    named_members: dict[str, int] = field(default_factory=dict)
    children: dict[object, int] = field(default_factory=dict)
    named_children: dict[str, int] = field(default_factory=dict)
    choices: dict[object, int] = field(default_factory=dict)
    references: dict[object, int] = field(default_factory=dict)
    # Each reference input's declaration -> the scope it reaches (None: unsupplied).
    targets: dict[object, int | None] = field(default_factory=dict)
    # A per-input export: key -> (reference input name, view node) in declared order.
    input_exports: dict[ViewKey[object], tuple[tuple[str, int], ...]] = field(default_factory=dict)
    # The value inputs nothing supplies here: a guard on their supply never holds.
    unsupplied: set[str] = field(default_factory=set)

    def freeze(self) -> Scope:
        return Scope(
            self.index,
            self.parent,
            self.name,
            self.effective.space_type,
            self.members,
            self.children,
            self.named_members,
            self.named_children,
            self.guard,
            self.choices,
            self.record,
            self.references,
        )


@dataclass
class ChoiceDraft:
    index: int
    scope: int
    key: str
    guard: int | None
    selector: int
    cases: list[tuple[str, int | None]] = field(default_factory=list)
    members: dict[str, int] = field(default_factory=dict)
    # Its guard never holds here: no candidate is placed, and it has no key; the
    # Space classes its candidates would place type what is read through it.
    never: bool = False
    space_types: dict[str, type[Space]] = field(default_factory=dict)

    def freeze(self) -> Choice:
        return Choice(
            self.index,
            self.scope,
            self.key,
            self.selector,
            tuple(self.cases),
            self.members,
            self.guard,
        )


@dataclass(frozen=True)
class MemberTask:
    index: int
    scope: int
    name: str
    declaration: Declaration
    binding: PlacementBinding | None
    # Where the owning scope of a named shared decision was inferred to be.
    owner_scope: int | None = None


@dataclass(frozen=True)
class GuardTask:
    index: int
    source_scope: int
    condition: ValueRef[bool]


@dataclass(frozen=True)
class ExpressionTask:
    index: int
    scope: int
    owner: str
    operator: IntOperator
    operands: tuple[int | ValueRef[int], ...]


@dataclass
class Table:
    """The nodes and scopes of one linked Space class, and what each phase passes on."""

    space_type: type[Space]
    nodes: list[Node] = field(default_factory=list)
    drafts: list[ScopeDraft] = field(default_factory=list)
    choice_drafts: list[ChoiceDraft] = field(default_factory=list)
    # Each Space class collected once: the declarations each member name stands for, and
    # each member's slot kind.
    effective: dict[type[Space], EffectiveSpace] = field(default_factory=dict)
    aliases: dict[type[Space], dict[str, list[Declaration]]] = field(default_factory=dict)
    kinds: dict[type[Space], dict[str, SlotKind]] = field(default_factory=dict)
    # Allocation -> lowering: every member to lower, every guard to link.
    members: list[MemberTask] = field(default_factory=list)
    guards: list[GuardTask] = field(default_factory=list)
    # The frozen scopes, once allocation has fixed their identities and maps.
    scopes: tuple[Scope, ...] = ()
    # Referenced scope -> the (user scope, input name) pairs that reference it.
    users: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    # A forwarded reference input: (scope, input) -> the input it forwards.
    forwarded: dict[tuple[int, str], tuple[int, str]] = field(default_factory=dict)
    # Allocation -> checks: selections linked for a member name several candidates share.
    shared: dict[int, tuple[int, str]] = field(default_factory=dict)
    # Lowering -> checks: integer expressions, and each callback argument's target.
    expression_tasks: list[ExpressionTask] = field(default_factory=list)
    argument_checks: list[tuple[BoundArgument, int, str]] = field(default_factory=list)
    # For the linked model. Formals supplied by a named shared Decision.
    editable_aliases: set[int] = field(default_factory=set)
    # Who set each supplied member (by node) and each replaced child (by scope).
    provenance: dict[int, Provenance] = field(default_factory=dict)
    scope_provenance: dict[int, Provenance] = field(default_factory=dict)
    pinned: dict[str, Provenance] = field(default_factory=dict)

    def collected(self, space_type: type[Space]) -> EffectiveSpace:
        effective = self.effective.get(space_type)
        if effective is None:
            effective = collect_space(space_type)
            self.effective[space_type] = effective
            aliases: dict[str, list[Declaration]] = {}
            for declaration, name in effective.aliases.items():
                aliases.setdefault(name, []).append(declaration)
            self.aliases[space_type] = aliases
            self.kinds[space_type] = {
                name: slot_kind(declaration) for name, declaration in effective.members.items()
            }
        return effective

    def reserve(
        self,
        scope: int,
        key: str,
        kind: NodeKind,
        semantics: ValueSemantics[object] | None = None,
        *,
        guard: int | None = None,
        source_owner: str | None = None,
        origin: str | None = None,
    ) -> int:
        index = len(self.nodes)
        self.nodes.append(
            Node(
                index,
                scope,
                key,
                kind,
                semantics,
                guard=guard,
                source_owner=source_owner,
                origin=origin,
            )
        )
        return index

    def guarded(
        self,
        source_scope: int,
        outer: int | None,
        condition: ValueRef[bool] | None,
        key: str,
        *,
        owner: str,
    ) -> int | None:
        """A guard node for ``condition`` inside ``outer``; its condition is linked later."""
        if condition is None:
            return outer
        index = self.reserve(source_scope, key, "guard", BOOL, guard=outer, source_owner=owner)
        self.guards.append(GuardTask(index, source_scope, condition))
        return index

    def local_name(self, scope: int, child: int) -> str:
        """A descendant scope's name relative to ``scope``."""
        prefix = self.drafts[scope].name
        name = self.drafts[child].name
        return name[len(prefix) + 1 :] if prefix else name
