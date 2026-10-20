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

**Template mode** (``template_of`` set; ``_template.link_template``): the table holds
one Space class's own scope and, for each Space it places, a **stub**: a scope that
allocates none of the child's members, only what the class's body writes for it (its
formals' bindings, and nodes reserved in it on the body's behalf). A reference into a
child is an ``import`` node in the stub, whose target is a node of the child class's
template; a deeper path goes through nested stubs. The phases branch on ``stub`` where
a scope is one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from ._bindings import PlacementBinding, Slot
from ._configuration import Space
from ._nodes import NodeDecl, SlotKind, slot_kind
from ._signatures import BoundArgument
from .collection import EffectiveSpace, collect_space
from .declarations import Declaration, ValueRef
from .errors import DefinitionError
from .expressions import IntOperator
from .ir import Choice, Node, NodeKind, Scope
from .semantics import BOOL, ValueSemantics

if TYPE_CHECKING:
    from ._template import Template, TemplateChoice

_OWNER = "space.table"


def member_key(scope: str, member: str) -> str:
    """A member's key: its scope's name and its own, dotted."""
    return f"{scope}.{member}" if scope else member


BindingForm = Literal["literal", "reference", "local-decision", "named", "unsupplied"]


@dataclass(frozen=True, slots=True)
class Binding:
    """How a placing body binds one formal of a child (template mode): a literal or a
    default (``value``, no node), a reference or a fresh Decision (its node in the stub),
    a named shared Decision (its use: an editable alias, ``node``), or nothing."""

    kind: BindingForm
    value: object = None
    node: int | None = None


UNSUPPLIED_BINDING = Binding("unsupplied")


@dataclass(frozen=True, slots=True)
class Read:
    """Template mode: one read of ``decision.member``, in the order the template lowers
    its reads: its reader's key, and the Decision over nodes it exists only where it
    applies (None: everywhere)."""

    owner: str
    requires: int | None


@dataclass
class Stub:
    """A child Space's scope in a template: its class's template, and what the placing
    body writes for it.

    A placed stub (a child node, a candidate) is a scope of the placing template; a
    nested stub stands for a scope deeper down (``a.b``), reached by a reference: its
    ``slot`` is its scope in its parent stub's template, and its bindings are that
    template's. ``requires`` is the placing template's Decision over nodes whose
    candidate the stub is (or is below), where that choice may never apply (a guard
    ``supplied(formal)``): the stub's nodes exist only where it applies.
    """

    template: Template
    slot: int | None = None
    bindings: dict[str, Binding] = field(default_factory=dict)
    imports: dict[int, int] = field(default_factory=dict)
    nested: dict[int, int] = field(default_factory=dict)
    requires: int | None = None
    # A nested stub's guard is an import of its placing template's guard, made when read.
    guard_imported: bool = False


@dataclass
class ScopeDraft:
    index: int
    parent: int | None
    name: str
    effective: EffectiveSpace
    guard: int | None
    source_scope: int
    record: NodeDecl | None
    slots: dict[str, Slot] = field(default_factory=dict)
    members: dict[object, int] = field(default_factory=dict)
    named_members: dict[str, int] = field(default_factory=dict)
    children: dict[object, int] = field(default_factory=dict)
    named_children: dict[str, int] = field(default_factory=dict)
    choices: dict[object, int] = field(default_factory=dict)
    # The value inputs nothing supplies here: a guard on their supply never holds.
    unsupplied: set[str] = field(default_factory=set)
    # The node answering this Space's validity, reserved with the scope.
    validity: int | None = None
    # Template mode: a child's scope that allocates none of its members.
    stub: Stub | None = None
    # Template mode: a candidate of a Decision over nodes of the own scope (choice, case).
    candidate: tuple[int, str] | None = None

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
            self.validity,
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
    # Template mode. The formal whose absence makes the choice never apply (its guard is
    # ``supplied(formal)``); every selection made for it (node -> member), those its
    # candidates share, each read of a member through it, and the type a selection takes
    # where the choice never applies (``never_declared``: whether a candidate declares
    # the member, else refused there); shared selections dropped for disagreeing types.
    never_formal: str | None = None
    selects: dict[int, str] = field(default_factory=dict)
    shared: set[int] = field(default_factory=set)
    readers: dict[str, list[Read]] = field(default_factory=dict)
    never_semantics: dict[str, ValueSemantics[object] | None] = field(default_factory=dict)
    never_declared: dict[str, bool] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)

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
    # Allocation -> checks: selections linked for a member name several candidates share.
    shared: dict[int, tuple[int, str]] = field(default_factory=dict)
    # Lowering -> checks: integer expressions, and each callback argument's target.
    expression_tasks: list[ExpressionTask] = field(default_factory=list)
    argument_checks: list[tuple[BoundArgument, int, str]] = field(default_factory=list)
    # For the linked model. Formals supplied by a named shared Decision.
    editable_aliases: set[int] = field(default_factory=set)
    # Template mode: each child class's template (linked first), the node standing for
    # the guard the own scope is placed under, each import's target, the Decision over
    # nodes each node exists only where it applies (and the one whose nodes are being
    # made), the named shared Decisions in order, and each selection made in a stub's
    # choice (node -> stub, the choice in the stub's template, member).
    template_of: Callable[[type[Space]], Template] | None = None
    placement_guard: int | None = None
    imports: dict[int, int] = field(default_factory=dict)
    requires: dict[int, int] = field(default_factory=dict)
    applying: int | None = None
    named: list[int] = field(default_factory=list)
    stub_selects: dict[int, tuple[int, int, str]] = field(default_factory=dict)
    stub_reads: dict[int, list[Read]] = field(default_factory=dict)

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
        if self.applying is not None:
            self.requires[index] = self.applying
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

    # -- template mode: stubs ---------------------------------------------------------------

    def stub(self, scope: int) -> Stub:
        stub = self.drafts[scope].stub
        assert stub is not None, "template mode: a stub expected"
        return stub

    def imported(self, scope: int, target: int) -> int:
        """The node of stub ``scope`` that stands for ``target``, a node of its class's
        template: the binding node of a formal the placing body binds by a node, the
        stub's guard for the guard the child is placed under, else one import per target."""
        stub = self.stub(scope)
        template = stub.template
        if target == template.placement_guard:
            return self.stub_guard(scope)
        formal = template.formal_names.get(target)
        if formal is not None and stub.slot is None:
            # A placed stub: the formal's binding node is this template's own.
            node = stub.bindings[formal].node
            if node is not None:
                return node
        found = stub.imports.get(target)
        if found is not None:
            return found
        source = template.nodes[target]
        found = self.reserve(
            scope, member_key(self.drafts[scope].name, source.key), "import", source.semantics
        )
        stub.imports[target] = found
        self.imports[found] = target
        return found

    def stub_guard(self, scope: int) -> int:
        """The guard a stub's child is placed under: a placed stub's own guard node, a
        nested stub's an import of its placing template's guard."""
        draft = self.drafts[scope]
        stub = self.stub(scope)
        if stub.slot is not None and not stub.guard_imported:
            parent = cast(int, draft.parent)
            placing = self.stub(parent).template.scopes[stub.slot]
            draft.guard = self.imported(parent, cast(int, placing.guard))
            stub.guard_imported = True
        return cast(int, draft.guard)

    def never_at(self, scope: int, choice: TemplateChoice) -> bool:
        """Whether a Decision over nodes of stub ``scope``'s template never applies there:
        its guard is the supply of a formal the placing body leaves unsupplied."""
        if choice.never_formal is None:
            return False
        return self.stub(scope).bindings[choice.never_formal].kind == "unsupplied"

    def nested(self, scope: int, child: int, *, owner: str) -> int:
        """The nested stub for scope ``child`` of stub ``scope``'s template, made once."""
        stub = self.stub(scope)
        found = stub.nested.get(child)
        if found is not None:
            return found
        template = stub.template
        placing = template.scopes[child]
        if placing.candidate is not None and self.never_at(
            scope, template.choices[placing.candidate[0]]
        ):
            raise DefinitionError.broken(
                "reference-not-placed",
                _OWNER,
                f"{owner}: {placing.record!r} is not placed in "
                f"{self.drafts[scope].name or '<root>'}",
                declaration=owner,
            )
        draft = self.drafts[scope]
        index = len(self.drafts)
        self.drafts.append(
            ScopeDraft(
                index,
                scope,
                member_key(draft.name, placing.name),
                self.collected(placing.space_type),
                None,
                draft.source_scope,
                placing.record,
                stub=Stub(
                    cast("Callable[[type[Space]], Template]", self.template_of)(placing.space_type),
                    slot=child,
                    bindings=dict(placing.bindings),
                    requires=stub.requires,
                ),
            )
        )
        stub.nested[child] = index
        return index

    def stub_member(self, scope: int, member: object) -> int:
        """A member of a stub's child, by declaration: a node the placing body reserved in
        it (a named shared Decision), else the child template's, imported."""
        draft = self.drafts[scope]
        stub = self.stub(scope)
        found = draft.members.get(member)
        if found is not None:
            return found
        if stub.slot is not None:
            parent = cast(int, draft.parent)
            placed = self.stub(parent).template.scopes[stub.slot].members.get(member)
            if placed is not None:
                return self.imported(parent, placed)
        return self.imported(scope, stub.template.members[member])

    def stub_named(self, scope: int, name: str) -> int | None:
        """A member of a stub's child, by name (as ``named_members``), or None."""
        draft = self.drafts[scope]
        stub = self.stub(scope)
        found = draft.named_members.get(name)
        if found is not None:
            return found
        if stub.slot is not None:
            parent = cast(int, draft.parent)
            placed = self.stub(parent).template.scopes[stub.slot].named_members.get(name)
            if placed is not None:
                return self.imported(parent, placed)
        target = stub.template.named_members.get(name)
        return None if target is None else self.imported(scope, target)
