# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The table of one Space class being compiled, which the linker's phases fill in turn.

The table holds the class's own scope, its records (its members, numbered within the
class, and the records it writes for the Spaces it places) and, for each Space it places
(a child node, a candidate of a Decision over nodes), its **slot**: the placed class's
compiled form (compiled first) and what this class's body writes for it (its formals'
bindings), never a member of the placed class. A reference into a placed Space is an edge
out of the class (``ir.Placed``, through its slots; ``ir.Selected`` for a selection of a
choice's member there), a reference to the guard the class is placed under is
``ir.PLACING``: neither is a record (``ir.NodeRef``). A reference resolves at a **place**
(``Place``): the class's own scope, a slot, or a scope deeper below a slot, reached through
the placed classes' compiled forms.

Allocation (``_allocate``) creates the scopes, a node for every member and the tasks
that lower them; lowering (``_lower``, resolving references through ``_names``) gives
each node its payload, reserving the nodes a reference needs; the checks composed
across slots (``_compose``) order it and ``_check`` validates the semantics.
``_template.link_template`` runs them in that order and makes the class's compiled form
of the table. The table holds what crosses a phase; what a phase keeps for itself stays in
that phase.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from ._bindings import PlacementBinding, Slot
from ._configuration import Space
from ._nodes import NodeDecl, SlotKind, slot_kind
from ._signatures import BoundArgument
from .collection import EffectiveSpace, collect_space
from .declarations import Declaration, ValueRef
from .expressions import IntOperator
from .ir import Node, NodeKind, NodeRef, Placed, Placing, Selected
from .semantics import BOOL, ValueSemantics

if TYPE_CHECKING:
    from ._template import Compiled


def member_key(scope: str, member: str) -> str:
    """A member's key: its scope's name and its own, dotted."""
    return f"{scope}.{member}" if scope else member


BindingForm = Literal["literal", "reference"]


@dataclass(frozen=True, slots=True)
class Binding:
    """How a placing body binds one formal of a child: a literal or a default (``value``,
    no record), or a reference (``node``: the record the placing class writes for it)."""

    kind: BindingForm
    value: object = None
    node: int | None = None


@dataclass(frozen=True, slots=True)
class Read:
    """One read of ``decision.member``: its reader's key, and the Decision over nodes it
    exists only where it applies (None: everywhere)."""

    owner: str
    requires: int | None


@dataclass
class SlotDraft:
    """A Space the class places, slot ``slot`` of its compiled form: the placed class's
    compiled form, and what the placing body writes for it (its formals' bindings).
    ``requires`` is the class's Decision over nodes whose candidate it is, where that
    choice may never apply (a guard ``supplied(formal)``): what the class writes for it
    exists only where it applies."""

    compiled: Compiled
    slot: int
    bindings: dict[str, Binding] = field(default_factory=dict)
    requires: int | None = None


@dataclass
class ScopeDraft:
    index: int
    parent: int | None
    name: str
    effective: EffectiveSpace
    guard: NodeRef | None
    source_scope: int
    record: NodeDecl | None
    slots: dict[str, Slot] = field(default_factory=dict)
    members: dict[object, int] = field(default_factory=dict)
    named_members: dict[str, int] = field(default_factory=dict)
    children: dict[object, int] = field(default_factory=dict)
    named_children: dict[str, int] = field(default_factory=dict)
    choices: dict[object, int] = field(default_factory=dict)
    # The node answering the own scope's validity, reserved with the scope.
    validity: int | None = None
    # A Space the class places: its slot (the own scope has none).
    placed: SlotDraft | None = None
    # A candidate of a Decision over nodes of the own scope (choice, case).
    candidate: tuple[int, str] | None = None


@dataclass(frozen=True, slots=True)
class Place:
    """Where a reference resolves: the own scope (draft 0, no steps), a slot (its draft, no
    steps), or a scope below a slot (its draft, then the slots of each placed class's
    compiled form on the way)."""

    draft: int
    steps: tuple[int, ...] = ()


@dataclass
class ChoiceDraft:
    index: int
    scope: int
    key: str
    guard: NodeRef | None
    selector: int
    cases: list[tuple[str, int | None]] = field(default_factory=list)
    members: dict[str, int] = field(default_factory=dict)
    # Every selection made for it (node -> member), those its candidates share, each read
    # of a member through it; shared selections dropped for disagreeing types.
    selects: dict[int, str] = field(default_factory=dict)
    shared: set[int] = field(default_factory=set)
    readers: dict[str, list[Read]] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class MemberTask:
    index: int
    scope: int
    name: str
    declaration: Declaration
    binding: PlacementBinding | None


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


# An edge out of the class a reference may name: a placed Space's member or selection.
Edge = Placed | Selected


@dataclass
class Table:
    """The records and scopes of one Space class being compiled, and what each phase passes
    on."""

    space_type: type[Space]
    # Each placed class's compiled form, compiled first.
    compiled_of: Callable[[type[Space]], Compiled]
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
    # The slots' drafts, in slot order.
    slot_drafts: list[int] = field(default_factory=list)
    # Allocation -> checks: selections linked for a member name several candidates share.
    shared: dict[int, tuple[int, str]] = field(default_factory=dict)
    # Lowering -> checks: integer expressions, and each callback argument's target.
    expression_tasks: list[ExpressionTask] = field(default_factory=list)
    argument_checks: list[tuple[BoundArgument, NodeRef, str]] = field(default_factory=list)
    # The Decision over nodes each node exists only where it applies, and the one whose
    # nodes are being made. Each edge out of the class, in the order first named, with the
    # Decision being applied then (an edge takes its first reader's applicability), and
    # the records and edges in the order they were made (``vertices``: the dependency
    # order's).
    requires: dict[int, int] = field(default_factory=dict)
    applying: int | None = None
    edges: dict[Edge, int | None] = field(default_factory=dict)
    vertices: list[int | Edge] = field(default_factory=list)
    # ``Select(by=...)``: each derived selector's node, its scope and what picks it.
    picks: list[tuple[int, int, object]] = field(default_factory=list)
    # Each selection made in a placed Space's choice: node -> (its slots, the choice in the
    # class those slots reach, member), and its reads.
    chooses: dict[int, tuple[tuple[int, ...], int, str]] = field(default_factory=dict)
    chosen_reads: dict[int, list[Read]] = field(default_factory=dict)

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
        guard: NodeRef | None = None,
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
        self.vertices.append(index)
        if self.applying is not None:
            self.requires[index] = self.applying
        return index

    def guarded(
        self,
        source_scope: int,
        outer: NodeRef | None,
        condition: ValueRef[bool] | None,
        key: str,
        *,
        owner: str,
    ) -> NodeRef | None:
        """A guard node for ``condition`` inside ``outer``; its condition is linked later."""
        if condition is None:
            return outer
        index = self.reserve(source_scope, key, "guard", BOOL, guard=outer, source_owner=owner)
        self.guards.append(GuardTask(index, source_scope, condition))
        return index

    # -- places ------------------------------------------------------------------------------

    def placing(self, scope: int) -> SlotDraft:
        placed = self.drafts[scope].placed
        assert placed is not None, "a slot expected"
        return placed

    def compiled_at(self, place: Place) -> Compiled:
        """The compiled form of the class placed at ``place`` (below a slot)."""
        return self.placing(place.draft).compiled.reached(place.steps)

    def path(self, place: Place) -> tuple[int, ...]:
        """The slots from the class to ``place``."""
        return (self.placing(place.draft).slot, *place.steps)

    def name(self, place: Place) -> str:
        """``place``'s scope's name, relative to the class."""
        name = self.drafts[place.draft].name
        current = self.placing(place.draft).compiled
        for step in place.steps:
            name = member_key(name, current.slots[step].name)
            current = current.slots[step].compiled
        return name

    def bindings(self, place: Place) -> Mapping[str, Binding]:
        """The bindings of the formals of the class placed at ``place``."""
        if not place.steps:
            return self.placing(place.draft).bindings
        above = self.compiled_at(Place(place.draft, place.steps[:-1]))
        return above.slots[place.steps[-1]].bindings

    def record(self, place: Place) -> NodeDecl | None:
        """The node declaration placed at ``place``."""
        if not place.steps:
            return self.drafts[place.draft].record
        above = self.compiled_at(Place(place.draft, place.steps[:-1]))
        return above.slots[place.steps[-1]].record

    def lift(self, place: Place, ref: int | Placing | Placed) -> NodeRef:
        """What ``ref``, a reference of the class placed at ``place`` (a record of it, the
        guard it is placed under, a member of a Space it places), stands for in this class:
        the record this class writes for a formal it binds by reference; the guard the
        placed scope is placed under; a selection of a choice's member in its choice's
        scope; else the member, through the slots."""
        if isinstance(ref, Placing):
            return self.placed_guard(place)
        if isinstance(ref, Placed):
            return self.lift(Place(place.draft, (*place.steps, *ref.path)), ref.member)
        compiled = self.compiled_at(place)
        if not place.steps:
            formal = compiled.formal_names.get(ref)
            if formal is not None:
                # A placed formal: the record this class writes for its binding, if any.
                node = self.placing(place.draft).bindings[formal].node
                if node is not None:
                    return node
        selection = compiled.selections.get(ref)
        edge: Edge
        if selection is not None:
            edge = Selected(self.path(place), *selection)
        else:
            edge = Placed(self.path(place), ref)
        return self.named(edge)

    def named(self, edge: Edge) -> Edge:
        """An edge out of the class, named: the first reader's applicability is its own."""
        if edge not in self.edges:
            self.edges[edge] = self.applying
            self.vertices.append(edge)
        return edge

    def placed_guard(self, place: Place) -> NodeRef:
        """The guard the scope at ``place`` is placed under: a slot's guard, a record of this
        class; deeper, the guard its placing class places it under, lifted."""
        if not place.steps:
            return cast(NodeRef, self.drafts[place.draft].guard)
        above = Place(place.draft, place.steps[:-1])
        compiled = self.compiled_at(above)
        return self.lift(
            above, cast("int | Placing", compiled.ref(compiled.slots[place.steps[-1]].guard))
        )

    def below(self, place: Place, slot: int) -> Place:
        """The scope of slot ``slot`` of the class placed at ``place``."""
        return Place(place.draft, (*place.steps, slot))

    def member_at(self, place: Place, member: object) -> NodeRef:
        """A member of the class placed at ``place``, by declaration."""
        return self.lift(place, self.compiled_at(place).members[member])

    def named_at(self, place: Place, name: str) -> NodeRef | None:
        """A member of the class placed at ``place``, by name, or None."""
        target = self.compiled_at(place).named_members.get(name)
        return None if target is None else self.lift(place, target)

    # -- what a reference names --------------------------------------------------------------

    def semantics(self, ref: NodeRef) -> ValueSemantics[object] | None:
        """The value semantics of what ``ref`` names: a record's, the placement guard's (a
        Boolean), or a placed Space's member's, as its class checked it."""
        if isinstance(ref, int):
            return self.nodes[ref].semantics
        if isinstance(ref, Placing):
            return BOOL
        current = self.placing(self.slot_drafts[ref.path[0]]).compiled.reached(ref.path[1:])
        if isinstance(ref, Selected):
            return current.records[current.own_selection[(ref.choice, ref.member)]].semantics
        return current.records[ref.member].semantics

    def edge_name(self, edge: Edge) -> str:
        """An edge's key relative to the class (the placed Space's member under its slots'
        names)."""
        draft = self.slot_drafts[edge.path[0]]
        place = Place(draft, edge.path[1:])
        current = self.compiled_at(place)
        member = (
            current.own_selection[(edge.choice, edge.member)]
            if isinstance(edge, Selected)
            else edge.member
        )
        return member_key(self.name(place), current.records[member].key)
