# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Allocate a Space class being compiled (``_table``): its own scope, every member a node,
and a slot for each Space it places.

The own scope is the Space class's, its formals its ``param`` nodes, its guard the guard
it is placed under (``ir.PLACING``, known where it is placed). A child node and a Decision
over nodes' candidates each place a slot, the placed class compiled first: a slot holds
the bindings this body writes for the child's formals (a literal or a default as a value,
no node; a reference as a node) and nothing of the child's own. A Decision given at a call
is refused (``_bindings.classify``): a Decision is its class's. Each member gets its node
and the task that lowers it. The own scope's validity node lists its constraints, its
Decisions and formals (a formal's piece is decided where placed, by its binding) and each
slot's validity. Afterwards scope identities and membership maps are fixed; later phases
replace node payloads only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from ._bindings import PlacementBinding, Slot, classify
from ._configuration import Space
from ._nodes import (
    NodeDecision,
    NodeDecl,
    fallback,
    missing_formal,
)
from ._ordering import dependency_order
from ._signatures import resolve_source
from ._table import (
    Binding,
    ChoiceDraft,
    MemberTask,
    Place,
    ScopeDraft,
    SlotDraft,
    Table,
    member_key,
)
from .declarations import (
    MISSING,
    CaseRef,
    ChoiceMemberRef,
    Constraint,
    Decision,
    Declaration,
    Derived,
    MemberRef,
    Members,
    Param,
    View,
)
from .domains import Domain, finite
from .errors import DefinitionError
from .ir import PLACING, Argument, NodeKind, NodeRef
from .results import reject
from .semantics import BOOL, STRING, ValueSemantics

if TYPE_CHECKING:
    from ._template import Compiled

_OWNER = "space.allocate"


def _names_case(expected: str, argument: str) -> Callable[..., object]:
    """A case guard: the selector's value names ``expected``. A chosen selector's reads
    the decision as ``selected``; a derived selector's reads its value as ``picked``."""

    def selected(*, selected: str) -> bool:
        return selected == expected

    def picked(*, picked: str) -> bool:
        return picked == expected

    return selected if argument == "selected" else picked


def _picks(cases: tuple[str, ...]) -> Callable[..., object]:
    """``Select(by=d)``'s value: ``d``'s, where it names a case; else refused at the
    selector, ``select-no-case`` (a picker never answers "nothing": a ``None`` case does)."""

    def pick(*, value: object) -> object:
        if type(value) is str and value in cases:
            return value
        return reject("select-no-case", f"{value!r} names no case of {list(cases)}")

    return pick


def _known(*, picked: object) -> bool:
    """A derived selector's guard: its value is known (it holds wherever it is read)."""
    return True


_BINDING_KINDS: Mapping[str, NodeKind] = {
    "literal": "const",
    "reference": "alias",
}


def allocate(space_type: type[Space], compiled_of: Callable[[type[Space]], Compiled]) -> Table:
    """A new table with ``space_type``'s own scope and a slot per Space it places, each
    placed class's compiled form from ``compiled_of``."""
    allocation = _Allocation(Table(space_type, compiled_of))
    allocation.check_recursion()
    allocation.allocate()
    return allocation.table


def _declared_kind(declaration: Declaration) -> NodeKind | None:
    """The node kind of a Space class's own member that is neither a formal nor a Decision."""
    if isinstance(declaration, Derived):
        return "derived"
    if isinstance(declaration, Constraint):
        return "constraint"
    if isinstance(declaration, View):
        return "view"
    if isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
        return "alias"
    if isinstance(declaration, Members):
        return "members"
    return None


class _Allocation:
    def __init__(self, table: Table) -> None:
        self.table = table
        # Required formals nothing supplies, reported together after allocation.
        self.missing: list[str] = []

    # -- Space classes -------------------------------------------------------------------

    def check_recursion(self) -> None:
        """Reject a Space class that places itself, before allocating. A class already
        linked places nothing recursively (its own linking checked it), so the walk stops
        there."""

        pending = [self.table.space_type]
        children_of: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in children_of:
                continue
            if space_type is not self.table.space_type and "_space_compiled" in space_type.__dict__:
                children_of[space_type] = ()
                continue
            effective = self.table.collected(space_type)
            children: list[type[Space]] = []
            for declaration in effective.members.values():
                if isinstance(declaration, NodeDecl):
                    children.append(declaration.space_type)
                elif isinstance(declaration, NodeDecision):
                    children.extend(
                        record.space_type
                        for record in declaration.candidates.values()
                        if record is not None
                    )
            children_of[space_type] = tuple(children)
            pending.extend(child for child in children if child not in children_of)
        indices = {space_type: index for index, space_type in enumerate(children_of)}
        structure = tuple(
            tuple(indices[child] for child in children_of[space_type]) for space_type in children_of
        )
        try:
            dependency_order(
                structure, tuple(space_type.__qualname__ for space_type in children_of)
            )
        except DefinitionError as cause:
            raise DefinitionError("recursive Space placement", findings=cause.findings) from cause

    # -- settings -------------------------------------------------------------------------

    def own_slots(self, draft: ScopeDraft) -> dict[str, Slot]:
        """What the body that declared this node supplied for each of its formals."""
        if draft.record is None:
            return {}
        return {
            member: Slot(value, draft.source_scope, origin, member_key(draft.name, member))
            for member, (value, origin) in draft.record.settings.items()
        }

    # -- allocation ---------------------------------------------------------------------

    def new_scope(
        self,
        space_type: type[Space],
        parent: int | None,
        name: str,
        guard: NodeRef | None,
        *,
        record: NodeDecl | None,
        source_scope: int,
    ) -> int:
        index = len(self.table.drafts)
        draft = ScopeDraft(
            index,
            parent,
            name,
            self.table.collected(space_type),
            guard,
            index if parent is None else source_scope,
            record,
        )
        self.table.drafts.append(draft)
        if parent is not None:
            # The child's compiled form, compiled first; a slot of this class.
            draft.placed = SlotDraft(
                self.table.compiled_of(space_type), len(self.table.slot_drafts)
            )
            self.table.slot_drafts.append(index)
        draft.slots = self.own_slots(draft)
        kinds = self.table.kinds[draft.effective.space_type]
        for member_name, declaration in draft.effective.members.items():
            kind = kinds[member_name]
            # Structure (nodes and choices) is allocated below.
            if kind not in ("node", "choice"):
                self.member(draft, member_name, declaration)
        if draft.placed is not None:
            return index
        for declaration, member_name in draft.effective.aliases.items():
            if member_name in draft.named_members:
                draft.members[declaration] = draft.named_members[member_name]
        key = member_key(name, "$validity")
        draft.validity = self.table.reserve(
            index, key, "validity", BOOL, guard=guard, source_owner=key
        )
        return index

    def member(self, draft: ScopeDraft, member_name: str, declaration: Declaration) -> None:
        """Reserve a value member's node; a formal is bound by what its body supplied."""
        key = member_key(draft.name, member_name)
        slot = draft.slots.get(member_name)
        binding: PlacementBinding | None = None
        if slot is not None:
            if not isinstance(declaration, Param):
                raise DefinitionError.broken(
                    "body-supplies-non-formal",
                    _OWNER,
                    f"{key} (assigned at {slot.origin}): a body supplies formals only; "
                    f"{type(declaration).__name__} belongs to the Space class",
                    declaration=key,
                )
            binding_kind, supplier = classify(slot)
            binding = PlacementBinding(supplier, binding_kind, slot.origin)
        placed = draft.placed
        if placed is not None and not isinstance(declaration, Param):
            return  # a slot holds the child's formals as this body binds them, nothing else
        node_kind: NodeKind | None
        if isinstance(declaration, Param):
            if binding is not None:
                node_kind = _BINDING_KINDS[binding.kind]
            elif draft.parent is None:
                node_kind = "param"
            else:
                binding, node_kind = self.unsupplied(draft, member_name, declaration), "const"
        elif isinstance(declaration, Decision):
            node_kind = "decision"
        else:
            node_kind = _declared_kind(declaration)
            if node_kind is None:
                raise DefinitionError.broken(
                    "declaration-unsupported",
                    _OWNER,
                    f"{key}: unsupported declaration",
                    declaration=key,
                )
        if placed is not None and node_kind == "const":
            # A literal or a default is a value on the slot: no node (the child's compiled
            # form has the formal's). A required formal nothing supplies has none: the
            # class is refused once allocated (``formal-unsupplied``).
            placed.bindings[member_name] = Binding(
                "literal", None if binding is None else binding.supplier
            )
            return
        node = self.table.reserve(
            draft.index,
            key,
            node_kind,
            draft.effective.semantics.get(declaration),
            guard=draft.guard,
            origin=declaration.origin,
        )
        if placed is not None:
            binding = cast(PlacementBinding, binding)
            placed.bindings[member_name] = Binding(binding.kind, node=node)
        else:
            draft.named_members[member_name] = node
        self.table.members.append(MemberTask(node, draft.index, member_name, declaration, binding))

    def unsupplied(
        self, draft: ScopeDraft, name: str, formal: Param[object]
    ) -> PlacementBinding | None:
        """A child's formal nothing supplies: its default (``None`` for an optional fact),
        or missing (none: a required formal)."""
        default = fallback(formal)
        if default is MISSING:
            self.missing.append(
                missing_formal(
                    member_key(draft.name, name),
                    formal,
                    draft.record,
                    self.table.space_type.__qualname__,
                )
            )
            return None
        return PlacementBinding(default, "literal")

    def place(
        self,
        scope: ScopeDraft,
        name: str,
        record: NodeDecl,
        guard: NodeRef | None,
        *,
        source_scope: int,
        keys: Iterable[object],
    ) -> int:
        child = self.new_scope(
            record.space_type,
            scope.index,
            member_key(scope.name, name),
            guard,
            record=record,
            source_scope=source_scope,
        )
        for key in keys:
            scope.children[key] = child
        return child

    def child(self, scope: ScopeDraft, name: str, declaration: NodeDecl) -> None:
        """A child node, placed in its Space class's body."""
        key = member_key(scope.name, name)
        guard = self.table.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(declaration),
            key + ".$guard",
            owner=key,
        )
        keys: tuple[object, ...] = tuple(self.table.aliases[scope.effective.space_type][name])
        child = self.place(scope, name, declaration, guard, source_scope=scope.index, keys=keys)
        scope.named_children[name] = child

    def choice(self, scope: ScopeDraft, name: str, decision: NodeDecision) -> None:
        """A selector: its value, and a guarded scope per case.

        Chosen (a Decision over nodes, ``Select`` without ``by=``), its value is the
        decision ``<key>`` and each case's scope is placed under ``<key>.<case>.$selected``,
        the decision naming the case. Picked (``Select(by=d)``), it has no key: its value
        ``<key>`` is a derived node reading ``d``, refused ``select-no-case`` where ``d``
        names no case; a guard ``<key>.$known`` holds wherever that value is known (refused
        where it refuses); and each case guard reads the value inside that guard, by an
        argument named ``picked`` so it is never taken for a chosen selector's case guard
        (``_scopes.Scopes.case_of``). A refused picker is one piece, in the selector's name:
        a piece is counted at the highest refused guard of its chain (``_running.evaluate``)."""
        key = member_key(scope.name, name)
        guard = self.table.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(decision),
            key + ".$guard",
            owner=key,
        )
        cases_guard: NodeRef | None
        if decision.by is None:
            selector = self.table.reserve(
                scope.index, key, "decision", STRING, guard=guard, origin=decision.origin
            )
            self.table.nodes[selector] = replace(
                self.table.nodes[selector],
                domain=cast(Domain[object], finite(decision.candidates, STRING)),
            )
            cases_guard, argument = guard, "selected"
        else:
            selector = self.table.reserve(
                scope.index, key, "derived", STRING, guard=guard, origin=decision.origin
            )
            self.table.nodes[selector] = replace(
                self.table.nodes[selector], function=_picks(tuple(decision.candidates))
            )
            by = resolve_source(decision.by, scope.effective, f"{key} by=")
            self.table.picks.append((selector, scope.index, by))
            cases_guard = self.table.reserve(
                scope.index, key + ".$known", "derived", BOOL, guard=guard, source_owner=key
            )
            self.table.nodes[cases_guard] = replace(
                self.table.nodes[cases_guard],
                function=_known,
                arguments=(Argument("picked", selector),),
            )
            argument = "picked"
        scope.named_members[name] = selector
        choice = ChoiceDraft(len(self.table.choice_drafts), scope.index, key, guard, selector)
        self.table.choice_drafts.append(choice)
        for alias in self.table.aliases[scope.effective.space_type][name]:
            scope.choices[alias] = choice.index
            scope.members[alias] = selector
        for case, record in decision.candidates.items():
            case_key = member_key(key, case)
            if record is None:
                choice.cases.append((case, None))
                continue
            case_guard = self.table.reserve(
                scope.index,
                case_key + ".$selected",
                "derived",
                BOOL,
                guard=cases_guard,
                source_owner=case_key,
            )
            self.table.nodes[case_guard] = replace(
                self.table.nodes[case_guard],
                function=_names_case(case, argument),
                arguments=(Argument(argument, selector),),
            )
            case_guard = cast(
                int,
                self.table.guarded(
                    scope.index,
                    case_guard,
                    scope.effective.guards.get(record, record.when),
                    case_key + ".$guard",
                    owner=case_key,
                ),
            )
            child = self.place(
                scope,
                f"{name}.{case}",
                record,
                case_guard,
                source_scope=scope.index,
                keys=(record,),
            )
            self.table.drafts[child].candidate = (choice.index, case)
            choice.cases.append((case, child))
        self.shared_members(scope, choice)

    def shared_members(self, scope: ScopeDraft, choice: ChoiceDraft) -> None:
        """Link ``decision.member`` for every member name several candidates share.

        A name only one candidate has needs no node: that candidate's member is
        present exactly when it is selected. A name with incompatible types in
        different candidates is dropped when semantics are checked.
        """
        found: dict[str, list[tuple[str, int, ValueSemantics[object] | None]]] = {}
        for case, child in choice.cases:
            if child is None:
                continue
            for member, semantics in self.candidate_members(child):
                found.setdefault(member, []).append((case, child, semantics))
        for member, entries in found.items():
            if len(entries) < 2:
                continue
            known = [semantics for _, _, semantics in entries if semantics is not None]
            if any(not known[0].is_compatible_with(other) for other in known[1:]):
                continue
            alternatives = tuple(
                (case, self.candidate_member(child, member)) for case, child, _ in entries
            )
            node = self.table.reserve(
                scope.index,
                f"{choice.key}.$member.{member}",
                "select",
                guard=choice.guard,
                source_owner=choice.key,
            )
            self.table.nodes[node] = replace(
                self.table.nodes[node], selector=choice.selector, alternatives=alternatives
            )
            choice.members[member] = node
            choice.selects[node] = member
            choice.shared.add(node)
            self.table.shared[node] = (choice.index, member)

    def candidate_members(self, child: int) -> Iterator[tuple[str, ValueSemantics[object] | None]]:
        """A placed candidate's value members (not its constraints), in member order, each
        with the type its class declares."""
        draft = self.table.drafts[child]
        kinds = self.table.kinds[draft.effective.space_type]
        for member, declaration in draft.effective.members.items():
            if kinds[member] not in ("node", "choice") and not isinstance(declaration, Constraint):
                yield member, draft.effective.semantics.get(declaration)

    def candidate_member(self, child: int, member: str) -> NodeRef:
        """A placed candidate's value member, as this class reads it (through its slot)."""
        return cast(NodeRef, self.table.named_at(Place(child), member))

    def allocate(self) -> None:
        # The own scope is placed under the guard it is placed under, known where placed.
        self.new_scope(
            self.table.space_type,
            None,
            "",
            PLACING,
            record=None,
            source_scope=0,
        )
        cursor = 0
        while cursor < len(self.table.drafts):
            scope = self.table.drafts[cursor]
            cursor += 1
            if scope.placed is not None:
                continue  # a child's compiled form places its own
            kinds = self.table.kinds[scope.effective.space_type]
            for name, declaration in scope.effective.members.items():
                kind = kinds[name]
                if kind == "node":
                    self.child(scope, name, cast(NodeDecl, declaration))
                elif kind == "choice":
                    self.choice(scope, name, cast(NodeDecision, declaration))
        if self.missing:
            raise DefinitionError.broken("formal-unsupplied", _OWNER, "; ".join(self.missing))
        self.validity_pieces()
        # All scope identities and membership maps are now stable. Later phases replace
        # node payloads, never scope identities.

    def validity_pieces(self) -> None:
        """The own scope's validity pieces: its constraints, its Decisions (each a piece
        only where its domain lists no value, or its guard is refused) and its formals
        (decided by their binding where the class is placed), in node order, then each
        slot's validity (its child's, through the slot), in placement order: a child node,
        or a selector's case."""
        nodes, drafts = self.table.nodes, self.table.drafts
        candidates = {"constraint", "decision", "param"}
        own: dict[int, list[NodeRef]] = {draft.index: [] for draft in drafts}
        for node in nodes:
            if node.kind in candidates:
                own[node.scope].append(node.index)
        for draft in drafts:
            if draft.placed is not None:
                validity = self.table.lift(Place(draft.index), draft.placed.compiled.validity)
                own[cast(int, draft.parent)].append(validity)
        for draft in drafts:
            if draft.placed is not None:
                continue
            index = cast(int, draft.validity)
            nodes[index] = replace(nodes[index], constraints=tuple(own[draft.index]))
