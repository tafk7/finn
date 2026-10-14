# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Allocate a Space class's occurrences: every placed node a scope, every member a node.

The root scope is the Space class's own, its formals runtime inputs; scopes are
then walked breadth-first. A child node and a Decision over nodes' candidates
each place a scope. Each member gets its node, bound by what the body that
declared the node supplied for it (a formal only), and the task that lowers it.
A named shared Decision then gets one owning node. Afterwards scope identities
and membership maps are fixed (``Table.scopes``); later phases replace node
payloads only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import cast

from ._bindings import PlacementBinding, Slot, classify
from ._configuration import Space
from ._nodes import (
    NodeDecision,
    NodeDecl,
    fallback,
    missing_formal,
)
from ._ordering import dependency_order
from ._table import (
    ChoiceDraft,
    MemberTask,
    ScopeDraft,
    Table,
    member_key,
)
from .declarations import (
    MISSING,
    UNSUPPLIED,
    CaseRef,
    ChoiceMemberRef,
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    Derived,
    MemberRef,
    Members,
    Param,
    Supplied,
    ValueRef,
    View,
    at,
)
from .domains import Domain, finite
from .errors import DefinitionError
from .expressions import Expr
from .ir import Argument, NodeKind
from .semantics import BOOL, STRING

_OWNER = "space.allocate"


def _matches(expected: str) -> Callable[..., object]:
    def matches(*, selected: str) -> bool:
        return selected == expected

    return matches


_BINDING_KINDS: Mapping[str, NodeKind] = {
    "literal": "const",
    "reference": "alias",
    "local-decision": "decision",
}


def allocate(space_type: type[Space]) -> Table:
    """A new table with every occurrence of ``space_type``'s design space allocated."""
    allocation = _Allocation(Table(space_type))
    allocation.check_recursion()
    allocation.allocate()
    return allocation.table


def _declared_kind(declaration: Declaration) -> NodeKind | None:
    """The node kind of a Space class's own member that is neither a formal nor a Decision."""
    if isinstance(declaration, Const):
        return "const"
    if isinstance(declaration, (Derived, Expr, Supplied)):
        return "derived"
    if isinstance(declaration, Constraint):
        return "constraint"
    if isinstance(declaration, View):
        return "view"
    if isinstance(declaration, ConstraintGroup):
        return "group"
    if isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
        return "alias"
    if isinstance(declaration, Members):
        return "members"
    return None


class _Allocation:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.member_positions: dict[int, int] = {}
        # Named Decisions that are not class attributes, by authoring scope: their uses.
        self.named: dict[tuple[int, int], list[int]] = {}
        self.named_declarations: dict[tuple[int, int], Decision[object]] = {}
        # Required formals nothing supplies, reported together after allocation.
        self.missing: list[str] = []

    # -- Space classes -------------------------------------------------------------------

    def check_recursion(self) -> None:
        """Reject a Space class that places itself, before allocating occurrences."""

        pending = [self.table.space_type]
        children_of: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in children_of:
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
        guard: int | None,
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
        draft.slots = self.own_slots(draft)
        kinds = self.table.kinds[draft.effective.space_type]
        for member_name, declaration in draft.effective.members.items():
            kind = kinds[member_name]
            # Structure (nodes and choices) is allocated below.
            if kind not in ("node", "choice"):
                self.member(draft, member_name, declaration)
        for declaration, member_name in draft.effective.aliases.items():
            if member_name in draft.named_members:
                draft.members[declaration] = draft.named_members[member_name]
        for export, declaration in draft.effective.exports.items():
            draft.members[export] = draft.members[declaration]
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
        node_kind: NodeKind | None
        if isinstance(declaration, Param):
            if binding is not None:
                node_kind = _BINDING_KINDS[binding.kind]
            elif draft.parent is None:
                node_kind = "param"
            else:
                binding, node_kind = self.unsupplied(draft, member_name, declaration)
                if binding is None:
                    draft.unsupplied.add(member_name)
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
        node = self.table.reserve(
            draft.index,
            key,
            node_kind,
            draft.effective.semantics.get(declaration),
            guard=draft.guard,
            origin=declaration.origin,
        )
        draft.named_members[member_name] = node
        if binding is not None and binding.kind == "local-decision":
            self.local_decision(draft, node, binding)
        self.member_positions[node] = len(self.table.members)
        self.table.members.append(MemberTask(node, draft.index, member_name, declaration, binding))

    def unsupplied(
        self, draft: ScopeDraft, name: str, formal: Param[object]
    ) -> tuple[PlacementBinding | None, NodeKind]:
        """A child's formal nothing supplies: its default, unsupplied, or missing."""
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
        if default is MISSING or default is UNSUPPLIED:
            return None, "present"  # no alternatives: unsupplied
        return PlacementBinding(default, "literal"), "const"

    def local_decision(self, draft: ScopeDraft, node: int, binding: PlacementBinding) -> None:
        """An inline Decision: one use keys it at the formal; a shared one needs a name."""
        decision = cast(Decision[object], binding.supplier)
        if decision.name is None:
            if len(decision.sites) > 1:
                raise DefinitionError.broken(
                    "shared-decision-unnamed",
                    _OWNER,
                    f"Decision{at(decision.origin)} supplies {len(decision.sites)} formals "
                    f"({', '.join(decision.sites)}): a shared decision must be named. Make it "
                    'a class attribute, or pass Decision(..., name="...")',
                )
            return
        identity = (draft.source_scope, id(decision))
        self.named.setdefault(identity, []).append(node)
        self.named_declarations[identity] = decision

    def place(
        self,
        scope: ScopeDraft,
        name: str,
        record: NodeDecl,
        guard: int | None,
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
        """A Decision over nodes: its selector, and a guarded scope per candidate."""
        key = member_key(scope.name, name)
        guard = self.table.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(decision),
            key + ".$guard",
            owner=key,
        )
        never = self.never_holds(scope, scope.effective.guards.get(decision))
        selector = self.table.reserve(
            scope.index,
            key,
            "const" if never else "decision",
            STRING,
            guard=guard,
            origin=decision.origin,
        )
        if never:
            # It can never apply here: the selector reads inapplicable and has no key,
            # and no candidate is compiled.
            self.table.nodes[selector] = replace(
                self.table.nodes[selector], value=next(iter(decision.candidates))
            )
        else:
            self.table.nodes[selector] = replace(
                self.table.nodes[selector],
                domain=cast(Domain[object], finite(decision.candidates, STRING)),
                required_choice=decision.required,
            )
        scope.named_members[name] = selector
        choice = ChoiceDraft(len(self.table.choice_drafts), scope.index, key, guard, selector)
        self.table.choice_drafts.append(choice)
        for alias in self.table.aliases[scope.effective.space_type][name]:
            scope.choices[alias] = choice.index
            scope.members[alias] = selector
        choice.never = never
        for case, record in decision.candidates.items():
            case_key = member_key(key, case)
            if never and record is not None:
                choice.space_types[case] = record.space_type
            if record is None or never:
                choice.cases.append((case, None))
                continue
            case_guard = self.table.reserve(
                scope.index,
                case_key + ".$selected",
                "derived",
                BOOL,
                guard=guard,
                source_owner=case_key,
            )
            self.table.nodes[case_guard] = replace(
                self.table.nodes[case_guard],
                function=_matches(case),
                arguments=(Argument("selected", selector),),
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
            choice.cases.append((case, child))
        self.shared_members(scope, choice)

    def never_holds(self, scope: ScopeDraft, condition: ValueRef[bool] | None) -> bool:
        """Whether a guard can never hold in ``scope``: the supply of a value input
        that nothing supplies here (``supplied``)."""
        if not isinstance(condition, Supplied):
            return False
        return scope.effective.aliases.get(condition.formal) in scope.unsupplied

    def shared_members(self, scope: ScopeDraft, choice: ChoiceDraft) -> None:
        """Link ``decision.member`` for every member name several candidates share.

        A name only one candidate has needs no node: that candidate's member is
        present exactly when it is selected. A name with incompatible types in
        different candidates is dropped when semantics are checked.
        """
        found: dict[str, list[tuple[str, int]]] = {}
        for case, child in choice.cases:
            if child is None:
                continue
            for member, index in self.table.drafts[child].named_members.items():
                if self.table.nodes[index].kind not in {"constraint", "group"}:
                    found.setdefault(member, []).append((case, index))
        for member, alternatives in found.items():
            if len(alternatives) < 2:
                continue
            known = [s for _, i in alternatives if (s := self.table.nodes[i].semantics) is not None]
            if any(not known[0].is_compatible_with(other) for other in known[1:]):
                continue
            node = self.table.reserve(
                scope.index,
                f"{choice.key}.$member.{member}",
                "select",
                guard=choice.guard,
                source_owner=choice.key,
            )
            self.table.nodes[node] = replace(
                self.table.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
            )
            choice.members[member] = node
            self.table.shared[node] = (choice.index, member)

    def allocate(self) -> None:
        self.new_scope(
            self.table.space_type,
            None,
            "",
            None,
            record=None,
            source_scope=0,
        )
        cursor = 0
        while cursor < len(self.table.drafts):
            scope = self.table.drafts[cursor]
            cursor += 1
            kinds = self.table.kinds[scope.effective.space_type]
            for name, declaration in scope.effective.members.items():
                kind = kinds[name]
                if kind == "node":
                    self.child(scope, name, cast(NodeDecl, declaration))
                elif kind == "choice":
                    self.choice(scope, name, cast(NodeDecision, declaration))
        if self.missing:
            raise DefinitionError.broken("formal-unsupplied", _OWNER, "; ".join(self.missing))
        self.own_named_decisions()
        # All occurrence identities and membership maps are now stable. Later
        # phases replace node payloads, never scope identities.
        self.table.scopes = tuple(draft.freeze() for draft in self.table.drafts)

    def own_named_decisions(self) -> None:
        """A named Decision that is not a class attribute is one decision.

        It is owned by the lowest scope containing every node it supplies, so
        it applies whenever that scope does, and it is keyed by that scope and
        its name. Every formal it supplies becomes an alias of it that keeps its
        own node's guard, and through which the decision may be edited.
        """

        for identity, uses in self.named.items():
            decision = self.named_declarations[identity]
            owner = self.lowest_common_scope([self.table.nodes[use].scope for use in uses])
            draft = self.table.drafts[owner]
            name = cast(str, decision.name)
            key = member_key(draft.name, name)
            if name in draft.named_members or name in draft.named_children:
                raise DefinitionError.broken(
                    "shared-decision-name-taken",
                    _OWNER,
                    f"{key}: the shared Decision{at(decision.origin)} is named like a member "
                    f"of {draft.effective.space_type.__qualname__}; choose another name",
                    declaration=key,
                )
            first = self.table.nodes[uses[0]]
            node = self.table.reserve(
                owner,
                key,
                "decision",
                first.semantics,
                guard=draft.guard,
                origin=decision.origin,
            )
            task = self.table.members[self.member_positions[first.index]]
            self.member_positions[node] = len(self.table.members)
            self.table.members.append(replace(task, index=node, owner_scope=owner))
            draft.members[decision] = node
            draft.named_members[name] = node
            for use in uses:
                position = self.member_positions[use]
                previous = cast(PlacementBinding, self.table.members[position].binding)
                self.table.members[position] = replace(
                    self.table.members[position],
                    binding=replace(previous, supplier=node, kind="reference"),
                )
                self.table.nodes[use] = replace(self.table.nodes[use], kind="alias")
                self.table.editable_aliases.add(use)

    def lowest_common_scope(self, scopes: list[int]) -> int:
        def ancestry(scope: int) -> list[int]:
            chain = [scope]
            while (parent := self.table.drafts[chain[-1]].parent) is not None:
                chain.append(parent)
            return chain[::-1]

        chains = [ancestry(scope) for scope in scopes]
        common = chains[0][0]
        for level in range(min(len(chain) for chain in chains)):
            candidates = {chain[level] for chain in chains}
            if len(candidates) != 1:
                break
            common = chains[0][level]
        return common
