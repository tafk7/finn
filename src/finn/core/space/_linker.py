# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Iterative occurrence allocation and linking into one owned node table.

Every declared node becomes a scope; every member a node of the evaluation
graph. A Decision over nodes becomes a selector decision, one guarded scope
per candidate, and a ``select`` node per member read through it. A reference
input becomes a presence node plus a scope reference (or, for a fresh node, a
placement there); ``Users`` becomes a ``members`` node over the exports of
the nodes whose inputs reference this one.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import cast

from ._bindings import PlacementBinding, PlacementPlan, Supply, fallback, placement_plan
from ._configuration import Space
from ._graph import dependency_order
from ._nodes import (
    FamilyFormal,
    NodeDecision,
    NodeDecl,
    family_formals,
    is_fresh,
    missing_formal,
    unwrap,
)
from ._signatures import BoundArgument, BoundFunction, validate_argument
from .collection import EffectiveSpace, collect_space
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
    LocatedParam,
    MemberRef,
    Members,
    Param,
    Present,
    Users,
    ValueRef,
    View,
    ViewKey,
    at,
)
from .domains import Domain, finite
from .errors import DefinitionError, RequestError
from .expressions import INTEGER_SEMANTICS, Expr, IntOperator, evaluator
from .graph import LOCATED, Located
from .ir import Argument, Choice, LinkedModel, Node, NodeKind, Scope
from .references import resolve_reference
from .semantics import ValueSemantics, default_semantics

_BOOL = default_semantics(bool)
_STRING = default_semantics(str)


@dataclass
class _ScopeDraft:
    index: int
    parent: int | None
    name: str
    effective: EffectiveSpace
    guard: int | None
    source_scope: int
    record: NodeDecl | None
    plan: PlacementPlan
    members: dict[object, int] = field(default_factory=dict)
    named_members: dict[str, int] = field(default_factory=dict)
    children: dict[object, int] = field(default_factory=dict)
    named_children: dict[str, int] = field(default_factory=dict)
    choices: dict[object, int] = field(default_factory=dict)
    references: dict[object, int] = field(default_factory=dict)
    # Each reference input's declaration -> the scope it reaches (None: unsupplied).
    targets: dict[object, int | None] = field(default_factory=dict)

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
class _ChoiceDraft:
    index: int
    scope: int
    key: str
    guard: int | None
    selector: int
    cases: list[tuple[str, int | None]] = field(default_factory=list)
    members: dict[str, int] = field(default_factory=dict)

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
class _MemberTask:
    index: int
    scope: int
    name: str
    declaration: Declaration
    binding: PlacementBinding | None
    # Where the owning scope of a named shared decision was inferred to be.
    owner_scope: int | None = None


@dataclass(frozen=True)
class _ReferenceTask:
    """A reference input to resolve once every node of its authoring scope is placed."""

    scope: int
    name: str
    supplier: NodeDecl
    source_scope: int
    presence: int


@dataclass(frozen=True)
class _GuardTask:
    index: int
    source_scope: int
    condition: ValueRef[bool]


@dataclass(frozen=True)
class _ExpressionTask:
    index: int
    scope: int
    owner: str
    operator: IntOperator
    operands: tuple[int | ValueRef[int], ...]


def _key(scope: str, member: str) -> str:
    return f"{scope}.{member}" if scope else member


def _matches(expected: str) -> Callable[..., object]:
    def matches(*, selected: str) -> bool:
        return selected == expected

    return matches


_BINDING_KINDS: Mapping[str, NodeKind] = {
    "literal": "const",
    "reference": "alias",
    "local-decision": "decision",
    "parameter": "param",
}


def _is_located(semantics: ValueSemantics[object] | None) -> bool:
    return semantics is not None and semantics.type_token is Located


class _Linker:
    def __init__(self, space_type: type[Space], root: NodeDecl | None) -> None:
        self.space_type = space_type
        self.root = root
        self.effective: dict[type[Space], EffectiveSpace] = {}
        self.aliases: dict[type[Space], dict[str, list[Declaration]]] = {}
        self.nodes: list[Node] = []
        self.drafts: list[_ScopeDraft] = []
        self.choice_drafts: list[_ChoiceDraft] = []
        self.members: list[_MemberTask] = []
        self.member_positions: dict[int, int] = {}
        self.guards: list[_GuardTask] = []
        self.argument_checks: list[tuple[BoundArgument, int, str]] = []
        self.expression_nodes: dict[tuple[int, Expr], int] = {}
        self.expression_tasks: list[_ExpressionTask] = []
        self.expression_counts: dict[str, int] = {}
        self.scopes: tuple[Scope, ...] = ()
        self.choices: tuple[Choice, ...] = ()
        self.present_nodes: dict[tuple[int, Present[object], bool], int] = {}
        self.choice_member_nodes: dict[tuple[int, str, bool], int] = {}
        # Named Decisions that are not class attributes, by authoring scope: their uses.
        self.named: dict[tuple[int, int], list[int]] = {}
        self.named_declarations: dict[tuple[int, int], Decision[object]] = {}
        self.editable_aliases: set[int] = set()
        self.formals: dict[type[Space], dict[str, Declaration]] = {}
        # Selections linked for a member name several candidates share.
        self.shared: dict[int, tuple[int, str]] = {}
        # Reference inputs waiting for their authoring scope to be placed.
        self.pending: list[_ReferenceTask] = []
        # Referenced scope -> the (user scope, input name) pairs that reference it.
        self.users: dict[int, list[tuple[int, str]]] = {}
        # Required formals nothing supplies, reported together after allocation.
        self.missing: list[str] = []

    # -- families -----------------------------------------------------------------------

    def family(self, space_type: type[Space]) -> EffectiveSpace:
        effective = self.effective.get(space_type)
        if effective is None:
            effective = collect_space(space_type)
            self.effective[space_type] = effective
            aliases: dict[str, list[Declaration]] = {}
            for declaration, name in effective.aliases.items():
                aliases.setdefault(name, []).append(declaration)
            self.aliases[space_type] = aliases
        return effective

    def check_recursion(self) -> None:
        """Reject a family that places itself, before allocating occurrences."""

        pending = [self.space_type]
        families: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in families:
                continue
            effective = self.family(space_type)
            children: list[type[Space]] = []
            for declaration in effective.members.values():
                if isinstance(declaration, FamilyFormal):
                    continue  # supplied per node, never by the family itself
                if isinstance(declaration, NodeDecl):
                    children.append(declaration.family)
                elif isinstance(declaration, NodeDecision):
                    children.extend(
                        record.family
                        for record in declaration.candidates.values()
                        if record is not None
                    )
            families[space_type] = tuple(children)
            pending.extend(child for child in children if child not in families)
        indices = {space_type: index for index, space_type in enumerate(families)}
        structure = tuple(
            tuple(indices[child] for child in families[space_type]) for space_type in families
        )
        try:
            dependency_order(structure, tuple(family.__qualname__ for family in families))
        except DefinitionError as cause:
            raise DefinitionError("recursive Space placement", findings=cause.findings) from cause

    # -- allocation ---------------------------------------------------------------------

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
        if condition is None:
            return outer
        index = self.reserve(
            source_scope,
            key,
            "guard",
            cast(ValueSemantics[object], _BOOL),
            guard=outer,
            source_owner=owner,
        )
        self.guards.append(_GuardTask(index, source_scope, condition))
        return index

    def nested_supplies(self, parent: int, record: NodeDecl | None, key: str) -> dict[str, Supply]:
        """Formals of ``record`` supplied through a path by an enclosing node.

        ``kernel.port.dtype = x`` in a body stores the binding on ``kernel`` for
        the path ``(port,)``; it applies to this placement of ``kernel`` only,
        and its supplier is read in the body that declared ``kernel``.
        """
        supplies: dict[str, Supply] = {}
        chain: list[NodeDecl | None] = [record]
        scope: int | None = parent
        while scope is not None:
            draft = self.drafts[scope]
            owner = draft.record
            if isinstance(owner, NodeDecl) and owner.nested and None not in chain:
                path = tuple(cast(list[NodeDecl], chain[::-1]))
                for name, (value, origin) in owner.nested.get(path, {}).items():
                    earlier = (
                        record.supplied_at.get(name)
                        if record is not None and name in record.bindings
                        else supplies[name][2]
                        if name in supplies
                        else False
                    )
                    if earlier is not False:
                        raise DefinitionError(
                            f"{_key(key, name)} (assigned at {origin}): the formal is already "
                            f"supplied at {earlier}; a formal has one supplier (write "
                            "alternatives as Present(a, b))"
                        )
                    supplies[name] = (value, draft.source_scope, origin)
            chain.append(owner)
            scope = draft.parent
        return supplies

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
        index = len(self.drafts)
        root = parent is None
        formals = self.formals.get(space_type)
        if formals is None:
            formals = self.formals[space_type] = family_formals(space_type)
        nested = None if parent is None else self.nested_supplies(parent, record, name)
        plan = placement_plan(space_type, record, root=root, formals=formals, nested=nested)
        draft = _ScopeDraft(
            index,
            parent,
            name,
            self.family(space_type),
            guard,
            index if root else source_scope,
            record,
            plan,
        )
        self.drafts.append(draft)
        for member_name, declaration in draft.effective.members.items():
            if isinstance(declaration, (NodeDecl, NodeDecision)):
                continue  # structure (and reference inputs): allocated below
            binding = plan.bindings.get(member_name)
            kind: NodeKind
            if isinstance(declaration, Param):
                if binding is not None:
                    kind = _BINDING_KINDS[binding.kind]
                elif root:
                    kind = "param"
                else:
                    binding, kind = self.unsupplied(draft, member_name, declaration)
            elif isinstance(declaration, Const):
                kind = "const"
            elif isinstance(declaration, Decision):
                kind = "decision"
            elif isinstance(declaration, (Derived, Expr)):
                kind = "derived"
            elif isinstance(declaration, Constraint):
                kind = "constraint"
            elif isinstance(declaration, View):
                kind = "view"
            elif isinstance(declaration, ConstraintGroup):
                kind = "group"
            elif isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
                kind = "alias"
            elif isinstance(declaration, Present):
                kind = "present"
            elif isinstance(declaration, Members):
                kind = "members"
            else:
                raise DefinitionError(f"{_key(name, member_name)}: unsupported declaration")
            node = self.reserve(
                index,
                _key(name, member_name),
                kind,
                draft.effective.semantics.get(declaration),
                guard=guard,
                origin=declaration.origin,
            )
            draft.named_members[member_name] = node
            if binding is not None and binding.kind == "local-decision":
                self.local_decision(draft, node, binding)
            self.member_positions[node] = len(self.members)
            self.members.append(_MemberTask(node, index, member_name, declaration, binding))
        for declaration, member_name in draft.effective.aliases.items():
            if member_name in draft.named_members:
                draft.members[declaration] = draft.named_members[member_name]
        for export, declaration in draft.effective.exports.items():
            draft.members[export] = draft.members[declaration]
        return index

    def unsupplied(
        self, draft: _ScopeDraft, name: str, formal: Declaration
    ) -> tuple[PlacementBinding | None, NodeKind]:
        """A child's formal nothing supplies: its default, unsupplied, or missing."""
        default = fallback(formal)
        if default is MISSING:
            self.missing.append(
                missing_formal(
                    _key(draft.name, name), formal, draft.record, self.space_type.__qualname__
                )
            )
        if default is MISSING or default is UNSUPPLIED:
            return None, "present"  # no alternatives: unsupplied
        return PlacementBinding(default, "literal"), "const"

    def local_decision(self, draft: _ScopeDraft, node: int, binding: PlacementBinding) -> None:
        """An inline Decision: one use keys it at the formal; a shared one needs a name."""
        decision = cast(Decision[object], binding.supplier)
        if decision.name is None:
            if len(decision.sites) > 1:
                raise DefinitionError(
                    f"Decision{at(decision.origin)} supplies {len(decision.sites)} formals "
                    f"({', '.join(decision.sites)}): a shared decision must be named. Make it "
                    'a class attribute, or pass Decision(..., name="...")'
                )
            return
        source = binding.source_scope if binding.source_scope is not None else draft.source_scope
        identity = (source, id(decision))
        self.named.setdefault(identity, []).append(node)
        self.named_declarations[identity] = decision

    def place(
        self,
        scope: _ScopeDraft,
        name: str,
        record: NodeDecl,
        guard: int | None,
        *,
        source_scope: int,
        keys: Iterable[object],
    ) -> int:
        child = self.new_scope(
            record.family,
            scope.index,
            _key(scope.name, name),
            guard,
            record=record,
            source_scope=source_scope,
        )
        for key in keys:
            scope.children[key] = child
        return child

    def choice(self, scope: _ScopeDraft, name: str, decision: NodeDecision) -> None:
        key = _key(scope.name, name)
        guard = self.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(decision),
            key + ".$guard",
            owner=key,
        )
        selector = self.reserve(
            scope.index,
            key,
            "decision",
            cast(ValueSemantics[object], _STRING),
            guard=guard,
            origin=decision.origin,
        )
        self.nodes[selector] = replace(
            self.nodes[selector],
            domain=cast(Domain[object], finite(decision.candidates, _STRING)),
        )
        scope.named_members[name] = selector
        choice = _ChoiceDraft(len(self.choice_drafts), scope.index, key, guard, selector)
        self.choice_drafts.append(choice)
        for alias in self.aliases[scope.effective.space_type][name]:
            scope.choices[alias] = choice.index
            scope.members[alias] = selector
        for case, record in decision.candidates.items():
            case_key = _key(key, case)
            if record is None:
                choice.cases.append((case, None))
                continue
            case_guard = self.reserve(
                scope.index,
                case_key + ".$selected",
                "derived",
                cast(ValueSemantics[object], _BOOL),
                guard=guard,
                source_owner=case_key,
            )
            self.nodes[case_guard] = replace(
                self.nodes[case_guard],
                function=_matches(case),
                arguments=(Argument("selected", selector),),
            )
            case_guard = cast(
                int,
                self.guarded(
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

    def shared_members(self, scope: _ScopeDraft, choice: _ChoiceDraft) -> None:
        """Link ``decision.member`` for every member name several candidates share.

        A name only one candidate has needs no node: that candidate's member is
        present exactly when it is selected. A name with incompatible types in
        different candidates is dropped when semantics are checked.
        """
        found: dict[str, list[tuple[str, int]]] = {}
        for case, child in choice.cases:
            if child is None:
                continue
            for member, index in self.drafts[child].named_members.items():
                if self.nodes[index].kind not in {"constraint", "group"}:
                    found.setdefault(member, []).append((case, index))
        for member, alternatives in found.items():
            if len(alternatives) < 2:
                continue
            known = [s for _, i in alternatives if (s := self.nodes[i].semantics) is not None]
            if any(not known[0].is_compatible_with(other) for other in known[1:]):
                continue
            node = self.reserve(
                scope.index,
                f"{choice.key}.$member.{member}",
                "select",
                guard=choice.guard,
                source_owner=choice.key,
            )
            self.nodes[node] = replace(
                self.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
            )
            choice.members[member] = node
            self.shared[node] = (choice.index, member)

    def reference_input(self, scope: _ScopeDraft, name: str, formal: FamilyFormal) -> None:
        """``output: Param[Stream]``: place a fresh node here, or reference a placed one.

        Either way the input gets a presence node: a constant guarded by the
        reached node's own guard, so reading the input of an absent node is
        inapplicable. An unsupplied optional input is unsupplied.
        """
        key = _key(scope.name, name)
        aliases = self.aliases[scope.effective.space_type][name]
        binding = scope.plan.bindings.get(name)
        presence = self.reserve(
            scope.index,
            key,
            "const" if binding is not None else "present",
            cast(ValueSemantics[object], _BOOL),
            origin=formal.origin,
        )
        scope.named_members[name] = presence
        for alias in aliases:
            scope.members[alias] = presence
        if binding is None:
            for alias in aliases:
                scope.targets[alias] = None
            if formal.required and scope.parent is not None:
                self.missing.append(
                    missing_formal(key, formal, scope.record, self.space_type.__qualname__)
                )
            return
        supplier = cast(NodeDecl, binding.supplier)
        source = binding.source_scope if binding.source_scope is not None else scope.source_scope
        if not is_fresh(supplier):
            self.pending.append(_ReferenceTask(scope.index, name, supplier, source, presence))
            return
        if len(supplier.sites) > 1:
            raise DefinitionError(
                f"{supplier.describe()} is supplied to {len(supplier.sites)} reference inputs "
                f"({', '.join(supplier.sites)}) and placed by none: place it (a class attribute "
                "or a composite member) so that each input references it"
            )
        guard = self.guarded(source, scope.guard, supplier.when, key + ".$guard", owner=key)
        child = self.place(
            scope, name, supplier, guard, source_scope=source, keys=(*aliases, supplier)
        )
        scope.named_children[name] = child
        for alias in aliases:
            scope.targets[alias] = child
        self.users.setdefault(child, []).append((scope.index, name))
        self.nodes[presence] = replace(self.nodes[presence], value=True, guard=guard)

    def resolve_references(self) -> None:
        """Resolve each reference input in the body that wrote it, in scope order.

        A reference names a node placed in that body (a sibling, a candidate)
        or forwards one of that body's own reference inputs. Nothing else is
        visible: a family reaches an ancestor's node only through its inputs.
        """
        for task in self.pending:
            draft = self.drafts[task.scope]
            source = self.drafts[task.source_scope]
            supplier = task.supplier
            key = _key(draft.name, task.name)
            target: int | None
            if isinstance(supplier, FamilyFormal):
                if supplier not in source.targets:
                    raise DefinitionError(
                        f"{key}: forwards a reference input{at(supplier.origin)} that is not an "
                        f"input of {source.effective.space_type.__qualname__}"
                    )
                target = source.targets[supplier]
            else:
                target = source.children.get(supplier)
                if target is None:
                    raise DefinitionError(
                        f"{key}: references {supplier.describe()}, which is not placed in "
                        f"{source.name or '<root>'}: a reference input names a node placed "
                        "beside it, or forwards an input of the enclosing family"
                    )
                self.users.setdefault(target, []).append((task.scope, task.name))
            for alias in self.aliases[draft.effective.space_type][task.name]:
                draft.targets[alias] = target
                if target is not None:
                    draft.references[alias] = target
            node = self.nodes[task.presence]
            if target is None:
                self.nodes[task.presence] = replace(node, kind="present")
            else:
                self.nodes[task.presence] = replace(
                    node, value=True, guard=self.drafts[target].guard
                )

    def allocate(self) -> None:
        self.new_scope(self.space_type, None, "", None, record=self.root, source_scope=0)
        cursor = 0
        while cursor < len(self.drafts):
            scope = self.drafts[cursor]
            cursor += 1
            for name, declaration in scope.effective.members.items():
                if isinstance(declaration, FamilyFormal):
                    self.reference_input(scope, name, declaration)
                elif isinstance(declaration, NodeDecl):
                    key = _key(scope.name, name)
                    guard = self.guarded(
                        scope.index,
                        scope.guard,
                        scope.effective.guards.get(declaration),
                        key + ".$guard",
                        owner=key,
                    )
                    child = self.place(
                        scope,
                        name,
                        declaration,
                        guard,
                        source_scope=scope.index,
                        keys=self.aliases[scope.effective.space_type][name],
                    )
                    scope.named_children[name] = child
                elif isinstance(declaration, NodeDecision):
                    self.choice(scope, name, declaration)
        self.resolve_references()
        if self.missing:
            raise DefinitionError("; ".join(self.missing))
        self.own_named_decisions()
        # All occurrence identities and membership maps are now stable. Later
        # phases replace node payloads, never scope identities.
        self.scopes = tuple(draft.freeze() for draft in self.drafts)

    def own_named_decisions(self) -> None:
        """A named Decision that is not a class attribute is one decision.

        It is owned by the lowest scope containing every node it supplies, so
        it applies whenever that scope does, and it is keyed by that scope and
        its name. Every formal it supplies becomes an alias of it that keeps its
        own node's guard, and through which the decision may be edited.
        """

        for identity, uses in self.named.items():
            decision = self.named_declarations[identity]
            owner = self.lowest_common_scope([self.nodes[use].scope for use in uses])
            draft = self.drafts[owner]
            name = cast(str, decision.name)
            key = _key(draft.name, name)
            if name in draft.named_members or name in draft.named_children:
                raise DefinitionError(
                    f"{key}: the shared Decision{at(decision.origin)} is named like a member "
                    f"of {draft.effective.space_type.__qualname__}; choose another name"
                )
            first = self.nodes[uses[0]]
            node = self.reserve(
                owner,
                key,
                "decision",
                first.semantics,
                guard=draft.guard,
                origin=decision.origin,
            )
            task = self.members[self.member_positions[first.index]]
            self.member_positions[node] = len(self.members)
            self.members.append(replace(task, index=node, owner_scope=owner))
            draft.members[decision] = node
            draft.named_members[name] = node
            for use in uses:
                position = self.member_positions[use]
                self.members[position] = replace(
                    self.members[position], binding=PlacementBinding(node, "reference")
                )
                self.nodes[use] = replace(self.nodes[use], kind="alias")
                self.editable_aliases.add(use)

    def lowest_common_scope(self, scopes: list[int]) -> int:
        def ancestry(scope: int) -> list[int]:
            chain = [scope]
            while (parent := self.drafts[chain[-1]].parent) is not None:
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

    # -- names --------------------------------------------------------------------------

    def local_name(self, scope: int, child: int) -> str:
        prefix = self.drafts[scope].name
        name = self.drafts[child].name
        return name[len(prefix) + 1 :] if prefix else name

    def located_name(self, scope: int, path: tuple[Declaration, ...]) -> str:
        """The node name of a path, relative to the scope that reads it."""
        names: list[str] = []
        current = scope
        for record in path:
            draft = self.drafts[current]
            if record is draft.record:
                continue
            child = draft.children.get(record)
            if child is not None:
                names.append(self.local_name(current, child))
            elif record in draft.references:
                # Through a reference input: named by the input, not by the node reached.
                child = draft.references[record]
                names.append(draft.effective.aliases[record])
            else:
                raise DefinitionError(f"{record.name}: node is not placed in this scope")
            current = child
        return ".".join(names)

    def members_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, int]]:
        """Each child node's contribution for ``key``, candidates in key order."""

        draft = self.drafts[scope]
        result: list[tuple[str, int]] = []
        for name, declaration in draft.effective.members.items():
            if isinstance(declaration, NodeDecl):
                if name not in draft.named_children:
                    continue  # a reference input: the node belongs where it is placed
                children = [draft.named_children[name]]
            elif isinstance(declaration, NodeDecision):
                cases = self.choice_drafts[draft.choices[declaration]].cases
                children = [child for _, child in cases if child is not None]
            else:
                continue
            for child in children:
                target = self.drafts[child].members.get(key)
                if target is None:
                    continue
                if self.nodes[target].kind != "view":
                    raise DefinitionError(
                        f"{draft.name or '<root>'}: member {key.name} must be a view"
                    )
                result.append((self.local_name(scope, child), target))
        return result

    def users_candidates(self, scope: int, key: ViewKey[object]) -> list[tuple[str, str, int]]:
        """Each user's export of ``key``: (its name beside this node, input, view)."""

        parent = self.drafts[scope].parent
        order = {
            user: list(self.drafts[user].effective.members) for user, _ in self.users.get(scope, ())
        }
        result: list[tuple[str, str, int]] = []
        for user, member in sorted(
            self.users.get(scope, ()), key=lambda item: (item[0], order[item[0]].index(item[1]))
        ):
            target = self.drafts[user].members.get(key)
            if target is None:
                continue
            if self.nodes[target].kind != "view":
                raise DefinitionError(
                    f"{self.drafts[user].name or '<root>'}: member {key.name} must be a view"
                )
            name = "" if parent is None or user == parent else self.local_name(parent, user)
            result.append((name, member, target))
        return result

    def fresh_number(self, owner: str) -> int:
        number = self.expression_counts.get(owner, 0)
        self.expression_counts[owner] = number + 1
        return number

    def locate(
        self,
        scope: int,
        source: object,
        *,
        owner: str,
        where: str | None = None,
        member: str | None = None,
    ) -> int:
        """Supply a located formal: the member's value with its node and member names."""

        source = unwrap(source)
        declared = cast("ValueSemantics[object] | None", getattr(source, "semantics", None))
        if _is_located(declared):
            return self.reference(scope, source, owner=owner)
        if isinstance(source, Present) and source.owner is None:
            return self.present(scope, cast(Present[object], source), owner=owner, locate=True)
        if isinstance(source, ChoiceMemberRef):
            return self.choice_member(scope, source, owner=owner, locate=True)
        if isinstance(source, Expr) and source.owner is None:
            raise DefinitionError(f"{owner}: an expression has no location to supply")
        target = self.reference(scope, source, owner=owner)
        if where is None and member is None:
            if isinstance(source, MemberRef):
                where = self.located_name(scope, source.path) or None
                member = cast(Declaration, unwrap(source.member)).name
            else:
                member = self.drafts[scope].effective.aliases.get(
                    cast(Declaration, source), getattr(source, "name", None)
                )
        index = self.reserve(
            scope,
            f"{owner}.$located.{self.fresh_number(owner)}",
            "locate",
            cast(ValueSemantics[object], LOCATED),
            guard=self.drafts[scope].guard,
            source_owner=owner,
        )
        self.nodes[index] = replace(self.nodes[index], output=target, value=(where, member))
        return index

    def present(
        self, scope: int, source: Present[object], *, owner: str, locate: bool = False
    ) -> int:
        identity = (scope, source, locate)
        if identity not in self.present_nodes:
            index = self.reserve(
                scope,
                f"{owner}.$present.{self.fresh_number(owner)}",
                "present",
                cast(ValueSemantics[object], LOCATED) if locate else source.semantics,
                guard=self.drafts[scope].guard,
                source_owner=owner,
            )
            self.present_nodes[identity] = index
            self.nodes[index] = replace(
                self.nodes[index],
                alternatives=tuple(
                    (f"{owner}.{position}", self.supply(scope, item, owner=owner, locate=locate))
                    for position, item in enumerate(source.sources)
                ),
            )
        return self.present_nodes[identity]

    def choice_member(
        self, scope: int, source: ChoiceMemberRef, *, owner: str, locate: bool = False
    ) -> int:
        """``decision.member``: select the selected candidate's member, by name."""

        target = self.descend(scope, source.path[:-1], owner=owner)
        decision = source.path[-1]
        index = self.drafts[target].choices.get(decision)
        if index is None:
            raise DefinitionError(f"{owner}: the Decision over nodes is not placed here")
        choice = self.choice_drafts[index]
        if not locate and source.member in choice.members:
            return choice.members[source.member]
        identity = (choice.index, source.member, locate)
        cached = self.choice_member_nodes.get(identity)
        if cached is not None:
            return cached
        alternatives: list[tuple[str, int]] = []
        for case, child in choice.cases:
            if child is None:
                continue
            member = self.drafts[child].named_members.get(source.member)
            if member is None:
                continue
            if locate:
                prefix = self.located_name(scope, source.path[:-1])
                where = self.local_name(target, child)
                member = self.locate(
                    child,
                    self.drafts[child].effective.members[source.member],
                    owner=owner,
                    where=f"{prefix}.{where}" if prefix else where,
                    member=source.member,
                )
            alternatives.append((case, member))
        if not alternatives:
            raise DefinitionError(
                f"{owner}: no candidate of {choice.key} has a member {source.member}"
            )
        node = self.reserve(
            target,
            f"{choice.key}.${'located' if locate else 'member'}.{source.member}",
            "select",
            cast(ValueSemantics[object], LOCATED) if locate else None,
            guard=choice.guard,
            source_owner=owner,
        )
        self.nodes[node] = replace(
            self.nodes[node], selector=choice.selector, alternatives=tuple(alternatives)
        )
        self.choice_member_nodes[identity] = node
        if not locate:
            choice.members[source.member] = node
        return node

    def descend(self, scope: int, path: Iterable[Declaration], *, owner: str) -> int:
        for record in path:
            draft = self.drafts[scope]
            if record is draft.record:
                continue
            child = draft.children.get(record, draft.references.get(record))
            if child is None:
                raise DefinitionError(
                    f"{owner}: {getattr(record, 'describe', lambda: record.name)()} is not "
                    f"placed in {self.drafts[scope].name or '<root>'}"
                )
            scope = child
        return scope

    def supply(self, scope: int, source: object, *, owner: str, locate: bool) -> int:
        return (
            self.locate(scope, source, owner=owner)
            if locate
            else self.reference(scope, source, owner=owner)
        )

    def reference(self, scope: int, source: object, *, owner: str) -> int:
        source = unwrap(source)
        if isinstance(source, Expr) and source.owner is None:
            return self.expression(scope, source, owner=owner)
        if isinstance(source, Present) and source.owner is None:
            return self.present(scope, cast(Present[object], source), owner=owner)
        if isinstance(source, ChoiceMemberRef) and source.owner is None:
            return self.choice_member(scope, source, owner=owner)
        if isinstance(source, CaseRef) and source.owner is None:
            target = self.descend(scope, source.path[:-1], owner=owner)
            index = self.drafts[target].choices.get(source.path[-1])
            if index is None:
                raise DefinitionError(f"{owner}: the Decision over nodes is not placed here")
            return self.choice_drafts[index].selector
        if isinstance(source, MemberRef) and source.owner is None:
            target = self.descend(scope, source.path, owner=owner)
            member = unwrap(source.member)
            try:
                return self.drafts[target].members[member]
            except (KeyError, TypeError) as cause:
                raise DefinitionError(
                    f"{owner}: {source!r} names no member of "
                    f"{self.drafts[target].effective.space_type.__qualname__}"
                ) from cause
        try:
            return self.drafts[scope].members[source]
        except (KeyError, TypeError):
            pass
        if isinstance(source, Declaration) and source.owner is None and source.origin:
            raise DefinitionError(
                f"{owner}: {type(source).__name__}{at(source.origin)} is not a member of "
                "this scope; name it as a class attribute, or supply a formal with it"
            )
        try:
            return resolve_reference(self.nodes, self.scopes, self.choices, scope, source)
        except (RequestError, IndexError) as cause:
            raise DefinitionError(f"{owner}: reference is not a member of this scope") from cause

    # -- expressions ----------------------------------------------------------------------

    def expression(
        self,
        scope: int,
        source: Expr,
        *,
        owner: str,
        index: int | None = None,
    ) -> int:
        """Register each source expression once per instantiated scope.

        A named expression owns its authored node. Anonymous shared expressions
        use their first consumer's source owner; every later consumer retains
        its own dependency edge and therefore its demanded evidence path.
        Their applicability comes from the scope, never the first consumer.
        """

        identity = (scope, source)
        previous = self.expression_nodes.get(identity)
        if previous is not None:
            return previous
        if index is None:
            index = self.reserve(
                scope,
                f"{owner}.$expr.{self.fresh_number(owner)}",
                "derived",
                cast(ValueSemantics[object], INTEGER_SEMANTICS),
                guard=self.drafts[scope].guard,
                source_owner=owner,
            )
        self.expression_nodes[identity] = index
        self.expression_tasks.append(
            _ExpressionTask(index, scope, owner, source.operator, tuple(source.operands))
        )
        return index

    def link_expressions(self) -> None:
        cursor = 0
        while cursor < len(self.expression_tasks):
            task = self.expression_tasks[cursor]
            cursor += 1
            if task.operator not in {"add", "sub", "mul", "floordiv", "mod", "neg"}:
                raise DefinitionError(f"{task.owner}: unsupported integer expression operator")
            names = ("operand",) if task.operator == "neg" else ("left", "right")
            if len(task.operands) != len(names):
                raise DefinitionError(f"{task.owner}: invalid integer expression arity")
            arguments: list[Argument] = []
            for name, operand in zip(names, task.operands):
                if type(operand) is int:
                    literal = self.reserve(
                        task.scope,
                        f"{self.nodes[task.index].key}.$literal.{name}",
                        "const",
                        cast(ValueSemantics[object], INTEGER_SEMANTICS),
                        source_owner=task.owner,
                    )
                    self.nodes[literal] = replace(self.nodes[literal], value=operand)
                    target = literal
                elif isinstance(operand, Const) and operand.owner is None:
                    semantics = cast(ValueSemantics[object], operand.semantics)
                    target = self.reserve(
                        task.scope,
                        f"{self.nodes[task.index].key}.$literal.{name}",
                        "const",
                        semantics,
                        source_owner=task.owner,
                    )
                    try:
                        value = semantics.freeze(operand.value)
                    except Exception as cause:
                        raise DefinitionError(f"{task.owner}: constant snapshot failed") from cause
                    self.nodes[target] = replace(self.nodes[target], value=value)
                elif isinstance(operand, ValueRef):
                    target = self.reference(task.scope, operand, owner=task.owner)
                else:
                    raise DefinitionError(
                        f"{task.owner}: integer expressions reject non-int literals"
                    )
                arguments.append(Argument(name, target))
            self.nodes[task.index] = replace(
                self.nodes[task.index],
                arguments=tuple(arguments),
                function=evaluator(task.operator),
            )

    def check_expression_semantics(self) -> None:
        """Resolve integer operand types after linked aliases have their semantics."""
        for task in self.expression_tasks:
            node = self.nodes[task.index]
            operands = (self.nodes[arg.node].semantics for arg in node.arguments)
            if any(semantics is None or semantics.type_token is not int for semantics in operands):
                raise DefinitionError(
                    f"{node.owner}: integer expression operands require int value semantics"
                )

    # -- members ------------------------------------------------------------------------

    def arguments(self, scope: int, function: BoundFunction, *, owner: str) -> tuple[Argument, ...]:
        arguments: list[Argument] = []
        for dependency in function.dependencies:
            target = self.reference(scope, dependency.source, owner=owner)
            arguments.append(Argument(dependency.name, target))
            self.argument_checks.append((dependency, target, owner))
        return tuple(arguments)

    def obligations(
        self,
        scope: int,
        sources: Iterable[object],
        allowed: tuple[type[object], ...],
        *,
        owner: str,
    ) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        for source in sources:
            if not isinstance(source, allowed):
                raise DefinitionError(f"{owner}: unsupported obligation declaration")
            if isinstance(source, Users):
                # Each user is obliged separately; one referencing twice counts once.
                key = cast(ViewKey[object], source.key)
                indices = list(
                    dict.fromkeys(target for *_, target in self.users_candidates(scope, key))
                )
            elif isinstance(source, Members):
                # A member family obliges each member's acceptance separately.
                indices = [
                    target
                    for _, target in self.members_candidates(
                        scope, cast(ViewKey[object], source.key)
                    )
                ]
            else:
                indices = [self.reference(scope, source, owner=owner)]
                if self.nodes[indices[0]].kind not in {"view", "constraint", "group"}:
                    raise DefinitionError(
                        f"{owner}: a view may require only constraints, groups, views, "
                        "references to views, Members and Users"
                    )
            for index in indices:
                if index in seen:
                    raise DefinitionError(f"{owner}: duplicate obligation")
                result.append(index)
                seen.add(index)
        return tuple(result)

    def domain(
        self,
        scope: int,
        declaration: Decision[object],
        semantics: ValueSemantics[object],
        *,
        owner: str,
    ) -> tuple[Domain[object], tuple[Argument, ...]]:
        try:
            domain = declaration.domain.with_semantics(semantics)
        except Exception as cause:
            raise DefinitionError(f"{owner}: invalid decision domain: {cause}") from cause
        arguments: list[Argument] = []
        for name, source in domain.dependencies:
            if not isinstance(source, ValueRef):
                raise DefinitionError(
                    f"{owner}: domain dependency {name} must be a value reference"
                )
            arguments.append(Argument(name, self.reference(scope, source, owner=owner)))
        supplied = dict.fromkeys(argument.name for argument in arguments)
        for role, function, values in (
            ("membership", domain.accepts, {"candidate": None, **supplied}),
            ("enumeration", domain.candidates, supplied),
        ):
            if function is not None:
                try:
                    inspect.signature(function).bind(**values)
                except (TypeError, ValueError) as cause:
                    raise DefinitionError(
                        f"{owner}: incompatible domain {role} signature"
                    ) from cause
        return domain, tuple(arguments)

    def callback(self, node: Node, function: BoundFunction) -> Node:
        return replace(
            node,
            function=function.function,
            call_style=function.call_style,
            arguments=self.arguments(node.scope, function, owner=node.owner),
        )

    def view_output(self, node: Node, declaration: View[object], name: str) -> int:
        """Both view forms lower to one accepted node with an explicit raw output."""
        if declaration.source is not None:
            return self.reference(node.scope, declaration.source, owner=node.key)
        function = self.drafts[node.scope].effective.functions[name]
        output = self.reserve(
            node.scope,
            node.key + ".$output",
            "derived",
            function.semantics,
            guard=node.guard,
            source_owner=node.key,
        )
        self.nodes[output] = self.callback(self.nodes[output], function)
        return output

    def link_member(self, task: _MemberTask) -> None:
        scope = self.drafts[task.scope]
        node = self.nodes[task.index]
        declaration = task.declaration
        binding = task.binding
        # A supplier is read in the body that wrote it: the node's own source
        # scope, or the enclosing node's for a binding assigned through a path.
        written = (
            scope.source_scope
            if binding is None or binding.source_scope is None
            else binding.source_scope
        )
        if binding is not None and binding.kind in {"literal", "reference"}:
            if binding.kind == "literal":
                self.nodes[node.index] = replace(node, value=binding.supplier)
            elif type(binding.supplier) is int and task.index in self.editable_aliases:
                self.nodes[node.index] = replace(node, output=binding.supplier)
            else:
                formal = cast(Param[object], declaration)
                located = isinstance(formal, LocatedParam)
                self.nodes[node.index] = replace(
                    node,
                    output=self.supply(written, binding.supplier, owner=node.key, locate=located),
                )
            return
        source_scope = scope.index
        condition: ValueRef[bool] | None = scope.effective.guards.get(declaration)
        guard_scope = scope
        if binding is not None and binding.kind == "local-decision":
            declaration = cast(Decision[object], binding.supplier)
            source_scope = written
            condition = declaration.when
            if task.owner_scope is not None:
                guard_scope = self.drafts[task.owner_scope]
        if binding is not None and binding.kind == "parameter":
            self.nodes[node.index] = replace(node, required=False)
            return
        guard = self.guarded(
            source_scope,
            guard_scope.guard,
            condition,
            node.key + ".$guard",
            owner=node.key,
        )
        node = replace(node, guard=guard)
        if isinstance(declaration, Param):
            node = replace(node, required=declaration.required)
        elif isinstance(declaration, Const):
            assert node.semantics is not None
            try:
                node = replace(node, value=node.semantics.freeze(declaration.value))
            except Exception as cause:
                raise DefinitionError(f"{node.key}: constant snapshot failed") from cause
        elif isinstance(declaration, Decision):
            assert node.semantics is not None
            domain, arguments = self.domain(
                source_scope, declaration, node.semantics, owner=node.key
            )
            node = replace(node, domain=domain, domain_arguments=arguments)
        elif isinstance(declaration, Expr):
            self.expression(scope.index, declaration, owner=node.key, index=node.index)
        elif isinstance(declaration, (Derived, Constraint)):
            node = self.callback(node, scope.effective.functions[task.name])
        elif isinstance(declaration, ConstraintGroup):
            node = replace(
                node,
                constraints=self.obligations(
                    scope.index,
                    declaration.constraints,
                    (Constraint,),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, View):
            node = replace(
                node,
                output=self.view_output(node, declaration, task.name),
                constraints=self.obligations(
                    scope.index,
                    declaration.requires,
                    (Constraint, ConstraintGroup, View, ValueRef, Members),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, Users):
            entries = self.users_candidates(scope.index, cast(ViewKey[object], declaration.key))
            node = replace(
                node,
                alternatives=tuple((name, target) for name, _, target in entries),
                value=tuple(member for _, member, _ in entries),
            )
        elif isinstance(declaration, Present):
            node = replace(
                node,
                alternatives=tuple(
                    (f"{node.key}.{position}", self.reference(scope.index, item, owner=node.key))
                    for position, item in enumerate(declaration.sources)
                ),
            )
        elif isinstance(declaration, Members):
            node = replace(
                node,
                alternatives=tuple(
                    self.members_candidates(scope.index, cast(ViewKey[object], declaration.key))
                ),
            )
            node = replace(node, value=(declaration.key.name,) * len(node.alternatives))
        elif isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
            anonymous = replace_owner(declaration)
            node = replace(node, output=self.reference(scope.index, anonymous, owner=node.key))
        self.nodes[node.index] = node

    # -- validation -----------------------------------------------------------------------

    def check_semantics(self, order: tuple[int, ...]) -> None:
        for index in order:
            node = self.nodes[index]
            if node.kind in {"view", "alias"} and node.output is not None:
                output = self.nodes[node.output].semantics
                if output is None:
                    raise DefinitionError(f"{node.key}: output has no value semantics")
                if node.semantics is not None and not node.semantics.is_compatible_with(output):
                    raise DefinitionError(f"{node.key}: output has incompatible value semantics")
                if node.semantics is None:
                    self.nodes[index] = node = replace(node, semantics=output)
            if node.kind in {"present", "select"} and node.semantics is None and node.alternatives:
                # Like an alias, these take their sources' linked semantics.
                self.nodes[index] = node = replace(
                    node, semantics=self.nodes[node.alternatives[0][1]].semantics
                )
            if node.kind in {"select", "present"}:
                for _, target in node.alternatives:
                    semantics = self.nodes[target].semantics
                    if (
                        node.semantics is None
                        or semantics is None
                        or not node.semantics.is_compatible_with(semantics)
                    ):
                        if index in self.shared and self.drop_shared(index):
                            break
                        raise DefinitionError(
                            f"{node.key}: alternatives have incompatible value semantics"
                        )
            if node.guard is not None:
                semantics = self.nodes[node.guard].semantics
                if semantics is None or not semantics.is_compatible_with(
                    cast(ValueSemantics[object], _BOOL)
                ):
                    raise DefinitionError(
                        f"{node.key}: applicability requires Boolean value semantics"
                    )
            if node.kind == "guard" and node.output is not None:
                semantics = self.nodes[node.output].semantics
                if semantics is None or not semantics.is_compatible_with(
                    cast(ValueSemantics[object], _BOOL)
                ):
                    raise DefinitionError(
                        f"{node.key}: applicability requires Boolean value semantics"
                    )
        for dependency, target, owner in self.argument_checks:
            semantics = self.nodes[target].semantics
            if semantics is None:
                raise DefinitionError(f"{owner}: dependency has no value semantics")
            validate_argument(dependency, semantics, owner=owner)
        for draft in self.drafts:
            for export, declaration in draft.effective.exports.items():
                semantics = self.nodes[draft.members[declaration]].semantics
                if semantics is None or not export.semantics.is_compatible_with(semantics):
                    raise DefinitionError(
                        f"{draft.name}: export {export.name} has incompatible semantics"
                    )

    def drop_shared(self, index: int) -> bool:
        """An unreferenced shared member whose candidates disagree on its type."""
        choice, member = self.shared.pop(index)
        if any(index in node.dependencies for node in self.nodes):
            return False  # a declaration reads it: the disagreement is an error
        del self.choice_drafts[choice].members[member]
        self.nodes[index] = replace(
            self.nodes[index], alternatives=(), semantics=cast(ValueSemantics[object], _STRING)
        )
        return True

    def build(self) -> LinkedModel:
        """Prepare templates, allocate occurrences, lower edges, then validate/freeze."""
        self.check_recursion()
        self.allocate()
        for task in self.members:
            self.link_member(task)
        for guard_task in self.guards:
            node = self.nodes[guard_task.index]
            self.nodes[guard_task.index] = replace(
                node,
                output=self.reference(
                    guard_task.source_scope, guard_task.condition, owner=node.owner
                ),
            )
        self.link_expressions()
        order = dependency_order(
            tuple(node.dependencies for node in self.nodes),
            tuple(node.key for node in self.nodes),
        )
        self.check_semantics(order)
        self.check_expression_semantics()
        self.choices = tuple(choice.freeze() for choice in self.choice_drafts)
        nodes = tuple(self.nodes)
        keys = {node.key: node.index for node in nodes}
        if len(keys) != len(nodes):
            raise DefinitionError("generated node names collide")
        return LinkedModel(
            nodes,
            self.scopes,
            order,
            tuple(node.index for node in nodes if node.kind == "param"),
            tuple(node.index for node in nodes if node.kind == "decision"),
            keys,
            self.choices,
            frozenset(self.editable_aliases),
        )


def replace_owner(reference: Declaration) -> Declaration:
    """A named reference member resolves as its anonymous structural twin."""
    if isinstance(reference, MemberRef):
        return MemberRef(reference.path, reference.member)
    if isinstance(reference, ChoiceMemberRef):
        return ChoiceMemberRef(reference.path, reference.member)
    if isinstance(reference, CaseRef):
        return CaseRef(reference.path)
    return reference


def link_space(space_type: type[Space], root: NodeDecl | None = None) -> LinkedModel:
    return _Linker(space_type, root).build()
