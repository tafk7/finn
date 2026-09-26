# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Iterative occurrence allocation and linking into one owned node table."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

from ._bindings import PlacementBinding, PlacementPlans
from ._configuration import Space
from ._graph import dependency_order
from ._signatures import BoundArgument, BoundFunction, validate_argument
from .collection import EffectiveSpace, collect_space
from .declarations import (
    AcceptedViewRef,
    Carried,
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    Declaration,
    Derived,
    Ends,
    Fold,
    Net,
    Param,
    Port,
    ScopedValueRef,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
)
from .domains import Domain, finite
from .errors import DefinitionError, RequestError
from .expressions import INTEGER_SEMANTICS, Expr, IntOperator, evaluator
from .graph import TOPOLOGY, Interpretation
from .ir import Argument, Choice, EndSlot, LinkedModel, Node, NodeKind, Scope, TopologyPlan
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
    bindings: Mapping[str, PlacementBinding]
    node_name: str | None = None
    # Typed as object: a Subspace is a descriptor, which a dataclass field would invoke.
    placement: object = None
    ports: Mapping[str, Net[Space]] = field(default_factory=dict)
    members: dict[object, int] = field(default_factory=dict)
    named_members: dict[str, int] = field(default_factory=dict)
    children: dict[object, int] = field(default_factory=dict)
    named_children: dict[str, int] = field(default_factory=dict)
    choices: dict[object, int] = field(default_factory=dict)

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
        )


@dataclass
class _ChoiceDraft:
    index: int
    scope: int
    key: str
    guard: int | None
    selector: int | None
    cases: list[tuple[str, int]] = field(default_factory=list)
    exports: dict[object, int] = field(default_factory=dict)

    def freeze(self) -> Choice:
        return Choice(
            self.index,
            self.scope,
            self.key,
            self.selector,
            tuple(self.cases),
            self.exports,
            self.guard,
        )


@dataclass(frozen=True)
class _MemberTask:
    index: int
    scope: int
    name: str
    declaration: Declaration
    binding: PlacementBinding | None
    binding_scope: int | None = None


@dataclass(frozen=True)
class _ParameterOverride:
    scope: int
    source_scope: int
    reference: ValueRef[object]
    binding: PlacementBinding


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


@dataclass
class _Attachment:
    """One potential end of a net: a child's port bound at placement, or the
    composite's own port attached from inside (``inside``)."""

    port: int
    declaration: Port[object]
    scope: int
    name: str
    node: str | None
    inside: bool
    publishes: bool = False


def _key(scope: str, member: str) -> str:
    return f"{scope}.{member}" if scope else member


def _matches(expected: str) -> Callable[..., object]:
    def matches(*, selected: str) -> bool:
        return selected == expected

    return matches


_BINDING_KINDS: Mapping[str, NodeKind] = {
    "literal": "const",
    "reference": "alias",
    "exposed-param": "param",
    "local-decision": "decision",
}


class _Linker:
    def __init__(self, space_type: type[Space]) -> None:
        self.space_type = space_type
        self.placements = PlacementPlans()
        self.effective: dict[type[Space], EffectiveSpace] = {}
        self.aliases: dict[type[Space], dict[str, list[Declaration]]] = {}
        self.nodes: list[Node] = []
        self.drafts: list[_ScopeDraft] = []
        self.choice_drafts: list[_ChoiceDraft] = []
        self.members: list[_MemberTask] = []
        self.member_positions: dict[int, int] = {}
        self.parameter_overrides: list[_ParameterOverride] = []
        self.guards: list[_GuardTask] = []
        self.argument_checks: list[tuple[BoundArgument, int, str]] = []
        self.expression_nodes: dict[tuple[int, Expr], int] = {}
        self.expression_tasks: list[_ExpressionTask] = []
        self.expression_counts: dict[str, int] = {}
        self.scopes: tuple[Scope, ...] = ()
        self.choices: tuple[Choice, ...] = ()
        # Graph composition: per net scope, its ends and its carried node; per
        # port node, the node its value aliases (None: unconnected).
        self.attachments: dict[int, list[_Attachment]] = {}
        self.carried: dict[int, int] = {}
        self.anchors: dict[int, list[tuple[str, int]]] = {}
        self.port_outputs: dict[int, int | None] = {}
        self.ends_nodes: dict[int, int] = {}
        self.folds: list[int] = []
        self.topologies: dict[tuple[int, int], int] = {}
        self.contributions: dict[int, tuple[int, ...]] = {}

    def collect(self) -> None:
        """Collect each family once, and reject structural recursion first."""

        pending = [self.space_type]
        children: dict[type[Space], tuple[type[Space], ...]] = {}
        cursor = 0
        while cursor < len(pending):
            space_type = pending[cursor]
            cursor += 1
            if space_type in self.effective:
                continue
            effective = collect_space(space_type, placements=self.placements)
            self.effective[space_type] = effective
            aliases: dict[str, list[Declaration]] = {}
            for declaration, name in effective.aliases.items():
                aliases.setdefault(name, []).append(declaration)
            self.aliases[space_type] = aliases
            descendants: list[type[Space]] = []
            for declaration in effective.members.values():
                if isinstance(declaration, Subspace):
                    descendants.append(declaration.space_type)
                elif isinstance(declaration, SubspaceChoice):
                    descendants.extend(
                        case.space_type for case in declaration.alternatives.values()
                    )
            children[space_type] = tuple(descendants)
            pending.extend(child for child in descendants if child not in self.effective)
        indices = {space_type: index for index, space_type in enumerate(self.effective)}
        structure = tuple(
            tuple(indices[child] for child in children[space_type]) for space_type in self.effective
        )
        try:
            dependency_order(structure, tuple(family.__qualname__ for family in self.effective))
        except DefinitionError as cause:
            raise DefinitionError("recursive Space placement", findings=cause.findings) from cause

    def reserve(
        self,
        scope: int,
        key: str,
        kind: NodeKind,
        semantics: ValueSemantics[object] | None = None,
        *,
        guard: int | None = None,
        source_owner: str | None = None,
    ) -> int:
        index = len(self.nodes)
        self.nodes.append(
            Node(index, scope, key, kind, semantics, guard=guard, source_owner=source_owner)
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

    def new_scope(
        self,
        space_type: type[Space],
        parent: int | None,
        name: str,
        guard: int | None,
        *,
        placement: Subspace[Space] | None = None,
        node_name: str | None = None,
    ) -> int:
        index = len(self.drafts)
        plan = self.placements.get(placement) if placement is not None else None
        bindings = plan.bindings if plan is not None else {}
        draft = _ScopeDraft(
            index,
            parent,
            name,
            self.effective[space_type],
            guard,
            0 if parent is None else parent,
            bindings,
            node_name,
            placement,
            plan.ports if plan is not None else {},
        )
        self.drafts.append(draft)
        if plan is not None:
            self.parameter_overrides.extend(
                _ParameterOverride(index, draft.source_scope, item.reference, item.binding)
                for item in plan.nested_bindings
            )
        for member_name, declaration in draft.effective.members.items():
            if isinstance(declaration, (Subspace, SubspaceChoice)):
                continue
            binding = bindings.get(member_name)
            kind: NodeKind
            if isinstance(declaration, Param):
                kind = "param" if binding is None else _BINDING_KINDS[binding.kind]
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
            elif isinstance(declaration, (ScopedValueRef, AcceptedViewRef, Carried, Ends)):
                kind = "alias"
            elif isinstance(declaration, Port):
                kind = "port"
            else:
                raise DefinitionError(f"{_key(name, member_name)}: unsupported declaration")
            node = self.reserve(
                index,
                _key(name, member_name),
                kind,
                draft.effective.semantics.get(declaration),
                guard=guard,
            )
            draft.named_members[member_name] = node
            self.member_positions[node] = len(self.members)
            self.members.append(_MemberTask(node, index, member_name, declaration, binding))
        for declaration, member_name in draft.effective.aliases.items():
            if member_name in draft.named_members:
                draft.members[declaration] = draft.named_members[member_name]
        for export, declaration in draft.effective.exports.items():
            draft.members[export] = draft.members[declaration]
        return index

    def choice(self, scope: _ScopeDraft, name: str, declaration: SubspaceChoice) -> None:
        key = _key(scope.name, name)
        guard = self.guarded(
            scope.index,
            scope.guard,
            scope.effective.guards.get(declaration),
            key + ".$guard",
            owner=key,
        )
        selector: int | None = None
        if len(declaration.alternatives) > 1:
            selector = self.reserve(
                scope.index,
                key + ".$selector",
                "decision",
                cast(ValueSemantics[object], _STRING),
                guard=guard,
                source_owner=key,
            )
            self.nodes[selector] = replace(
                self.nodes[selector],
                domain=cast(Domain[object], finite(declaration.alternatives, _STRING)),
            )
        choice = _ChoiceDraft(len(self.choice_drafts), scope.index, key, guard, selector)
        self.choice_drafts.append(choice)
        for alias in self.aliases[scope.effective.space_type][name]:
            scope.choices[alias] = choice.index
        for case, placement in declaration.alternatives.items():
            case_key = _key(key, case)
            case_guard = guard
            if selector is not None:
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
            case_guard = self.guarded(
                scope.index,
                case_guard,
                placement.when,
                case_key + ".$guard",
                owner=case_key,
            )
            child = self.new_scope(
                placement.space_type,
                scope.index,
                case_key,
                case_guard,
                placement=placement,
                node_name=name,
            )
            choice.cases.append((case, child))
        for export in declaration.exports:
            choice.exports[export] = self.reserve(
                scope.index,
                key + ".$export." + export.name,
                "select",
                export.semantics,
                guard=guard,
                source_owner=key,
            )

    def allocate(self) -> None:
        self.new_scope(self.space_type, None, "", None)
        cursor = 0
        while cursor < len(self.drafts):
            scope = self.drafts[cursor]
            cursor += 1
            for name, declaration in scope.effective.members.items():
                if isinstance(declaration, Subspace):
                    key = _key(scope.name, name)
                    guard = self.guarded(
                        scope.index,
                        scope.guard,
                        scope.effective.guards.get(declaration),
                        key + ".$guard",
                        owner=key,
                    )
                    child = self.new_scope(
                        declaration.space_type,
                        scope.index,
                        key,
                        guard,
                        placement=declaration,
                        node_name=name,
                    )
                    scope.named_children[name] = child
                    for alias in self.aliases[scope.effective.space_type][name]:
                        scope.children[alias] = child
                elif isinstance(declaration, SubspaceChoice):
                    self.choice(scope, name, declaration)
        # All occurrence identities and membership maps are now stable. Later
        # phases replace node payloads, never scope identities.
        self.scopes = tuple(draft.freeze() for draft in self.drafts)
        self.choices = tuple(choice.freeze() for choice in self.choice_drafts)

    def apply_parameter_overrides(self) -> None:
        """Bind exposed leaf slots from inner definitions to enclosing placements.

        Node and scope identities are already allocated. Only binding records
        change here, before domain/evaluator linking. Each supplier retains the
        scope where its outer placement was authored.
        """

        seen: set[tuple[int, int]] = set()
        for override in sorted(self.parameter_overrides, key=lambda item: item.scope, reverse=True):
            owner = self.scopes[override.scope].name
            target = self.reference(override.scope, override.reference, owner=owner)
            identity = (override.scope, target)
            if identity in seen:
                raise DefinitionError(f"{owner}: duplicate nested parameter binding")
            seen.add(identity)
            node = self.nodes[target]
            if node.kind != "param":
                raise DefinitionError(f"{node.key}: nested target is not deliberately exposed")
            position = self.member_positions[target]
            task = self.members[position]
            self.members[position] = replace(
                task,
                binding=override.binding,
                binding_scope=override.source_scope,
            )
            kind = _BINDING_KINDS[override.binding.kind]
            self.nodes[target] = replace(node, kind=kind)

    def attach(self) -> None:
        """Collect every net's ends, then decide which of them anchor its value.

        Inner nets are resolved first: whether a composite child's port
        publishes outward depends on whether its inside anchors the value.
        """

        nets = [draft.index for draft in self.drafts if isinstance(draft.placement, Net)]
        inside: list[tuple[int, _Attachment]] = []
        outside: list[tuple[int, _Attachment]] = []
        for draft in self.drafts:
            for name, declaration in draft.effective.members.items():
                if isinstance(declaration, Port) and declaration.net is not None:
                    net = draft.children.get(declaration.net)
                    if net is None or not isinstance(self.drafts[net].placement, Net):
                        raise DefinitionError(
                            f"{_key(draft.name, name)}: net= must name a Net of the same Space"
                        )
                    port = draft.named_members[name]
                    inside.append(
                        (net, _Attachment(port, declaration, draft.index, name, None, True))
                    )
            if draft.ports:
                assert draft.parent is not None
                parent = self.drafts[draft.parent]
                for name, net_declaration in draft.ports.items():
                    net = parent.children.get(net_declaration)
                    if net is None or not isinstance(self.drafts[net].placement, Net):
                        raise DefinitionError(
                            f"{_key(draft.name, name)}: a port binds to a Net of its parent"
                        )
                    declaration = cast(Port[object], draft.effective.members[name])
                    port = draft.named_members[name]
                    outside.append(
                        (
                            net,
                            _Attachment(
                                port, declaration, draft.index, name, draft.node_name, False
                            ),
                        )
                    )
        self.attachments = {net: [] for net in nets}
        for net, attachment in (*inside, *outside):
            self.attachments[net].append(attachment)
        outer = {attachment.port: net for net, attachment in outside}
        for net in nets:
            draft = self.drafts[net]
            interfaces = {
                id(a.declaration.interface): a.declaration.interface for a in self.attachments[net]
            }
            for declaration in draft.effective.members.values():
                if isinstance(declaration, (Carried, Ends)):
                    interfaces.setdefault(id(declaration.interface), declaration.interface)
            if len(interfaces) > 1:
                raise DefinitionError(f"{draft.name}: a net joins ports of one interface")
            if not interfaces:
                continue
            (interface,) = interfaces.values()
            self.carried[net] = self.reserve(
                net,
                draft.name + ".$carried",
                "unify",
                interface.carried,
                guard=draft.guard,
                source_owner=draft.name,
            )
            self.ends_nodes[net] = self.reserve(
                net,
                draft.name + ".$ends",
                "ends",
                cast(ValueSemantics[object], default_semantics(tuple)),
                guard=draft.guard,
                source_owner=draft.name,
            )
        exposes: set[int] = set()
        for net in sorted(nets, reverse=True):
            if net not in self.carried:
                continue
            draft = self.drafts[net]
            assert draft.parent is not None
            candidates: list[tuple[str, int]] = []
            carry = cast(Net[Space], draft.placement).carry
            if carry is not None:
                candidates.append((draft.name + ".carry", self.anchor(draft, carry)))
            for attachment in self.attachments[net]:
                declaration = attachment.declaration
                if declaration.carry is not None or (
                    not attachment.inside and attachment.port in exposes
                ):
                    attachment.publishes = True
                    candidates.append((self.nodes[attachment.port].key, attachment.port))
            for attachment in self.attachments[net]:
                if attachment.inside and attachment.declaration.carry is None:
                    if candidates and not attachment.publishes:
                        # The inside anchors this value: the port publishes it outward.
                        exposes.add(attachment.port)
                        self.port_outputs[attachment.port] = self.carried[net]
                    else:
                        # Nothing inside anchors it: the port relays it from outside.
                        attachment.publishes = True
                        candidates.append((self.nodes[attachment.port].key, attachment.port))
                        self.port_outputs[attachment.port] = (
                            self.carried[outer[attachment.port]]
                            if attachment.port in outer
                            else None
                        )
            if not candidates:
                raise DefinitionError(
                    f"{draft.name}: no end publishes the carried value; bind carry= or "
                    "publish it from a port"
                )
            self.anchors[net] = candidates
        for draft in self.drafts:
            for name, declaration in draft.effective.members.items():
                if not isinstance(declaration, Port):
                    continue
                port = draft.named_members[name]
                if declaration.carry is not None:
                    self.port_outputs[port] = self.reference(
                        draft.index, declaration.carry, owner=self.nodes[port].key
                    )
                elif port not in self.port_outputs:
                    net = outer.get(port)
                    self.port_outputs[port] = None if net is None else self.carried.get(net)

    def anchor(self, draft: _ScopeDraft, carry: object) -> int:
        """The parent's own anchor of a net: a reference, a literal or a fresh Decision."""

        assert draft.parent is not None
        semantics = cast(ValueSemantics[object], self.nodes[self.carried[draft.index]].semantics)
        key = draft.name + ".carry"
        if isinstance(carry, Decision) and carry.owner is None:
            index = self.reserve(draft.parent, key, "decision", semantics, guard=draft.guard)
            domain, arguments = self.domain(
                draft.parent, cast(Decision[object], carry), semantics, owner=key
            )
            self.nodes[index] = replace(
                self.nodes[index], domain=domain, domain_arguments=arguments
            )
            return index
        if isinstance(carry, ValueRef):
            return self.reference(draft.parent, carry, owner=key)
        index = self.reserve(draft.parent, key, "const", semantics, source_owner=draft.name)
        try:
            value = semantics.freeze(carry)
        except Exception as cause:
            raise DefinitionError(f"{key}: {cause}") from cause
        self.nodes[index] = replace(self.nodes[index], value=value)
        return index

    def link_graph(self) -> None:
        """Fill each net's anchors and ends, then each fold's topology."""

        for net, carried in self.carried.items():
            self.nodes[carried] = replace(
                self.nodes[carried],
                alternatives=tuple(self.anchors[net]),
            )
            slots: list[EndSlot] = []
            for attachment in self.attachments[net]:
                declaration = attachment.declaration
                direction = declaration.direction
                if attachment.inside:
                    direction = "out" if direction == "in" else "in"
                offer = (
                    None
                    if attachment.inside or declaration.offer is None
                    else self.reference(
                        attachment.scope, declaration.offer, owner=self.nodes[attachment.port].key
                    )
                )
                slots.append(
                    EndSlot(
                        attachment.node,
                        attachment.name,
                        direction,
                        attachment.publishes,
                        self.nodes[attachment.port].guard,
                        offer,
                    )
                )
            ends = self.ends_nodes[net]
            self.nodes[ends] = replace(self.nodes[ends], ends=tuple(slots))
        for index in self.folds:
            self.link_fold(index)

    def topology(self, scope: int, interpretation: Interpretation[Any, Any, Any]) -> int:
        identity = (scope, id(interpretation))
        if identity in self.topologies:
            return self.topologies[identity]
        draft = self.drafts[scope]
        key = _key(draft.name, "$topology." + interpretation.name)
        nodes: list[tuple[str, tuple[int, ...]]] = []
        nets: list[tuple[str, int, tuple[EndSlot, ...]]] = []
        ports: list[tuple[str, str, int | None]] = []
        obligations: list[int] = []

        def contribution(child: int, export: object) -> int | None:
            target = self.drafts[child].members.get(export) if export is not None else None
            if target is None:
                return None
            if self.nodes[target].kind != "view":
                raise DefinitionError(f"{key}: a contribution must be an exported view")
            return target

        for name, declaration in draft.effective.members.items():
            if isinstance(declaration, Net):
                net = draft.named_children[name]
                target = contribution(net, interpretation.net)
                if target is not None:
                    slots = self.nodes[self.ends_nodes[net]].ends if net in self.ends_nodes else ()
                    nets.append((name, target, tuple(replace(s, offer=None) for s in slots)))
                    obligations.append(target)
            elif isinstance(declaration, Subspace):
                target = contribution(draft.named_children[name], interpretation.node)
                if target is not None:
                    nodes.append((name, (target,)))
                    obligations.append(target)
            elif isinstance(declaration, SubspaceChoice):
                choice = self.choice_drafts[draft.choices[declaration]]
                candidates = tuple(
                    target
                    for _, case in choice.cases
                    if (target := contribution(case, interpretation.node)) is not None
                )
                if candidates:
                    nodes.append((name, candidates))
                    obligations.extend(candidates)
            elif isinstance(declaration, Port):
                port = draft.named_members[name]
                ports.append((name, declaration.direction, self.nodes[port].guard))
        index = self.reserve(
            scope,
            key,
            "topology",
            cast(ValueSemantics[object], TOPOLOGY),
            guard=draft.guard,
            source_owner=draft.name or None,
        )
        self.nodes[index] = replace(
            self.nodes[index],
            topology=TopologyPlan(tuple(nodes), tuple(nets), tuple(ports)),
        )
        self.contributions[index] = tuple(obligations)
        self.topologies[identity] = index
        return index

    def link_fold(self, index: int) -> None:
        node = self.nodes[index]
        draft = self.drafts[node.scope]
        declaration = cast(Fold[object], draft.effective.members[node.key.rsplit(".", 1)[-1]])
        interpretation = declaration.interpretation
        topology = self.topology(node.scope, interpretation)
        arguments = [Argument("topology", topology)]
        for name, supplied in declaration.arguments.items():
            if isinstance(supplied, ValueRef):
                target = self.reference(node.scope, supplied, owner=node.key)
            else:
                target = self.reserve(
                    node.scope,
                    f"{node.key}.$literal.{name}",
                    "const",
                    cast(ValueSemantics[object], default_semantics(type(supplied))),
                    source_owner=node.key,
                )
                self.nodes[target] = replace(self.nodes[target], value=supplied)
            arguments.append(Argument(name, target))
        output = self.reserve(
            node.scope,
            node.key + ".$output",
            "derived",
            interpretation.result,
            guard=node.guard,
            source_owner=node.key,
        )
        self.nodes[output] = replace(
            self.nodes[output], function=interpretation.reduce, arguments=tuple(arguments)
        )
        declared = self.obligations(
            node.scope,
            declaration.constraints,
            (Constraint, ConstraintGroup, View, AcceptedViewRef),
            owner=node.key,
        )
        self.nodes[index] = replace(
            node, output=output, constraints=(*declared, *self.contributions[topology])
        )

    def reference(self, scope: int, source: object, *, owner: str) -> int:
        if isinstance(source, Expr) and source.owner is None:
            return self.expression(scope, source, owner=owner)
        try:
            return resolve_reference(self.nodes, self.scopes, self.choices, scope, source)
        except RequestError as cause:
            raise DefinitionError(f"{owner}: {cause}") from cause

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
            number = self.expression_counts.get(owner, 0)
            self.expression_counts[owner] = number + 1
            index = self.reserve(
                scope,
                f"{owner}.$expr.{number}",
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
            index = self.reference(scope, source, owner=owner)
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

    def member_source(self, task: _MemberTask) -> tuple[Declaration, int] | None:
        """Apply a child supplier; return the declaration and its authoring scope."""
        scope = self.drafts[task.scope]
        node = self.nodes[task.index]
        declaration = task.declaration
        source_scope = scope.index
        if task.binding is not None:
            binding = task.binding
            source_scope = scope.source_scope if task.binding_scope is None else task.binding_scope
            if binding.kind == "literal":
                self.nodes[node.index] = replace(node, value=binding.supplier)
                return None
            if binding.kind == "reference":
                self.nodes[node.index] = replace(
                    node, output=self.reference(source_scope, binding.supplier, owner=node.key)
                )
                return None
            if not isinstance(binding.supplier, (Param, Decision)):
                raise DefinitionError(f"{node.key}: malformed exposed parameter binding")
            declaration = binding.supplier

        return declaration, source_scope

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
        source = self.member_source(task)
        if source is None:
            return
        declaration, source_scope = source
        scope = self.drafts[task.scope]
        node = self.nodes[task.index]
        condition = (
            getattr(declaration, "when", None)
            if task.binding is not None
            else scope.effective.guards.get(declaration)
        )
        guard = self.guarded(
            source_scope,
            scope.guard,
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
        elif isinstance(declaration, Port):
            output = self.port_outputs.get(node.index)
            if output is not None:
                node = replace(node, kind="alias", output=output)
        elif isinstance(declaration, (Carried, Ends)):
            table = self.carried if isinstance(declaration, Carried) else self.ends_nodes
            if task.scope not in table:
                raise DefinitionError(f"{node.key}: Carried and Ends belong to a placed Net")
            node = replace(node, output=table[task.scope])
        elif isinstance(declaration, Fold):
            self.folds.append(node.index)
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
                    declaration.constraints,
                    (Constraint, ConstraintGroup, View, AcceptedViewRef),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, (ScopedValueRef, AcceptedViewRef)):
            try:
                output = resolve_reference(
                    self.nodes, self.scopes, self.choices, scope.index, declaration, expand=True
                )
            except RequestError as cause:
                raise DefinitionError(f"{node.key}: {cause}") from cause
            node = replace(node, output=output)
        self.nodes[node.index] = node

    def link_choices(self) -> None:
        for choice in self.choice_drafts:
            for export, output in choice.exports.items():
                alternatives: list[tuple[str, int]] = []
                for case, scope in choice.cases:
                    try:
                        target = self.scopes[scope].members[export]
                    except KeyError as cause:
                        raise DefinitionError(
                            f"{choice.key}.{case}: missing choice export "
                            f"{cast(ValueKey[object], export).name}"
                        ) from cause
                    kind = self.nodes[target].kind
                    if isinstance(export, ViewKey) and kind != "view":
                        raise DefinitionError(
                            f"{choice.key}.{case}: accepted export must be a view"
                        )
                    if isinstance(export, ValueKey) and kind in {
                        "view",
                        "constraint",
                        "group",
                        "readiness",
                    }:
                        raise DefinitionError(
                            f"{choice.key}.{case}: value export has the wrong kind"
                        )
                    alternatives.append((case, target))
                self.nodes[output] = replace(
                    self.nodes[output], selector=choice.selector, alternatives=tuple(alternatives)
                )

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
            if node.kind in {"select", "unify"}:
                for _, target in node.alternatives:
                    semantics = self.nodes[target].semantics
                    if (
                        node.semantics is None
                        or semantics is None
                        or not node.semantics.is_compatible_with(semantics)
                    ):
                        raise DefinitionError(
                            f"{node.key}: choice export has incompatible value semantics"
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

    def build(self) -> LinkedModel:
        """Prepare templates, allocate occurrences, lower edges, then validate/freeze."""
        self.collect()
        self.allocate()
        self.apply_parameter_overrides()
        self.attach()
        for task in self.members:
            self.link_member(task)
        self.link_graph()
        for guard_task in self.guards:
            node = self.nodes[guard_task.index]
            self.nodes[guard_task.index] = replace(
                node,
                output=self.reference(
                    guard_task.source_scope, guard_task.condition, owner=node.owner
                ),
            )
        self.link_choices()
        self.link_expressions()
        order = dependency_order(
            tuple(node.dependencies for node in self.nodes),
            tuple(node.key for node in self.nodes),
        )
        self.check_semantics(order)
        self.check_expression_semantics()
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
        )


def link_space(space_type: type[Space]) -> LinkedModel:
    return _Linker(space_type).build()
