# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Iterative occurrence allocation and linking into one owned node table."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import inspect
from typing import cast

from .collection import (
    BoundArgument,
    BoundFunction,
    EffectiveSpace,
    PlacementBinding,
    collect_placement,
    collect_space,
    validate_argument,
)
from .declarations import (
    AcceptedViewRef,
    Const,
    Constraint,
    ConstraintGroup,
    Declaration,
    Decision,
    DecisionRef,
    Derived,
    Param,
    Readiness,
    ScopedValueRef,
    Space,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
)
from .domains import Domain, finite
from .errors import DefinitionError, RequestError
from .expressions import Expr, INTEGER_SEMANTICS, IntOperator, apply_integer, evaluator
from .ir import Argument, Choice, LinkedModel, Node, NodeKind, Scope
from .semantics import ValueSemantics, default_semantics

_BOOL = default_semantics(bool)
_STRING = default_semantics(str)


def resolve_reference(
    nodes: Sequence[Node],
    scopes: Sequence[Scope],
    choices: Sequence[Choice],
    scope: int,
    reference: object,
    *,
    expand: bool = False,
) -> int:
    """Interpret fresh typed paths using frozen occurrence maps, never classes."""

    if type(scope) is not int or not 0 <= scope < len(scopes):
        raise RequestError("reference scope does not belong to this model")
    seen: set[tuple[int, int]] = set()
    require_view = False
    decision_scope: int | None = None
    original = reference
    while True:
        if not expand:
            try:
                index = scopes[scope].members[reference]
                break
            except (KeyError, TypeError):
                pass
        expand = False
        if not isinstance(reference, (ScopedValueRef, AcceptedViewRef)):
            raise RequestError("reference is not a member of this compiled scope")
        step = (scope, id(reference))
        if step in seen:
            raise RequestError("cyclic scoped reference")
        seen.add(step)
        placement = reference.placement
        if isinstance(reference, AcceptedViewRef):
            require_view = True
        child = scopes[scope].children.get(placement)
        if child is not None:
            scope = child
            if isinstance(reference, DecisionRef):
                decision_scope = scope
            reference = reference.member
            continue
        choice_index = scopes[scope].choices.get(placement)
        if choice_index is not None:
            if isinstance(reference, DecisionRef):
                raise RequestError("a DecisionRef must name a concrete locally owned decision")
            member = reference.member
            if (require_view and not isinstance(member, ViewKey)) or (
                not require_view and not isinstance(member, ValueKey)
            ):
                raise RequestError("choice reference has the wrong export kind")
            try:
                return choices[choice_index].exports[member]
            except KeyError as cause:
                raise RequestError("reference is not an export of this compiled choice") from cause
        raise RequestError("placement is not a child or choice of this compiled scope")
    node = nodes[index]
    # Named editable handles keep the frozen alias target, too. A fresh handle
    # is checked while descending; declared aliases do not reread its fields.
    if isinstance(original, DecisionRef):
        seen_aliases: set[int] = set()
        while node.kind == "alias" and node.output is not None:
            if index in seen_aliases:
                raise RequestError("cyclic decision alias")
            seen_aliases.add(index)
            index = node.output
            node = nodes[index]
        if node.kind != "decision":
            raise RequestError("a Param alias is not a locally owned Decision")
    if require_view and node.kind != "view":
        raise RequestError("an accepted reference must name a view")
    if decision_scope is not None and (node.kind != "decision" or node.scope != scope):
        raise RequestError("a Param alias is not a locally owned Decision")
    return index


@dataclass
class _ScopeDraft:
    index: int
    parent: int | None
    name: str
    effective: EffectiveSpace
    guard: int | None
    source_scope: int
    bindings: Mapping[str, PlacementBinding]
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
    declaration: SubspaceChoice
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


def _key(scope: str, member: str) -> str:
    return f"{scope}.{member}" if scope else member


def _matches(expected: str) -> Callable[..., object]:
    def matches(*, selected: str) -> bool:
        return selected == expected

    return matches


class _Linker:
    def __init__(
        self,
        space_type: type[Space],
        validate: Callable[[tuple[Node, ...]], tuple[int, ...]],
    ) -> None:
        self.space_type = space_type
        self.validate = validate
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
        self.expression_operators: dict[int, IntOperator] = {}
        self.expression_counts: dict[str, int] = {}
        self.scopes: tuple[Scope, ...] = ()
        self.choices: tuple[Choice, ...] = ()

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
            effective = collect_space(space_type)
            self.effective[space_type] = effective
            aliases: dict[str, list[Declaration]] = {}
            for declaration, name in effective.aliases.items():
                aliases.setdefault(name, []).append(declaration)
            self.aliases[space_type] = aliases
            descendants: list[type[Space]] = []
            for record in effective.members.values():
                declaration = record.declaration
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
            Node(
                index,
                0,
                space_type.__qualname__,
                "group",
                requires=tuple(indices[child] for child in children[space_type]),
            )
            for space_type, index in indices.items()
        )
        try:
            self.validate(structure)
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
    ) -> int:
        index = len(self.drafts)
        plan = collect_placement(placement) if placement is not None else None
        bindings = plan.bindings if plan is not None else {}
        draft = _ScopeDraft(
            index,
            parent,
            name,
            self.effective[space_type],
            guard,
            0 if parent is None else parent,
            bindings,
        )
        self.drafts.append(draft)
        if plan is not None:
            self.parameter_overrides.extend(
                _ParameterOverride(index, draft.source_scope, item.reference, item.binding)
                for item in plan.nested_bindings
            )
        for member_name, record in draft.effective.members.items():
            declaration = record.declaration
            if isinstance(declaration, (Subspace, SubspaceChoice)):
                continue
            binding = bindings.get(member_name)
            kind: NodeKind
            if isinstance(declaration, Param):
                kind = (
                    "param"
                    if binding is None or binding.kind == "exposed-param"
                    else "decision"
                    if binding.kind == "local-decision"
                    else "const"
                    if binding.kind == "literal"
                    else "alias"
                )
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
            elif isinstance(declaration, Readiness):
                kind = "readiness"
            elif isinstance(declaration, ConstraintGroup):
                kind = "group"
            elif isinstance(declaration, (ScopedValueRef, AcceptedViewRef)):
                kind = "alias"
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
        choice = _ChoiceDraft(
            len(self.choice_drafts), scope.index, key, declaration, guard, selector
        )
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
                placement.space_type, scope.index, case_key, case_guard, placement=placement
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
            for name, record in scope.effective.members.items():
                declaration = record.declaration
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
                        declaration.space_type, scope.index, key, guard, placement=declaration
                    )
                    scope.named_children[name] = child
                    for alias in self.aliases[scope.effective.space_type][name]:
                        scope.children[alias] = child
                elif isinstance(declaration, SubspaceChoice):
                    self.choice(scope, name, declaration)
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
            kind = cast(
                NodeKind,
                {
                    "literal": "const",
                    "reference": "alias",
                    "exposed-param": "param",
                    "local-decision": "decision",
                }[override.binding.kind],
            )
            self.nodes[target] = replace(node, kind=kind)

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
            self.expression_operators[task.index] = task.operator

    def fold_expressions(self, order: tuple[int, ...]) -> None:
        """Fold successful builtin arithmetic on unguarded definition constants."""

        for index in order:
            operator = self.expression_operators.get(index)
            if operator is None:
                continue
            node = self.nodes[index]
            operands = tuple(self.nodes[argument.node] for argument in node.arguments)
            if any(
                operand.semantics is None or operand.semantics.type_token is not int
                for operand in operands
            ):
                raise DefinitionError(
                    f"{node.owner}: integer expression operands require int value semantics"
                )
            if not all(
                operand.kind == "const" and operand.guard is None and type(operand.value) is int
                for operand in operands
            ):
                continue
            try:
                value = apply_integer(
                    operator, tuple(cast(int, operand.value) for operand in operands)
                )
            except ArithmeticError:
                # Invalid arithmetic is a contextual runtime error only if its
                # body is demanded. Inactive guarded expressions remain safe.
                continue
            assert node.semantics is not None
            self.nodes[index] = replace(
                node,
                kind="const",
                value=node.semantics.freeze(value),
                function=None,
            )

    def arguments(self, scope: int, function: BoundFunction, *, owner: str) -> tuple[Argument, ...]:
        arguments: list[Argument] = []
        for dependency in function.dependencies:
            target = self.reference(scope, dependency.source, owner=owner)
            arguments.append(Argument(dependency.name, target, dependency.mode))
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

    def link_member(self, task: _MemberTask) -> None:
        scope = self.drafts[task.scope]
        node = self.nodes[task.index]
        declaration = task.declaration
        source_scope = scope.index
        if task.binding is not None:
            binding = task.binding
            source_scope = scope.source_scope if task.binding_scope is None else task.binding_scope
            if binding.kind == "literal":
                self.nodes[node.index] = replace(node, value=binding.supplier)
                return
            if binding.kind == "reference":
                self.nodes[node.index] = replace(
                    node, output=self.reference(source_scope, binding.supplier, owner=node.key)
                )
                return
            if not isinstance(binding.supplier, (Param, Decision)):
                raise DefinitionError(f"{node.key}: malformed exposed parameter binding")
            declaration = binding.supplier

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
            function = scope.effective.functions[task.name]
            node = replace(
                node,
                function=function.function,
                arguments=self.arguments(scope.index, function, owner=node.key),
            )
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
        elif isinstance(declaration, Readiness):
            node = replace(
                node,
                requires=self.obligations(
                    scope.index,
                    declaration.requires,
                    (ValueRef, Constraint, ConstraintGroup),
                    owner=node.key,
                ),
            )
        elif isinstance(declaration, View):
            if declaration.source is not None:
                output = self.reference(scope.index, declaration.source, owner=node.key)
            else:
                function = scope.effective.functions[task.name]
                output = self.reserve(
                    scope.index,
                    node.key + ".$output",
                    "derived",
                    function.semantics,
                    guard=guard,
                    source_owner=node.key,
                )
                self.nodes[output] = replace(
                    self.nodes[output],
                    function=function.function,
                    arguments=self.arguments(scope.index, function, owner=node.key),
                )
            node = replace(
                node,
                output=output,
                constraints=self.obligations(
                    scope.index,
                    declaration.constraints,
                    (Constraint, ConstraintGroup),
                    owner=node.key,
                ),
                requires=self.obligations(
                    scope.index,
                    declaration.requires,
                    (ValueRef, Constraint, ConstraintGroup, Readiness),
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
            if node.kind == "select":
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
        self.collect()
        self.allocate()
        self.apply_parameter_overrides()
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
        self.link_choices()
        self.link_expressions()
        order = self.validate(tuple(self.nodes))
        self.check_semantics(order)
        self.fold_expressions(order)
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


def link_space(
    space_type: type[Space],
    validate: Callable[[tuple[Node, ...]], tuple[int, ...]],
) -> LinkedModel:
    return _Linker(space_type, validate).build()
