# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower each allocated member into its node's payload.

A supplied member becomes its binding (a literal, an alias of what it
references, a pin checked against the declared domain); a family's own member
becomes its kind's payload: a formal's requirement, a constant, a Decision's
domain and its arguments, a callback with its arguments, a view's output and
obligations, ``Members``, ``Users`` and ``Present`` alternatives. Guards and
integer expressions are linked after every member, references resolving
through ``_names``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, cast

from ._configuration import Space
from ._names import Names
from ._signatures import BoundFunction
from ._table import MemberTask, ScopeDraft, Table
from .declarations import (
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
    Supplied,
    Users,
    ValueRef,
    View,
    ViewKey,
)
from .domains import Domain, requirement_argument
from .errors import DefinitionError
from .expressions import INTEGER_SEMANTICS, Expr, evaluator
from .ir import Argument, Node
from .semantics import ValueSemantics


def lower(table: Table) -> None:
    """Lower every member, then link every guard's condition and every expression."""
    _Lowering(table).run()


def _supply(formal: Param[object]) -> Callable[..., object]:
    def supplied(point: Space) -> bool:
        return point.present(formal)

    return supplied


def _constant(semantics: ValueSemantics[object], constant: Const[object], *, owner: str) -> object:
    """The model's own snapshot of a constant, detached from its declaration.

    ``Const`` snapshots its value when declared; a compiled model takes another,
    so a later change to the declaration's value never reaches it.
    """
    try:
        return semantics.freeze(constant.value)
    except Exception as cause:
        raise DefinitionError(f"{owner}: constant snapshot failed") from cause


class _Lowering:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.names = Names(table)

    def run(self) -> None:
        for task in self.table.members:
            self.link_member(task)
        for guard_task in self.table.guards:
            node = self.table.nodes[guard_task.index]
            self.table.nodes[guard_task.index] = replace(
                node,
                output=self.names.reference(
                    guard_task.source_scope, guard_task.condition, owner=node.owner
                ),
            )
        self.link_expressions()

    # -- expressions ----------------------------------------------------------------------

    def link_expressions(self) -> None:
        cursor = 0
        while cursor < len(self.table.expression_tasks):
            task = self.table.expression_tasks[cursor]
            cursor += 1
            if task.operator not in {"add", "sub", "mul", "floordiv", "mod", "neg"}:
                raise DefinitionError(f"{task.owner}: unsupported integer expression operator")
            names = ("operand",) if task.operator == "neg" else ("left", "right")
            if len(task.operands) != len(names):
                raise DefinitionError(f"{task.owner}: invalid integer expression arity")
            arguments: list[Argument] = []
            for name, operand in zip(names, task.operands):
                if type(operand) is int:
                    literal = self.table.reserve(
                        task.scope,
                        f"{self.table.nodes[task.index].key}.$literal.{name}",
                        "const",
                        cast(ValueSemantics[object], INTEGER_SEMANTICS),
                        source_owner=task.owner,
                    )
                    self.table.nodes[literal] = replace(self.table.nodes[literal], value=operand)
                    target = literal
                elif isinstance(operand, Const) and operand.owner is None:
                    semantics = cast(ValueSemantics[object], operand.semantics)
                    target = self.table.reserve(
                        task.scope,
                        f"{self.table.nodes[task.index].key}.$literal.{name}",
                        "const",
                        semantics,
                        source_owner=task.owner,
                    )
                    self.table.nodes[target] = replace(
                        self.table.nodes[target],
                        value=_constant(semantics, operand, owner=task.owner),
                    )
                elif isinstance(operand, ValueRef):
                    target = self.names.reference(task.scope, operand, owner=task.owner)
                else:
                    raise DefinitionError(
                        f"{task.owner}: integer expressions reject non-int literals"
                    )
                arguments.append(Argument(name, target))
            self.table.nodes[task.index] = replace(
                self.table.nodes[task.index],
                arguments=tuple(arguments),
                function=evaluator(task.operator),
            )

    # -- members ------------------------------------------------------------------------

    def arguments(self, scope: int, function: BoundFunction, *, owner: str) -> tuple[Argument, ...]:
        arguments: list[Argument] = []
        for dependency in function.dependencies:
            target = self.names.reference(scope, dependency.source, owner=owner)
            arguments.append(Argument(dependency.name, target))
            self.table.argument_checks.append((dependency, target, owner))
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
                    dict.fromkeys(target for *_, target in self.names.users_candidates(scope, key))
                )
            elif isinstance(source, Members):
                # A member family obliges each member's acceptance separately.
                indices = [
                    target
                    for *_, target in self.names.members_candidates(
                        scope, cast(ViewKey[object], source.key)
                    )
                ]
            else:
                indices = [self.names.reference(scope, source, owner=owner)]
                if self.table.nodes[indices[0]].kind not in {"view", "constraint", "group"}:
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
            arguments.append(Argument(name, self.names.reference(scope, source, owner=owner)))
        supplied = dict.fromkeys(argument.name for argument in arguments)
        for position, requirement in enumerate(domain.requirements):
            if not isinstance(requirement.fact, ValueRef):
                raise DefinitionError(
                    f"{owner}: requirement {requirement.code}'s fact must be a value reference"
                )
            arguments.append(
                Argument(
                    requirement_argument(position),
                    self.names.reference(scope, requirement.fact, owner=owner),
                )
            )
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

    def contract(
        self, scope: ScopeDraft, decision: Decision[object], node: Node
    ) -> tuple[Domain[object], tuple[Argument, ...]]:
        """The declared domain of an overridden Decision, read in its family's body."""
        semantics = scope.effective.semantics.get(decision, node.semantics)
        if semantics is None:
            raise DefinitionError(f"{node.key}: the declared Decision has no value semantics")
        return self.domain(scope.index, decision, semantics, owner=node.key)

    def callback(self, node: Node, function: BoundFunction) -> dict[str, Any]:
        """A callback's payload: its function, call style and resolved arguments."""
        return {
            "function": function.function,
            "call_style": function.call_style,
            "arguments": self.arguments(node.scope, function, owner=node.owner),
        }

    def view_output(
        self, node: Node, guard: int | None, declaration: View[object], name: str
    ) -> int:
        """Both view forms lower to one accepted node with an explicit raw output."""
        if declaration.source is not None:
            return self.names.reference(node.scope, declaration.source, owner=node.key)
        function = self.table.drafts[node.scope].effective.functions[name]
        output = self.table.reserve(
            node.scope,
            node.key + ".$output",
            "derived",
            function.semantics,
            guard=guard,
            source_owner=node.key,
        )
        nodes = self.table.nodes
        nodes[output] = replace(nodes[output], **self.callback(nodes[output], function))
        return output

    def link_member(self, task: MemberTask) -> None:
        """Lower one member: every field its node gains, replaced in one step."""
        scope = self.table.drafts[task.scope]
        node = self.table.nodes[task.index]
        declaration = task.declaration
        binding = task.binding
        changes: dict[str, Any] = {}
        # A supplier is read in the body that wrote it: the node's own source
        # scope, or the enclosing node's for a binding assigned through a path.
        written = (
            scope.source_scope
            if binding is None or binding.source_scope is None
            else binding.source_scope
        )
        if binding is not None and binding.kind in {"literal", "reference", "pin", "pin-reference"}:
            if binding.kind in {"literal", "pin"}:
                changes["value"] = binding.supplier
            elif type(binding.supplier) is int and task.index in self.table.editable_aliases:
                changes["output"] = binding.supplier
            else:
                located = isinstance(declaration, LocatedParam)
                changes["output"] = self.names.supply(
                    written, binding.supplier, owner=node.key, locate=located
                )
            if binding.contract is not None:
                # A pinned coordinate keeps its declared guard and domain: the
                # family's contract checks whatever an enclosing body supplies.
                changes["guard"] = self.table.guarded(
                    scope.index,
                    scope.guard,
                    scope.effective.guards.get(binding.contract),
                    node.key + ".$guard",
                    owner=node.key,
                )
                changes["domain"], changes["domain_arguments"] = self.contract(
                    scope, cast(Decision[object], binding.contract), node
                )
                changes["note"] = (
                    binding.provenance.text() if binding.provenance is not None else None
                )
            self.table.nodes[node.index] = replace(node, **changes)
            return
        source_scope = scope.index
        condition: ValueRef[bool] | None = scope.effective.guards.get(declaration)
        guard_scope = scope
        outer = scope.guard
        replaced: Decision[object] | None = None
        if binding is not None and binding.kind == "local-decision":
            replaced = cast("Decision[object] | None", binding.contract)
            declaration = cast(Decision[object], binding.supplier)
            source_scope = written
            condition = declaration.when
            if task.owner_scope is not None:
                guard_scope = self.table.drafts[task.owner_scope]
            outer = guard_scope.guard
            if replaced is not None:
                outer = self.table.guarded(
                    scope.index,
                    outer,
                    scope.effective.guards.get(replaced),
                    node.key + ".$contract",
                    owner=node.key,
                )
        if binding is not None and binding.kind == "parameter":
            self.table.nodes[node.index] = replace(node, required=False)
            return
        guard = self.table.guarded(
            source_scope,
            outer,
            condition,
            node.key + ".$guard",
            owner=node.key,
        )
        changes["guard"] = guard
        if isinstance(declaration, Param):
            changes["required"] = declaration.required
        elif isinstance(declaration, Const):
            assert node.semantics is not None
            changes["value"] = _constant(node.semantics, declaration, owner=node.key)
        elif isinstance(declaration, Decision):
            assert node.semantics is not None
            changes["domain"], changes["domain_arguments"] = self.domain(
                source_scope, declaration, node.semantics, owner=node.key
            )
            if replaced is not None:
                changes["contract"], changes["contract_arguments"] = self.contract(
                    scope, replaced, node
                )
                changes["note"] = (
                    binding.provenance.text()
                    if binding is not None and binding.provenance is not None
                    else None
                )
        elif isinstance(declaration, Expr):
            self.names.expression(scope.index, declaration, owner=node.key, index=node.index)
        elif isinstance(declaration, (Derived, Constraint)):
            changes.update(self.callback(node, scope.effective.functions[task.name]))
        elif isinstance(declaration, Supplied):
            changes["function"] = _supply(declaration.formal)
            changes["call_style"] = "self"
        elif isinstance(declaration, ConstraintGroup):
            changes["constraints"] = self.obligations(
                scope.index,
                declaration.constraints,
                (Constraint,),
                owner=node.key,
            )
        elif isinstance(declaration, View):
            changes["output"] = self.view_output(node, guard, declaration, task.name)
            changes["constraints"] = self.obligations(
                scope.index,
                declaration.requires,
                (Constraint, ConstraintGroup, View, ValueRef, Members),
                owner=node.key,
            )
        elif isinstance(declaration, Users):
            entries = self.names.users_candidates(
                scope.index, cast(ViewKey[object], declaration.key)
            )
            changes["alternatives"] = tuple((name, target) for name, _, target in entries)
            changes["value"] = tuple(member for _, member, _ in entries)
        elif isinstance(declaration, Present):
            changes["alternatives"] = tuple(
                (
                    f"{node.key}.{position}",
                    self.names.reference(scope.index, item, owner=node.key),
                )
                for position, item in enumerate(declaration.sources)
            )
        elif isinstance(declaration, Members):
            entries = self.names.members_candidates(
                scope.index, cast(ViewKey[object], declaration.key)
            )
            changes["alternatives"] = tuple((name, target) for name, _, target in entries)
            changes["value"] = tuple(member for _, member, _ in entries)
        elif isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
            anonymous = replace_owner(declaration)
            changes["output"] = self.names.reference(scope.index, anonymous, owner=node.key)
        self.table.nodes[node.index] = replace(node, **changes)


def replace_owner(reference: Declaration) -> Declaration:
    """A named reference member resolves as its anonymous structural twin."""
    if isinstance(reference, MemberRef):
        return MemberRef(reference.path, reference.member)
    if isinstance(reference, ChoiceMemberRef):
        return ChoiceMemberRef(reference.path, reference.member)
    if isinstance(reference, CaseRef):
        return CaseRef(reference.path)
    return reference
