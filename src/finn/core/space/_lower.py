# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower each member of a template's table (``_table``) into its node's payload.

A supplied formal becomes its binding (a literal, or an alias of what it
references); a Space class's own member becomes its kind's payload: a formal's
requirement, a constant, a Decision's domain and its arguments, a callback with
its arguments, a view's output and its Space's validity, and ``Members`` entries. Guards
and integer expressions are linked after every member, references resolving
through ``_names``. What a task makes exists only where its own node does
(``Table.requires``): a candidate's binding, or a case's guard, of a Decision over nodes
that may never apply.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from ._bindings import PlacementBinding
from ._configuration import Space
from ._names import Names
from ._nodes import selector_value
from ._signatures import BoundFunction
from ._table import MemberTask, ScopeDraft, Table
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Const,
    Constraint,
    Decision,
    Declaration,
    Derived,
    MemberRef,
    Members,
    Param,
    Supplied,
    ValueRef,
    View,
    ViewKey,
)
from .domains import Domain
from .errors import DefinitionError
from .expressions import INTEGER_SEMANTICS, Expr, evaluator
from .ir import Argument, Node
from .semantics import ValueSemantics

_OWNER = "space.lower"


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
        raise DefinitionError.broken(
            "constant-snapshot-failed",
            _OWNER,
            f"{owner}: constant snapshot failed",
            declaration=owner,
        ) from cause


class _Lowering:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.names = Names(table)

    def run(self) -> None:
        requires = self.table.requires
        for task in self.table.members:
            self.table.applying = requires.get(task.index)
            self.link_member(task)
        for guard_task in self.table.guards:
            self.table.applying = requires.get(guard_task.index)
            node = self.table.nodes[guard_task.index]
            self.table.nodes[guard_task.index] = replace(
                node,
                output=self.names.reference(
                    guard_task.source_scope, guard_task.condition, owner=node.owner
                ),
            )
        for selector, source_scope, by in self.table.picks:
            self.table.applying = requires.get(selector)
            node = self.table.nodes[selector]
            target = self.names.reference(source_scope, by, owner=node.owner)
            self.table.nodes[selector] = replace(node, arguments=(Argument("value", target),))
        self.link_expressions()
        self.table.applying = None

    # -- expressions ----------------------------------------------------------------------

    def link_expressions(self) -> None:
        cursor = 0
        while cursor < len(self.table.expression_tasks):
            task = self.table.expression_tasks[cursor]
            cursor += 1
            self.table.applying = self.table.requires.get(task.index)
            if task.operator not in {"add", "sub", "mul", "floordiv", "mod", "neg"}:
                raise DefinitionError.broken(
                    "expression-operator-unsupported",
                    _OWNER,
                    f"{task.owner}: unsupported integer expression operator",
                    declaration=task.owner,
                )
            names = ("operand",) if task.operator == "neg" else ("left", "right")
            if len(task.operands) != len(names):
                raise DefinitionError.broken(
                    "expression-arity-invalid",
                    _OWNER,
                    f"{task.owner}: invalid integer expression arity",
                    declaration=task.owner,
                )
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
                    raise DefinitionError.broken(
                        "expression-literal-not-int",
                        _OWNER,
                        f"{task.owner}: integer expressions reject non-int literals",
                        declaration=task.owner,
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
            raise DefinitionError.broken(
                "domain-invalid",
                _OWNER,
                f"{owner}: invalid decision domain: {cause}",
                declaration=owner,
            ) from cause
        arguments: list[Argument] = []
        for name, given in domain.dependencies:
            source = selector_value(given)
            if not isinstance(source, ValueRef):
                raise DefinitionError.broken(
                    "domain-dependency-not-value",
                    _OWNER,
                    f"{owner}: domain dependency {name} must be a value reference",
                    declaration=owner,
                )
            arguments.append(Argument(name, self.names.reference(scope, source, owner=owner)))
        declared = {argument.name for argument in arguments}
        for role, function, named in domain.signatures():
            # Each callable takes the declared inputs its signature names.
            values = dict.fromkeys(named)
            positional: tuple[object, ...] = ()
            if role == "membership":
                values["candidate"] = None
            elif role == "order":
                positional = (None,)  # the values it ranks
            try:
                if not set(named) <= declared:
                    raise TypeError(f"names {', '.join(sorted(set(named) - declared))}")
                inspect.signature(function).bind(*positional, **values)
            except (TypeError, ValueError) as cause:
                raise DefinitionError.broken(
                    "domain-signature-incompatible",
                    _OWNER,
                    f"{owner}: incompatible domain {role} signature",
                    declaration=owner,
                ) from cause
        read = {name for _, _, named in domain.signatures() for name in named}
        if declared - read:
            raise DefinitionError.broken(
                "domain-signature-incompatible",
                _OWNER,
                f"{owner}: domain input {', '.join(sorted(declared - read))} is read by none "
                "of its callables",
                declaration=owner,
            )
        return domain, tuple(arguments)

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
        binding = task.binding
        # A supplier is read in the body that wrote it: the node's source scope.
        written = scope.source_scope
        changes: dict[str, Any]
        if binding is not None and binding.kind in {"literal", "reference"}:
            changes = self.supplied(task, node, binding, written)
        else:
            changes = self.declared(task, scope, node, written)
        self.table.nodes[node.index] = replace(node, **changes)

    def supplied(
        self, task: MemberTask, node: Node, binding: PlacementBinding, written: int
    ) -> dict[str, Any]:
        """A formal its declaring body supplied: its value, or what it references."""
        changes: dict[str, Any] = {}
        if binding.kind == "literal":
            changes["value"] = binding.supplier
        elif type(binding.supplier) is int and task.index in self.table.editable_aliases:
            changes["output"] = binding.supplier
        else:
            changes["output"] = self.names.reference(written, binding.supplier, owner=node.key)
        return changes

    def declared(
        self, task: MemberTask, scope: ScopeDraft, node: Node, written: int
    ) -> dict[str, Any]:
        """A Space class's own member, or a fresh Decision supplied for it: guard and payload."""
        declaration = task.declaration
        binding = task.binding
        source_scope = scope.index
        condition: ValueRef[bool] | None = scope.effective.guards.get(declaration)
        guard_scope = scope
        outer = scope.guard
        if binding is not None and binding.kind == "local-decision":
            declaration = cast(Decision[object], binding.supplier)
            source_scope = written
            condition = declaration.when
            if task.owner_scope is not None:
                guard_scope = self.table.drafts[task.owner_scope]
            outer = guard_scope.guard
        guard = self.table.guarded(
            source_scope,
            outer,
            condition,
            node.key + ".$guard",
            owner=node.key,
        )
        changes: dict[str, Any] = {"guard": guard}
        changes.update(self.payload(task, declaration, node, guard, source_scope))
        return changes

    def payload(
        self,
        task: MemberTask,
        declaration: Declaration,
        node: Node,
        guard: int | None,
        source_scope: int,
    ) -> dict[str, Any]:
        """What a declared member's node holds, by the kind of its declaration."""
        scope = self.table.drafts[task.scope]
        if isinstance(declaration, Param):
            return {"required": declaration.required}
        if isinstance(declaration, Const):
            assert node.semantics is not None
            return {"value": _constant(node.semantics, declaration, owner=node.key)}
        if isinstance(declaration, Decision):
            assert node.semantics is not None
            changes: dict[str, Any] = {}
            changes["domain"], changes["domain_arguments"] = self.domain(
                source_scope, declaration, node.semantics, owner=node.key
            )
            changes["required_choice"] = declaration.required
            return changes
        if isinstance(declaration, Expr):
            self.names.expression(scope.index, declaration, owner=node.key, index=node.index)
            return {}
        if isinstance(declaration, (Derived, Constraint)):
            return self.callback(node, scope.effective.functions[task.name])
        if isinstance(declaration, Supplied):
            return {"function": _supply(declaration.formal), "call_style": "self"}
        if isinstance(declaration, View):
            return {
                "output": self.view_output(node, guard, declaration, task.name),
                "constraints": (cast(int, self.table.drafts[node.scope].validity),),
            }
        if isinstance(declaration, Members):
            entries = self.names.members_candidates(
                scope.index, cast(ViewKey[object], declaration.key)
            )
            return {
                "alternatives": tuple((name, target) for name, _, target in entries),
                "value": tuple(member for _, member, _ in entries),
            }
        if isinstance(declaration, (MemberRef, ChoiceMemberRef, CaseRef)):
            anonymous = replace_owner(declaration)
            return {"output": self.names.reference(scope.index, anonymous, owner=node.key)}
        return {}


def replace_owner(reference: Declaration) -> Declaration:
    """A named reference member resolves as its anonymous structural twin."""
    if isinstance(reference, MemberRef):
        return MemberRef(reference.path, reference.member)
    if isinstance(reference, ChoiceMemberRef):
        return ChoiceMemberRef(reference.path, reference.member)
    if isinstance(reference, CaseRef):
        return CaseRef(reference.path)
    return reference
