# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Check a lowered table's value semantics across scopes, in dependency order.

An alias, a view, a presence and a selection take their sources' semantics
when they declare none; then every output, alternative, guard, callback
argument, export and integer operand must agree with what reads it. A shared
``decision.member`` whose candidates disagree on its type is dropped when
nothing reads it.
"""

from __future__ import annotations

from dataclasses import replace

from ._signatures import validate_argument
from ._table import Table
from .errors import DefinitionError
from .semantics import BOOL, STRING


def check(table: Table, order: tuple[int, ...]) -> None:
    _check_nodes(table, order)
    _check_arguments(table)
    _check_exports(table)
    _check_expressions(table)


def _check_nodes(table: Table, order: tuple[int, ...]) -> None:
    nodes = table.nodes
    for index in order:
        node = nodes[index]
        if node.kind in {"view", "alias"} and node.output is not None:
            output = nodes[node.output].semantics
            if output is None:
                raise DefinitionError(f"{node.key}: output has no value semantics")
            if node.semantics is not None and not node.semantics.is_compatible_with(output):
                raise DefinitionError(f"{node.key}: output has incompatible value semantics")
            if node.semantics is None:
                nodes[index] = node = replace(node, semantics=output)
        if node.kind in {"present", "select"} and node.semantics is None and node.alternatives:
            # Like an alias, these take their sources' linked semantics.
            nodes[index] = node = replace(node, semantics=nodes[node.alternatives[0][1]].semantics)
        if node.kind in {"select", "present"}:
            for _, target in node.alternatives:
                semantics = nodes[target].semantics
                if (
                    node.semantics is None
                    or semantics is None
                    or not node.semantics.is_compatible_with(semantics)
                ):
                    if index in table.shared and _drop_shared(table, index):
                        break
                    raise DefinitionError(
                        f"{node.key}: alternatives have incompatible value semantics"
                    )
        if node.guard is not None:
            semantics = nodes[node.guard].semantics
            if semantics is None or not semantics.is_compatible_with(BOOL):
                raise DefinitionError(f"{node.key}: applicability requires Boolean value semantics")
        if node.kind == "guard" and node.output is not None:
            semantics = nodes[node.output].semantics
            if semantics is None or not semantics.is_compatible_with(BOOL):
                raise DefinitionError(f"{node.key}: applicability requires Boolean value semantics")


def _drop_shared(table: Table, index: int) -> bool:
    """An unreferenced shared member whose candidates disagree on its type."""
    choice, member = table.shared.pop(index)
    if any(index in node.dependencies for node in table.nodes):
        return False  # a declaration reads it: the disagreement is an error
    del table.choice_drafts[choice].members[member]
    table.nodes[index] = replace(table.nodes[index], alternatives=(), semantics=STRING)
    return True


def _check_arguments(table: Table) -> None:
    for dependency, target, owner in table.argument_checks:
        semantics = table.nodes[target].semantics
        if semantics is None:
            raise DefinitionError(f"{owner}: dependency has no value semantics")
        validate_argument(dependency, semantics, owner=owner)


def _check_exports(table: Table) -> None:
    for draft in table.drafts:
        exported = [
            (export, draft.members[declaration])
            for export, declaration in draft.effective.exports.items()
        ]
        exported += [
            (export, target)
            for export, entries in draft.input_exports.items()
            for _, target in entries
        ]
        for export, target in exported:
            semantics = table.nodes[target].semantics
            if semantics is None or not export.semantics.is_compatible_with(semantics):
                raise DefinitionError(
                    f"{draft.name}: export {export.name} has incompatible semantics"
                )


def _check_expressions(table: Table) -> None:
    """Integer operands, once linked aliases have their semantics."""
    for task in table.expression_tasks:
        node = table.nodes[task.index]
        operands = (table.nodes[arg.node].semantics for arg in node.arguments)
        if any(semantics is None or semantics.type_token is not int for semantics in operands):
            raise DefinitionError(
                f"{node.owner}: integer expression operands require int value semantics"
            )
