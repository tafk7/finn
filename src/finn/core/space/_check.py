# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Check a template's lowered table's value semantics, in dependency order.

An alias, a view, a presence and a selection take their sources' semantics
when they declare none; then every output, alternative, guard, callback
argument, export and integer operand must agree with what reads it. A shared
``decision.member`` whose candidates disagree on its type is dropped when
nothing reads it. An import has its target's checked semantics, and a child's exports
are checked in its own template.
"""

from __future__ import annotations

from dataclasses import replace

from ._signatures import validate_argument
from ._table import Table
from .errors import DefinitionError
from .semantics import BOOL, STRING

_OWNER = "space.check"


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
                raise DefinitionError.broken(
                    "output-semantics-missing",
                    _OWNER,
                    f"{node.key}: output has no value semantics",
                    declaration=node.key,
                )
            if node.semantics is not None and not node.semantics.is_compatible_with(output):
                raise DefinitionError.broken(
                    "output-semantics-incompatible",
                    _OWNER,
                    f"{node.key}: output has incompatible value semantics",
                    declaration=node.key,
                )
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
                    raise DefinitionError.broken(
                        "alternatives-semantics-incompatible",
                        _OWNER,
                        f"{node.key}: alternatives have incompatible value semantics",
                        declaration=node.key,
                    )
        if node.guard is not None:
            semantics = nodes[node.guard].semantics
            if semantics is None or not semantics.is_compatible_with(BOOL):
                raise DefinitionError.broken(
                    "applicability-not-boolean",
                    _OWNER,
                    f"{node.key}: applicability requires Boolean value semantics",
                    declaration=node.key,
                )
        if node.kind == "guard" and node.output is not None:
            semantics = nodes[node.output].semantics
            if semantics is None or not semantics.is_compatible_with(BOOL):
                raise DefinitionError.broken(
                    "applicability-not-boolean",
                    _OWNER,
                    f"{node.key}: applicability requires Boolean value semantics",
                    declaration=node.key,
                )


def _drop_shared(table: Table, index: int) -> bool:
    """An unreferenced shared member whose candidates disagree on its type."""
    choice, member = table.shared.pop(index)
    if any(index in node.dependencies for node in table.nodes):
        return False  # a declaration reads it: the disagreement is an error
    del table.choice_drafts[choice].members[member]
    table.choice_drafts[choice].dropped.add(member)
    table.nodes[index] = replace(table.nodes[index], alternatives=(), semantics=STRING)
    return True


def _check_arguments(table: Table) -> None:
    for dependency, target, owner in table.argument_checks:
        semantics = table.nodes[target].semantics
        if semantics is None:
            raise DefinitionError.broken(
                "dependency-semantics-missing",
                _OWNER,
                f"{owner}: dependency has no value semantics",
                declaration=owner,
            )
        validate_argument(dependency, semantics, owner=owner)


def _check_exports(table: Table) -> None:
    for draft in table.drafts:
        if draft.stub is not None:
            continue  # a child's exports are checked in its template
        exported = [
            (export, draft.members[declaration])
            for export, declaration in draft.effective.exports.items()
        ]
        for export, target in exported:
            semantics = table.nodes[target].semantics
            if semantics is None or not export.semantics.is_compatible_with(semantics):
                raise DefinitionError.broken(
                    "export-semantics-incompatible",
                    _OWNER,
                    f"{draft.name}: export {export.name} has incompatible semantics",
                    declaration=draft.name,
                )


def _check_expressions(table: Table) -> None:
    """Integer operands, once linked aliases have their semantics."""
    for task in table.expression_tasks:
        node = table.nodes[task.index]
        operands = (table.nodes[arg.node].semantics for arg in node.arguments)
        if any(semantics is None or semantics.type_token is not int for semantics in operands):
            raise DefinitionError.broken(
                "expression-operand-not-int",
                _OWNER,
                f"{node.owner}: integer expression operands require int value semantics",
                declaration=node.owner,
            )
