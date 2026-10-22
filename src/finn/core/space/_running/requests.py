# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change's requests, normalized: every form ``with_choices`` takes (a ``Change`` made by
a bound accessor, a ``{reference: value}`` mapping, a keyword of the scope's own choices)
and every fact a change of facts names, as node ids, before anything is evaluated. The
questions (``questions.change``, ``questions.bind``) take node ids.

Each request is read against a design space read at one of its scopes (``surface.At``): the
receiver. A ``Change`` holds the design space it was made on (``Change.base``), and may be
applied to any design space equal to it: one made on a child's scope applies at the root
as well, as both read the same design space. A malformed or foreign item is a
``RequestError``.

A node is read for its own fields (kind, scope, key): its header, as the compiled model
states it (``Linked.header``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from .._failure import driver_only
from ..edits import Change, ChangeRequest
from ..errors import RequestError
from ..ir import NodeId

if TYPE_CHECKING:
    from .surface import At

T = TypeVar("T")


def normalize(current: At, changes: Sequence[Change[object]]) -> dict[NodeId, Change[object]]:
    """Each change by its choice: every one of this design space (made on a design space
    equal to it), naming an owned Decision of its scope, once. Its scope and node are asked
    of the model (``holds_scope``, ``holds_node``: an addressed scope, a node of one), never
    compared with a length. A choice has one identity, the one its handles give: an id
    standing for another node (an import, a formal's binding: ``_structure``) is no node,
    refused here and never followed into the node it stands for, so the duplicate check and
    every map keyed by a change's node see one id per choice."""
    linked = current.ds.model.linked
    pending: dict[NodeId, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound decision field")
        if item.base != current.ds:
            raise RequestError("all changes must target this exact base snapshot")
        if not linked.holds_scope(item.scope):
            raise RequestError("change scope does not belong to this model")
        if not linked.holds_node(item.node):
            raise RequestError("change node does not belong to this model")
        header = linked.header(item.node)
        if header.kind != "decision" or header.scope != item.scope:
            raise RequestError("change does not identify an owned decision in its scope")
        if item.node in pending:
            raise RequestError(f"duplicate change for {header.key}")
        pending[item.node] = item
    return pending


def change(current: At, reference: object, value: T) -> Change[T]:
    """A bound accessor's change: ``reference``'s Decision, read at ``current``'s scope, set
    to ``value``."""
    driver_only("change construction")
    model = current.ds.model
    index = model.decision(current.scope, reference)
    return Change(current.ds, model.linked.header(index).scope, index, value)


def clear(current: At, reference: object) -> Change[object]:
    """A bound accessor's clear of ``reference``'s Decision, read at ``current``'s scope."""
    driver_only("clear construction")
    model = current.ds.model
    index = model.decision(current.scope, reference)
    return Change(current.ds, model.linked.header(index).scope, index, remove=True)


def _keyword_change(current: At, name: str, value: object) -> Change[object]:
    """A keyword change: ``name``, a choice owned by ``current``'s scope, set to ``value``."""
    linked = current.ds.model.linked
    scope = linked.scopes[current.scope]
    index = scope.named_members.get(name)
    if index is None:
        raise RequestError(f"unknown direct choice {name!r}")
    header = linked.header(index)
    if header.kind != "decision" or header.scope != current.scope:
        raise RequestError(f"{header.key} is not a direct owned choice in this scope")
    return Change(current.ds, header.scope, index, value)


def _requests(
    current: At, items: tuple[ChangeRequest | Mapping[Any, object], ...]
) -> list[Change[object]]:
    """Expand ``{reference: value}`` mappings in order, each read at ``current``'s scope;
    Change objects pass through."""
    model = current.ds.model
    result: list[Change[object]] = []
    for item in items:
        if isinstance(item, Mapping):
            for reference, value in item.items():
                index = model.decision(current.scope, reference)
                result.append(Change(current.ds, model.linked.header(index).scope, index, value))
        else:
            result.append(item)
    return result


def _fact_nodes(current: At, facts: Mapping[Any, object]) -> dict[NodeId, object]:
    """``{formal: value}`` by node, each read at ``current``'s scope: each formal a root
    formal the model exposes."""
    model = current.ds.model
    linked = model.linked
    nodes: dict[NodeId, object] = {}
    for reference, value in facts.items():
        index = model.resolve(current.scope, reference)
        if index not in linked.parameters:
            raise RequestError(f"{linked.header(index).key} is not an exposed parameter")
        if index in nodes:
            raise RequestError(f"duplicate fact for {linked.header(index).key}")
        nodes[index] = value
    return nodes


__all__ = ["change", "clear", "normalize"]
