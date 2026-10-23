# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change's requests, normalized: every form ``with_choices`` takes (a ``Change`` made by
a bound accessor, a ``{reference: value}`` mapping, a keyword of the scope's own choices)
and every fact a change of facts names, as addresses, before anything is evaluated. The
questions (``questions.change``, ``questions.bind``) take addresses.

Each request is read against a design space read at one of its scopes (``surface.At``): the
receiver. A ``Change`` holds the design space it was made on (``Change.base``), and may be
applied to any design space equal to it: one made on a child's scope applies at the root
as well, as both read the same design space. A malformed or foreign item is a
``RequestError``.

An address is read for its own fields (kind, scope, key): its member record, as its scope
reads it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from .._failure import driver_only
from ..edits import Change, ChangeRequest
from ..errors import RequestError

if TYPE_CHECKING:
    from .._scopes import Address
    from .surface import At

T = TypeVar("T")


def normalize(current: At, changes: Sequence[Change[object]]) -> dict[Address, Change[object]]:
    """Each change by its choice: every one of this design space (made on a design space
    equal to it), naming an owned Decision of its scope, once. Its scope and address are
    asked of the model (``holds``, ``holds_address``: a scope record of it, a canonical
    address of one). A choice has one address, the one its handles give: an address
    standing for another (a formal's binding, a selection made elsewhere: ``_scopes``) is
    none, refused here and never followed into the address it stands for, so the duplicate
    check and every map keyed by a change's address see one address per choice."""
    scopes = current.ds.model.scopes
    pending: dict[Address, Change[object]] = {}
    for item in changes:
        if not isinstance(item, Change):
            raise RequestError("changes must come from a bound decision field")
        if item.base != current.ds:
            raise RequestError("all changes must target this exact base snapshot")
        if not scopes.holds(item.scope):
            raise RequestError("change scope does not belong to this model")
        if not scopes.holds_address(item.node):
            raise RequestError("change node does not belong to this model")
        node = item.node
        if node.kind != "decision" or node.scope is not item.scope:
            raise RequestError("change does not identify an owned decision in its scope")
        if node in pending:
            raise RequestError(f"duplicate change for {node.key}")
        pending[node] = item
    return pending


def change(current: At, reference: object, value: T) -> Change[T]:
    """A bound accessor's change: ``reference``'s Decision, read at ``current``'s scope, set
    to ``value``."""
    driver_only("change construction")
    index = current.ds.model.decision(current.scope, reference)
    return Change(current.ds, index.scope, index, value)


def clear(current: At, reference: object) -> Change[object]:
    """A bound accessor's clear of ``reference``'s Decision, read at ``current``'s scope."""
    driver_only("clear construction")
    index = current.ds.model.decision(current.scope, reference)
    return Change(current.ds, index.scope, index, remove=True)


def _keyword_change(current: At, name: str, value: object) -> Change[object]:
    """A keyword change: ``name``, a choice owned by ``current``'s scope, set to ``value``."""
    index = current.scope.named_members.get(name)
    if index is None:
        raise RequestError(f"unknown direct choice {name!r}")
    if index.kind != "decision" or index.scope is not current.scope:
        raise RequestError(f"{index.key} is not a direct owned choice in this scope")
    return Change(current.ds, index.scope, index, value)


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
                result.append(Change(current.ds, index.scope, index, value))
        else:
            result.append(item)
    return result


def _fact_nodes(current: At, facts: Mapping[Any, object]) -> dict[Address, object]:
    """``{formal: value}`` by address, each read at ``current``'s scope: each formal a root
    formal the model exposes."""
    model = current.ds.model
    exposed = model.scopes.parameters
    nodes: dict[Address, object] = {}
    for reference, value in facts.items():
        index = model.resolve(current.scope, reference)
        if index not in exposed:
            raise RequestError(f"{index.key} is not an exposed parameter")
        if index in nodes:
            raise RequestError(f"duplicate fact for {index.key}")
        nodes[index] = value
    return nodes


__all__ = ["change", "clear", "normalize"]
