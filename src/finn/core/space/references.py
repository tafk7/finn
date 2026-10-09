# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable typed references to nodes of one compiled Space model, and resolution.

Use the factories in :mod:`inspection` to bind authored references or discover
handles. Handles retain compilation, never a configuration, assignments or caches.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from ._nodes import NodeChoice, unwrap
from .declarations import (
    CaseRef,
    ChoiceMemberRef,
    Decision,
    Declaration,
    MemberRef,
    ValueRef,
)
from .errors import RequestError
from .ir import Choice, LinkedModel, Node, Provenance, Scope
from .semantics import ValueSemantics

T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ValueHandle(ValueRef[T], Generic[T]):
    """A value reference whose interpretation belongs to an exact compilation."""

    _record_origin = False
    _linked: LinkedModel = field(repr=False)
    _node: int = field(repr=False)
    semantics: ValueSemantics[T] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self._node) is not int or not 0 <= self._node < len(self._linked.nodes):
            raise RequestError("handle does not identify a compiled node")
        object.__setattr__(
            self,
            "semantics",
            cast(ValueSemantics[T] | None, self._linked.nodes[self._node].semantics),
        )

    def _resolve(self, linked: LinkedModel) -> int:
        if self._linked is not linked:
            raise RequestError("handle belongs to a different compiled model")
        return self._node

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ValueHandle)
            and self._linked is other._linked
            and self._node == other._node
        )

    def __hash__(self) -> int:
        return hash((id(self._linked), self._node))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._linked.nodes[self._node].key!r})"


@dataclass(frozen=True, eq=False, repr=False)
class DecisionHandle(ValueHandle[T], Generic[T]):
    """An editable handle to an owning Decision, including a structural one."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._linked.nodes[self._node].kind != "decision":
            raise RequestError("a decision handle requires an owning Decision")


def decision_key(linked: LinkedModel, index: int) -> str:
    """A decision's stable key is its dotted declaration path."""
    return linked.nodes[index].key


def descend(scopes: Sequence[Scope], scope: int, path: Sequence[Declaration]) -> int:
    for record in path:
        current = scopes[scope]
        if record is current.record:
            continue  # the node this scope instantiates
        child = current.children.get(record, current.references.get(record))
        if child is None:
            name = record.name if record.name is not None else type(record).__name__
            raise RequestError(
                f"node {name} is not placed in {current.name or '<root>'} of this compiled model"
            )
        scope = child
    return scope


def _choice(
    scopes: Sequence[Scope], choices: Sequence[Choice], scope: int, path: Sequence[Declaration]
) -> Choice:
    scope = descend(scopes, scope, path[:-1])
    index = scopes[scope].choices.get(path[-1])
    if index is None:
        raise RequestError("the Decision over nodes is not part of this compiled scope")
    return choices[index]


def resolve_reference(
    nodes: Sequence[Node],
    scopes: Sequence[Scope],
    choices: Sequence[Choice],
    scope: int,
    reference: object,
    *,
    candidate: bool = False,
) -> int:
    """Interpret a member, a symbolic reference or a choice face, never through classes.

    ``decision.member`` resolves to the selection linked for it, or else to the
    one candidate that has the member; ``candidate=True`` (for edits) always
    names that candidate's own member.
    """

    if type(scope) is not int or not 0 <= scope < len(scopes):
        raise RequestError("reference scope does not belong to this model")
    if isinstance(reference, NodeChoice) and len(reference._space_path) > 1:
        # ``Site.plant.heating``: a Decision over nodes reached through a node.
        scope = descend(scopes, scope, reference._space_path[:-1])
    reference = unwrap(reference)
    try:
        return scopes[scope].members[reference]
    except (KeyError, TypeError):
        pass
    if isinstance(reference, MemberRef):
        target = descend(scopes, scope, reference.path)
        member = unwrap(reference.member)
        try:
            return scopes[target].members[member]
        except (KeyError, TypeError) as cause:
            raise RequestError(
                f"{member.name if isinstance(member, Declaration) else member!r} is not a "
                f"member of {scopes[target].space_type.__qualname__}"
            ) from cause
    if isinstance(reference, ChoiceMemberRef):
        choice = _choice(scopes, choices, scope, reference.path)
        if not candidate and reference.member in choice.members:
            return choice.members[reference.member]
        found = [
            (case, scopes[child].named_members[reference.member])
            for case, child in choice.cases
            if child is not None and reference.member in scopes[child].named_members
        ]
        if len(found) == 1:
            # Only one candidate has the member: it is present exactly when
            # that candidate is selected, so its own node answers the same.
            return found[0][1]
        if not found:
            raise RequestError(f"no candidate of {choice.key} has a member {reference.member}")
        raise RequestError(
            f"{choice.key}.{reference.member} names a member of candidates "
            f"{[case for case, _ in found]}; reach one candidate through its node handle"
            + ("" if candidate else " (or read it in a declaration, which links a selection)")
        )
    if isinstance(reference, CaseRef):
        return _choice(scopes, choices, scope, reference.path).selector
    raise RequestError("reference is not a member of this compiled scope")


def resolve_decision(
    nodes: Sequence[Node],
    scopes: Sequence[Scope],
    choices: Sequence[Choice],
    scope: int,
    reference: object,
    editable_aliases: Collection[int] = (),
    linked_pinned: Mapping[str, Provenance] | None = None,
) -> int:
    """The owning decision a reference may edit.

    A reference to a Decision member follows declared aliases of it. A
    reference to a formal edits only a decision that formal owns: a fresh
    decision it was bound to, possibly shared. A formal bound to another
    member's decision is a plain alias and is not independently editable.
    """

    index = resolve_reference(nodes, scopes, choices, scope, reference, candidate=True)
    member = unwrap(reference.member) if isinstance(reference, MemberRef) else unwrap(reference)
    follow = isinstance(member, Decision) or isinstance(reference, DecisionHandle)
    seen: set[int] = set()
    node = nodes[index]
    while node.kind == "alias" and node.output is not None:
        if not follow and index not in editable_aliases:
            break
        if index in seen:
            raise RequestError("cyclic decision alias")
        seen.add(index)
        index = node.output
        node = nodes[index]
    if node.kind != "decision":
        pinned = linked_pinned.get(node.key) if linked_pinned else None
        if pinned is not None:
            raise RequestError(
                f"{node.key} is not an owned Decision: an enclosing body pinned it "
                f"({pinned.text()})"
            )
        raise RequestError(
            f"{node.key} is not an owned Decision: a formal bound to a supplier is not "
            "independently editable"
        )
    return index


__all__ = ["DecisionHandle", "ValueHandle"]
