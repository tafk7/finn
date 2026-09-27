# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable typed references to nodes of one compiled Space family.

Use the factories in :mod:`inspection` to bind authored references or discover
handles. Handles retain compilation, never a configuration, assignments or caches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from .declarations import AcceptedViewRef, DecisionRef, ScopedValueRef, ValueKey, ValueRef, ViewKey
from .errors import RequestError
from .ir import Choice, LinkedModel, Node, Scope
from .semantics import ValueSemantics

T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ValueHandle(ValueRef[T], Generic[T]):
    """A value reference whose interpretation belongs to an exact compilation."""

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
class DecisionHandle(ValueHandle[T], DecisionRef[T], Generic[T]):
    """An editable handle to an owning Decision, including a choice selector."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._linked.nodes[self._node].kind != "decision":
            raise RequestError("a decision handle requires an owning Decision")


def decision_key(linked: LinkedModel, index: int) -> str:
    """Use the authored choice key for selectors, never generated node names."""
    choice = linked.selector_choices.get(index)
    return linked.nodes[index].key if choice is None else linked.choices[choice].key


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


__all__ = ["DecisionHandle", "ValueHandle"]
