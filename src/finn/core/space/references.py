# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable typed references to nodes of one compiled Space model, and resolution.

Use the factories in :mod:`inspection` to bind authored references or discover
handles. Handles retain compilation, never a configuration, assignments or caches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

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
from .semantics import ValueSemantics

if TYPE_CHECKING:
    from ._scopes import Address, ChoiceRecord, ScopeRecord, Scopes
    from .compiler import Model

T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class ValueHandle(ValueRef[T], Generic[T]):
    """A value reference whose interpretation belongs to an exact compilation: a model and
    one of its addresses."""

    _record_origin = False
    _model: Model[Any] = field(repr=False)
    _node: Address = field(repr=False)
    semantics: ValueSemantics[T] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # An address that stands for another, or none of this model, is no member
        # (``Scopes.holds_address``): a handle holds a member's own address, so equality
        # and hash are by it.
        if not self._model.scopes.holds_address(self._node):
            raise RequestError("handle does not identify a compiled node")
        object.__setattr__(self, "semantics", cast(ValueSemantics[T] | None, self._node.semantics))

    def _resolve(self, scopes: Scopes) -> Address:
        if self._model.scopes is not scopes:
            raise RequestError("handle belongs to a different compiled model")
        return self._node

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ValueHandle)
            and self._model.scopes is other._model.scopes
            and self._node == other._node
        )

    def __hash__(self) -> int:
        return hash(self._node)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._node.key!r})"


@dataclass(frozen=True, eq=False, repr=False)
class DecisionHandle(ValueHandle[T], Generic[T]):
    """An editable handle to an owning Decision, including a structural one."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._node.kind != "decision":
            raise RequestError("a decision handle requires an owning Decision")


def decision_key(index: Address) -> str:
    """A decision's stable key is its dotted declaration path."""
    return index.key


def descend(scope: ScopeRecord, path: Sequence[Declaration]) -> ScopeRecord:
    for record in path:
        if record is scope.record:
            continue  # the node this scope instantiates
        child = scope.children.get(record)
        if child is None:
            name = record.name if record.name is not None else type(record).__name__
            raise RequestError(
                f"node {name} is not placed in {scope.name or '<root>'} of this compiled model"
            )
        scope = child
    return scope


def _choice(scope: ScopeRecord, path: Sequence[Declaration]) -> ChoiceRecord:
    scope = descend(scope, path[:-1])
    found = scope.choices.get(path[-1])
    if found is None:
        raise RequestError("the selector is not part of this compiled scope")
    return found


def resolve_reference(
    scopes: Scopes,
    scope: ScopeRecord,
    reference: object,
    *,
    candidate: bool = False,
) -> Address:
    """Interpret a member, a symbolic reference or a choice face, never through classes.

    ``decision.member`` resolves to the selection linked for it, or else to the
    one candidate that has the member; ``candidate=True`` (for edits) always
    names that candidate's own member.
    """

    if not scopes.holds(scope):
        raise RequestError("reference scope does not belong to this model")
    if isinstance(reference, NodeChoice) and len(reference._space_path) > 1:
        # ``Site.plant.heating``: a Decision over nodes reached through a node.
        scope = descend(scope, reference._space_path[:-1])
    reference = unwrap(reference)
    try:
        return scope.members[reference]
    except (KeyError, TypeError):
        pass
    if isinstance(reference, MemberRef):
        target = descend(scope, reference.path)
        member = unwrap(reference.member)
        try:
            return target.members[member]
        except (KeyError, TypeError) as cause:
            raise RequestError(
                f"{member.name if isinstance(member, Declaration) else member!r} is not a "
                f"member of {target.space_type.__qualname__}"
            ) from cause
    if isinstance(reference, ChoiceMemberRef):
        choice = _choice(scope, reference.path)
        if not candidate and reference.member in choice.members:
            return choice.members[reference.member]
        found = [
            (case, child.named_members[reference.member])
            for case, child in choice.cases
            if child is not None and reference.member in child.named_members
        ]
        if len(found) == 1:
            # Only one candidate has the member: it is present exactly when
            # that candidate is selected, so its own node answers the same.
            return found[0][1]
        if not found:
            raise RequestError(f"no case of {choice.key} has a member {reference.member}")
        raise RequestError(
            f"{choice.key}.{reference.member} names a member of cases "
            f"{[case for case, _ in found]}; reach one case through its path"
            + ("" if candidate else " (or read it in a declaration, which links a selection)")
        )
    if isinstance(reference, CaseRef):
        return _choice(scope, reference.path).selector
    raise RequestError("reference is not a member of this compiled scope")


def resolve_decision(scopes: Scopes, scope: ScopeRecord, reference: object) -> Address:
    """The owning decision a reference may edit.

    A reference to a Decision member follows declared aliases of it, hop by hop, to the
    chain's source (transparency within its scope: each alias read as itself, its output
    followed). A reference to a formal edits only a decision that formal owns. A formal
    bound to another member's decision is a plain alias and is not independently editable.
    """

    index = resolve_reference(scopes, scope, reference, candidate=True)
    member = unwrap(reference.member) if isinstance(reference, MemberRef) else unwrap(reference)
    follow = isinstance(member, Decision) or isinstance(reference, DecisionHandle)
    seen: set[Address] = set()
    while follow and index.kind == "alias":
        output = index.output
        if output is None:
            break
        if index in seen:
            raise RequestError("cyclic decision alias")
        seen.add(index)
        index = output
    if index.kind != "decision":
        raise RequestError(
            f"{index.key} is not an owned Decision: a formal bound to a supplier is not "
            "independently editable"
        )
    return index


__all__ = ["DecisionHandle", "ValueHandle"]
