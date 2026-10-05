# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How each member of one placed node is supplied, layer by layer.

A member may be set by its Space class's declaration (a Param default, a Decision,
a child node), by the body that declared the node (at the call or by
assignment), and by every enclosing body through a path. The outermost setting
wins; the others are kept as its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from ._nodes import NodeDecision, NodeDecl
from .declarations import (
    MISSING,
    UNSUPPLIED,
    CaseRef,
    ChoiceMemberRef,
    Const,
    Decision,
    Declaration,
    MemberRef,
    Param,
    Present,
    ValueRef,
    View,
    _path_text,
    at,
)
from .domains import Domain
from .errors import DefinitionError
from .expressions import Expr
from .ir import Layer, Provenance

BindingKind = Literal["literal", "reference", "local-decision", "parameter", "pin", "pin-reference"]

# The writer of a root declaration's own settings: the design_space() call.
ROOT_WRITER = -1
ROOT_BODY = "the root declaration"


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    supplier: object
    kind: BindingKind
    # The scope whose body wrote the supplier: references are read there.
    source_scope: int | None = None
    origin: str | None = None
    # A Decision member overridden by a value or a narrower Decision keeps its
    # declared Decision as the contract every supplier is checked against.
    # (Typed as a Declaration: a Decision-typed attribute reads through __get__.)
    contract: Declaration | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True, slots=True)
class Slot:
    """The effective setting of one member of one placed node, with its history."""

    supplier: object
    writer: int  # ROOT_WRITER, or the scope whose body wrote it
    scope: int  # where its references are read
    origin: str | None
    provenance: Provenance
    # The suppliers of the inner layers it overrides, innermost first.
    overridden: tuple[object, ...] = ()

    @property
    def opened(self) -> bool:
        """Whether an overridden inner layer opened a coordinate (a fresh Decision)."""
        return any(
            isinstance(item, Decision) and item.owner is None and item.name is None
            for item in self.overridden
        )


def supply_text(value: object) -> str:
    """A supplier as provenance text: a literal, a reference path, a node or a Decision."""
    if isinstance(value, NodeDecl):
        return f"{value.space_type.__qualname__} node"
    if isinstance(value, NodeDecision):
        return f"Decision over {sorted(value.candidates)}"
    if isinstance(value, Decision):
        return decision_text(cast(Decision[object], value))
    if isinstance(value, MemberRef):
        return f"{_path_text(value.path)}.{value.member.name}"
    if isinstance(value, ChoiceMemberRef):
        return f"{_path_text(value.path)}.{value.member}"
    if isinstance(value, CaseRef):
        return f"selected({_path_text(value.path)})"
    if isinstance(value, Present):
        return f"Present({', '.join(supply_text(item) for item in value.sources)})"
    if isinstance(value, Expr):
        return "an expression"
    if isinstance(value, Const):
        return repr(value.value)
    if isinstance(value, (Declaration, View)):
        return str(getattr(value, "name", None) or type(value).__name__)
    return repr(value)


def decision_text(decision: Decision[object]) -> str:
    domain: Domain[object] = decision.domain
    values = domain._finite_values
    if values is not None:
        return f"Decision(values={tuple(values)!r})"
    return "Decision(domain=...)"


def declared_layer(declaration: Declaration) -> Layer | None:
    """The Space class's own setting of a member, if it has one."""
    body = declaration.owner.__name__ if declaration.owner is not None else "?"
    if isinstance(declaration, Param):
        default = declaration.default
        if default is MISSING or default is UNSUPPLIED:
            return None
        return Layer(body, declaration.origin, repr(default), declared=True)
    if isinstance(declaration, (NodeDecl, NodeDecision, Decision)):
        return Layer(body, declaration.origin, supply_text(declaration), declared=True)
    return None


def classify(
    member: Declaration, slot: Slot, *, root_own: bool
) -> tuple[BindingKind, object, Decision[object] | None]:
    """How a value member is supplied: its binding kind, supplier and contract.

    A root's own plain values stay runtime inputs (``parameter``). A Decision
    member overridden by a value is pinned; by a reference, pinned to it; by a
    fresh Decision, replaced under the same key. Either keeps the declared
    Decision as its contract.
    """
    value = slot.supplier
    if isinstance(value, Param) and value.owner is None:
        raise DefinitionError(
            f"{slot.provenance.key}: an inline Param cannot supply a member; declare the "
            "formal on the enclosing Space class and bind it"
        )
    if isinstance(value, NodeDecl):
        raise DefinitionError(
            f"{slot.provenance.key}: a node or a Decision over nodes is not a value; bind one "
            "of its members"
        )
    if isinstance(value, Param) and value.reference_space_type() is not None:
        raise DefinitionError(
            f"{slot.provenance.key}: forwards the reference input {value.name}"
            f"{at(value.origin)}, but the formal takes a value"
        )
    fresh_decision = (
        isinstance(value, Decision) and value.owner is None and not isinstance(value, NodeDecision)
    )
    if isinstance(value, Const) and value.owner is None:
        value, literal = value.value, True
    else:
        literal = not isinstance(value, (ValueRef, View))
    if isinstance(member, Decision):
        decision = cast(Decision[object], member)
        if fresh_decision:
            return "local-decision", value, decision
        return ("pin" if literal else "pin-reference"), value, decision
    if fresh_decision:
        return "local-decision", value, None
    if literal:
        return ("parameter" if root_own else "literal"), value, None
    return "reference", value, None


__all__ = [
    "BindingKind",
    "PlacementBinding",
    "ROOT_BODY",
    "ROOT_WRITER",
    "Slot",
    "classify",
    "declared_layer",
    "decision_text",
    "supply_text",
]
