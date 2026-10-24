# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How each formal of one placed node is supplied.

A formal is supplied by the body that declared the node (at the call or by
assignment), once, or else by its Space class's declaration (a default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._nodes import NodeDecision, NodeDecl, inline_decision
from .declarations import Decision, Param, ValueRef, View
from .errors import DefinitionError

_OWNER = "space.bindings"

BindingKind = Literal["literal", "reference"]


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    supplier: object
    kind: BindingKind
    origin: str | None = None


@dataclass(frozen=True, slots=True)
class Slot:
    """The supplier of one formal of one placed node, as its declaring body wrote it."""

    supplier: object
    # Where its references are read: the scope whose body declared the node.
    scope: int
    origin: str | None
    key: str


def classify(slot: Slot) -> tuple[BindingKind, object]:
    """How a value formal is supplied: its binding kind and supplier.

    A literal is a constant, and anything else forwards a
    reference. A Decision that is no class attribute (given at the call) is refused.
    """
    value = slot.supplier
    if isinstance(value, Param) and value.owner is None:
        raise DefinitionError.broken(
            "inline-param-supplies-member",
            _OWNER,
            f"{slot.key}: an inline Param cannot supply a member; declare the "
            "formal on the enclosing Space class and bind it",
            declaration=slot.key,
        )
    if isinstance(value, NodeDecl):
        raise DefinitionError.broken(
            "node-not-value",
            _OWNER,
            f"{slot.key}: a node or a selector is not a value; bind one of its members",
            declaration=slot.key,
        )
    if isinstance(value, Decision) and value.owner is None and not isinstance(value, NodeDecision):
        raise inline_decision(slot.key)
    if isinstance(value, (ValueRef, View)):
        return "reference", value
    return "literal", value


__all__ = ["BindingKind", "PlacementBinding", "Slot", "classify"]
