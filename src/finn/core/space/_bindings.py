# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How each formal of one placed node is supplied.

A formal is supplied by the body that declared the node (at the call or by
assignment), once, or else by its Space class's declaration (a default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._nodes import NodeDecision, NodeDecl
from .declarations import Const, Decision, Param, ValueRef, View
from .errors import DefinitionError

BindingKind = Literal["literal", "reference", "local-decision"]


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

    A literal (or an inline ``Const``) is a constant, a fresh Decision opens a
    choice keyed at the formal, and anything else forwards a reference.
    """
    value = slot.supplier
    if isinstance(value, Param) and value.owner is None:
        raise DefinitionError(
            f"{slot.key}: an inline Param cannot supply a member; declare the "
            "formal on the enclosing Space class and bind it"
        )
    if isinstance(value, NodeDecl):
        raise DefinitionError(
            f"{slot.key}: a node or a Decision over nodes is not a value; bind one of its members"
        )
    if isinstance(value, Decision) and value.owner is None and not isinstance(value, NodeDecision):
        return "local-decision", value
    if isinstance(value, Const) and value.owner is None:
        return "literal", value.value
    if isinstance(value, (ValueRef, View)):
        return "reference", value
    return "literal", value


__all__ = ["BindingKind", "PlacementBinding", "Slot", "classify"]
