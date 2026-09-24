# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable typed references to nodes of one compiled Space family.

Use the factories in :mod:`inspection` to bind authored references or discover
handles. Handles retain compilation, never a configuration, assignments or caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from .declarations import DecisionRef, ValueRef
from .errors import RequestError
from .ir import LinkedModel
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


__all__ = ["DecisionHandle", "ValueHandle"]
