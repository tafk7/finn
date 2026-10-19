# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration changes and checked publication reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from .results import QueryResult

if TYPE_CHECKING:
    from ._configuration import Space

T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")


@dataclass(frozen=True, slots=True)
class Change(Generic[T_co]):
    """One typed choice patch tied to the exact configuration it changes."""

    base: object = field(repr=False)
    scope: int
    node: int
    value: T_co | None = field(default=None, repr=False)
    remove: bool = False


# A batch accepts only Change objects, with heterogeneous payloads. Any erases
# the payload here without widening type inference in nested change(...) calls.
# Factories and individual Change[T] objects retain their precise value type.
ChangeRequest: TypeAlias = Change[Any]


@dataclass(frozen=True, slots=True)
class ChangeOutcome:
    """What a change did to one choice or fact.

    ``result``: for a choice set, its type check, the domain over the facts: true where
    it holds the value; refused outside it, or through an input of the domain refusing
    or not applying; unresolved while a fact the domain reads is missing (the choice is
    stored and waits on it). A choice whose scope is absent or waits is checked where its
    selectors read its case; where a guard over something else decides its presence,
    ``result`` is why it was not checked (inapplicable: absent; unresolved: its presence
    waits), and it is checked when a change brings it into play. For a fact, or a clear,
    ``Available(True)``.

    ``status``: ``"changed"``, ``"unchanged"`` or ``"removed"`` in a published change;
    in a refused one, ``"refused"`` for what refuses it and ``"checked"`` for the rest.
    ``requested`` is false for a stored choice the change did not name and checked
    again (a facts change, or one bringing it into play)."""

    owner: str
    result: QueryResult[bool]
    status: Literal["unchanged", "changed", "removed", "checked", "refused"]
    requested: bool = True


@dataclass(frozen=True, slots=True)
class ConfigurationResult(Generic[S]):
    instance: S
    accepted: bool
    outcomes: tuple[ChangeOutcome, ...]


__all__ = [
    "Change",
    "ChangeOutcome",
    "ChangeRequest",
    "ConfigurationResult",
]
