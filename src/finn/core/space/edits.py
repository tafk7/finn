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
    """One typed choice patch tied to an exact immutable base snapshot."""

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
    owner: str
    result: QueryResult[bool]
    status: Literal["unchanged", "admissible", "refused", "changed", "removed"]
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
