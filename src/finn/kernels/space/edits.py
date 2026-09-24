# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration changes and checked publication reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

from .results import QueryResult

if TYPE_CHECKING:
    from .declarations import Space

T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")


class ChangeRequest(Protocol):
    @property
    def base(self) -> object: ...

    @property
    def scope(self) -> int: ...

    @property
    def node(self) -> int: ...

    @property
    def remove(self) -> bool: ...

    @property
    def value(self) -> object: ...


@dataclass(frozen=True, slots=True)
class Change(Generic[T_co]):
    """One typed choice patch tied to an exact immutable base snapshot."""

    base: object = field(repr=False)
    scope: int
    node: int
    value: T_co | None = field(default=None, repr=False)
    remove: bool = False


@dataclass(frozen=True, slots=True)
class ChangeOutcome:
    owner: str
    result: QueryResult[bool]
    status: Literal["unchanged", "admissible", "refused", "changed", "removed", "committed"]
    requested: bool = True


@dataclass(frozen=True, slots=True)
class CommitmentReport(Generic[S]):
    instance: S
    accepted: bool
    outcomes: tuple[ChangeOutcome, ...]


@dataclass(frozen=True, slots=True)
class ConfigurationResult(Generic[S]):
    instance: S
    accepted: bool
    outcomes: tuple[ChangeOutcome, ...]


__all__ = [
    "Change",
    "ChangeOutcome",
    "ChangeRequest",
    "CommitmentReport",
    "ConfigurationResult",
]
