# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed scoped requests and atomic refinement outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

from typing_extensions import NamedTuple

from .results import Answer

if TYPE_CHECKING:
    from .declarations import Space

T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S", bound="Space")


class EditRequest(Protocol):
    """Read-only shape of a typed edit, allowing heterogeneous atomic batches."""

    @property
    def base(self) -> object: ...

    @property
    def scope(self) -> int: ...

    @property
    def node(self) -> int: ...

    @property
    def value(self) -> object: ...


class Edit(NamedTuple, Generic[T_co]):
    """One scoped candidate tied to an exact immutable base snapshot."""

    base: object
    scope: int
    node: int
    value: T_co


@dataclass(frozen=True, slots=True)
class EditOutcome:
    owner: str
    answer: Answer[bool]
    status: Literal["unchanged", "provisional", "refused", "committed"]


@dataclass(frozen=True, slots=True)
class RefinementReport(Generic[S]):
    point: S
    accepted: bool
    outcomes: tuple[EditOutcome, ...]
