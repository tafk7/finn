# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit fault-capturing support for interactive or service boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, NoReturn, TypeAlias, TypeVar

from .errors import DesignSpaceError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SafeSuccess(Generic[T]):
    value: T

    @property
    def succeeded(self) -> Literal[True]:
        return True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class SafeFailure:
    error: DesignSpaceError

    @property
    def succeeded(self) -> Literal[False]:
        return False

    def unwrap(self) -> NoReturn:
        raise self.error


SafeResult: TypeAlias = SafeSuccess[T] | SafeFailure


def capture(operation: Callable[[], T]) -> SafeResult[T]:
    """Run one engine operation and capture only contextual engine failures."""

    try:
        return SafeSuccess(operation())
    except DesignSpaceError as exc:
        return SafeFailure(exc)


__all__ = ["SafeFailure", "SafeResult", "SafeSuccess", "capture"]
