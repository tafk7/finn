# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Values that make a composite's structure observable to its computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .semantics import ValueSemantics, default_semantics

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Located(Generic[T]):
    """A value and where it lives: ``node`` is the child's declaration name in the
    Space that read it (None for that Space itself); ``member`` names the member
    or export key."""

    node: str | None
    member: str
    value: T


LOCATED: ValueSemantics[Located[object]] = default_semantics(Located)

__all__ = ["LOCATED", "Located"]
