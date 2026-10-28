# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Temporary read declaration used only by the independently owned budget lane.

KernelDeclaration/lowering/setter/parallel callback tables are retired. The
three remaining callers are reserved budget producers, whose accepted separate
migration is assembled by desk; this function is not a kernel authoring API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])


def declared_reads(*paths: str) -> Callable[[_F], _F]:
    """Declare what a method reads by names its source computes (``__reads__``: dotted
    paths from ``self``), which stand in for its source in the reads the engine infers; the
    engine holds the method to them when it runs."""

    def declare(function: _F) -> _F:
        function.__reads__ = paths  # type: ignore[attr-defined]
        return function

    return declare


__all__ = ["declared_reads"]
