# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Read and replay a configuration's commitments by key in the space tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from finn.core.space import Available, Space, inspection

S = TypeVar("S", bound=Space)


def committed(point: Space) -> dict[str, object]:
    """Every Decision ``point`` commits, by key, in key order (a forced one is never
    committed)."""
    found: dict[str, object] = {}
    for item in inspection.decisions(point):
        state = point.field(item.reference).state
        if isinstance(state, Available) and state.value.status == "committed":
            found[item.key] = state.value.value
    return found


def replay(base: S, choices: Mapping[str, object]) -> S:
    """``base`` with ``choices`` (by key) committed in one batch."""
    owned = {item.key: item.reference for item in inspection.decisions(base)}
    return base.with_choices({owned[key]: value for key, value in choices.items()})
