# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Locate real dispatcher interruption boundaries without production test hooks."""

import inspect
from collections.abc import Callable


def source_line(function: Callable[..., object], marker: str) -> int:
    source, start = inspect.getsourcelines(function)
    matches = [start + offset for offset, line in enumerate(source) if marker in line]
    assert len(matches) == 1, f"expected one interruption boundary for {marker!r}: {matches}"
    return matches[0]
