# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A board a build may name: its name and its part, nothing about any flow. The boards
are declared as data (``finn.platform.declarations``), and the shells that support a
board say so, each with what its linker needs of it (``finn.platform.shells``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Board:
    """A board: its FINN ``name``, for one board revision, and its ``part``."""

    name: str
    part: str


__all__ = ["Board"]
