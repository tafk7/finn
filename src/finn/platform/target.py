# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build target: the platform its kernels are built for (``finn.platform.capabilities``),
with the part, the shell and the board, if one is named. ``finn.platform.resolve`` resolves
it, and the flow states it in the model (``finn.core.graph.annotations.platform``).
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.platform.capabilities import Platform


@dataclass(frozen=True, kw_only=True)
class Target:
    """The build target: the part, the shell that integrates the partition (``ip``: the
    packaged IP, integrated by its user), the board if one is named, and the platform
    its kernels are built for."""

    part: str
    platform: Platform
    shell: str
    board: str | None = None


__all__ = ["Target"]
