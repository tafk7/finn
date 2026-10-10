# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build asks of its target, before it is resolved: the one place a build
configuration names its board or part, its clock and its shell."""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.target import Target
from finn.platform.resolve import resolve_target
from finn.platform.shells import IP


@dataclass(frozen=True, kw_only=True)
class TargetRequest:
    """What a build configuration says about its target, before resolution: the clock
    period, a board or a part, and the shell (``ip`` by default)."""

    period_ns: float
    board: str | None = None
    part: str | None = None
    shell: str = IP

    def resolve(self) -> Target:
        """Resolve this request to its target."""
        return resolve_target(
            period_ns=self.period_ns, part=self.part, board=self.board, shell=self.shell
        )


__all__ = ["TargetRequest"]
