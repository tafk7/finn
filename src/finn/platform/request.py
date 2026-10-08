# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build states of its target, before it is resolved: the one place a build
configuration names its board or part, its clock and its shell."""

from __future__ import annotations

from dataclasses import dataclass

from finn.kernels.target import Target
from finn.platform.resolve import resolve_target
from finn.platform.shells import IP


@dataclass(frozen=True, kw_only=True)
class TargetRequest:
    """The target a build asks for: the clock period it is built at (``period_ns``), a
    ``board`` or a ``part`` (a board gives its part; a part stated beside it is an
    assertion), and the ``shell`` that integrates it (``ip`` unless one is stated).
    ``resolve()`` is its target (``resolve_target``), every refusal named."""

    period_ns: float
    board: str | None = None
    part: str | None = None
    shell: str = IP

    def resolve(self) -> Target:
        """The target this request names (``resolve_target``)."""
        return resolve_target(
            period_ns=self.period_ns, part=self.part, board=self.board, shell=self.shell
        )


__all__ = ["TargetRequest"]
