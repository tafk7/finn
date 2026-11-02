# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a build asks of its target, before it is resolved: the one place a build
configuration names its board or part, its clock, its shell and the runtime it runs on."""

from __future__ import annotations

from dataclasses import dataclass

from finn.platform.resolve import resolve_runtime, resolve_target
from finn.platform.shells import IP
from finn.platform.target import Target


@dataclass(frozen=True, kw_only=True)
class TargetRequest:
    """What a build configuration says about its target, before resolution: the clock
    period, a board or a part, the shell (``ip`` by default) and the runtime of those the
    shell offers (the first by default)."""

    period_ns: float
    board: str | None = None
    part: str | None = None
    shell: str = IP
    runtime: str | None = None

    def resolve(self) -> Target:
        """Resolve this request to its target; a runtime its shell does not offer is
        refused here too (``resolve_runtime``), though the target does not hold it."""
        target = resolve_target(
            period_ns=self.period_ns, part=self.part, board=self.board, shell=self.shell
        )
        self.resolve_runtime()
        return target

    def resolve_runtime(self) -> str | None:
        """The runtime this request's build runs on its board (``resolve_runtime``)."""
        return resolve_runtime(self.shell, self.runtime)


__all__ = ["TargetRequest"]
