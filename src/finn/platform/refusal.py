# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build target that cannot be resolved, refused by name."""

from __future__ import annotations

from finn.core.space import Finding, FindingKind


class TargetRefused(ValueError):
    """A part, board, shell or clock the resolution refuses: its one finding
    (``findings``), whose code says which refusal (``unknown-part``,
    ``board-part-mismatch``, ...); the message starts with the code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.findings = (Finding(FindingKind.REJECTION, code, "finn.platform", message),)


__all__ = ["TargetRefused"]
