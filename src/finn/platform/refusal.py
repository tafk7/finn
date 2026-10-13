# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build target that cannot be resolved, refused by name."""

from __future__ import annotations

from finn.core.reporting.findings import Finding, FindingKind, Refusal


class TargetRefused(Refusal):
    """A part, board, shell or clock the resolution refuses: its one finding
    (``findings``), whose code says which refusal (``unknown-part``,
    ``board-part-mismatch``, ...), owned by the configuration's field it refuses
    (``configuration.target.part``), as a build's configuration asks for its target."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),))


__all__ = ["TargetRefused"]
