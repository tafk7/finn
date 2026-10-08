# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A build target that cannot be resolved, refused by name."""

from __future__ import annotations


class TargetRefused(ValueError):
    """A part, board, shell or clock the resolution refuses: ``name`` says which
    refusal (``unknown-part``, ``board-part-mismatch``, ...), and the message starts
    with it."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name}: {message}")
        self.name = name


__all__ = ["TargetRefused"]
