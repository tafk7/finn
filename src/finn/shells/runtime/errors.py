# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What the shared driver refuses: one error, its ``code`` and its message."""

from __future__ import annotations


class Refused(Exception):
    """A runtime file, a value or a call the driver refuses: its ``code`` and its message,
    which says what to change."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


__all__ = ["Refused"]
