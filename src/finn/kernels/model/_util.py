############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Small shared helpers for the kernel model layer."""

from __future__ import annotations


def prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out
