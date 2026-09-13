# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical weight-supply choices shared by MVAU construction and its Design."""

from enum import Enum


class WeightSupply(str, Enum):
    """How the matrix reaches the compute Region."""

    EXTERNAL = "external"
    EMBEDDED = "embedded"
    DECOUPLED = "decoupled"


__all__ = ["WeightSupply"]
