# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU-specific computation contracts kept separate from region structure."""

from __future__ import annotations

from enum import Enum


class MVAUComputationProfile(str, Enum):
    """Evidenced source-computation profiles supported by MVAU bindings."""

    ACCUMULATOR_INTEGER = "accumulator_integer"
    BIPOLAR_XNOR_ACCUMULATOR = "bipolar_xnor_accumulator"
    FUSED_THRESHOLD = "fused_threshold"


__all__ = ["MVAUComputationProfile"]
