############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Thresholding kernel: identity + HLS/RTL implementations (embedded-only)."""

from ...adapter import register_identity
from ...registry import registry
from .impl_hls import ThresholdingHLS
from .impl_rtl import ThresholdingRTL
from .thresholding import THRESHOLDING_SCHEMA, ThresholdingOp

registry.register(ThresholdingHLS)
registry.register(ThresholdingRTL)
register_identity("Thresholding", ThresholdingOp)

__all__ = [
    "ThresholdingOp",
    "THRESHOLDING_SCHEMA",
    "ThresholdingHLS",
    "ThresholdingRTL",
]
