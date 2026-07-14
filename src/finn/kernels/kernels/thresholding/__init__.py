############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Thresholding kernel: identity + HLS/RTL implementations (embedded-only)."""

from .thresholding import THRESHOLDING_SCHEMA, ThresholdingOp

__all__ = ["ThresholdingOp", "THRESHOLDING_SCHEMA"]
