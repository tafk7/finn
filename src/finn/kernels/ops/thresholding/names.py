############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Shared Thresholding string constants — pool-member identities and tensor names.

Note: unlike MVAU/VVAU there is NO mem_mode delivery cluster modeled here (deferred
this task). The parameter tensor is named ``thresholds`` (never ``external``).
"""

from __future__ import annotations

# Pool-member identities (values of the root `implementation` axis).
THRESHOLDING_HLS = "thresholding_hls"
THRESHOLDING_RTL = "thresholding_rtl"

# Context tensor names this schema resolves against.
THRESHOLDS = "thresholds"
INPUT = "inp"
OUTPUT = "out"
