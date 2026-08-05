############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
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

# The BLOCK->STREAM lowering shared by both compute impls: PE folds the channel dim
# (NumChannels) on the input, output, and the threshold block's leading (channel) extent.
# The threshold's step dim is unfolded (a whole row per beat).
#
# Lives HERE rather than in op.py because each backend declares it (STREAM is backend-owned,
# BLOCK is op-owned) and op.py imports the backends -- so op.py cannot be its home without a
# cycle. It is re-exported from op.py for callers that expect it there.
COMPUTE_STREAM = {
    INPUT: [1, "PE"],
    OUTPUT: [1, "PE"],
    THRESHOLDS: ["PE", 1],
}
