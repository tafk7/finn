############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Shared VVAU string constants — pool-member identities, mem-mode values, tensor
names. The op-agnostic DSP ``VERSION`` map lives in ``fixtures/_dsp_rtl.py``.
"""

from __future__ import annotations

# Pool-member identities (values of the root `implementation` axis).
VVAU_HLS = "vvau_hls"
VVAU_RTL = "vvau_rtl"

# mem_mode domain values (same delivery cluster as MVAU; shared by both impls).
DECOUPLED = "internal_decoupled"
EMBEDDED = "internal_embedded"
EXTERNAL = "external"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"
