############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Shared MVAU string constants — pool-member identities, mem-mode values, tensor
names, and the device DSP-version map. Kept in one tiny module so every bundle and
the op-level shared code reference the same literals without importing each other.
"""

from __future__ import annotations

# Pool-member identities (values of the root `implementation` axis). A new backend
# defines its own name in its own bundle file; these three are the built-ins.
MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

# mem_mode domain values.
DECOUPLED = "internal_decoupled"
EMBEDDED = "internal_embedded"
EXTERNAL = "external"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"

# DSP block -> $VERSION$ (silicon consequence, FORCED from fpgapart; §1.2.2).
VERSION = {"DSP48E1": 1, "DSP48E2": 2, "DSP58": 3}
