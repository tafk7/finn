############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Implementation bundle: ``thresholding_hls`` — the HLS backend.

Universal (no device/dtype feasibility gate). Its delivery cluster (mem_mode /
ram_style) is DEFERRED this task, so this bundle currently contributes only the
``language`` marker — deliberately thin, to make the point that the HLS and RTL
bundles carry DISJOINT impl-local axes (the RTL bundle carries depth-triggers etc.,
this one would carry mem_mode when the delivery cluster is modeled).
"""

from __future__ import annotations

from finn.kernels.space import Derived, Implementation

from .names import THRESHOLDING_HLS
from .registry import register


@register
def hls_bundle() -> Implementation:
    return Implementation(
        name=THRESHOLDING_HLS,
        # HLS builds anywhere; no dtype gate (identical envelope to RTL).
        derived=(Derived("language", lambda p, ctx: "hls"),),
        sources=("thresholding_hls.py",),
    )
