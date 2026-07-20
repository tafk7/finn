############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Implementation bundle: ``vvau_hls`` — the HLS compute core.

The universal fallback: no device/dtype feasibility gate (builds anywhere). Owns the
``resType`` lever. Self-contained and self-registering.
"""

from __future__ import annotations

from finn.kernels.space import Derived, Implementation, discrete_axis

from .names import VVAU_HLS
from .registry import register


@register
def hls_bundle() -> Implementation:
    return Implementation(
        name=VVAU_HLS,
        # HLS builds anywhere — no feasibility gate.
        axes=(discrete_axis("resType", {"lut", "dsp"}, "lut"),),
        derived=(Derived("language", lambda p, ctx: "hls"),),
        predicates=(),
        sources=("vectorvectoractivation_hls.py",),
    )
