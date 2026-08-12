############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend backend: ``thresholding_hls`` — the HLS backend.

Universal (no device/dtype feasibility gate). Bakes thresholds into a ``thresh.h``
``ThresholdsActivation`` ROM (the constant/embedded topology) via the shared parameter
serializer — the SEPARABLE, static-schedule threshold memory. The decoupled
threshold-stream cluster (mem_mode/ram_style) is DEFERRED, mirroring the MVAU HLS core's
deferred weight-stream path; the HLS and RTL backends carry DISJOINT backend-local axes (the
RTL backend carries depth-triggers etc.).
"""

from __future__ import annotations

from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.param_names import EMBEDDED

from .emit_hls import emit_thresholding_hls
from .names import COMPUTE_STREAM, THRESHOLDING_HLS, THRESHOLDS
from .registry import register


@register
def hls_bundle() -> Backend:
    return Backend(
        name=THRESHOLDING_HLS,
        language="hls",
        # HLS builds anywhere; no dtype gate (identical envelope to RTL).
        sources=("thresholding_hls.py",),
        emit=emit_thresholding_hls,
        # The HLS core bakes thresholds into thresh.h — it consumes them in EMBEDDED mode
        # only (embedded ROM, no stream port). base FINN: internal_embedded is HLS-only.
        # The stream declares the BLOCK→STREAM fold, from which the tiling engine generates
        # the PE dial, the divisibility rules and stream_width.<iface>.
        ports=ports_from(stream=COMPUTE_STREAM, mem_modes={THRESHOLDS: {EMBEDDED}}),
    )
