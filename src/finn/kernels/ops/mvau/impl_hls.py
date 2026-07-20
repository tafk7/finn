############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Implementation bundle: ``mvau_hls`` — the HLS compute core.

Self-contained: its axes/derived/predicates/feasibility/sources live here and
nowhere else. Registers itself via the registry decorator; adding or removing this
backend touches no other file.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.kernels.space import Derived, Implementation, discrete_axis, predicate

from .emit_hls import emit_mvau_hls
from .names import INPUT, MVAU_HLS, WEIGHTS
from .registry import register


@predicate("HLS: SIMD >= MW/1024")
def _hls_simd_lower_bound(p, ctx):
    if p.SIMD < p.MW / 1024:
        return f"HLS array-partition limit: SIMD={p.SIMD} < MW/1024={p.MW / 1024} (hls:216)"
    return None


@predicate("true-binary (non-bipolar) inputs/weights unsupported")
def _no_true_binary(p, ctx):
    # F3 — reject when (input binary OR weight binary) AND NOT binaryXnorMode
    # (matrixvectoractivation_hls.py:167). The old fixture checked only the input and
    # ignored both the xnor escape and binary weights.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    inp_binary = idt == DataType["BINARY"]
    wt_binary = wdt == DataType["BINARY"]
    if (inp_binary or wt_binary) and p.binaryXnorMode != 1:
        return "true binary (non-bipolar) inputs/weights are not supported (hls:167)"
    return None


@register
def hls_bundle() -> Implementation:
    return Implementation(
        name=MVAU_HLS,
        # HLS has no device/dtype feasibility gate — it builds anywhere.
        axes=(
            # resType: a real HLS user lever (hls:58 default lut, dsp available).
            discrete_axis("resType", {"lut", "dsp"}, "lut"),
        ),
        derived=(Derived("language", lambda p, ctx: "hls"),),
        predicates=(_hls_simd_lower_bound, _no_true_binary),
        sources=("matrixvectoractivation_hls.py",),  # HLS codegen owns its template
        emit=emit_mvau_hls,
    )
