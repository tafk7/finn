############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend bundle: ``mvau_hls`` — the HLS compute core.

Self-contained: its axes/derived/predicates/feasibility/sources live here and
nowhere else. Registers itself via the registry decorator; adding or removing this
backend touches no other file.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.kernels.space import Backend, discrete_axis, predicate
from finn.kernels.model.param_names import CONSTANT, STREAM

from .emit_hls import emit_mvau_hls
from .op import COMPUTE_STREAM, INPUT, MVAU_HLS, THRESHOLDS, WEIGHTS
from .registry import register


@predicate("HLS: SIMD >= MW/1024")
def _hls_simd_lower_bound(p, ctx):
    mw = ctx.tensor_shape(WEIGHTS)[0]  # the weight block's reduction extent
    if p.SIMD < mw / 1024:
        return f"HLS array-partition limit: SIMD={p.SIMD} < MW/1024={mw / 1024} (hls:216)"
    return None


@predicate("HLS-MVU requires integer input and weight datatypes")
def _hls_integer_iw(p, ctx):
    # The HLS MVU compute core is a quantized-integer matmul: a float32 i/w tensor has no
    # legal HLS realization. This is the backend-OWNED feasibility fact the frontend claim
    # used to encode (op.py's is_integer literal) — pushed down so the pool is the SoT and
    # can_infer_from can delegate to it (D-R5).
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    if not (idt.is_integer() and wdt.is_integer()):
        return f"HLS-MVU requires integer input/weights (got idt={idt}, wdt={wdt})"
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
def hls_bundle() -> Backend:
    return Backend(
        name=MVAU_HLS,
        language="hls",
        # HLS has no device/dtype feasibility gate — it builds anywhere.
        axes=(
            # resType: a real HLS user lever (hls:58 default lut, dsp available).
            discrete_axis("resType", {"lut", "dsp"}, "lut"),
        ),
        predicates=(_hls_integer_iw, _hls_simd_lower_bound, _no_true_binary),
        sources=("matrixvectoractivation_hls.py",),  # HLS codegen owns its template
        emit=emit_mvau_hls,
        stream=COMPUTE_STREAM,
        # The HLS core takes weights either baked (params.h) or streamed (memstream), and
        # bakes thresholds into thresh.h — so it consumes weights in BOTH modes, thresholds
        # constant-only. (base FINN: internal_embedded is HLS-only; the fused HLS core is the
        # only MVU that supports embedded thresholds.)
        consumes={WEIGHTS: {CONSTANT, STREAM}, THRESHOLDS: {CONSTANT}},
    )
