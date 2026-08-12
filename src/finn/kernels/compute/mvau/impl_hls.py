############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend backend: ``mvau_hls`` — the HLS compute core.

Self-contained: its axes/derived/predicates/feasibility/sources live here and
nowhere else. Registers itself via the registry decorator; adding or removing this
backend touches no other file.
"""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.kernels.engine.axis import discrete_axis
from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
from finn.kernels.engine.predicate import predicate
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.param_names import DECOUPLED, EMBEDDED

from .emit_hls import emit_mvau_hls
from .backends import COMPUTE_STREAM, mvau_out_dtype, mvau_register_dtypes
from .kernel import INPUT, MVAU_HLS, OUTPUT, THRESHOLDS, WEIGHTS
from .registry import register

# The HLS MVU compute core is a quantized-integer matmul: a float32 i/w tensor has no legal
# HLS realization. Declared per port so the pool's UNION of supported datatypes is the single
# source of truth can_infer_from delegates to (a float MatMul is rejected FOR THE RIGHT
# REASON — no backend supports it); a future float backend widens the claim with no op edit.
_INTEGER = DatatypeSupport(kind=DatatypeKind.INTEGER)


@predicate("HLS: SIMD >= MW/1024", deps={"SIMD"})
def _hls_simd_lower_bound(p, ctx):
    mw = ctx.tensor_shape(WEIGHTS)[0]  # the weight block's reduction extent
    if p.SIMD < mw / 1024:
        return f"HLS array-partition limit: SIMD={p.SIMD} < MW/1024={mw / 1024} (hls:216)"
    return None


@predicate("true-binary (non-bipolar) inputs/weights unsupported")
def _no_true_binary(p, ctx):
    # F3 — reject when input binary OR weight binary (matrixvectoractivation_hls.py:167).
    # HLS's only binary gate: without a dtype-support gate, removing this would let a BINARY
    # node resolve and emit wrong hardware. BIPOLAR {-1,+1} operands (the old binaryXnorMode=1
    # escape) re-enter as backend datatype support, not a resurrected flag.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    inp_binary = idt == DataType["BINARY"]
    wt_binary = wdt == DataType["BINARY"]
    if inp_binary or wt_binary:
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
        predicates=(_hls_simd_lower_bound, _no_true_binary),
        sources=("matrixvectoractivation_hls.py",),  # HLS codegen owns its template
        emit=emit_mvau_hls,
        # Backend-scoped internal-register dtypes (acc/weight). All MVAU cores narrow
        # identically today; a future core diverges by supplying different specs.
        derived_dtypes=mvau_register_dtypes(),
        # The HLS core takes weights either baked (params.h) or streamed (memstream), and
        # bakes thresholds into thresh.h — so it consumes weights in BOTH modes, thresholds
        # constant-only. (base FINN: internal_embedded is HLS-only; the fused HLS core is the
        # only MVU that supports embedded thresholds.) Integer i/w declared as datatype support;
        # the out port's produced dtype rides derived_dtype (acc-or-graph rule).
        ports=ports_from(
            stream=COMPUTE_STREAM,
            mem_modes={WEIGHTS: {EMBEDDED, DECOUPLED}, THRESHOLDS: {EMBEDDED}},
            accepted_dtypes={INPUT: _INTEGER, WEIGHTS: _INTEGER},
            derived_dtype={OUTPUT: mvau_out_dtype()},
        ),
    )
