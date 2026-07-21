############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU routed through the KernelOp façade — the pilot proving the Tier-3 surface
agrees with the real op's battle-tested emit-side derivations.

``mvau_kernel_op()`` wraps the SAME shared axes/derived/predicates + compute pool +
parameters composition that ``mvau_schema()`` always used (they are now the same
object — ``mvau_schema`` delegates to the KernelOp). This test proves:

  * the KernelOp getters (folded shapes, stream widths) agree EXACTLY with the
    op-level ``instream_width``/``outstream_width`` derived that emit reads — the two
    surfaces are consistent, not a reimplementation that drifts;
  * the impl-owned ``tiling`` (SIMD folds inp, PE folds out, weights=PE*SIMD) resolves
    per compute impl;
  * the op-level ``cost_model`` gives the reduction-coupled product nf*sf*n_vecs.
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import Context, Illegal
from finn.kernels.ops.mvau import mvau_kernel_op

MW, MH = 128, 64


def _ctx() -> Context:
    return Context(
        shapes={"inp": (1, MW), "weights": (MW, MH), "out": (1, MH)},
        datatypes={
            "inp": DataType["INT8"],
            "weights": DataType["INT8"],
            "out": DataType["INT32"],
        },
        initializers={"weights": np.ones((MW, MH), dtype=np.float32)},
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


def _configure(simd, pe, impl="mvau_hls"):
    op, ctx = mvau_kernel_op(), _ctx()
    pt = op.configure(ctx, {"implementation": impl, "SIMD": simd, "PE": pe})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    return op, ctx, pt


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64), (1, 1)])
def test_kernelop_getters_agree_with_op_derived_widths(simd, pe):
    # The KernelOp stream-width getters must equal the emit-side op-derived widths.
    op, ctx, pt = _configure(simd, pe)
    assert op.get_instream_width(pt, ctx, 0) == pt.instream_width
    assert op.get_outstream_width(pt, ctx, 0) == pt.outstream_width


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_folded_data_shapes(simd, pe):
    op, ctx, pt = _configure(simd, pe)
    assert op.get_folded_input_shape(pt, ctx, 0) == (1, MW // simd, simd)
    assert op.get_folded_output_shape(pt, ctx, 0) == (1, MH // pe, pe)


def test_weight_port_width_is_pe_simd_and_shape_raises():
    op, ctx, pt = _configure(16, 4)
    from finn.kernels.space import KernelOpError

    # weight WIDTH = PE*SIMD*wbits resolves via the tiling evaluator...
    assert op.get_instream_width(pt, ctx, 1) == (4 * 16) * 8
    # ...but its folded SHAPE is not a tensor-axis reshape -> raises, not fakes.
    with pytest.raises(KernelOpError, match="does not fold a tensor axis"):
        op.get_folded_input_shape(pt, ctx, 1)


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_cost_model_is_reduction_product(simd, pe):
    op, ctx, pt = _configure(simd, pe)
    # nf * sf * n_vecs (n_vecs == 1 here) — the reduction x output product.
    assert op.get_exp_cycles(pt, ctx) == (MH // pe) * (MW // simd)


def test_all_three_compute_impls_carry_tiling():
    op, ctx = mvau_kernel_op(), _ctx()
    for impl in ("mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"):
        pt = op.configure(ctx, {"implementation": impl, "SIMD": 8, "PE": 8})
        if isinstance(pt, Illegal):
            # a DSP impl may be infeasible on this part; skip — tiling is still declared.
            continue
        assert op.get_instream_width(pt, ctx, 0) == 8 * 8


def test_mvau_schema_delegates_to_kernel_op():
    # mvau_schema() must be exactly the KernelOp's schema (same axis set), so all the
    # existing emit/composition tests exercise the refactored assembly.
    from finn.kernels.ops.mvau import mvau_schema

    assert mvau_schema().axis_names == mvau_kernel_op().schema().axis_names
