############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVU — the Tier-3 Kernel acid test (kernelop-tensor-block-stream.md §4, §5.1).

MVU is the case that forces everything LayerNorm didn't:

  * PE and SIMD are ONE dial each on DIFFERENT interfaces (SIMD folds act's reduction
    dim MW; PE folds out's free dim MH) — not two dials on one port.
  * The weight PARAM port's stream WIDTH is a cross-interface expr WSIMD = PE*SIMD/TH,
    NOT a fold of the weight tensor's own axes — so width resolves via the evaluator,
    but a folded SHAPE request raises (folds_last_axis=False).
  * TH exists ONLY on mvau_rtl_tiled — the weight-port expr's deps are backend-local.
  * The cost nf*sf*n_vecs*TH is a PRODUCT of the reduction trip (act) and output trip
    (out): the generic max-over-interfaces floor under-counts it, proving the op-level
    cost_model seam is necessary, not decorative (§2.2 reduction coupling).

Validated against FINN matrixvectoractivation.py:
  normal in0  = [vecs, MW]        folded in0  = [vecs, MW/SIMD, SIMD]
  normal out  = [vecs, MH]        folded out  = [vecs, MH/PE, PE]
  instream_width(0) = i_bits*SIMD      outstream_width = o_bits*PE
  instream_width(1) = (PE*SIMD*w_bits)//TH   (decoupled weight port, WSIMD)
  get_exp_cycles = (MH/PE)*(MW/SIMD)*prod(vecs)*TH
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Context,
    Direction,
    Illegal,
    Backend,
    Interface,
    Kernel,
    KernelError,
    KernelSchema,
    derive,
    discrete_axis,
    fixed_axis,
    param,
)

MW = 128  # reduction dim (act last axis)
MH = 64   # output dim (out last axis)
VECS = (1,)  # numInputVectors leading dims


# ---------------------------------------------------------------------------
# The MVU Kernel.
# ---------------------------------------------------------------------------


def _mvu_cost(point, context):
    # nf * sf * prod(vecs) * TH — reduction trip x output trip x vectors x tiling passes.
    sf = point.MW // point.SIMD
    nf = point.MH // point.PE
    n_vecs = 1
    for d in context.tensor_shape("inp")[:-1]:
        n_vecs *= int(d)
    th = point.get("TH", 1)
    return nf * sf * n_vecs * th


def _mvu_op() -> Kernel:
    from finn.kernels.space import FULL

    mw = fixed_axis("MW", lambda p, ctx: ctx.tensor_shape("inp")[-1])
    mh = fixed_axis("MH", lambda p, ctx: ctx.tensor_shape("out")[-1])

    # inp folds MW by SIMD (last axis); out folds MH by PE; weights is a 2-D block streamed
    # SIMD·PE. The engine derives the SIMD/PE dials + divisibility from these stream folds.
    untiled_stream = {
        "inp": [1, "SIMD"],
        "out": [1, "PE"],
        "weights": ["SIMD", "PE"],
    }

    hls = Backend(name="mvau_hls", stream=untiled_stream)
    rtl_untiled = Backend(name="mvau_rtl_untiled", stream=untiled_stream)
    # tiled backend OWNS TH; weight delivered as ONE cross-interface expr (PE*SIMD)/TH —
    # the acid test that a backend can STREAM DIFFERENTLY from its peers (impl-owned) while
    # the BLOCK is identical. This weight position is an expr (not a plain dial) so its
    # folded SHAPE is width-only (raises), but its width resolves.
    rtl_tiled = Backend(
        name="mvau_rtl_tiled",
        axes=(discrete_axis("TH", {1, 2, 4}, 2),),
        stream={
            "inp": [1, "SIMD"],
            "out": [1, "PE"],
            "weights": [derive("PE") * derive("SIMD") / param("TH")],
        },
    )

    return Kernel(
        identity=KernelSchema(
            name="MVU",
            interfaces=(
                Interface("inp", Direction.IN, block=[1, FULL]),
                Interface("weights", Direction.IN, block=[FULL, FULL]),
                Interface("out", Direction.OUT, block=[1, FULL]),
            ),
            op_axes=(mw, mh),
            cost_model=_mvu_cost,
        ),
        pool=(hls, rtl_untiled, rtl_tiled),
    )


def _ctx() -> Context:
    return Context(
        shapes={"inp": VECS + (MW,), "weights": (MW, MH), "out": VECS + (MH,)},
        datatypes={
            "inp": DataType["INT8"],
            "weights": DataType["INT8"],
            "out": DataType["INT32"],
        },
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


# ---------------------------------------------------------------------------
# PE / SIMD split across interfaces; folded shapes match FINN.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simd,pe", [(8, 8), (16, 4), (128, 64), (1, 1)])
def test_folded_data_shapes_match_finn(simd, pe):
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": simd, "PE": pe})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # act folds MW by SIMD; out folds MH by PE.
    assert op.get_folded_input_shape(pt, ctx, 0) == VECS + (MW // simd, simd)
    assert op.get_folded_output_shape(pt, ctx, 0) == VECS + (MH // pe, pe)


@pytest.mark.parametrize("simd,pe", [(8, 8), (16, 4), (128, 64)])
def test_data_stream_widths_match_finn(simd, pe):
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": simd, "PE": pe})
    assert op.get_instream_width(pt, ctx, 0) == 8 * simd    # act: i_bits*SIMD
    assert op.get_outstream_width(pt, ctx, 0) == 32 * pe    # out: o_bits*PE


# ---------------------------------------------------------------------------
# The weight PARAM port: width = WSIMD (derived), shape request raises.
# ---------------------------------------------------------------------------


def test_untiled_weight_width_is_pe_simd():
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": 16, "PE": 4})
    # WSIMD = PE*SIMD; width = WSIMD * w_bits.
    assert op.get_instream_width(pt, ctx, 1) == (4 * 16) * 8


@pytest.mark.parametrize("th", [1, 2, 4])
def test_tiled_weight_width_is_wsimd_over_th(th):
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_tiled", "SIMD": 16, "PE": 4, "TH": th})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # WSIMD = (PE*SIMD)/TH; width = WSIMD * w_bits.
    assert op.get_instream_width(pt, ctx, 1) == ((4 * 16) // th) * 8


def test_untiled_weight_folds_as_2d_block():
    # The untiled backend streams weights [SIMD, PE] — a proper 2-D block fold of (MW, MH).
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": 16, "PE": 4})
    assert op.get_folded_input_shape(pt, ctx, 1) == (MW // 16, MH // 4, 16, 4)


def test_tiled_weight_folded_shape_raises_not_fakes():
    # The tiled backend delivers weights as ONE cross-interface expr (PE*SIMD/TH), not a
    # per-dim reshape — so its folded SHAPE raises (width still resolves).
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_tiled", "SIMD": 16, "PE": 4, "TH": 2})
    with pytest.raises(KernelError, match="does not fold a tensor axis"):
        op.get_folded_input_shape(pt, ctx, 1)


def test_TH_is_backend_local():
    # TH exists only on mvau_rtl_tiled; the untiled backend has no TH axis.
    op, ctx = _mvu_op(), _ctx()
    tiled = op.configure(ctx, {"implementation": "mvau_rtl_tiled", "SIMD": 16, "PE": 4, "TH": 2})
    assert "TH" in tiled
    untiled = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": 16, "PE": 4})
    assert "TH" not in untiled
    # Assigning TH under the untiled impl is a caller error (absent axis).
    bad = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": 16, "PE": 4, "TH": 2})
    assert isinstance(bad, Illegal)


# ---------------------------------------------------------------------------
# Cost: the op-level cost_model captures the reduction x output product.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simd,pe,th", [(16, 4, 1), (8, 8, 2), (128, 64, 4)])
def test_exp_cycles_match_finn(simd, pe, th):
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_tiled", "SIMD": simd, "PE": pe, "TH": th})
    expected = (MH // pe) * (MW // simd) * 1 * th
    assert op.get_exp_cycles(pt, ctx) == expected


def test_cost_model_beats_generic_floor():
    # The generic max-over-interfaces floor would give max(MW/SIMD, MH/PE); the true MVU
    # cost is the PRODUCT (MH/PE)*(MW/SIMD). Confirm the cost_model returns the product,
    # which is strictly larger — the reduction coupling the floor cannot see.
    op, ctx = _mvu_op(), _ctx()
    pt = op.configure(ctx, {"implementation": "mvau_rtl_untiled", "SIMD": 16, "PE": 4})
    sf, nf = MW // 16, MH // 4  # 8, 16
    assert op.get_exp_cycles(pt, ctx) == sf * nf       # 128 (product)
    assert sf * nf > max(sf, nf)                        # strictly beats the floor


# ---------------------------------------------------------------------------
# Legality.
# ---------------------------------------------------------------------------


def test_non_dividing_folds_illegal():
    op, ctx = _mvu_op(), _ctx()
    # SIMD=5 does not divide MW=128 (divisor_axis rejects it at resolve).
    assert isinstance(op.configure(ctx, {"SIMD": 5, "PE": 4}), Illegal)
    # PE=7 does not divide MH=64.
    assert isinstance(op.configure(ctx, {"SIMD": 8, "PE": 7}), Illegal)
