############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The tiling ENGINE — ``generate_tiling`` joins the op interfaces' BLOCK extents with an
Backend's STREAM folds (positionally) and derives the fold-dial axes (+ divisor
ranges), divisibility predicates, stream-width deriveds, and the fold map. Exercised
directly, isolated from any op.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    FULL,
    Context,
    Derived,
    Direction,
    Illegal,
    Backend,
    Interface,
    Kernel,
    KernelSchema,
    derive,
    fixed_axis,
    param,
)
from finn.kernels.space.point import Point
from finn.kernels.space.tiling import TileError, generate_tiling


# MVU-shaped interfaces: inp (1, MW), weights (MW, MH), out (1, MH). No semantic role —
# direction only; weight-vs-activation is emergent from context.
MVU_IFACES = (
    Interface("inp", Direction.IN, block=[1, FULL]),
    Interface("weights", Direction.IN, block=[FULL, FULL]),
    Interface("out", Direction.OUT, block=[1, FULL]),
)


def _ctx(mw=128, mh=64):
    return Context(
        shapes={"inp": (1, mw), "weights": (mw, mh), "out": (1, mh)},
        datatypes={
            "inp": DataType["INT8"],
            "weights": DataType["INT8"],
            "out": DataType["INT32"],
        },
    )


# ---------------------------------------------------------------------------
# 1. A bare-dial stream fold generates the divisor-domain axis (from the block extent).
# ---------------------------------------------------------------------------


def test_fold_generates_divisor_axis():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]}
    g = generate_tiling(MVU_IFACES, stream)
    by = {a.name: a for a in g.axes}
    assert set(by) == {"SIMD", "PE"}
    ctx = _ctx(mw=128, mh=64)
    # SIMD folds MW=128 (inp[1] and weights[0], gcd stays 128); PE folds MH=64.
    assert by["SIMD"].domain(Point({}), ctx).values == tuple(
        d for d in range(1, 129) if 128 % d == 0
    )
    assert by["PE"].domain(Point({}), ctx).values[-1] == 64


# ---------------------------------------------------------------------------
# 2. The divisibility predicate fires on a non-dividing dial.
# ---------------------------------------------------------------------------


def test_divisibility_predicate_fires():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"]}
    g = generate_tiling(MVU_IFACES, stream)
    ctx = _ctx(mw=128, mh=64)
    simd_preds = [p for p in g.predicates if "SIMD" in p.description]
    assert simd_preds
    assert simd_preds[0].check(Point({"SIMD": 5}), ctx) is not None
    assert simd_preds[0].check(Point({"SIMD": 16}), ctx) is None


# ---------------------------------------------------------------------------
# 3. A cross-interface expr fold width (not a bare dial) sources no range and does not
#    reshape — but its width still resolves. TH is a real impl axis, not engine-generated.
# ---------------------------------------------------------------------------


def test_expr_fold_is_widthonly_no_range_source():
    # weights delivered as ONE cross-interface expr position (PE*SIMD/TH) rather than
    # per-dim dials: it folds the tensor to a width but not a plain reshape.
    stream = {
        "inp": [1, "SIMD"],
        "out": [1, "PE"],
        "weights": [derive("PE") * derive("SIMD") / param("TH")],
    }
    g = generate_tiling(MVU_IFACES, stream)
    assert {a.name for a in g.axes} == {"SIMD", "PE"}  # TH NOT engine-generated
    assert g.reshapes["weights"] is False              # cross-iface expr ⇒ not a reshape


# ---------------------------------------------------------------------------
# 4. A dial folding multiple interfaces takes the GCD of their block extents.
# ---------------------------------------------------------------------------


def test_multi_fold_gcd_domain():
    ifaces = (
        Interface("a", Direction.IN, block=[1, FULL]),
        Interface("b", Direction.OUT, block=[1, FULL]),
    )
    stream = {"a": [1, "PE"], "b": [1, "PE"]}
    ctx = Context(
        shapes={"a": (1, 48), "b": (1, 64)},
        datatypes={"a": DataType["INT8"], "b": DataType["INT8"]},
    )
    g = generate_tiling(ifaces, stream)
    pe = {a.name: a for a in g.axes}["PE"]
    # PE must divide both 48 and 64 -> divisors of gcd(48,64)=16.
    assert pe.domain(Point({}), ctx).values == (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# 5. The weight port as a proper 2-D block fold (through the Kernel facade).
# ---------------------------------------------------------------------------


def _mvu_kernel():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]}
    impl = Backend(name="mvu", stream=stream)
    return Kernel(
        identity=KernelSchema(name="MVU", interfaces=MVU_IFACES, op_axes=()),
        pool=(impl,),
    )


def test_weight_2d_block_folds_and_width():
    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"backend": "mvu", "SIMD": 16, "PE": 4})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # weight stream WIDTH = SIMD*PE*wbits = 16*4*8 = 512.
    assert k.get_instream_width(pt, ctx, 1) == 16 * 4 * 8
    # 2-D block (MW,MH) streamed [SIMD,PE] -> (MW/SIMD, MH/PE, SIMD, PE).
    assert k.get_folded_input_shape(pt, ctx, 1) == (128 // 16, 64 // 4, 16, 4)


def test_folded_shape_folds_the_named_dim():
    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"backend": "mvu", "SIMD": 16, "PE": 4})
    assert k.get_folded_input_shape(pt, ctx, 0) == (1, 8, 16)   # inp (1,128) folds MW by SIMD
    assert k.get_folded_output_shape(pt, ctx, 0) == (1, 16, 4)  # out (1,64) folds MH by PE


# ---------------------------------------------------------------------------
# 6. Cost: the reduction product falls out of the generic floor — no cost_model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_exp_cycles_is_reduction_product_from_floor(simd, pe):
    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"backend": "mvu", "SIMD": simd, "PE": pe})
    # weights = MW*MH/(SIMD*PE) = sf*nf is the largest interface term; no override needed.
    assert k.get_exp_cycles(pt, ctx) == (128 // simd) * (64 // pe)


# ---------------------------------------------------------------------------
# 7. role implies direction; index derived from position.
# ---------------------------------------------------------------------------


def test_direction_is_declared_not_role():
    # Direction is an explicit declared field (a node-slot fact), no longer derived from a
    # semantic role. An interface carries no role at all — weight-vs-activation is emergent.
    assert Interface("a", Direction.IN).direction == Direction.IN
    assert Interface("w", Direction.IN).direction == Direction.IN
    assert Interface("o", Direction.OUT).direction == Direction.OUT


def test_index_derived_from_position():
    k = _mvu_kernel()
    # inputs() are position-ordered: inp (index 0), weights (index 1).
    assert [i.name for i in k.inputs()] == ["inp", "weights"]
    assert k.get_input_datatype(_ctx(), 0) == DataType["INT8"]   # inp
    assert k.get_input_datatype(_ctx(), 1) == DataType["INT8"]   # weights


# ---------------------------------------------------------------------------
# 8. dtype_source: the width uses a derived dtype, not the raw tensor dtype.
# ---------------------------------------------------------------------------


def test_width_uses_dtype_source():
    ifaces = (
        Interface("inp", Direction.IN, block=[1, FULL]),
        Interface("out", Direction.OUT, block=[1, FULL], dtype_source="acc"),
    )
    impl = Backend(name="k", stream={"inp": [1, "SIMD"], "out": [1, "PE"]})
    k = Kernel(
        identity=KernelSchema(
            name="K",
            interfaces=ifaces,
            op_derived=(Derived("acc", lambda p, ctx: DataType["INT16"]),),
        ),
        pool=(impl,),
    )
    ctx = Context(
        shapes={"inp": (1, 128), "out": (1, 64)},
        datatypes={"inp": DataType["INT8"], "out": DataType["INT32"]},
    )
    pt = k.configure(ctx, {"backend": "k", "SIMD": 16, "PE": 4})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # PE=4 elements * INT16 (from dtype_source "acc"), NOT INT32 (the tensor dtype).
    assert k.get_outstream_width(pt, ctx, 0) == 4 * 16
    assert pt["stream_width.out"] == 4 * 16
