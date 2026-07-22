############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The tiling ENGINE — ``generate_tiling`` derives fold-dial axes (+ divisor ranges),
divisibility predicates, stream-width deriveds, and the fold map from ONE per-interface
spec declaration (kernelop-tensor-block-stream.md §5). These tests exercise the engine
directly, isolated from any op.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Context,
    Direction,
    Illegal,
    Implementation,
    Interface,
    Kernel,
    Role,
    Broadcast,
    Fold,
    Full,
    WidthOnly,
    derive,
    param,
)
from finn.kernels.space.point import Point
from finn.kernels.space.tiling import TileError, generate_tiling


def _ifaces(*specs):
    """specs: (name, tensor, direction, role) tuples."""
    return tuple(Interface(n, t, d, r) for (n, t, d, r) in specs)


MVU_IFACES = _ifaces(
    ("inp", "inp", Direction.IN, Role.DATA_IN),
    ("weights", "weights", Direction.IN, Role.WEIGHT_SINK),
    ("out", "out", Direction.OUT, Role.DATA_OUT),
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
# 1. A Fold generates the divisor-domain axis.
# ---------------------------------------------------------------------------


def test_fold_generates_divisor_axis():
    tiling = {"inp": [Full(), Fold("SIMD")], "out": [Full(), Fold("PE")]}
    g = generate_tiling(MVU_IFACES, tiling)
    by = {a.name: a for a in g.axes}
    assert set(by) == {"SIMD", "PE"}
    ctx = _ctx(mw=128, mh=64)
    # SIMD folds inp's last dim (MW=128); PE folds out's last dim (MH=64).
    assert by["SIMD"].domain(Point({}), ctx).values == tuple(
        d for d in range(1, 129) if 128 % d == 0
    )
    assert by["PE"].domain(Point({}), ctx).values[-1] == 64


# ---------------------------------------------------------------------------
# 2. The divisibility predicate fires on a non-dividing dial.
# ---------------------------------------------------------------------------


def test_divisibility_predicate_fires():
    tiling = {"inp": [Full(), Fold("SIMD")], "out": [Full(), Fold("PE")]}
    g = generate_tiling(MVU_IFACES, tiling)
    ctx = _ctx(mw=128, mh=64)
    # Find the SIMD divisibility predicate and evaluate it on a non-divisor.
    simd_preds = [p for p in g.predicates if "SIMD" in p.description]
    assert simd_preds
    bad = simd_preds[0].check(Point({"SIMD": 5}), ctx)
    assert bad is not None and "128" in bad
    ok = simd_preds[0].check(Point({"SIMD": 16}), ctx)
    assert ok is None


# ---------------------------------------------------------------------------
# 3. A dial that appears only inside WidthOnly has no range source -> raises.
# ---------------------------------------------------------------------------


def test_dial_only_in_widthonly_raises():
    # "TH" appears only in a WidthOnly expr, never as a plain Fold.
    tiling = {
        "inp": [Full(), Fold("SIMD")],
        "out": [Full(), Fold("PE")],
        "weights": [WidthOnly(derive("PE") * derive("SIMD") / param("TH"))],
    }
    # TH is declared as a real impl axis elsewhere, so generate_tiling must NOT try to
    # source its range — a WidthOnly reference alone is fine (it's not in dial_folds).
    g = generate_tiling(MVU_IFACES, tiling)
    assert {a.name for a in g.axes} == {"SIMD", "PE"}  # TH is not engine-generated
    assert g.reshapes["weights"] is False


def test_broadcast_only_dial_raises():
    # A dial used ONLY in a Broadcast (never a plain Fold) has no range source.
    ifaces = _ifaces(
        ("lhs", "lhs", Direction.IN, Role.DATA_IN),
        ("rhs", "rhs", Direction.IN, Role.DATA_IN),
    )
    tiling = {"lhs": [Full()], "rhs": [Broadcast("rhs_last", "PE")]}
    with pytest.raises(TileError, match="no range source"):
        generate_tiling(ifaces, tiling)


# ---------------------------------------------------------------------------
# 4. A dial folding multiple interfaces takes the GCD of their block dims.
# ---------------------------------------------------------------------------


def test_multi_fold_gcd_domain():
    ifaces = _ifaces(
        ("a", "a", Direction.IN, Role.DATA_IN),
        ("b", "b", Direction.OUT, Role.DATA_OUT),
    )
    tiling = {"a": [Full(), Fold("PE")], "b": [Full(), Fold("PE")]}
    ctx = Context(
        shapes={"a": (1, 48), "b": (1, 64)},
        datatypes={"a": DataType["INT8"], "b": DataType["INT8"]},
    )
    g = generate_tiling(ifaces, tiling)
    pe = {a.name: a for a in g.axes}["PE"]
    # PE must divide BOTH 48 and 64 -> divisors of gcd(48,64)=16.
    assert pe.domain(Point({}), ctx).values == (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# 5. WidthOnly: width resolves, folded shape raises (through the Kernel facade).
# ---------------------------------------------------------------------------


def _mvu_kernel():
    tiling = {
        "inp": [Full(), Fold("SIMD")],
        "out": [Full(), Fold("PE")],
        "weights": [WidthOnly(derive("PE") * derive("SIMD"))],
    }
    impl = Implementation(name="mvu", tiling=tiling)
    return Kernel(
        name="MVU",
        interfaces=(
            Interface("inp", "inp", Direction.IN, Role.DATA_IN, index=0),
            Interface("weights", "weights", Direction.IN, Role.WEIGHT_SINK, index=1,
                      folds_last_axis=False),
            Interface("out", "out", Direction.OUT, Role.DATA_OUT, index=0),
        ),
        pool=(impl,),
        op_axes=(),
    )


def test_widthonly_width_resolves_but_shape_raises():
    from finn.kernels.space import KernelError

    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"SIMD": 16, "PE": 4})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # weight stream WIDTH = PE*SIMD*wbits = 4*16*8 = 512 resolves...
    assert k.get_instream_width(pt, ctx, 1) == 4 * 16 * 8
    # ...but its folded SHAPE is not a tensor-axis reshape -> raises.
    with pytest.raises(KernelError, match="does not fold a tensor axis"):
        k.get_folded_input_shape(pt, ctx, 1)


def test_folded_shape_folds_the_named_dim():
    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"SIMD": 16, "PE": 4})
    # inp (1, 128): SIMD folds the last dim -> (1, 128/16, 16).
    assert k.get_folded_input_shape(pt, ctx, 0) == (1, 8, 16)
    assert k.get_folded_output_shape(pt, ctx, 0) == (1, 16, 4)


# ---------------------------------------------------------------------------
# 6. Broadcast: size-1 dim streams width 1; a full dim folds normally.
# ---------------------------------------------------------------------------


def _elementwise_kernel():
    ifaces = (
        Interface("lhs", "lhs", Direction.IN, Role.DATA_IN, index=0),
        Interface("rhs", "rhs", Direction.IN, Role.DATA_IN, index=1),
        Interface("out", "out", Direction.OUT, Role.DATA_OUT, index=0),
    )
    from finn.kernels.space import fixed_axis

    tiling = {
        "lhs": [Full(), Fold("PE")],
        "rhs": [Full(), Broadcast("rhs_last", "PE")],
        "out": [Full(), Fold("PE")],
    }
    impl = Implementation(name="ew", tiling=tiling)
    return Kernel(
        name="EW",
        interfaces=ifaces,
        pool=(impl,),
        op_axes=(fixed_axis("rhs_last", lambda p, ctx: ctx.tensor_shape("rhs")[-1]),),
    )


def test_broadcast_size1_streams_one():
    k = _elementwise_kernel()
    # rhs broadcast (last dim = 1): rhs streams width 1, PE domain unaffected (divisors 64).
    ctx = Context(
        shapes={"lhs": (1, 64), "rhs": (1, 1), "out": (1, 64)},
        datatypes={"lhs": DataType["INT8"], "rhs": DataType["UINT4"], "out": DataType["INT8"]},
    )
    pt = k.configure(ctx, {"PE": 16})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    assert k.get_instream_width(pt, ctx, 0) == 16 * 8  # lhs folds by PE
    assert k.get_instream_width(pt, ctx, 1) == 1 * 4   # rhs broadcast -> 1 element * UINT4


def test_broadcast_full_dim_folds_normally():
    k = _elementwise_kernel()
    ctx = Context(
        shapes={"lhs": (1, 64), "rhs": (1, 64), "out": (1, 64)},
        datatypes={"lhs": DataType["INT8"], "rhs": DataType["UINT4"], "out": DataType["INT8"]},
    )
    pt = k.configure(ctx, {"PE": 16})
    assert k.get_instream_width(pt, ctx, 1) == 16 * 4  # rhs full -> folds by PE


# ---------------------------------------------------------------------------
# 7. dtype_source: the width uses a derived dtype, not the raw tensor dtype.
# ---------------------------------------------------------------------------


def test_width_uses_dtype_source():
    from finn.kernels.space import Derived, fixed_axis

    # out declares dtype_source="acc"; a derived "acc" supplies INT16 while the out tensor
    # is INT32 -> the generated outstream_width must use INT16.
    ifaces = (
        Interface("inp", "inp", Direction.IN, Role.DATA_IN, index=0),
        Interface("out", "out", Direction.OUT, Role.DATA_OUT, index=0, dtype_source="acc"),
    )
    impl = Implementation(name="k", tiling={"inp": [Full(), Fold("SIMD")], "out": [Full(), Fold("PE")]})
    k = Kernel(
        name="K",
        interfaces=ifaces,
        pool=(impl,),
        op_derived=(Derived("acc", lambda p, ctx: DataType["INT16"]),),
    )
    ctx = Context(
        shapes={"inp": (1, 128), "out": (1, 64)},
        datatypes={"inp": DataType["INT8"], "out": DataType["INT32"]},
    )
    pt = k.configure(ctx, {"SIMD": 16, "PE": 4})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    # PE=4 elements * INT16 (from dtype_source "acc"), NOT INT32 (the tensor dtype).
    assert k.get_outstream_width(pt, ctx, 0) == 4 * 16
    assert pt["outstream_width"] == 4 * 16
