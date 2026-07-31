############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Tiling — TileExpr algebra + the BLOCK→STREAM engine (T1-T7).

Two halves. The expr AST (Const/Ref/Mul/Div/BroadcastAware) is exact + introspectable:
``.deps()`` returns exactly the point keys read (drives ordering), Div raises on a
non-exact fold, BroadcastAware replicates a size-1 axis. The engine (``generate_tiling``)
joins an op's BLOCK extents with a Backend's STREAM folds and derives fold-dial axes
(divisor-of-GCD domains), divisibility predicates, per-interface stream-width deriveds,
and the fold map — the one-declaration-generates-four contract. Folded shapes are a plain
reshape iff all folds are named dials/1; a cross-interface width expr raises.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.point import Illegal, Point
from finn.kernels.model.backend import Backend, ports_from
from finn.kernels.model.kernel import InterfaceSchema, Kernel, KernelSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import (
    FULL,
    Const,
    TileError,
    broadcast_aware,
    const,
    derive,
    entry_deps,
    eval_entry,
    generate_tiling,
    param,
)


# ===========================================================================
# T2/T3 — TileExpr algebra: exact folds, introspectable deps, broadcast.
# ===========================================================================


def test_mvu_tiled_weight_port_wsimd():
    # A point as mvau_rtl_tiled would resolve it: PE, SIMD folds + backend-local TH.
    p = Point({"PE": 4, "SIMD": 8, "TH": 2})
    wsimd = derive("PE") * derive("SIMD") / param("TH")
    assert eval_entry(wsimd, p) == (4 * 8) // 2  # == 16


def test_mvu_untiled_weight_port_is_pe_simd():
    p = Point({"PE": 4, "SIMD": 8})
    assert eval_entry(derive("PE") * derive("SIMD"), p) == 32


def test_weight_port_deps_are_backend_local():
    wsimd = derive("PE") * derive("SIMD") / param("TH")
    assert wsimd.deps() == frozenset({"PE", "SIMD", "TH"})
    assert entry_deps("SIMD") == frozenset({"SIMD"})
    assert entry_deps(1) == frozenset()


def test_div_requires_exact_fold():
    p = Point({"PE": 3, "SIMD": 8, "TH": 5})
    with pytest.raises(TileError, match="not an exact integer fold"):
        eval_entry(derive("PE") * derive("SIMD") / param("TH"), p)


def test_div_by_zero_raises():
    p = Point({"PE": 4, "TH": 0})
    with pytest.raises(TileError, match="zero"):
        eval_entry(derive("PE") / param("TH"), p)


def test_broadcast_aware_folds_when_not_broadcast():
    p = Point({"rhs_len": 128, "PE": 16})
    assert eval_entry(broadcast_aware("rhs_len", derive("PE")), p) == 16


def test_broadcast_aware_replicates_when_size_one():
    p = Point({"rhs_len": 1, "PE": 16})
    assert eval_entry(broadcast_aware("rhs_len", derive("PE")), p) == 1


def test_broadcast_aware_deps_include_extent():
    expr = broadcast_aware("rhs_len", derive("PE"))
    assert expr.deps() == frozenset({"rhs_len", "PE"})


def test_absent_dep_is_diagnosed():
    p = Point({"PE": 4})  # SIMD guarded out / not declared
    with pytest.raises(TileError, match="SIMD"):
        eval_entry(derive("PE") * derive("SIMD"), p)


def test_non_integer_value_rejected():
    p = Point({"PE": "wide"})
    with pytest.raises(TileError, match="expected an integer"):
        eval_entry(derive("PE"), p)


def test_bare_int_and_name_coerce():
    p = Point({"SIMD": 8})
    assert eval_entry("SIMD", p) == 8
    assert eval_entry(4, p) == 4
    assert isinstance(const(4), Const)


def test_rmul_and_const_compose():
    p = Point({"PE": 8})
    assert eval_entry(2 * derive("PE"), p) == 16


# ===========================================================================
# T1/T6/T7 — the tiling engine: generate_tiling.
# ===========================================================================

# MVU-shaped interfaces: inp (1, MW), weights (MW, MH), out (1, MH). Direction only —
# weight-vs-activation is emergent from context.
MVU_IFACES = (
    InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
    InterfaceSchema("weights", Direction.IN, block=[FULL, FULL]),
    InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
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


def test_fold_generates_divisor_axis():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]}
    g = generate_tiling(MVU_IFACES, stream)
    by = {a.name: a for a in g.axes}
    assert set(by) == {"SIMD", "PE"}
    ctx = _ctx(mw=128, mh=64)
    # SIMD folds MW=128; PE folds MH=64.
    assert by["SIMD"].domain(Point({}), ctx).values == tuple(
        d for d in range(1, 129) if 128 % d == 0
    )
    assert by["PE"].domain(Point({}), ctx).values[-1] == 64


def test_divisibility_predicate_fires():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"]}
    g = generate_tiling(MVU_IFACES, stream)
    ctx = _ctx(mw=128, mh=64)
    simd_preds = [p for p in g.predicates if "SIMD" in p.description]
    assert simd_preds
    assert simd_preds[0].check(Point({"SIMD": 5}), ctx) is not None
    assert simd_preds[0].check(Point({"SIMD": 16}), ctx) is None


def test_expr_fold_is_widthonly_no_range_source():
    # weights delivered as ONE cross-interface expr position (PE*SIMD/TH): folds a width
    # but not a plain reshape, and sources no dial range (TH is a real impl axis).
    stream = {
        "inp": [1, "SIMD"],
        "out": [1, "PE"],
        "weights": [derive("PE") * derive("SIMD") / param("TH")],
    }
    g = generate_tiling(MVU_IFACES, stream)
    assert {a.name for a in g.axes} == {"SIMD", "PE"}  # TH NOT engine-generated
    assert g.reshapes["weights"] is False


def test_multi_fold_gcd_domain():
    ifaces = (
        InterfaceSchema("a", Direction.IN, block=[1, FULL]),
        InterfaceSchema("b", Direction.OUT, block=[1, FULL]),
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


def test_stream_names_unknown_interface_raises():
    with pytest.raises(TileError, match="not in the kernel"):
        generate_tiling(MVU_IFACES, {"ghost": [1, "PE"]})


# ===========================================================================
# T4/T5/T7 — folded shapes + widths through the Kernel facade.
# ===========================================================================


def _mvu_kernel():
    stream = {"inp": [1, "SIMD"], "out": [1, "PE"], "weights": ["SIMD", "PE"]}
    impl = Backend(name="mvu", ports=ports_from(stream=stream))
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


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_exp_cycles_is_reduction_product_from_floor(simd, pe):
    k, ctx = _mvu_kernel(), _ctx(mw=128, mh=64)
    pt = k.configure(ctx, {"backend": "mvu", "SIMD": simd, "PE": pe})
    # weights = MW*MH/(SIMD*PE) = sf*nf is the largest interface term; no override needed.
    assert k.get_exp_cycles(pt, ctx) == (128 // simd) * (64 // pe)


def test_direction_is_declared_not_role():
    assert InterfaceSchema("a", Direction.IN).direction == Direction.IN
    assert InterfaceSchema("o", Direction.OUT).direction == Direction.OUT


def test_index_derived_from_position():
    k = _mvu_kernel()
    assert [i.name for i in k.inputs()] == ["inp", "weights"]
    assert k.get_input_datatype(_ctx(), 0) == DataType["INT8"]   # inp
    assert k.get_input_datatype(_ctx(), 1) == DataType["INT8"]   # weights


def test_width_uses_dtype_source():
    ifaces = (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
        InterfaceSchema("out", Direction.OUT, block=[1, FULL], dtype_source="acc"),
    )
    impl = Backend(name="k", ports=ports_from(stream={"inp": [1, "SIMD"], "out": [1, "PE"]}))
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


# ===========================================================================
# Interface dtype/memory fields — direction-exclusivity enforced at assembly.
# ===========================================================================


def _kernel_with_port(iface_name, direction, **port_kwargs):
    from finn.kernels.model.backend import Interface
    from finn.kernels.model.kernel import KernelError

    ifaces = (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )
    impl = Backend(
        name="k",
        ports={iface_name: Interface(**port_kwargs)},
    )
    return Kernel(identity=KernelSchema(name="K", interfaces=ifaces), pool=(impl,))


def test_derived_dtype_on_input_port_rejected():
    from finn.kernels.model.kernel import KernelError

    with pytest.raises(KernelError, match="derived_dtype set on INPUT"):
        _kernel_with_port("inp", Direction.IN, derived_dtype=DataType["INT8"])


def test_accepted_dtypes_on_output_port_rejected():
    from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
    from finn.kernels.model.kernel import KernelError

    with pytest.raises(KernelError, match="accepted_dtypes set on OUTPUT"):
        _kernel_with_port(
            "out", Direction.OUT, accepted_dtypes=DatatypeSupport(kind=DatatypeKind.INTEGER)
        )


def test_mem_modes_on_output_port_rejected():
    from finn.kernels.model.kernel import KernelError

    with pytest.raises(KernelError, match="mem_modes set on OUTPUT"):
        _kernel_with_port("out", Direction.OUT, mem_modes={"embedded"})


def test_derived_dtype_on_output_and_accepted_on_input_ok():
    from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
    from finn.kernels.model.backend import Interface

    ifaces = (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )
    impl = Backend(
        name="k",
        ports={
            "inp": Interface(accepted_dtypes=DatatypeSupport(kind=DatatypeKind.INTEGER)),
            "out": Interface(derived_dtype=DataType["INT16"]),
        },
    )
    # Constructs without raising — correct-direction facts are legal.
    Kernel(identity=KernelSchema(name="K", interfaces=ifaces), pool=(impl,))
