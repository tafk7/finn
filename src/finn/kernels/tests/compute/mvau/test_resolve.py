############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU resolve — the depth-first flagship (the richest op, end to end).

Exercises E (engine) + S (pool selection) + T (tiling) + D (delivery) through the real
mvau_kernel: guarded axes, device pool pruning (DSP58/NUM_LANES/weight-width feasibility),
forced-derived values (dsp_primitive/accDataType), URAM combination predicate, data-
dependent accumulator dtype (static vs runtime-writeable), composability (a 4th bundle
adds with zero edits; registry self-registration), the optional thresholds interface,
consumption-mode gating (DSP cores are stream-weight-only), and per-interface cadence.

Also folds in the F2/D-R5 feasibility contract: each backend's feasibility is the single
source of truth leveraged at claim time (the brittle op.py source-grep guard is dropped).
"""

import itertools
import logging

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.kernels.engine.point import AbsentAxisError, Illegal, Point
from finn.kernels.engine.context import Context
from finn.kernels.engine.resolve import resolve
from finn.kernels.compute.mvau import (
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    mvau_pool,
    mvau_space,
)
from finn.kernels.compute.mvau.op import MvauKernelOp, mvau_kernel
from finn.kernels.dataflow.parameters.names import (
    DECOUPLED as PARAM_DECOUPLED,
    EMBEDDED as PARAM_EMBEDDED,
    WEIGHTS as PARAM_WEIGHTS,
)
from finn.kernels.model.param_names import (
    pumped_memory_key,
    ram_style_key,
    runtime_writeable_key,
    param_datatype_key,
    topology_key,
)
from finn.util.basic import is_versal


def _language_of(point, pool=None):
    by_name = {b.name: b for b in (pool if pool is not None else mvau_pool())}
    return by_name[point["backend"]].language


# The parameters pool is composed per parameter interface; MVAU's live one is ``weights``.
PARAM_TOPOLOGY = topology_key(PARAM_WEIGHTS)
PARAM_RAM_STYLE = ram_style_key(PARAM_WEIGHTS)
PARAM_RUNTIME_WRITEABLE = runtime_writeable_key(PARAM_WEIGHTS)
PARAM_PUMPED_MEMORY = pumped_memory_key(PARAM_WEIGHTS)
PARAM_DTYPE = param_datatype_key(PARAM_WEIGHTS)

SEVEN_SERIES = "xc7z020clg400-1"  # Zynq-7000, DSP48E1, not Versal
ULTRASCALE = "xcku040-ffva1156-2-e"  # Kintex UltraScale, DSP48E2, not Versal
VERSAL = "xcvc1902-vsva2197-2MP-e-S"  # Versal, DSP58


@pytest.fixture
def schema():
    return mvau_space()


def narrow_weights(shape=(6, 8), wdt="INT4"):
    lo = int(DataType[wdt].min()) + 1
    hi = int(DataType[wdt].max())
    rng = np.random.RandomState(0)
    return rng.randint(lo, hi + 1, size=shape).astype(np.float32)


def make_context(fpgapart=SEVEN_SERIES, weights=None, wdt="INT4", idt="INT4"):
    if weights is None:
        rng = np.random.RandomState(0)
        weights = rng.randint(-8, 8, size=(6, 8)).astype(np.float32)
    return Context(
        shapes={"weights": weights.shape, "inp": (1, weights.shape[0]), "out": (1, weights.shape[1])},
        datatypes={"weights": DataType[wdt], "inp": DataType[idt], "out": DataType["INT16"]},
        initializers={"weights": weights},
        fpgapart=fpgapart,
        clk_ns=5.0,
    )


def base_assignment(**overrides):
    a = {"backend": MVAU_HLS, "PE": 4, "SIMD": 2, "mem_mode": "internal_decoupled"}
    a.update(overrides)
    return _translate_parameters(a)


_MEM_MODE_TO_TOPOLOGY = {
    "internal_embedded": PARAM_EMBEDDED,
    "internal_decoupled": PARAM_DECOUPLED,
}
_DELIVERY_KEYS = {
    "ram_style": PARAM_RAM_STYLE,
    "runtime_writeable_weights": PARAM_RUNTIME_WRITEABLE,
    "pumpedMemory": PARAM_PUMPED_MEMORY,
}


def _translate_parameters(a):
    out = {}
    for k, v in a.items():
        if k == "noActivation":
            continue
        if k == "mem_mode":
            out[PARAM_TOPOLOGY] = _MEM_MODE_TO_TOPOLOGY[v]
        elif k in _DELIVERY_KEYS:
            out[_DELIVERY_KEYS[k]] = v
        else:
            out[k] = v
    return out


# --- #1 guarded axis absent, reading errors --------------------------------


def test_ram_style_absent_when_not_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r, Point)
    assert PARAM_RAM_STYLE not in r
    assert PARAM_RUNTIME_WRITEABLE not in r
    with pytest.raises(AbsentAxisError):
        _ = r[PARAM_RAM_STYLE]


def test_ram_style_present_when_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_decoupled"))
    assert isinstance(r, Point)
    assert PARAM_RAM_STYLE in r
    assert r[PARAM_RAM_STYLE] == "auto"


# --- #2 device prunes the implementation pool ------------------------------


def test_packed_impl_illegal_on_seven_series(schema):
    r = resolve(
        schema,
        make_context(SEVEN_SERIES, weights=narrow_weights()),
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r, Illegal)
    assert any("DSP58" in reason for reason in r.reasons)


def test_other_impls_remain_on_seven_series(schema):
    r_sv = resolve(
        schema,
        make_context(SEVEN_SERIES, weights=narrow_weights()),
        base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r_sv, Point)
    r_hls = resolve(schema, make_context(SEVEN_SERIES), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r_hls, Point)


def test_packed_impl_legal_on_versal(schema):
    r = resolve(
        schema,
        make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8"),
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r, Point)
    assert r.dsp_primitive == "DSP58"


def test_packed_impl_illegal_on_versal_with_wide_weights(schema):
    ctx = make_context(VERSAL, weights=narrow_weights(shape=(6, 8), wdt="INT16"), wdt="INT16")
    r = resolve(
        schema, ctx,
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r, Illegal)
    assert any("weight_width<=8" in reason for reason in r.reasons)
    r2 = resolve(
        schema, ctx,
        base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r2, Point)


# --- #3 forced-derived values computed, never axes -------------------------


def test_dsp_primitive_forced_from_fpgapart(schema):
    for part, expected in [(SEVEN_SERIES, "DSP48E1"), (ULTRASCALE, "DSP48E2"), (VERSAL, "DSP58")]:
        r = resolve(
            schema,
            make_context(part, weights=narrow_weights()),
            base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_decoupled"),
        )
        assert isinstance(r, Point), r
        assert r.dsp_primitive == expected


def test_forced_and_derived_names_are_not_axes(schema):
    axis_names = schema.axis_names
    for name in ("dsp_primitive", "accDataType", "language", "SEGMENTLEN"):
        assert name not in axis_names, f"{name} must be Derived, not an Axis"


def test_acc_datatype_data_dependent_static(schema):
    # accDataType sizes from the storage owner's PUBLISHED range: static weights -> the owner
    # is visible -> per-value narrowing, so a smaller matrix yields a smaller accumulator.
    small = np.ones((6, 8), dtype=np.float32)
    r_small = resolve(schema, make_context(weights=small), base_assignment())
    big = np.full((6, 8), -8.0, dtype=np.float32)
    r_big = resolve(schema, make_context(weights=big), base_assignment())
    assert isinstance(r_small, Point) and isinstance(r_big, Point)
    # The mechanism: the owner published a visible ParamDatatype; acc read it, not raw weights.
    assert r_small[PARAM_DTYPE].values_visible is True
    assert r_small.accDataType.bitwidth() < r_big.accDataType.bitwidth()


def test_acc_datatype_worst_case_when_runtime_writeable(schema):
    # Runtime-writable -> the storage owner is BLIND -> it publishes the dtype envelope
    # (values_visible=False), so acc sizes worst-case, never tighter than the static case.
    small = np.ones((6, 8), dtype=np.float32)
    r_static = resolve(schema, make_context(weights=small), base_assignment(noActivation=1))
    r_rtw = resolve(
        schema,
        make_context(weights=small),
        base_assignment(noActivation=1, runtime_writeable_weights=1),
    )
    assert isinstance(r_static, Point) and isinstance(r_rtw, Point)
    # The mechanism: the runtime-writable owner withheld trust, flipping acc to the envelope.
    assert r_static[PARAM_DTYPE].values_visible is True
    assert r_rtw[PARAM_DTYPE].values_visible is False
    assert r_rtw.accDataType.bitwidth() >= r_static.accDataType.bitwidth()


def test_acc_datatype_resolves_after_param_datatype(schema):
    # R-order guard: accDataType reads a parameters-pool derived (the param datatype), so the
    # unified topo-sort MUST order it after that ParamDatatype on the composed MVAU point. Without
    # A1's derived deps this would resolve in list order (acc before storage) and crash.
    names = [d.name for d in schema.ordered_derived()]
    assert names.index("accDataType") > names.index(PARAM_DTYPE)
    # And the out-port stream width, which transitively reads acc under noActivation, likewise.
    assert names.index("stream_width.out") > names.index(PARAM_DTYPE)


# --- #4 the combination predicate (config + device) fires ------------------


def test_uram_requires_runtime_writeable_on_ultrascale(schema):
    illegal = resolve(
        schema, make_context(ULTRASCALE),
        base_assignment(ram_style="ultra", runtime_writeable_weights=0),
    )
    assert isinstance(illegal, Illegal)
    assert any("URAM" in reason for reason in illegal.reasons)
    legal = resolve(
        schema, make_context(ULTRASCALE),
        base_assignment(ram_style="ultra", runtime_writeable_weights=1),
    )
    assert isinstance(legal, Point)


def test_uram_ok_on_versal_without_runtime_writeable(schema):
    r = resolve(
        schema, make_context(VERSAL),
        base_assignment(ram_style="ultra", runtime_writeable_weights=0),
    )
    assert isinstance(r, Point)


# --- #5 legal -> Point with derived; illegal -> Illegal --------------------


def test_legal_point_carries_derived(schema):
    from finn.kernels.model.fold_depth import weight_fold_depth
    from finn.kernels.compute.mvau.op import WEIGHTS

    ctx = make_context()
    r = resolve(schema, ctx, base_assignment(PE=4, SIMD=2))
    assert isinstance(r, Point)
    assert weight_fold_depth(r, ctx, WEIGHTS) == 6 * 8 // (4 * 2)
    assert _language_of(r) == "hls"
    from finn.kernels.compute.mvau.op import mvau_out_dtype, OUTPUT
    from finn.kernels.engine.datatype_spec import resolve_datatype_spec

    odt = resolve_datatype_spec(mvau_out_dtype(), iface=OUTPUT, point=r, context=ctx)
    assert r["stream_width.out"] == odt.bitwidth() * 4


def test_kernel_attrs_pinned_onto_point_never_explored(schema):
    # A kernel_attr reaches the Point at its assigned value (backends read it), but resolve
    # never treats it as an explorable dial — it is pinned to the assignment/default.
    ctx = make_context()
    r = resolve(schema, ctx, base_assignment(ActVal=7, mlo_max_iter=3))
    assert isinstance(r, Point)
    assert r["ActVal"] == 7
    assert r["mlo_max_iter"] == 3
    # Unassigned ⇒ the declared default (0), not a swept value.
    r0 = resolve(schema, ctx, base_assignment())
    assert r0["ActVal"] == 0
    assert r0["mlo_max_iter"] == 0


def test_domain_violation_illegal(schema):
    r = resolve(schema, make_context(), base_assignment(PE=5))  # 5 ∤ MH=8
    assert isinstance(r, Illegal)
    assert "PE" in r.reasons[0]


def test_predicate_violation_illegal(schema):
    r = resolve(
        schema, make_context(),
        base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp", SIMD=1, pumpedCompute=1),
    )
    assert isinstance(r, Illegal)
    assert any("pumpedCompute" in reason for reason in r.reasons)


# --- #6 guards compress the space ------------------------------------------


def test_guards_compress_the_space(schema):
    ctx = make_context()
    topologies = ["internal_embedded", "internal_decoupled"]
    ram_styles = ["auto", "block", "distributed"]
    rw = [0, 1]
    naive = len(topologies) * len(ram_styles) * len(rw)
    seen = set()
    for mm, rs, w in itertools.product(topologies, ram_styles, rw):
        assignment = base_assignment(mem_mode=mm)
        if mm == "internal_decoupled":
            assignment[PARAM_RAM_STYLE] = rs
            assignment[PARAM_RUNTIME_WRITEABLE] = w
        r = resolve(schema, ctx, assignment)
        if isinstance(r, Point):
            seen.add((r[PARAM_TOPOLOGY], r.get(PARAM_RAM_STYLE), r.get(PARAM_RUNTIME_WRITEABLE)))
    dependent = len(seen)
    assert dependent == 6 + 1
    assert dependent < naive


# --- F1 packed feasibility computes NUM_LANES for real ---------------------


def test_packed_num_lanes_gate(schema):
    ctx = make_context(VERSAL, weights=narrow_weights(wdt="INT4"), wdt="INT4", idt="INT4")
    r = resolve(
        schema, ctx,
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r, Illegal)
    assert any("NUM_LANES" in reason for reason in r.reasons)
    r_sv = resolve(
        schema, ctx,
        base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r_sv, Point)


def test_packed_num_lanes_ok_for_int8(schema):
    ctx = make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8")
    r = resolve(
        schema, ctx,
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_decoupled"),
    )
    assert isinstance(r, Point)


# --- F3 true-binary rejection reads the full condition ---------------------


def test_hls_rejects_binary_weights(schema):
    ctx = make_context(weights=np.ones((6, 8), dtype=np.float32), wdt="BINARY", idt="INT4")
    r = resolve(schema, ctx, base_assignment(backend=MVAU_HLS))
    assert isinstance(r, Illegal)
    assert any("binary" in reason.lower() for reason in r.reasons)


# --- composability: a 4th bundle adds with zero edits ----------------------


def test_fourth_implementation_composes_additively():
    from finn.kernels.engine.predicate import predicate
    from finn.kernels.model.backend import Backend, ports_from
    from finn.kernels.compute.mvau import mvau_kernel, mvau_pool
    from finn.kernels.compute.mvau.op import COMPUTE_STREAM

    @predicate("mvau_lut_rtl targets non-Versal parts only (hypothetical)")
    def lut_rtl_feasible(p, ctx):
        if is_versal(ctx.fpgapart):
            return "mvau_lut_rtl targets non-Versal parts only (hypothetical)"
        return None

    lut_rtl = Backend(
        name="mvau_lut_rtl", language="rtl",
        axes=(), predicates=(lut_rtl_feasible,), sources=("mvu_lut.sv",),
        ports=ports_from(stream=COMPUTE_STREAM),
    )
    from dataclasses import replace

    base = mvau_kernel()
    kernel4 = replace(base, pool=mvau_pool() + (lut_rtl,))
    schema4 = kernel4.compile()

    legal = resolve(
        schema4, make_context(SEVEN_SERIES),
        base_assignment(backend="mvau_lut_rtl", mem_mode="internal_embedded"),
    )
    assert isinstance(legal, Point)
    assert _language_of(legal, kernel4.pool) == "rtl"
    assert legal.sources == ("mvu_lut.sv",)

    illegal = resolve(
        schema4, make_context(VERSAL),
        base_assignment(backend="mvau_lut_rtl", mem_mode="internal_embedded"),
    )
    assert isinstance(illegal, Illegal)
    assert any("non-Versal" in reason for reason in illegal.reasons)

    r_hls = resolve(schema4, make_context(SEVEN_SERIES), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r_hls, Point)
    assert _language_of(r_hls, kernel4.pool) == "hls"


def test_registry_makes_addition_structural():
    from finn.kernels.model.backend import Backend, ports_from
    from finn.kernels.compute.mvau import mvau_pool, mvau_space
    from finn.kernels.compute.mvau.op import COMPUTE_STREAM
    from finn.kernels.compute.mvau.registry import register, unregister

    before = {b.name for b in mvau_pool()}
    assert "mvau_stub_backend" not in before

    @register
    def _stub_bundle():
        return Backend(name="mvau_stub_backend", sources=("stub.sv",),
                       ports=ports_from(stream=COMPUTE_STREAM))

    try:
        after = {b.name for b in mvau_pool()}
        assert after == before | {"mvau_stub_backend"}
        r = resolve(
            mvau_space(), make_context(SEVEN_SERIES),
            base_assignment(backend="mvau_stub_backend", mem_mode="internal_embedded"),
        )
        assert isinstance(r, Point)
        assert r.sources == ("stub.sv",)
    finally:
        unregister("mvau_stub_backend")


# --- thresholds: the optional 3rd interface --------------------------------


def make_thresh_context(fpgapart=SEVEN_SERIES, weights=None, steps=7, tdt="INT16", idt="INT4"):
    if weights is None:
        rng = np.random.RandomState(0)
        weights = rng.randint(-8, 8, size=(6, 8)).astype(np.float32)
    mh = weights.shape[1]
    thr = np.sort(np.random.RandomState(1).randint(0, 100, size=(mh, steps)).astype(np.float32), axis=1)
    return Context(
        shapes={"weights": weights.shape, "inp": (1, weights.shape[0]), "out": (1, mh), "thresholds": (mh, steps)},
        datatypes={
            "weights": DataType["INT4"], "inp": DataType[idt],
            "out": DataType["INT16"], "thresholds": DataType[tdt],
        },
        initializers={"weights": weights, "thresholds": thr},
        fpgapart=fpgapart, clk_ns=5.0,
    )


def test_no_threshold_node_leaves_threshold_identity_none(schema):
    r = resolve(schema, make_context(), base_assignment())
    assert isinstance(r, Point)
    assert r["thresholdDataType"] is None


def test_thresholded_node_populates_threshold_identity(schema):
    r = resolve(schema, make_thresh_context(steps=7), base_assignment())
    assert isinstance(r, Point)
    assert r["thresholdDataType"] is not None


def test_malformed_threshold_tensor_is_illegal(schema):
    ctx = make_thresh_context(steps=7)
    n_ch = ctx.shapes["thresholds"][0]
    bad = Context(
        shapes={**ctx.shapes, "thresholds": (n_ch,)},
        datatypes=ctx.datatypes,
        initializers={**ctx.initializers, "thresholds": np.zeros((n_ch,), np.float32)},
        fpgapart=ctx.fpgapart, clk_ns=5.0,
    )
    r = resolve(schema, bad, base_assignment())
    assert isinstance(r, Illegal)
    assert any("rank 2" in reason or "2d" in reason.lower() for reason in r.reasons)


def test_unsigned_input_requires_nonneg_thresholds(schema):
    ctx = make_thresh_context(steps=4, idt="UINT4")
    neg = np.array(ctx.initializer("thresholds"))
    neg[0, 0] = -1.0
    ctx = Context(
        shapes=ctx.shapes, datatypes=ctx.datatypes,
        initializers={**ctx.initializers, "thresholds": neg},
        fpgapart=ctx.fpgapart, clk_ns=5.0,
    )
    r = resolve(schema, ctx, base_assignment())
    assert isinstance(r, Illegal)
    assert any("thresholds >= 0" in reason or "non-negative" in reason.lower() for reason in r.reasons)


# --- mem_modes: DSP cores are decoupled-weight-only, no thresholds -------------


def test_dsp_core_rejects_embedded_weights(schema):
    r = resolve(
        schema,
        make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8"),
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("embedded" in reason for reason in r.reasons)


def test_dsp_core_default_weight_topology_is_decoupled(schema):
    r = resolve(
        schema,
        make_context(VERSAL, weights=narrow_weights(wdt="INT8"), wdt="INT8"),
        base_assignment(backend=MVAU_DSP_PACKED, resType="dsp"),
    )
    assert isinstance(r, Point)
    assert r[PARAM_TOPOLOGY] == PARAM_DECOUPLED


def test_dsp_core_rejects_thresholded_node(schema):
    r = resolve(
        schema,
        make_thresh_context(VERSAL, weights=narrow_weights(wdt="INT8")),
        base_assignment(backend=MVAU_DSP_SOFTVEC, resType="dsp"),
    )
    assert isinstance(r, Illegal)
    assert any("threshold" in reason.lower() for reason in r.reasons)


# ===========================================================================
# F2 / D-R5 — backend feasibility is the single source of truth (claim time).
# The old op.py source-grep guard is intentionally dropped (build-discipline,
# not behavior); the feasibility DELEGATION below is the durable contract.
# ===========================================================================

MW, MH = 128, 64


def _feas_ctx(idt="INT8", wdt="INT8", part=VERSAL):
    return Context(
        shapes={"inp": (1, MW), "weights": (MW, MH), "out": (1, MH)},
        datatypes={"inp": DataType[idt], "weights": DataType[wdt], "out": DataType["INT32"]},
        initializers={"weights": np.ones((MW, MH), dtype=np.float32)},
        fpgapart=part,
    )


def _matmul_model(idt="INT8", wdt="INT8"):
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul], "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[idt])
    model.set_tensor_datatype("weights", DataType[wdt])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


@pytest.mark.parametrize("impl", ["mvau_hls", "mvau_dsp_softvec", "mvau_dsp_packed"])
def test_each_backend_rejects_float_iw(impl):
    k = mvau_kernel()
    float_pt = resolve(k.compile(), _feas_ctx(idt="FLOAT32", wdt="FLOAT32"), {"backend": impl})
    assert isinstance(float_pt, Illegal)
    int_pt = resolve(k.compile(), _feas_ctx(idt="INT8", wdt="INT8"), {"backend": impl})
    assert isinstance(int_pt, Point)


def test_has_feasible_point_true_for_int_false_for_float():
    k = mvau_kernel()
    assert k.has_feasible_point(_feas_ctx(idt="INT8", wdt="INT8")) is True
    assert k.has_feasible_point(_feas_ctx(idt="FLOAT32", wdt="FLOAT32")) is False


def test_first_feasible_backend_names_first_pool_member_or_none():
    # The SELECTION query behind PerNodePolicy(first_feasible): pool order is precedence, so
    # the integer node selects the first member (mvau_hls); a float node has no feasible
    # backend and returns None. has_feasible_point is the boolean over this same query.
    k = mvau_kernel()
    assert k.first_feasible_backend(_feas_ctx(idt="INT8", wdt="INT8")) == "mvau_hls"
    assert k.first_feasible_backend(_feas_ctx(idt="FLOAT32", wdt="FLOAT32")) is None


def test_can_infer_from_rejects_float_matmul_for_no_feasible_backend(caplog):
    with caplog.at_level(logging.INFO):
        claimed = MvauKernelOp.can_infer_from(
            _matmul_model(idt="FLOAT32", wdt="FLOAT32").graph.node[0],
            _matmul_model(idt="FLOAT32", wdt="FLOAT32"),
        )
    assert claimed is False
    assert any("no backend has a feasible point" in r.message for r in caplog.records)


def test_can_infer_from_claims_integer_matmul():
    model = _matmul_model(idt="INT8", wdt="INT8")
    assert MvauKernelOp.can_infer_from(model.graph.node[0], model) is True


def test_float_backend_widens_the_accepted_datatypes_union():
    # The viability the frontend claim reads is the UNION of the pool's declared datatype
    # support. Adding a float-supporting backend must make a float MatMul feasible with ZERO
    # edits to the op / can_infer_from — the whole point of declarative per-backend support.
    from dataclasses import replace
    from finn.kernels.model.backend import Backend, ports_from
    from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
    from finn.kernels.compute.mvau import mvau_kernel, mvau_pool
    from finn.kernels.compute.mvau.op import COMPUTE_STREAM, INPUT, WEIGHTS

    base = mvau_kernel()
    # Baseline: the all-integer pool has no feasible point for a float node.
    assert base.has_feasible_point(_feas_ctx(idt="FLOAT32", wdt="FLOAT32")) is False

    fp = DatatypeSupport(kind=DatatypeKind.FLOAT)
    float_backend = Backend(
        name="mvau_fp16", language="hls", sources=("mvu_fp.sv",),
        ports=ports_from(stream=COMPUTE_STREAM, accepted_dtypes={INPUT: fp, WEIGHTS: fp}),
    )
    widened = replace(base, pool=mvau_pool() + (float_backend,))
    assert widened.has_feasible_point(_feas_ctx(idt="FLOAT32", wdt="FLOAT32")) is True
    # integer nodes still feasible (the union only grew).
    assert widened.has_feasible_point(_feas_ctx(idt="INT8", wdt="INT8")) is True


def test_operand_map_shared_by_claim_and_build():
    model = _matmul_model()
    node = model.graph.node[0]
    mapping = MvauKernelOp._operand_map(node)
    assert mapping == {"inp": "inp", "weights": "weights", "out": "out"}
    trial = MvauKernelOp._trial_context(node, model)
    assert set(trial.shapes) == {"inp", "weights", "out"}
