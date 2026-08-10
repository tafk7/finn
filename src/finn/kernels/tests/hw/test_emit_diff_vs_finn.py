############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Differential checks: our hermetic emit vs real FINN codegen (finn_codegen tier).

Validates the REAL "we reproduce upstream FINN codegen" contract that the unit-tier
string-golden tests only approximate. Needs the FINN classic op backends (HWCustomOp /
ModelWrapper / SpecializeLayers) but NO Vivado/HLS/XSI — so it runs in the docker env, not
gated behind --runslow. Drives FINN's real generate_hdl/generate_params on a single-node
model, runs our design-space emit on the equivalent (Point, Context), and diffs the
artifacts (RTL wrapper params + text, HLS params.h/thresh.h/top_*.cpp, memstream wrapper +
memblock.dat, RTL threshs_*.dat) — byte/param equivalence, toolchain-agnostic.

WHAT THE FIXTURES ARE FOR. Three real divergences from baseline FINN shipped green past an
earlier version of this file. None was a missing test — each was a test pinned at the ONE
point where the bug is invisible. Whenever you add a case here, ask what makes the
behaviour observable, not just what exercises the code path:

  * a threshold table whose values SATURATE its declared dtype cannot show a wrong
    (narrowed) threshold dtype — the narrowing is a no-op there;
  * a step count equal to ``2**o_bits - 1`` cannot show wrong narrow-quant params — FINN's
    zero-padding is a no-op there;
  * ``pumpedCompute`` at SIMD=2 cannot move SEGMENTLEN — both arms floor to 1;
  * comparing only ``parameter`` lines cannot show a wrong instantiated core module.

Each of those blind spots now has a live arm beside it. The no-op arms are kept
deliberately, so the boundary between "covered" and "invisible" stays pinned.

Two deltas from FINN are KNOWN and deliberate, excluded at the assertion with the reason
inline rather than normalized away silently: ``AP_INT_MAX_W`` (a wider upper bound, two
formula deltas) and the forked per-core RTL module name (the 2c source split).
"""

import difflib
import os
import tempfile

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

pytestmark = pytest.mark.finn_codegen

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"  # Versal / DSP58
CLK_NS = 5.0


def _norm(s):
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()


# ===========================================================================
# MVAU
# ===========================================================================

from finn.kernels.dataflow.parameters.names import WEIGHTS
from finn.kernels.model.param_names import ram_style_key, topology_key

MVAU_TOPOLOGY = topology_key(WEIGHTS)
MVAU_RAM_STYLE = ram_style_key(WEIGHTS)


def _make_mvau_model(W, pe, simd, wdt, idt, odt, T=None, tdt=None, actval=0):
    """A single-node MVAU model. With ``T``, the FUSED 3-input form.

    ``noActivation`` picks the arity: 1 means (inp, weights) and a PassThroughActivation;
    0 means (inp, weights, thresholds) and a baked ThresholdsActivation, which FINN's
    verify_node enforces. The 3-input form is what emits ``thresh.h``.
    """
    mw, mh = W.shape
    inputs = ["inp", "weights"] if T is None else ["inp", "weights", "thresholds"]
    node = helper.make_node(
        "MVAU", inputs, ["mid"],
        domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
        MW=mw, MH=mh, SIMD=simd, PE=pe,
        inputDataType=idt.name, weightDataType=wdt.name, outputDataType=odt.name,
        ActVal=actval, binaryXnorMode=0, noActivation=1 if T is None else 0,
        mem_mode="internal_embedded",
    )
    tail = helper.make_node("Identity", ["mid"], ["outp"])
    graph = helper.make_graph(
        [node, tail], "mvau_graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-diff"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresholds", tdt)
        model.set_initializer("thresholds", T)
    return model


def _mvau_context_point(W, pe, simd, wdt, idt, odt, impl, restype,
                        T=None, tdt=None, actval=None, **extra):
    from finn.kernels.engine.context import Context
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.mvau import mvau_space
    from finn.kernels.dataflow.parameters.names import DECOUPLED

    mw, mh = W.shape
    shapes = {"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)}
    datatypes = {"weights": wdt, "inp": idt, "out": odt}
    initializers = {"weights": W}
    if T is not None:
        # A threshold INITIALIZER is what makes the fused form exist — `has_thresh` in
        # emit_hls.py is emergent from it, not from a flag.
        shapes["thresholds"] = T.shape
        datatypes["thresholds"] = tdt
        initializers["thresholds"] = T
    ctx = Context(
        shapes=shapes, datatypes=datatypes, initializers=initializers,
        fpgapart=extra.pop("fpgapart", FPGAPART), clk_ns=CLK_NS,
    )
    assignment = {
        "backend": impl, "PE": pe, "SIMD": simd, "resType": restype, MVAU_TOPOLOGY: DECOUPLED,
    }
    if actval is not None:
        assignment["ActVal"] = actval
    assignment.update(extra)
    point = resolve(mvau_space(), ctx, assignment)
    return ctx, point


def _params_of(text):
    """Parse ``parameter NAME = VALUE`` lines into a dict.

    The comma is stripped AFTER the trailing comment is dropped, not before. A line like
    ``parameter VERSION = 3,\t// Allowed versions - 1: DSP48E1, ...`` carries the comma in the
    MIDDLE, so the original leading ``rstrip(",")`` did nothing and ``VERSION`` parsed as
    ``"3,"``. Both sides parsed it the same way, so nothing failed — but comparing ``"3,"`` to
    ``"3,"`` is one comment-rewording away from a spurious mismatch, and it silently widens
    what counts as "the value".
    """
    d = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("parameter"):
            parts = line.replace("parameter", "").split("=")
            if len(parts) == 2:
                name = parts[0].strip()
                val = parts[1].split("//")[0].strip().rstrip(",").strip()
                if val.endswith(".0"):
                    val = val[:-2]
                d[name] = val
    return d


# --- Case D: whole-file comparison, not just the `parameter` lines ------------
# `_params_of` reads ONLY lines starting with `parameter`, so everything else in the
# wrapper is invisible to it — port declarations, the generate/ifdef structure, and
# most importantly the INSTANTIATED CORE MODULE NAME. A backend-selection or
# module-naming regression would keep every parameter identical and still be wrong.
#
# Both dict and text comparison are kept: the dict gives a readable per-param diff when
# a value moves, the text catches everything the dict cannot see.


def _norm_v(text):
    """Normalize a Verilog wrapper to comparable lines.

    Three things are deliberately erased, each non-semantic for a specific reason:
      * the vendored ``/* ... */`` license banner FINN's templates carry and ours do not,
      * whitespace runs and blank lines,
      * directory prefixes inside string literals (INIT_FILE is an absolute path into a
        per-run tmpdir on FINN's side, so it can never match literally).

    With those erased the memstream wrapper matches FINN exactly, and the compute wrapper
    differs only on the two lines named at ``_V_CORE_INSTANTIATION`` and ``SEGMENTLEN``
    below — i.e. the normalization is not wide enough to be hiding anything else.
    """
    import re

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    out = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r'"[^"]*/([^"/]+)"', r'"\1"', line)
        if line:
            out.append(line)
    return out


# The ONE known, deliberate divergence in the compute wrapper's body. Our emit instantiates
# a PER-CORE wrapper (mvu_vvu_axi_softvec / _packed, from the selected backend's
# `rtl_core_module`), where FINN instantiates the fused `mvu_vvu_axi` and forks internally
# with a generate block. That is the 2c source split (compute/mvau/__init__.py:40-59), which
# deliberately forks vendored HDL — so the byte oracle cannot apply to this line, and
# rtlsim bit-equivalence (rtlsim_split_equiv_mvau.py) is its behavioural replacement.
#
# It is EXCLUDED BY EXACT PREFIX rather than by a loose pattern, so any OTHER instantiation
# change still fails the diff. Note SEGMENTLEN is not excluded here: FINN renders it as the
# float "2.0" and us as "2", which _params_of already normalizes, so the text comparison is
# scoped to the module body below the parameter block.
_V_CORE_INSTANTIATION = ("mvu_vvu_axi #(", "mvu_vvu_axi_softvec #(", "mvu_vvu_axi_packed #(")


def _assert_wrapper_text_matches(finn_v, ours_v, *, skip_params=True):
    """Compare wrapper text line-for-line, excluding the deliberate core-name fork.

    ``skip_params`` drops the ``parameter`` lines, which the caller has already compared
    as a dict — they differ only in numeric FORMATTING ("2.0" vs "2"), and re-checking
    them textually would fail on that alone.
    """
    def keep(lines):
        return [
            ln for ln in lines
            if not ln.startswith(_V_CORE_INSTANTIATION)
            and not (skip_params and ln.startswith("parameter"))
        ]

    f, o = keep(_norm_v(finn_v)), keep(_norm_v(ours_v))
    assert f == o, "wrapper text differs:\n" + "\n".join(
        difflib.unified_diff(f, o, "FINN", "OURS", lineterm="", n=1)
    )


# --- Case C axes ------------------------------------------------------------
# The RTL wrapper diff ran at one part with pumpedCompute unset, which is exactly the
# corner where the derived-order-sensitive slots are least interesting: $VERSION$ is
# FORCED by the DSP block (_dsp_rtl.py:26), and $SEGMENTLEN$ reads pumpedCompute to pick
# ref_clk = clk/2 and simd_factor = 6 instead of clk and 3 (_dsp_rtl.py:48-60).
#
# Everything here is expected GREEN — this is regression insurance for those two
# derivations, not a bug hunt. runtime_writeable is deliberately NOT crossed in: on
# DSP48E1 it legitimately makes the RTL backends infeasible (a deliberate divergence,
# 4d68aeb35), so the arms would not be comparable. test_resolve.py pins that separately.
_MVAU_RTL_PARTS = [
    pytest.param("xcvc1902-vsva2197-2MP-e-S", id="versal_dsp58"),
    pytest.param("xczu3eg-sbva484-1-e", id="ultrascale_dsp48e2"),
    pytest.param("xc7z020clg400-1", id="zynq7_dsp48e1"),
]

# SIMD=6, not 2. SEGMENTLEN is min(critical_path_dsps, ceil(SIMD/simd_factor)) and
# pumpedCompute moves only the simd_factor term (3 -> 6). At SIMD=2 both arms floor to 1,
# so the pumped axis is INERT and the six cases collapse to three distinct points. At
# SIMD=6 SEGMENTLEN goes 2 -> 1 with the flag. SIMD=1 would be rejected outright
# (rtl:334). test_mvau_rtl_arms_are_distinct below keeps this honest.
_MVAU_RTL_PE, _MVAU_RTL_SIMD = 2, 6


@pytest.mark.parametrize("fpgapart", _MVAU_RTL_PARTS)
@pytest.mark.parametrize("pumped", [0, 1])
def test_mvau_rtl_wrapper_params_match_finn(fpgapart, pumped):
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth,
    )
    from finn.kernels.model.backend import emit_point
    from finn.kernels.compute.mvau import mvau_pool, MVAU_DSP_SOFTVEC

    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    pe, simd = _MVAU_RTL_PE, _MVAU_RTL_SIMD

    model = _make_mvau_model(W, pe, simd, wdt, idt, odt).transform(SpecializeLayers(fpgapart))
    model = model.transform(MinimizeAccumulatorWidth())
    node = model.graph.node[0]
    assert node.op_type == "MVAU_rtl"
    inst = getCustomOp(node)
    inst.set_nodeattr("pumpedCompute", pumped)
    # SpecializeLayers drops the node name, which FINN uses for the module name and the
    # `endmodule //` trailer. Match our emit's default so the text diff below compares
    # codegen rather than naming.
    node.name = "mvau_top"
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_hdl(model, fpgapart, CLK_NS)
        top = inst.get_nodeattr("gen_top_module")
        finn_v = open(os.path.join(d, top + "_wrapper.v")).read()

    ctx, point = _mvau_context_point(
        W, pe, simd, wdt, idt, odt, MVAU_DSP_SOFTVEC, "dsp",
        fpgapart=fpgapart, pumpedCompute=pumped,
    )
    ours_v = emit_point(mvau_pool(), point, ctx).generated[0].content()

    fp, op = _params_of(finn_v), _params_of(ours_v)
    # The slots this parametrization exists to move. If a rename ever drops them from the
    # wrapper the diff below would vacuously pass on the intersection.
    for k in ("VERSION", "SEGMENTLEN", "ACCU_WIDTH"):
        assert k in fp and k in op, f"{k} missing (FINN={k in fp}, ours={k in op})"
    shared = set(fp) & set(op)
    mism = {k: (fp[k], op[k]) for k in shared if fp[k] != op[k]}
    assert not mism, f"param mismatches: {mism}"
    assert not (set(fp) - set(op)), f"FINN-only params: {set(fp) - set(op)}"
    _assert_wrapper_text_matches(finn_v, ours_v)


def test_mvau_rtl_arms_are_distinct():
    """The part x pumpedCompute arms must land on SIX distinct wrapper configurations.

    A parametrization whose arms all resolve to the same point is six copies of one test
    that reads as broad coverage. That is precisely how this file's original blind spots
    survived, so the spread is asserted rather than assumed: at SIMD=2 the pumped axis is
    inert and this drops to 3, which is the regression this guards.
    """
    from finn.kernels.engine.point import Illegal
    from finn.kernels.compute.mvau import mvau_pool, MVAU_DSP_SOFTVEC
    from finn.kernels.model.backend import emit_point

    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]

    seen = {}
    for pumped in (0, 1):
        for part in (p.values[0] for p in _MVAU_RTL_PARTS):
            ctx, point = _mvau_context_point(
                W, _MVAU_RTL_PE, _MVAU_RTL_SIMD, wdt, idt, odt, MVAU_DSP_SOFTVEC, "dsp",
                fpgapart=part, pumpedCompute=pumped,
            )
            assert not isinstance(point, Illegal), (part, pumped, point)
            params = _params_of(emit_point(mvau_pool(), point, ctx).generated[0].content())
            seen[(part, pumped)] = tuple(params[k] for k in ("VERSION", "SEGMENTLEN"))

    assert len(set(seen.values())) == 6, f"arms collapsed: {seen}"
    # VERSION is forced by the DSP block, so it must move with the part and NOT with the
    # pumped flag; SEGMENTLEN is the converse. Asserting both directions catches an arm
    # that stays distinct for the wrong reason.
    assert len({v[0] for v in seen.values()}) == 3, "VERSION does not track the part"
    assert len({v[1] for v in seen.values()}) == 2, "SEGMENTLEN does not track pumpedCompute"


def test_mvau_hls_params_h_matches_finn():
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.compute.mvau import MVAU_HLS
    from finn.kernels.compute.mvau.emit_hls import _params_h
    from finn.kernels.compute.mvau.geometry import mvau_geometry

    rng = np.random.RandomState(0)
    W = rng.randint(-7, 7, size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]

    model = _make_mvau_model(W, 2, 2, wdt, idt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "MVAU_hls"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model, d)
        finn_params = open(os.path.join(d, "params.h")).read()

    ctx, point = _mvau_context_point(W, 2, 2, wdt, idt, odt, MVAU_HLS, "lut")
    ours_params = _params_h(point, ctx, mvau_geometry(point, ctx))
    assert _norm(finn_params) == _norm(ours_params)


# --- Case A: the FUSED (3-input) MVAU-HLS path ------------------------------
# Nothing diffed a thresholded MVAU against FINN, so the whole `has_thresh` branch of
# compute/mvau/emit_hls.py — the fused thresh.h and the top_*.cpp that includes it — was
# unchecked. The MVAU half of the threshold-dtype divergence (e3e005793) lived here.
#
# The axis that matters is the threshold VALUES against the DECLARED dtype, for the same
# reason as Case B: narrowing a dtype the values already saturate is invisible.

_MVAU_THRESH_CASES = [
    # (id, top threshold value, declared tdt) — signed `wide_declared` also moves the
    # comparator type, not just the width.
    pytest.param(255, "UINT8", id="saturating"),       # narrowing is a no-op here
    pytest.param(49, "UINT8", id="non_saturating"),    # used to diverge: ap_uint<8> vs <5>
    pytest.param(30, "INT16", id="wide_declared"),     # used to diverge: ap_int<16> vs <7>
]


def _mvau_thresholds(mh, steps, top, tdt):
    """Ascending per-output-channel threshold rows topping out at ``top``.

    Signed declared types get a negative floor so the values genuinely need a signed
    container; the top is pinned so the saturating/non-saturating regime is exact.
    """
    lo = -top if DataType[tdt].signed() else 0
    rows = np.sort(
        np.random.RandomState(3).randint(lo, top, size=(mh, steps)).astype(np.float32), axis=1
    )
    rows[:, -1] = top
    return rows


@pytest.mark.parametrize("top,tdt_name", _MVAU_THRESH_CASES)
def test_mvau_hls_fused_thresh_h_matches_finn(top, tdt_name):
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.compute.mvau import MVAU_HLS
    from finn.kernels.compute.mvau.emit_hls import _thresh_h
    from finn.kernels.compute.mvau.geometry import mvau_geometry

    rng = np.random.RandomState(0)
    W = rng.randint(-7, 7, size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["UINT3"]
    tdt = DataType[tdt_name]
    pe, simd, steps = 2, 2, 7
    T = _mvau_thresholds(W.shape[1], steps, top, tdt_name)

    model = _make_mvau_model(W, pe, simd, wdt, idt, odt, T=T, tdt=tdt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    # Guards: a 2-input node would emit no thresh.h at all and this test would cover
    # nothing while passing.
    assert node.op_type == "MVAU_hls"
    assert len(node.input) == 3, f"expected the fused 3-input MVAU, got {len(node.input)}"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model, d)
        finn_thresh = open(os.path.join(d, "thresh.h")).read()
    assert "ThresholdsActivation<" in finn_thresh, "FINN emitted no fused activation"

    ctx, point = _mvau_context_point(W, pe, simd, wdt, idt, odt, MVAU_HLS, "lut", T=T, tdt=tdt)
    ours_thresh = _thresh_h(point, ctx, mvau_geometry(point, ctx))
    assert _norm_header_ws(finn_thresh) == _norm_header_ws(ours_thresh)


def _norm_cpp(s):
    """Collapse all whitespace runs per line, dropping blanks.

    FINN builds top_*.cpp by string-replacing a template with '\\n'-joined lists, which
    leaves ragged leading indentation on the $DEFINES$ block. That is formatting, not
    codegen — normalizing it is what makes a token-level diff meaningful.
    """
    import re

    return [re.sub(r"\s+", " ", ln).strip() for ln in s.strip().splitlines() if ln.strip()]


# AP_INT_MAX_W is a KNOWN, deliberate divergence and is excluded from the diff below.
# It is the ONLY line that differs; on the fused fixture FINN emits 16 and we emit 32,
# from two independent formula deltas:
#
#   * FINN's MVAU_hls override (matrixvectoractivation_hls.py:466) takes
#     max(io, weightstream, SIMD*weight_bits) — a SINGLE PE's weight entry.
#     `_ap_int_max_w` (emit_hls.py:180) uses PE*SIMD*wdt, i.e. all PEs' entries.
#   * we additionally floor the result at 32; FINN's MVAU applies no floor (only
#     checksum_hls/lookup_hls do).
#
# Both make our bound WIDER, and the knob is an upper bound on ap_uint width — a larger
# value costs HLS compile time, not hardware. So this is not being forced to match here;
# it is recorded as a real, non-gating delta rather than silently normalized away.
_CPP_EXCLUDED = ("#define AP_INT_MAX_W",)


def test_mvau_hls_fused_top_cpp_matches_finn():
    """Diffs the fused top_*.cpp — the .cpp half of the has_thresh branch.

    Nothing diffed this file before, for either arity. It carries the thresh.h include,
    the threshold ARRAY_PARTITION pragmas and the `threshs` activation argument, all of
    which are point-derived and none of which thresh.h itself would catch.
    """
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.compute.mvau import MVAU_HLS
    from finn.kernels.compute.mvau.emit_hls import emit_mvau_hls

    rng = np.random.RandomState(0)
    W = rng.randint(-7, 7, size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt, tdt = DataType["UINT3"], DataType["UINT8"]
    pe, simd = 2, 2
    name = "mvau_top"  # our emit's default module_name
    T = _mvau_thresholds(W.shape[1], 7, 49, "UINT8")

    model = _make_mvau_model(W, pe, simd, wdt, idt, odt, T=T, tdt=tdt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "MVAU_hls"
    assert len(node.input) == 3, f"expected the fused 3-input MVAU, got {len(node.input)}"
    # SpecializeLayers drops the node name, and FINN builds the blackbox signature from it.
    # Without this the two sides differ on the function name for a purely harness reason.
    node.name = name
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.code_generation_ipgen(model, FPGAPART, CLK_NS)
        finn_cpp = open(os.path.join(d, f"top_{name}.cpp")).read()

    ctx, point = _mvau_context_point(W, pe, simd, wdt, idt, odt, MVAU_HLS, "lut", T=T, tdt=tdt)
    ours_cpp = emit_mvau_hls(point, ctx, module_name=name).generated[0].content()

    assert '#include "thresh.h"' in finn_cpp, "FINN took the no-activation path"

    def keep(lines):
        return [ln for ln in lines if not ln.startswith(_CPP_EXCLUDED)]

    f, o = keep(_norm_cpp(finn_cpp)), keep(_norm_cpp(ours_cpp))
    assert f == o, "top .cpp differs:\n" + "\n".join(
        difflib.unified_diff(f, o, "FINN", "OURS", lineterm="", n=1)
    )


def _memstream_params(text):
    d = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("parameter") and "=" in line:
            body = line.replace("parameter", "").split("//")[0]
            name, val = body.split("=", 1)
            name, val = name.strip(), val.strip()
            if name in ("AXILITE_ADDR_WIDTH", "SET_BITS"):
                continue
            if name == "INIT_FILE":
                val = '"' + val.strip('"').split("/")[-1] + '"'
            d[name] = val
    return d


def test_mvau_memstream_wrapper_and_dat_match_finn():
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.transformation.fpgadataflow.minimize_accumulator_width import (
        MinimizeAccumulatorWidth,
    )
    from finn.kernels.engine.context import Context
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.model.backend import emit_point
    from finn.kernels.compute.mvau import mvau_space, MVAU_HLS
    from finn.kernels.dataflow.parameters import parameters_pool
    from finn.kernels.dataflow.parameters.names import DECOUPLED

    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    pe, simd = 2, 2

    model = _make_mvau_model(W, pe, simd, wdt, idt, odt).transform(SpecializeLayers(FPGAPART))
    model = model.transform(MinimizeAccumulatorWidth())
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("mem_mode", "internal_decoupled")
    inst.set_nodeattr("ram_style", "block")
    # SpecializeLayers drops the node name; FINN derives the memstream module name from it.
    # Match our emit's default so the text diff compares codegen rather than naming.
    node.name = "mvau_top"
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_hdl_memstream(FPGAPART, 0)
        finn_v = open(os.path.join(d, node.name + "_memstream_wrapper.v")).read()
        inst.generate_params(model, d)
        finn_dat = open(os.path.join(d, "memblock.dat")).read()

    ctx = Context(
        shapes={"weights": W.shape, "inp": (1, W.shape[0]), "out": (1, W.shape[1])},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_space(), ctx, {
        "backend": MVAU_HLS, "PE": pe, "SIMD": simd, "resType": "lut",
        MVAU_TOPOLOGY: DECOUPLED, MVAU_RAM_STYLE: "block",
    })
    arts = emit_point(parameters_pool(), point, ctx, root=MVAU_TOPOLOGY)
    ours_v = arts.generated[0].content()
    ours_dat = [f for f in arts.data_files if f.filename == "memblock.dat"][0].content

    fp, op = _memstream_params(finn_v), _memstream_params(ours_v)
    shared = set(fp) & set(op)
    mism = {k: (fp[k], op[k]) for k in shared if fp[k] != op[k]}
    assert not mism, f"wrapper param mismatches: {mism}"
    assert not (set(fp) - set(op)), f"FINN-only params: {set(fp) - set(op)}"
    # The memstream wrapper instantiates no forked core, so nothing is excluded here and
    # the whole body is compared — including the `memstream_axi_wrapper` instantiation
    # that _memstream_params could not see. It matches exactly.
    _assert_wrapper_text_matches(finn_v, ours_v)
    assert _norm(finn_dat) == _norm(ours_dat)


# ===========================================================================
# Thresholding
# ===========================================================================


def _make_thresholding_model(T, pe, idt, tdt, odt):
    channels, steps = T.shape
    node = helper.make_node(
        "Thresholding", ["inp", "thresholds"], ["mid"],
        domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
        NumChannels=channels, PE=pe, numSteps=steps,
        inputDataType=idt.name, weightDataType=tdt.name, outputDataType=odt.name,
        ActVal=0, numInputVectors=[1],
    )
    tail = helper.make_node("Identity", ["mid"], ["outp"])
    graph = helper.make_graph(
        [node, tail], "thresholding_graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, channels])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, channels])],
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresholding-diff"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("thresholds", tdt)
    model.set_initializer("thresholds", T)
    return model


def _thresh_ctx(T, idt, tdt, odt):
    from finn.kernels.engine.context import Context

    channels, steps = T.shape
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": tdt, "inp": idt, "out": odt},
        initializers={"thresholds": T},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )


# --- Case B axes -----------------------------------------------------------
# Two INDEPENDENT axes used to run at one point each, which is what let both halves of
# the narrow-quant divergence (728a2a52d) and the threshold-dtype divergence (e3e005793)
# ship green:
#
#   * STEP COUNT vs output width. At steps == 2**o_bits - 1 FINN's zero-padding in
#     make_weight_file is a no-op, so the padded and unpadded tables are identical and
#     $N$/$WT$/$O_BITS$ cannot disagree. That was the only case covered.
#   * THRESHOLD VALUES vs the DECLARED dtype. When the values saturate the declared dtype,
#     value-narrowing it is a no-op — so a consumer that narrows and one that does not emit
#     the same ThresholdsActivation<>. That was also the only case covered.
#
# The `padding_noop_*` / `saturating` ids below ARE those old blind spots, kept so the
# no-op arms stay pinned alongside the arms that exercise the real behaviour.

_THRESH_STEP_CASES = [
    pytest.param(7, "UINT3", id="padding_noop_7_uint3"),   # 2**3-1 == 7: pad is a no-op
    pytest.param(6, "UINT3", id="narrow_6_uint3"),         # used to diverge: FINN N=6 WT=8
    pytest.param(2, "UINT2", id="narrow_2_uint2"),
    pytest.param(14, "UINT4", id="narrow_14_uint4"),
]

# The declared dtype for every case below. `saturating` values reach its max; the
# `non_saturating` ones do not, which is what makes narrowing observable.
_THRESH_DECLARED = "UINT8"
_THRESH_REGIME_TOP = {"saturating": 255, "non_saturating": 49}


def _thresh_table(channels, steps, regime):
    """Ascending per-channel threshold rows in the requested value regime.

    ``saturating`` reaches UINT8's max, so narrowing the declared dtype to fit the values
    is a no-op; ``non_saturating`` tops out near 49, so a narrowed dtype would be visibly
    narrower than the declared one. The top of each row is PINNED rather than left to the
    RNG so the regime is exact, not probabilistic.
    """
    hi = _THRESH_REGIME_TOP[regime]
    rows = np.sort(
        np.random.RandomState(0).randint(0, hi, size=(channels, steps)).astype(np.float32),
        axis=1,
    )
    rows[:, -1] = hi
    return rows


def _norm_header_ws(s):
    import re

    lines = _norm(s).splitlines()
    if lines:
        lines[0] = re.sub(r"[ \t]+", " ", lines[0])
    return "\n".join(lines)


@pytest.mark.parametrize("steps,odt_name", _THRESH_STEP_CASES)
@pytest.mark.parametrize("regime", ["saturating", "non_saturating"])
def test_thresholding_hls_thresh_h_matches_finn(steps, odt_name, regime):
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.thresholding import THRESHOLDING_HLS, thresholding_space
    from finn.kernels.compute.thresholding.emit_hls import _thresh_h

    T = _thresh_table(4, steps, regime)
    idt = DataType["UINT8"]
    tdt = DataType[_THRESH_DECLARED]
    odt = DataType[odt_name]

    model = _make_thresholding_model(T, 2, idt, tdt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "Thresholding_hls"
    inst = getCustomOp(node)
    inst.set_nodeattr("mem_mode", "internal_embedded")
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model, d)
        finn_thresh = open(os.path.join(d, "thresh.h")).read()

    ctx = _thresh_ctx(T, idt, tdt, odt)
    point = resolve(thresholding_space(), ctx, {"backend": THRESHOLDING_HLS, "PE": 2})
    ours_thresh = _thresh_h(point, ctx)
    assert "ThresholdsActivation<" in finn_thresh, "FINN emitted no activation to diff against"
    assert _norm_header_ws(finn_thresh) == _norm_header_ws(ours_thresh)


# The wrapper params that carry the narrow-quant decision: $N$ (raw step count), $WT$
# (threshold width), $O_BITS$ (output container). $WI$/$C$/$BIAS$/$PE$ ride along as the
# rest of the geometry. FINN keys these with $-delimiters in prepare_codegen_rtl_values.
_THRESH_RTL_PARAMS = ("N", "WT", "WI", "C", "BIAS", "PE", "O_BITS")


@pytest.mark.parametrize("steps,odt_name", _THRESH_STEP_CASES)
def test_thresholding_rtl_dat_and_params_match_finn(steps, odt_name):
    """Diffs the wrapper PARAMS as well as the .dat.

    The .dat alone was not enough: at the old single fixture (steps == 2**o_bits - 1) FINN's
    zero-padding is a no-op, so a wrong $N$/$WT$ produced a byte-identical table. Both halves
    of the narrow-quant divergence lived in the params, which nothing looked at.
    """
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.model.backend import emit_point
    from finn.kernels.compute.thresholding import (
        THRESHOLDING_RTL,
        thresholding_space,
        thresholding_pool,
    )

    T = _thresh_table(4, steps, "non_saturating")
    idt, tdt = DataType["UINT8"], DataType[_THRESH_DECLARED]
    odt = DataType[odt_name]

    model = _make_thresholding_model(T, 2, idt, tdt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "rtl"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "Thresholding_rtl"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        # prepare_codegen_rtl_values takes the MODEL (not a path) and calls generate_params
        # internally, so code_gen_dir_ipgen must already be set — hence the ordering here.
        finn_codegen = inst.prepare_codegen_rtl_values(model)
        finn_dats = {}
        for f in os.listdir(d):
            if "threshs_" in f and f.endswith(".dat"):
                finn_dats[f[f.index("threshs_"):]] = open(os.path.join(d, f)).read()

    ctx = _thresh_ctx(T, idt, tdt, odt)
    point = resolve(thresholding_space(), ctx, {"backend": THRESHOLDING_RTL, "PE": 2})
    arts = emit_point(thresholding_pool(), point, ctx)
    ours_dats = {}
    for f in arts.data_files:
        c = f.content
        ours_dats[f.filename] = c() if callable(c) else c

    fp = {k: str(finn_codegen[f"${k}$"][0]) for k in _THRESH_RTL_PARAMS}
    op = _params_of(arts.generated[0].content())
    missing = [k for k in _THRESH_RTL_PARAMS if k not in op]
    assert not missing, f"our wrapper emitted no {missing}"
    mism = {k: (fp[k], op[k]) for k in _THRESH_RTL_PARAMS if fp[k] != op[k]}
    assert not mism, f"wrapper param mismatches: {mism}"

    assert set(finn_dats) == set(ours_dats), (
        f"file set mismatch FINN={sorted(finn_dats)} OURS={sorted(ours_dats)}"
    )
    for name in sorted(finn_dats):
        assert _norm(finn_dats[name]) == _norm(ours_dats[name]), f"mismatch in {name}"
