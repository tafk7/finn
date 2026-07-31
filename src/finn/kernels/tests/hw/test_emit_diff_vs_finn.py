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
artifacts (RTL wrapper params, HLS params.h/thresh.h, memstream wrapper + memblock.dat,
RTL threshs_*.dat) — byte/param equivalence, toolchain-agnostic.
"""

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


def _make_mvau_model(W, pe, simd, wdt, idt, odt):
    mw, mh = W.shape
    node = helper.make_node(
        "MVAU", ["inp", "weights"], ["mid"],
        domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
        MW=mw, MH=mh, SIMD=simd, PE=pe,
        inputDataType=idt.name, weightDataType=wdt.name, outputDataType=odt.name,
        ActVal=0, binaryXnorMode=0, noActivation=1, mem_mode="internal_embedded",
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
    return model


def _mvau_context_point(W, pe, simd, wdt, idt, odt, impl, restype):
    from finn.kernels.engine.context import Context
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.mvau import mvau_space
    from finn.kernels.dataflow.parameters.names import DECOUPLED

    mw, mh = W.shape
    ctx = Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_space(), ctx, {
        "backend": impl, "PE": pe, "SIMD": simd, "resType": restype, MVAU_TOPOLOGY: DECOUPLED,
    })
    return ctx, point


def _params_of(text):
    d = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("parameter"):
            parts = line.replace("parameter", "").split("=")
            if len(parts) == 2:
                name = parts[0].strip()
                val = parts[1].split("//")[0].strip()
                if val.endswith(".0"):
                    val = val[:-2]
                d[name] = val
    return d


def test_mvau_rtl_wrapper_params_match_finn():
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

    model = _make_mvau_model(W, 2, 2, wdt, idt, odt).transform(SpecializeLayers(FPGAPART))
    model = model.transform(MinimizeAccumulatorWidth())
    node = model.graph.node[0]
    assert node.op_type == "MVAU_rtl"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_hdl(model, FPGAPART, CLK_NS)
        top = inst.get_nodeattr("gen_top_module")
        finn_v = open(os.path.join(d, top + "_wrapper.v")).read()

    ctx, point = _mvau_context_point(W, 2, 2, wdt, idt, odt, MVAU_DSP_SOFTVEC, "dsp")
    ours_v = emit_point(mvau_pool(), point, ctx).generated[0].content()

    fp, op = _params_of(finn_v), _params_of(ours_v)
    shared = set(fp) & set(op)
    mism = {k: (fp[k], op[k]) for k in shared if fp[k] != op[k]}
    assert not mism, f"param mismatches: {mism}"
    assert not (set(fp) - set(op)), f"FINN-only params: {set(fp) - set(op)}"


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


def _norm_header_ws(s):
    import re

    lines = _norm(s).splitlines()
    if lines:
        lines[0] = re.sub(r"[ \t]+", " ", lines[0])
    return "\n".join(lines)


def test_thresholding_hls_thresh_h_matches_finn():
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.thresholding import THRESHOLDING_HLS, thresholding_kernel_space
    from finn.kernels.compute.thresholding.emit_hls import _thresh_h

    T = np.sort(np.array([[0, 40, 80, 120, 160, 200, 255]] * 4, dtype=np.float32), axis=-1)
    idt, tdt, odt = DataType["UINT8"], DataType["UINT8"], DataType["UINT3"]

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
    point = resolve(thresholding_kernel_space(), ctx, {"backend": THRESHOLDING_HLS, "PE": 2})
    ours_thresh = _thresh_h(point, ctx)
    assert _norm_header_ws(finn_thresh) == _norm_header_ws(ours_thresh)


def test_thresholding_rtl_dat_matches_finn():
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.model.backend import emit_point
    from finn.kernels.compute.thresholding import (
        THRESHOLDING_RTL,
        thresholding_kernel_space,
        thresholding_pool,
    )

    T = np.sort(np.random.RandomState(0).randint(0, 50, size=(4, 7)).astype(np.float32), axis=-1)
    idt, tdt, odt = DataType["UINT8"], DataType["UINT8"], DataType["UINT3"]

    model = _make_thresholding_model(T, 2, idt, tdt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "rtl"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "Thresholding_rtl"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model, d)
        finn_dats = {}
        for f in os.listdir(d):
            if "threshs_" in f and f.endswith(".dat"):
                finn_dats[f[f.index("threshs_"):]] = open(os.path.join(d, f)).read()

    ctx = _thresh_ctx(T, idt, tdt, odt)
    point = resolve(thresholding_kernel_space(), ctx, {"backend": THRESHOLDING_RTL, "PE": 2})
    arts = emit_point(thresholding_pool(), point, ctx)
    ours_dats = {}
    for f in arts.data_files:
        c = f.content
        ours_dats[f.filename] = c() if callable(c) else c

    assert set(finn_dats) == set(ours_dats), (
        f"file set mismatch FINN={sorted(finn_dats)} OURS={sorted(ours_dats)}"
    )
    for name in sorted(finn_dats):
        assert _norm(finn_dats[name]) == _norm(ours_dats[name]), f"mismatch in {name}"
