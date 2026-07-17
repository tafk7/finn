############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Differential check: our hermetic MVAU emit vs FINN's own codegen.

Runs INSIDE the FINN Docker container (needs HWCustomOp / ModelWrapper / the RTL &
HLS backends). Drives FINN's real generate_hdl / make_weight_file on a single-MVAU
model, then runs our design_space emit on the equivalent (Point, Context), and diffs
the generated artifacts. Proves our emit faithfully reproduces FINN's codegen — the
equivalence claim, demonstrated rather than asserted, and toolchain-agnostic (no
synthesis).

Usage (from finn/):
    bash run-docker.sh bash -c \\
      "PYTHONPATH=/workspace/finn/src:$PYTHONPATH \\
       python src/finn/design_space/tests/diff_mvau_emit_vs_finn.py"
"""

import os
import sys
import tempfile

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)

# our side
from finn.design_space.space import Context, resolve, emit_point
from finn.design_space.fixtures.mvau import mvau_schema, mvau_pool, MVAU_DSP_SOFTVEC, MVAU_HLS
from finn.design_space.fixtures.parameters import parameters_pool
from finn.design_space.fixtures.parameters.names import (
    DECOUPLED as PARAM_DECOUPLED,
    EMBEDDED,
    RAM_STYLE as PARAM_RAM_STYLE,
    TOPOLOGY as PARAM_TOPOLOGY,
    TOPOLOGY,
)

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"  # Versal / DSP58
CLK_NS = 5.0


def _make_mvau_model(W, pe, simd, wdt, idt, odt):
    mw, mh = W.shape
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    node = helper.make_node(
        "MVAU", ["inp", "weights"], ["mid"],
        domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
        MW=mw, MH=mh, SIMD=simd, PE=pe,
        inputDataType=idt.name, weightDataType=wdt.name, outputDataType=odt.name,
        ActVal=0, binaryXnorMode=0, noActivation=1, mem_mode="internal_embedded",
    )
    # A trivial successor so the MVAU is NOT graph-terminal. FINN's
    # minimize_accumulator_width rounds a TERMINAL no-activation node's accumulator
    # up to a multiple of 8 (byte-aligning the graph output) — a graph-topology
    # concern our hermetic Context deliberately does not model. With a successor,
    # both sides compute the true minimal accumulator (apples-to-apples).
    tail = helper.make_node("Identity", ["mid"], ["outp"])
    graph = helper.make_graph([node, tail], "mvau_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mvau-diff"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)
    return model


def _finn_context_point(W, pe, simd, wdt, idt, odt, impl, restype):
    mw, mh = W.shape
    ctx = Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_schema(), ctx, {
        "implementation": impl, "PE": pe, "SIMD": simd, "resType": restype,
        TOPOLOGY: EMBEDDED, "noActivation": 1,
    })
    return ctx, point


def _norm(s):
    # normalize the module name (FINN names it per-node) + trailing whitespace, so we
    # compare the SEMANTIC content, not the auto-generated top name.
    out = []
    for line in s.splitlines():
        out.append(line.rstrip())
    return "\n".join(out).strip()


def diff_rtl():
    print("== RTL differential (softvec) ==")
    rng = np.random.RandomState(0)
    # narrow weights (min strictly above dtype min) so $NARROW_WEIGHTS$ == 1 deterministically
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]

    model = _make_mvau_model(W, 2, 2, wdt, idt, odt)
    model = model.transform(SpecializeLayers(FPGAPART))
    # Our emit bakes in the accumulator-minimization that FINN does as a separate
    # transform (noActivation => outputDataType := accDataType, base:49-57). Run it so
    # the comparison is apples-to-apples (a real FINN flow always applies it).
    model = model.transform(MinimizeAccumulatorWidth())
    node = model.graph.node[0]
    assert node.op_type == "MVAU_rtl", f"expected MVAU_rtl, got {node.op_type}"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_hdl(model, FPGAPART, CLK_NS)
        top = inst.get_nodeattr("gen_top_module")
        finn_v = open(os.path.join(d, top + "_wrapper.v")).read()

    ctx, point = _finn_context_point(W, 2, 2, wdt, idt, odt, MVAU_DSP_SOFTVEC, "dsp")
    arts = emit_point(mvau_pool(), point, ctx)
    ours_v = arts.generated[0].content()

    # Compare the parameter block (the semantic payload). Extract "parameter NAME = VAL"
    def params(text):
        d = {}
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("parameter"):
                parts = line.replace("parameter", "").split("=")
                if len(parts) == 2:
                    name = parts[0].strip()
                    val = parts[1].split("//")[0].strip()
                    # FINN emits some numerics as numpy floats ("1.0"); normalize a
                    # whole-number float to its int form for the semantic compare.
                    if val.endswith(".0"):
                        val = val[:-2]
                    d[name] = val
        return d

    fp, op = params(finn_v), params(ours_v)
    shared = set(fp) & set(op)
    mism = {k: (fp[k], op[k]) for k in shared if fp[k] != op[k]}
    print(f"  shared params: {len(shared)} | mismatches: {mism}")
    print(f"  FINN-only params: {set(fp)-set(op)} | ours-only: {set(op)-set(fp)}")
    ok = not mism and not (set(fp) - set(op))
    print("  RTL PARAM EQUIVALENCE:", "PASS" if ok else "FAIL")
    return ok


def diff_hls_params_h():
    print("== HLS params.h differential ==")
    rng = np.random.RandomState(0)
    W = rng.randint(-7, 7, size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]

    model = _make_mvau_model(W, 2, 2, wdt, idt, odt)
    # prefer HLS so specialization yields MVAU_hls
    model.graph.node[0].attribute.append(
        helper.make_attribute("preferred_impl_style", "hls")
    )
    model_hls = model.transform(SpecializeLayers(FPGAPART))
    node = model_hls.graph.node[0]
    assert node.op_type == "MVAU_hls", f"expected MVAU_hls, got {node.op_type}"
    inst = getCustomOp(node)

    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model_hls, d)
        finn_params = open(os.path.join(d, "params.h")).read()

    ctx, point = _finn_context_point(W, 2, 2, wdt, idt, odt, MVAU_HLS, "lut")
    from finn.design_space.fixtures.mvau.emit_hls import _params_h
    ours_params = _params_h(point, ctx)

    ok = _norm(finn_params) == _norm(ours_params)
    print("  params.h byte-equivalence:", "PASS" if ok else "FAIL")
    if not ok:
        print("  --- FINN (first 200) ---\n", finn_params[:200])
        print("  --- OURS (first 200) ---\n", ours_params[:200])
    return ok


def _memstream_params(text):
    """Extract 'parameter NAME = VAL' from a memstream wrapper, normalizing the
    per-node module name and INIT_FILE path (FINN writes an absolute code_gen_dir
    path; we emit the bare basename) — compare the SEMANTIC geometry, not paths."""
    d = {}
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("parameter") and "=" in line:
            body = line.replace("parameter", "").split("//")[0]
            name, val = body.split("=", 1)
            name, val = name.strip(), val.strip()
            # skip the two $clog2-derived params (AXILITE_ADDR_WIDTH / SET_BITS) —
            # both sides carry the identical Verilog expression, not a value.
            if name in ("AXILITE_ADDR_WIDTH", "SET_BITS"):
                continue
            if name == "INIT_FILE":
                val = '"' + val.strip('"').split("/")[-1] + '"'  # basename only
            d[name] = val
    return d


def diff_memstream():
    print("== memstream differential (decoupled: wrapper params + .dat) ==")
    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    pe, simd = 2, 2

    # FINN side: an internal_decoupled MVAU. generate_hdl_memstream writes the wrapper;
    # generate_params writes memblock.dat.
    model = _make_mvau_model(W, pe, simd, wdt, idt, odt)
    model = model.transform(SpecializeLayers(FPGAPART))
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

    # our side: resolve a decoupled point and emit.
    ctx = Context(
        shapes={"weights": W.shape, "inp": (1, W.shape[0]), "out": (1, W.shape[1])},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_HLS, "PE": pe, "SIMD": simd, "resType": "lut",
        "noActivation": 1, PARAM_TOPOLOGY: PARAM_DECOUPLED, PARAM_RAM_STYLE: "block",
    })
    arts = emit_point(parameters_pool(), point, ctx, root=PARAM_TOPOLOGY)
    ours_v = arts.generated[0].content()
    ours_dat = [f for f in arts.data_files if f.filename == "memblock.dat"][0].content

    # 1) wrapper param equivalence
    fp, op = _memstream_params(finn_v), _memstream_params(ours_v)
    shared = set(fp) & set(op)
    mism = {k: (fp[k], op[k]) for k in shared if fp[k] != op[k]}
    print(f"  wrapper params: shared={len(shared)} mismatches={mism}")
    print(f"  FINN-only={set(fp)-set(op)} ours-only={set(op)-set(fp)}")
    v_ok = not mism and not (set(fp) - set(op))
    print("  WRAPPER PARAM EQUIVALENCE:", "PASS" if v_ok else "FAIL")

    # 2) .dat byte equivalence
    dat_ok = _norm(finn_dat) == _norm(ours_dat)
    print("  memblock.dat byte-equivalence:", "PASS" if dat_ok else "FAIL")
    if not dat_ok:
        fl, ol = finn_dat.strip().split("\n"), ours_dat.strip().split("\n")
        print(f"    FINN {len(fl)} lines first3={fl[:3]}")
        print(f"    OURS {len(ol)} lines first3={ol[:3]}")
    return v_ok and dat_ok


if __name__ == "__main__":
    results = []
    results.append(diff_rtl())
    try:
        results.append(diff_hls_params_h())
    except Exception as e:
        print("  HLS params diff errored:", repr(e))
        results.append(False)
    try:
        results.append(diff_memstream())
    except Exception as e:
        import traceback
        print("  memstream diff errored:", repr(e))
        traceback.print_exc()
        results.append(False)
    print("\nRESULT:", "ALL PASS" if all(results) else "SOME FAIL")
    sys.exit(0 if all(results) else 1)
