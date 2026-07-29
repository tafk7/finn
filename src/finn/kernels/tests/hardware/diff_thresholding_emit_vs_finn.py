############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Differential check: our hermetic Thresholding emit vs FINN's own codegen.

Runs INSIDE the FINN Docker container. Drives FINN's real Thresholding_hls /
Thresholding_rtl codegen on a single-node model, then runs our design_space emit on the
equivalent (Point, Context), and diffs the generated artifacts:

* HLS thresh.h — our shared ``layout`` (THRESHOLD traversal, CPP_HEADER) vs FINN's
  ``make_weight_file("hls_header")`` (the SEPARABLE static-schedule ROM). Byte-equivalence.
* RTL threshs_*.dat — our ``_threshold_dat_files`` (the data-dependent binary-search
  scatter, FUSED, NOT via layout) vs FINN's ``make_weight_file`` embedded branch.

Usage (from finn/):
    bash run-docker.sh bash src/finn/kernels/tests/hardware/run_diff_thresholding.sh
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

from finn.kernels.space import Context, emit_point, resolve
from finn.kernels.compute.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_kernel_schema,
    thresholding_pool,
)

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
CLK_NS = 5.0


def _sorted_thresholds(channels, steps, hi=50):
    return np.sort(
        np.random.RandomState(0).randint(0, hi, size=(channels, steps)).astype(np.float32),
        axis=-1,
    )


def _make_thresholding_model(T, pe, idt, tdt, odt):
    channels, steps = T.shape
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, channels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, channels])
    node = helper.make_node(
        "Thresholding", ["inp", "thresholds"], ["mid"],
        domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
        NumChannels=channels, PE=pe, numSteps=steps,
        inputDataType=idt.name, weightDataType=tdt.name, outputDataType=odt.name,
        ActVal=0, numInputVectors=[1],
    )
    tail = helper.make_node("Identity", ["mid"], ["outp"])
    graph = helper.make_graph([node, tail], "thresholding_graph", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresholding-diff"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("thresholds", tdt)
    model.set_initializer("thresholds", T)
    return model


def _our_ctx(T, idt, tdt, odt):
    channels, steps = T.shape
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": tdt, "inp": idt, "out": odt},
        initializers={"thresholds": T},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )


def _norm(s):
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()


def _norm_header_ws(s):
    # FINN's thresh.h header uses a multi-line f-string with a "threshs \" continuation
    # that leaves a run of spaces before "=". Collapse internal whitespace runs on the
    # FIRST (declaration) line so the SEMANTIC header + the initializer body compare equal;
    # the body lines (which _norm already rstrips) are unaffected.
    import re
    lines = _norm(s).splitlines()
    if lines:
        lines[0] = re.sub(r"[ \t]+", " ", lines[0])
    return "\n".join(lines)


def diff_hls_thresh_h():
    print("== HLS thresh.h differential ==")
    # Full-range threshold values (spanning the UINT8 domain) so our value-narrowing
    # thresholdDataType derive lands on the same dtype FINN's header uses (UINT8) — this
    # isolates the SERIALIZER (the layout body), not the dtype policy. A narrowed-value
    # case would diverge only on the header dtype, which is intentional kernel-op behavior.
    T = np.sort(
        np.array([[0, 40, 80, 120, 160, 200, 255],
                  [0, 40, 80, 120, 160, 200, 255],
                  [0, 40, 80, 120, 160, 200, 255],
                  [0, 40, 80, 120, 160, 200, 255]], dtype=np.float32),
        axis=-1,
    )
    idt, tdt, odt = DataType["UINT8"], DataType["UINT8"], DataType["UINT3"]

    model = _make_thresholding_model(T, 2, idt, tdt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "hls"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "Thresholding_hls", f"expected Thresholding_hls, got {node.op_type}"
    inst = getCustomOp(node)
    inst.set_nodeattr("mem_mode", "internal_embedded")
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        inst.generate_params(model, d)
        finn_thresh = open(os.path.join(d, "thresh.h")).read()

    ctx = _our_ctx(T, idt, tdt, odt)
    point = resolve(thresholding_kernel_schema(), ctx, {"backend": THRESHOLDING_HLS, "PE": 2})
    from finn.kernels.compute.thresholding.emit_hls import _thresh_h
    ours_thresh = _thresh_h(point, ctx)

    ok = _norm_header_ws(finn_thresh) == _norm_header_ws(ours_thresh)
    print("  thresh.h byte-equivalence:", "PASS" if ok else "FAIL")
    if not ok:
        print("  --- FINN (first 240) ---\n", finn_thresh[:240])
        print("  --- OURS (first 240) ---\n", ours_thresh[:240])
    return ok


def diff_rtl_dat():
    print("== RTL threshs_*.dat differential ==")
    T = _sorted_thresholds(4, 7)
    idt, tdt, odt = DataType["UINT8"], DataType["UINT8"], DataType["UINT3"]

    model = _make_thresholding_model(T, 2, idt, tdt, odt)
    model.graph.node[0].attribute.append(helper.make_attribute("preferred_impl_style", "rtl"))
    model = model.transform(SpecializeLayers(FPGAPART))
    node = model.graph.node[0]
    assert node.op_type == "Thresholding_rtl", f"expected Thresholding_rtl, got {node.op_type}"
    inst = getCustomOp(node)
    with tempfile.TemporaryDirectory() as d:
        inst.set_nodeattr("code_gen_dir_ipgen", d)
        # FINN writes {node_name}_threshs_{pe}_{stage}.dat via generate_params/make_weight_file.
        inst.generate_params(model, d)
        # Strip the node-name prefix so names align with our bare threshs_{pe}_{stage}.dat.
        finn_dats = {}
        for f in os.listdir(d):
            if "threshs_" in f and f.endswith(".dat"):
                bare = f[f.index("threshs_"):]
                finn_dats[bare] = open(os.path.join(d, f)).read()

    ctx = _our_ctx(T, idt, tdt, odt)
    point = resolve(thresholding_kernel_schema(), ctx, {"backend": THRESHOLDING_RTL, "PE": 2})
    arts = emit_point(thresholding_pool(), point, ctx)
    ours_dats = {}
    for f in arts.data_files:
        c = f.content
        ours_dats[f.filename] = c() if callable(c) else c

    same_names = set(finn_dats) == set(ours_dats)
    print(f"  file set: FINN={len(finn_dats)} ours={len(ours_dats)} match={same_names}")
    if not same_names:
        print(f"    FINN names: {sorted(finn_dats)}")
        print(f"    OURS names: {sorted(ours_dats)}")
    all_ok = same_names
    for name in sorted(set(finn_dats) & set(ours_dats)):
        if _norm(finn_dats[name]) != _norm(ours_dats[name]):
            all_ok = False
            print(f"  MISMATCH {name}: FINN={_norm(finn_dats[name])!r} OURS={_norm(ours_dats[name])!r}")
    print("  threshs_*.dat byte-equivalence:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    results = []
    for fn in (diff_hls_thresh_h, diff_rtl_dat):
        try:
            results.append(fn())
        except Exception as e:
            import traceback
            print(f"  {fn.__name__} errored:", repr(e))
            traceback.print_exc()
            results.append(False)
    print("\nRESULT:", "ALL PASS" if all(results) else "SOME FAIL")
    sys.exit(0 if all(results) else 1)
