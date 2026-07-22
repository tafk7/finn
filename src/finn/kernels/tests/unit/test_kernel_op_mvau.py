############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The MVAU FINN adapter (``MvauKernelOp``) on a real ModelWrapper — proving the
Kernel engine satisfies FINN's estimate-only (Tier 0-3) consumer surface with the
*stock* analysis transforms, no Vivado, no codegen.

Phase A (this file): the passive estimate surface — getters, the datatype write-back,
the AnnotateCycles↔dataflow_performance handshake (R7), res_estimation, shape-compat +
InferShapes, and pickle/rehydrate (R10). Phase B (folding via SetFolding) lives in
``test_kernel_op_mvau_folding.py``.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.kernels.space import Context
from finn.kernels.ops.mvau import mvau_kernel

MW, MH = 128, 64
OP_TYPE = "MVAUKernel_hls"
DOMAIN = "finn.custom_op.fpgadataflow.hls"
FPGAPART = "xcvc1902-vsva2197-2MP-e-S"


def _build_model(simd=16, pe=4, noActivation=0, annotate_out=True):
    """A single-node MVAUKernel_hls graph with geometry baked into nodeattrs."""
    node = helper.make_node(
        OP_TYPE,
        ["inp", "weights"],
        ["out"],
        domain=DOMAIN,
        backend="fpgadataflow",
        implementation="mvau_hls",
        SIMD=simd,
        PE=pe,
        noActivation=noActivation,
        inp_shape=[1, MW],
        inp_dtype="INT8",
        weights_shape=[MW, MH],
        weights_dtype="INT8",
        out_shape=[1, MH],
        out_dtype="INT32",
    )
    graph = helper.make_graph(
        [node],
        "mvau_kernel_graph",
        [
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    if annotate_out:
        model.set_tensor_datatype("out", DataType["INT32"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _ctx():
    """The equivalent pure Context, to cross-check the adapter against the engine."""
    return Context(
        shapes={"inp": (1, MW), "weights": (MW, MH), "out": (1, MH)},
        datatypes={
            "inp": DataType["INT8"],
            "weights": DataType["INT8"],
            "out": DataType["INT32"],
        },
        initializers={"weights": np.ones((MW, MH), dtype=np.float32)},
        fpgapart=FPGAPART,
    )


def _inst(model):
    return getCustomOp(model.graph.node[0])


# ---------------------------------------------------------------------------
# 1. Registration + nodeattr schema (R12 dissolved)
# ---------------------------------------------------------------------------


def test_getcustomop_resolves_to_mvau_kernel_op():
    inst = _inst(_build_model())
    assert type(inst).__name__ == "MvauKernelOp"


def test_nodeattr_types_include_axes_and_geometry():
    attrs = _inst(_build_model()).get_nodeattr_types()
    for key in ("implementation", "PE", "SIMD", "MW", "MH", "noActivation"):
        assert key in attrs, f"design axis {key} missing from nodeattr schema"
    for key in ("inp_shape", "inp_dtype", "weights_shape", "out_dtype"):
        assert key in attrs, f"geometry attr {key} missing"


# ---------------------------------------------------------------------------
# 2. The bridge is faithful: adapter getters == direct Kernel getters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simd,pe", [(16, 4), (8, 8), (128, 64)])
def test_getters_agree_with_engine(simd, pe):
    inst = _inst(_build_model(simd=simd, pe=pe))
    kernel, ctx = mvau_kernel(), _ctx()
    point = kernel.configure(ctx, {"implementation": "mvau_hls", "SIMD": simd, "PE": pe})

    assert inst.get_instream_width(0) == kernel.get_instream_width(point, ctx, 0)
    assert inst.get_instream_width(1) == kernel.get_instream_width(point, ctx, 1)
    assert inst.get_outstream_width(0) == kernel.get_outstream_width(point, ctx, 0)
    assert tuple(inst.get_folded_input_shape(0)) == kernel.get_folded_input_shape(point, ctx, 0)
    assert tuple(inst.get_folded_output_shape(0)) == kernel.get_folded_output_shape(point, ctx, 0)
    assert inst.get_exp_cycles() == kernel.get_exp_cycles(point, ctx)


def test_datatype_getters():
    inst = _inst(_build_model())
    assert inst.get_input_datatype(0) == DataType["INT8"]
    assert inst.get_input_datatype(1) == DataType["INT8"]
    assert inst.get_output_datatype(0) == DataType["INT32"]


def test_number_output_values():
    # prod(folded_out[:-1]) = MH/PE * (leading dims) = 64/4 = 16.
    assert _inst(_build_model(pe=4)).get_number_output_values() == MH // 4


# ---------------------------------------------------------------------------
# 3. infer_node_datatype: derive + propagate, idempotent
# ---------------------------------------------------------------------------


def test_infer_datatype_forwards_graph_dtype():
    model = _build_model(noActivation=0, annotate_out=True)
    _inst(model).infer_node_datatype(model)
    assert model.get_tensor_datatype("out") == DataType["INT32"]


def test_infer_datatype_derives_accumulator_under_noactivation():
    model = _build_model(noActivation=1, annotate_out=False)
    inst = _inst(model)
    inst.infer_node_datatype(model)
    first = model.get_tensor_datatype("out")
    # weight-derived accumulator type (all-ones INT8 weights, INT8 acts).
    assert first.is_integer() and first != DataType["FLOAT32"]
    inst.infer_node_datatype(model)  # idempotent
    assert model.get_tensor_datatype("out") == first


# ---------------------------------------------------------------------------
# 4. The AnnotateCycles -> dataflow_performance handshake (R7)
# ---------------------------------------------------------------------------


def test_annotate_cycles_and_dataflow_performance():
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance

    model = _build_model(simd=16, pe=4)
    model = model.transform(AnnotateCycles())
    stamped = _inst(model).get_nodeattr("cycles_estimate")
    assert stamped == _inst(model).get_exp_cycles()

    perf = model.analysis(dataflow_performance)
    assert perf["max_cycles"] == stamped
    assert perf["max_cycles_node_name"] == model.graph.node[0].name


def test_exp_cycles_per_layer():
    from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

    model = _build_model()
    cyc = model.analysis(exp_cycles_per_layer)
    assert cyc[model.graph.node[0].name] == _inst(model).get_exp_cycles()


# ---------------------------------------------------------------------------
# 5. res_estimation
# ---------------------------------------------------------------------------


def test_res_estimation():
    from finn.analysis.fpgadataflow.res_estimation import res_estimation

    model = _build_model()
    res = model.analysis(lambda m: res_estimation(m, fpgapart=FPGAPART))
    node_res = res[model.graph.node[0].name]
    for key in ("BRAM_18K", "LUT", "URAM", "DSP"):
        assert key in node_res


# ---------------------------------------------------------------------------
# 6. make_shape_compatible_op + qonnx InferShapes
# ---------------------------------------------------------------------------


def test_shape_compatible_and_infer_shapes():
    from qonnx.transformation.infer_shapes import InferShapes

    model = _build_model()
    model = model.transform(InferShapes())
    assert tuple(model.get_tensor_shape("out")) == (1, MH)


# ---------------------------------------------------------------------------
# 7. Pickle / rehydrate (R10): serialize -> reload -> identical getters
# ---------------------------------------------------------------------------


def test_serialize_rehydrate_roundtrip(tmp_path):
    model = _build_model(simd=16, pe=4)
    before = {
        "instream0": _inst(model).get_instream_width(0),
        "folded_out": tuple(_inst(model).get_folded_output_shape(0)),
        "exp_cycles": _inst(model).get_exp_cycles(),
    }
    path = str(tmp_path / "mvau_kernel.onnx")
    model.save(path)
    reloaded = ModelWrapper(path)
    inst = _inst(reloaded)
    assert inst.get_instream_width(0) == before["instream0"]
    assert tuple(inst.get_folded_output_shape(0)) == before["folded_out"]
    assert inst.get_exp_cycles() == before["exp_cycles"]
