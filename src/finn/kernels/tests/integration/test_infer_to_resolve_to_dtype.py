############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The grand path — INFER → INSTANTIATE → RESOLVE → ROUTE, self-consistent (NEW).

The single flow that proves the seams compose: a frontend MatMul(+MultiThreshold) is
inferred to a finn.kernels node (INFER), instantiated model-aware (INSTANTIATE), specialized
by committing a backend (RESOLVE), and the routing + published output dtype + folded
shapes/stream widths/exp cycles are all mutually consistent (ROUTE). Highest
coverage-per-test in the suite: a single traversal catches cross-seam contract mismatches a
dozen isolated getter tests would miss.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.infer_kernels import InferKernels
from finn.kernels.compute.mvau.op import MvauKernelOp
from finn.kernels.compute.thresholding.op import ThresholdingKernelOp
from finn.kernels.ir.routing import is_specialized, kernel_hw_language
from finn.util.fpgadataflow import is_fpgadataflow_node, is_hls_node, is_rtl_node

pytestmark = pytest.mark.integration

MW, MH = 128, 64
NUM_STEPS = 7
KERNEL_DOMAIN = "finn.kernels"


def _pool():
    return InferKernels([MvauKernelOp, ThresholdingKernelOp])


def _matmul_only_model():
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul], "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _matmul_threshold_model():
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold", ["mm_out", "thresholds"], ["out"],
        domain="qonnx.custom_op.general", out_dtype="INT4", out_scale=1.0, out_bias=-8.0, name="mt0",
    )
    graph = helper.make_graph(
        [matmul, mt], "mm_threshold",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["INT4"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_initializer(
        "thresholds",
        np.sort(np.random.RandomState(2).randint(0, 100, size=(MH, NUM_STEPS)).astype(np.float32), axis=1),
    )
    return model


def _kernel_node(model):
    return [n for n in model.graph.node if n.domain == KERNEL_DOMAIN][0]


def test_matmul_only_grand_path_self_consistent():
    # INFER: MatMul → unspecialized finn.kernels MVAU.
    model = _matmul_only_model().transform(_pool())
    node = _kernel_node(model)
    assert node.op_type == "MVAU"
    # ROUTE (unspecialized): a family member, but not HW-ready in any language.
    assert is_specialized(node) is False
    assert kernel_hw_language(node) is None
    assert is_fpgadataflow_node(node) is True

    # INSTANTIATE: output dtype is BACKEND-SCOPED, so it is DEFERRED on an unspecialized
    # node — infer_node_datatype publishes the raw graph output dtype (no committed backend
    # ⇒ no realized accumulator type yet).
    inst = model.get_customop_wrapper(node)
    inst.infer_node_datatype(model)
    odt_unspec = model.get_tensor_datatype(node.output[0])
    assert odt_unspec == DataType["FLOAT32"]  # the graph's declared out dtype, unrefined

    # RESOLVE: commit an HLS backend + fold dials.
    inst.set_nodeattr("backend", "mvau_hls")
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 8)

    # ROUTE now agrees: specialized, HLS language, routes as an HLS node.
    assert is_specialized(node) is True
    assert kernel_hw_language(node) == "hls"
    assert is_hls_node(node) is True and is_rtl_node(node) is False

    # RESOLVE consistency: folded shapes/widths/cycles are mutually consistent.
    inst = model.get_customop_wrapper(node)
    folded_out = tuple(inst.get_folded_output_shape(0))
    assert folded_out[-1] == 8  # PE folds MH → last dim = PE
    assert inst.get_instream_width(0) > 0
    assert inst.get_outstream_width(0) > 0
    assert inst.get_exp_cycles() == (MW // 8) * (MH // 8)
    # RESOLVE refines the published output dtype: now the backend-derived weight/accumulator
    # type (integer, no thresholds), replacing the deferred raw graph dtype.
    inst.infer_node_datatype(model)
    odt_spec = model.get_tensor_datatype(node.output[0])
    assert odt_spec.is_integer()
    assert odt_spec != odt_unspec


def test_fused_grand_path_publishes_graph_output_dtype():
    # INFER: MatMul+MultiThreshold → fused MVAU (thresholds absorbed).
    model = _matmul_threshold_model().transform(_pool())
    model = model.transform(InferShapes()).transform(InferDataTypes())
    node = _kernel_node(model)
    assert list(node.input) == ["inp", "weights", "thresholds"]

    # RESOLVE: specialize to an RTL DSP core (thresholds are baked/constant on the core).
    inst = model.get_customop_wrapper(node)
    inst.set_nodeattr("backend", "mvau_hls")  # HLS core supports fused thresholds
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 8)

    # ROUTE agrees; the fused node forwards the graph output dtype (thresholds map down).
    assert kernel_hw_language(node) == "hls"
    inst = model.get_customop_wrapper(node)
    inst.infer_node_datatype(model)
    assert model.get_tensor_datatype(node.output[0]) == DataType["INT4"]
    assert tuple(model.get_tensor_shape(node.output[0])) == (1, MH)
    # folded output width reads the same stream width the getter derives — no divergence.
    assert tuple(inst.get_folded_output_shape(0))[-1] == 8
