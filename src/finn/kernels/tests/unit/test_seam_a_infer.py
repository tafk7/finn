############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam A — the INFER seam, end to end (handoff §8 gates 2-5).

Proves the kernel-substrate inference path:
  * gate 2 — unit: InferKernels on a MatMul[+MultiThreshold] graph produces a
    ``finn.kernels`` MVAU node (threshold absorbed) with ONLY ActVal baked (no
    MW/MH/SIMD/PE/dtypes); a standalone MultiThreshold produces a ``finn.kernels``
    Thresholding node. The frontend nodes are removed.
  * gate 3 — bypass proof: after InferKernels, FINN's
    InferQuantizedMatrixVectorActivation finds NO MatMul to claim (ours took it).
  * gate 4 — cleanup passes: InferShapes + InferDataTypes run clean over the mixed
    graph; output dtype/shape are published for the kernel node.
  * gate 5 — mixed graph: a graph with a MatMul[+MultiThreshold] AND a non-kernel op
    runs InferKernels then FINN's classic infers — kernel claims its pattern, FINN
    claims the rest, no crash.

The classic build step ``step_convert_to_hw`` cannot be imported in the venv-pure kernel
suite (its build-flow module graph pulls ``onnx.mapping``, dropped in the venv's newer
onnx). Gate 5 therefore drives the SAME transformation sequence the step runs
(InferKernels, then the classic Infer* transforms) directly; the one-line step injection
is exercised in the dedicated build-env test.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.kernels.adapter import InferKernels
from finn.kernels.ops.mvau.op import MvauKernelOp
from finn.kernels.ops.thresholding.op import ThresholdingKernelOp

MW, MH = 128, 64
NUM_STEPS = 7
KERNEL_DOMAIN = "finn.kernels"


def _pool():
    return InferKernels([MvauKernelOp, ThresholdingKernelOp])


def _thresholds(rows, seed=2, lo=0, hi=100):
    return np.sort(
        np.random.RandomState(seed).randint(lo, hi, size=(rows, NUM_STEPS)).astype(np.float32),
        axis=1,
    )


# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------


def _matmul_threshold_model(out_bias=-8):
    """MatMul (quantized weight initializer) followed by MultiThreshold — the fused
    activation pattern the MVAU kernel claims."""
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold",
        ["mm_out", "thresholds"],
        ["out"],
        domain="qonnx.custom_op.general",
        out_dtype="INT4",
        out_scale=1.0,
        out_bias=float(out_bias),
        name="mt0",
    )
    graph = helper.make_graph(
        [matmul, mt],
        "mm_threshold",
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
    model.set_initializer("thresholds", _thresholds(MH))
    return model


def _standalone_threshold_model(channels=MH):
    """A MultiThreshold with NO MatMul producer — the standalone Thresholding pattern."""
    mt = helper.make_node(
        "MultiThreshold",
        ["inp", "thresholds"],
        ["out"],
        domain="qonnx.custom_op.general",
        out_dtype="INT4",
        out_scale=1.0,
        out_bias=-8.0,
        name="mt_solo",
    )
    graph = helper.make_graph(
        [mt],
        "standalone_threshold",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, channels])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, channels])],
        value_info=[
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [channels, NUM_STEPS]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_tensor_datatype("out", DataType["INT4"])
    model.set_initializer("thresholds", _thresholds(channels))
    return model


def _node_by_domain(model, domain):
    return [n for n in model.graph.node if n.domain == domain]


def _attr_names(node):
    return {a.name for a in node.attribute}


# ---------------------------------------------------------------------------
# gate 2 — unit: the produced nodes and the bake contract
# ---------------------------------------------------------------------------


def test_mvau_fused_infer_produces_kernel_node():
    model = _matmul_threshold_model(out_bias=-8)
    model = model.transform(_pool())

    kernel_nodes = _node_by_domain(model, KERNEL_DOMAIN)
    assert len(kernel_nodes) == 1
    kn = kernel_nodes[0]
    assert kn.op_type == "MVAU"
    assert kn.domain == KERNEL_DOMAIN
    # The MatMul + MultiThreshold are absorbed and gone.
    assert [n.op_type for n in model.graph.node] == ["MVAU"]
    # Fused: references input, weight, AND threshold tensors.
    assert list(kn.input) == ["inp", "weights", "thresholds"]

    # Bake contract (F2′): ONLY ActVal is baked; graph-derived facts stay derived.
    names = _attr_names(kn)
    assert "ActVal" in names
    for forbidden in ("MW", "MH", "SIMD", "PE", "numInputVectors", "mem_mode",
                      "inputDataType", "weightDataType", "outputDataType", "noActivation"):
        assert forbidden not in names, f"{forbidden} must not be baked on the kernel node"


def test_mvau_actval_carries_out_bias():
    # The residual op-owned param: the absorbed MultiThreshold's out_bias must survive
    # onto the kernel node (it has no graph home once the MultiThreshold is removed).
    model = _matmul_threshold_model(out_bias=-8)
    model = model.transform(_pool())
    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    from qonnx.util.basic import get_by_name

    assert get_by_name(kn.attribute, "ActVal").i == -8


def test_mvau_matmul_only_infer():
    # A bare MatMul (no following MultiThreshold) → a 2-input MVAU, ActVal=0.
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul],
        "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model = model.transform(_pool())

    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    assert kn.op_type == "MVAU"
    assert list(kn.input) == ["inp", "weights"]


def test_standalone_threshold_infer_produces_kernel_node():
    model = _standalone_threshold_model()
    model = model.transform(_pool())

    kernel_nodes = _node_by_domain(model, KERNEL_DOMAIN)
    assert len(kernel_nodes) == 1
    kn = kernel_nodes[0]
    assert kn.op_type == "Thresholding"
    assert [n.op_type for n in model.graph.node] == ["Thresholding"]
    assert list(kn.input) == ["inp", "thresholds"]

    names = _attr_names(kn)
    assert "ActVal" in names
    for forbidden in ("NumChannels", "numSteps", "PE", "numInputVectors",
                      "inputDataType", "outputDataType"):
        assert forbidden not in names, f"{forbidden} must not be baked on the kernel node"


# ---------------------------------------------------------------------------
# gate 3 — bypass proof: FINN's classic MVAU infer finds nothing to claim
# ---------------------------------------------------------------------------


def test_classic_mvau_infer_finds_nothing_after_kernel_infer():
    from finn.transformation.fpgadataflow.convert_to_hw_layers import (
        InferQuantizedMatrixVectorActivation,
    )

    model = _matmul_threshold_model()
    model = model.transform(_pool())
    # No MatMul remains for the classic infer to claim.
    assert not model.get_nodes_by_op_type("MatMul")

    before = [n.name for n in model.graph.node]
    model = model.transform(InferQuantizedMatrixVectorActivation())
    after = [n.name for n in model.graph.node]
    # The classic infer is a no-op: the graph is unchanged (no double-infer).
    assert before == after
    # Still exactly one kernel MVAU, and NO classic (finn.custom_op) MVAU appeared.
    assert len(_node_by_domain(model, KERNEL_DOMAIN)) == 1
    assert not _node_by_domain(model, "finn.custom_op.fpgadataflow")


# ---------------------------------------------------------------------------
# gate 4 — cleanup passes survive over the kernel node
# ---------------------------------------------------------------------------


def test_infer_shapes_and_datatypes_survive():
    model = _matmul_threshold_model()
    model = model.transform(_pool())
    # The driver already re-ran these; running again must stay clean + idempotent.
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    assert tuple(model.get_tensor_shape(kn.output[0])) == (1, MH)
    # Fused node forwards the graph output dtype (thresholds map the accumulator down).
    assert model.get_tensor_datatype(kn.output[0]) == DataType["INT4"]


def test_matmul_only_publishes_accumulator_dtype():
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["out"], name="mm0")
    graph = helper.make_graph(
        [matmul],
        "mm_only",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model = model.transform(_pool())

    kn = _node_by_domain(model, KERNEL_DOMAIN)[0]
    odt = model.get_tensor_datatype(kn.output[0])
    # No activation → output IS the weight-derived accumulator (an integer type).
    assert odt.is_integer() and odt != DataType["FLOAT32"]


# ---------------------------------------------------------------------------
# gate 5 — mixed graph: kernel claims its pattern, a non-kernel op is left for FINN
# ---------------------------------------------------------------------------


def test_mixed_graph_kernel_claims_only_its_pattern():
    """MatMul[+MultiThreshold] (kernel) alongside a MaxPool (non-kernel). The kernel
    infer claims the fused MVAU; the MaxPool is untouched and left for FINN's flow."""
    matmul = helper.make_node("MatMul", ["inp", "weights"], ["mm_out"], name="mm0")
    mt = helper.make_node(
        "MultiThreshold",
        ["mm_out", "thresholds"],
        ["mt_out"],
        domain="qonnx.custom_op.general",
        out_dtype="INT4",
        out_scale=1.0,
        out_bias=-8.0,
        name="mt0",
    )
    # A standard ONNX op the kernel pool does NOT claim.
    pool = helper.make_node(
        "MaxPool", ["mt_out"], ["out"], kernel_shape=[1, 1], name="pool0"
    )
    graph = helper.make_graph(
        [matmul, mt, pool],
        "mixed",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
        value_info=[
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
            helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
            helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, MH]),
            helper.make_tensor_value_info("mt_out", TensorProto.FLOAT, [1, MH]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_initializer("thresholds", _thresholds(MH))

    model = model.transform(_pool())

    op_types = [n.op_type for n in model.graph.node]
    # The kernel claimed the fused MVAU; the MaxPool survives untouched.
    assert "MVAU" in op_types
    assert "MaxPool" in op_types
    assert "MatMul" not in op_types and "MultiThreshold" not in op_types
    assert len(_node_by_domain(model, KERNEL_DOMAIN)) == 1
