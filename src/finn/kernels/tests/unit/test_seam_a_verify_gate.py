############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam A VERIFY-FIRST GATE (handoff §4).

The atomicity of the infer seam rests on ONE fact: an UNRESOLVED kernel node
(post-infer, pre-resolve) must be instantiable enough to answer its normal output
shape and output dtype, because the generic InferShapes/InferDataTypes passes that run
after inference call ``make_shape_compatible_op(model)`` and ``infer_node_datatype(model)``
on it.

This mimics exactly what ``infer_from`` will produce: a ``finn.kernels`` MVAU node
referencing the frontend tensors, with NO ``implementation``/``SIMD``/``PE`` nodeattrs
set (only ``ActVal``, the one residual op-owned param baked from the absorbed
MultiThreshold's out_bias). The node is instantiated model-aware and both cleanup-pass
entrypoints must succeed on it.

Expected: PASS → the seam is atomic; proceed to build the driver.
If either REQUIRES a resolved axis → STOP: dtype publication must defer to resolve
(Seam B), which changes the plan.

This is a THROWAWAY gate test — it constructs the node by hand (the driver + registration
do not exist yet), so it instantiates ``MvauKernelOp`` directly rather than via
``getCustomOp``.
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.kernels.compute.mvau.op import MvauKernelOp

MW, MH = 128, 64
NUM_STEPS = 7


def _fused_model():
    """A frontend-shaped graph AFTER inference would have run: a single unresolved
    ``finn.kernels`` MVAU node (fused MatMul+MultiThreshold) referencing the input,
    weight, and threshold tensors. NO implementation/SIMD/PE — only ActVal baked."""
    node = helper.make_node(
        "MVAU",
        ["inp", "weights", "thresholds"],
        ["out"],
        domain="finn.kernels",
        # The ONE residual op-owned param (absorbed MultiThreshold out_bias). Deliberately
        # nonzero to prove a signed-activation node carries it through the unresolved node.
        ActVal=-4,
        name="MVAU_fused",
    )
    value_info = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
        helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [MH, NUM_STEPS]),
    ]
    graph = helper.make_graph(
        [node],
        "seam_a_gate_fused",
        value_info,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_tensor_datatype("out", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    model.set_tensor_datatype("thresholds", DataType["INT16"])
    thr = np.sort(
        np.random.RandomState(2).randint(-50, 50, size=(MH, NUM_STEPS)).astype(np.float32),
        axis=1,
    )
    model.set_initializer("thresholds", thr)
    return model


def _matmul_only_model():
    """The no-activation case: a 2-input unresolved MVAU node (bare MatMul, no thresholds).
    Its output IS the weight-derived accumulator — the harder case for dtype publication,
    since the output dtype is value-derived, not a graph forward."""
    node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["out"],
        domain="finn.kernels",
        ActVal=0,
        name="MVAU_matmul_only",
    )
    value_info = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, MW]),
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, [MW, MH]),
    ]
    graph = helper.make_graph(
        [node],
        "seam_a_gate_matmul",
        value_info,
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, MH])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("weights", DataType["INT8"])
    model.set_initializer("weights", np.ones((MW, MH), dtype=np.float32))
    return model


def _assert_no_folding_axes(node):
    """The unresolved node carries NONE of the resolve-time axes."""
    attr_names = {a.name for a in node.attribute}
    for forbidden in ("backend", "SIMD", "PE"):
        assert forbidden not in attr_names, f"{forbidden} must be unset on an unresolved node"


def test_gate_fused_unresolved_answers_shape_and_dtype():
    model = _fused_model()
    node = model.graph.node[0]
    _assert_no_folding_axes(node)

    inst = MvauKernelOp(node).attach_model(model)

    # (a) make_shape_compatible_op(model) → a valid const-shape op.
    shape_op = inst.make_shape_compatible_op(model)
    assert shape_op is not None
    # It is a Constant node projecting the normal output shape (1, MH).
    assert inst.get_normal_output_shape(0) == (1, MH)

    # (b) infer_node_datatype(model) → publishes a sane output dtype, no crash.
    inst.infer_node_datatype(model)
    odt = model.get_tensor_datatype("out")
    assert odt is not None
    # Fused (has thresholds) → the activation maps the accumulator down to the graph dtype.
    assert odt == DataType["INT8"]


def test_gate_matmul_only_unresolved_derives_accumulator():
    model = _matmul_only_model()
    node = model.graph.node[0]
    _assert_no_folding_axes(node)

    inst = MvauKernelOp(node).attach_model(model)

    shape_op = inst.make_shape_compatible_op(model)
    assert shape_op is not None
    assert inst.get_normal_output_shape(0) == (1, MH)

    # No thresholds → output IS the weight-derived accumulator; must publish a real int type
    # on the UNRESOLVED node (independent of the unset implementation/folding axes).
    inst.infer_node_datatype(model)
    odt = model.get_tensor_datatype("out")
    assert odt.is_integer() and odt != DataType["FLOAT32"]
