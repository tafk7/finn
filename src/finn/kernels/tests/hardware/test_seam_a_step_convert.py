############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Seam A — the REAL build-step injection (handoff §8 gate 5).

Drives the actual ``step_convert_to_hw`` build-flow function (not just the transformation
sequence) over a mixed graph, proving the one-line ``InferKernels`` injection claims the
kernel pattern before FINN's classic infers, which then handle the remainder.

This lives under ``tests/hardware/`` because importing ``step_convert_to_hw`` pulls FINN's
full build-flow module graph (which needs the pinned ``onnx==1.17`` providing
``onnx.mapping``) — unavailable in the venv-pure unit suite. Run it in the FINN build env
(the docker image or a pinned venv).
"""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

MW, MH = 128, 64
NUM_STEPS = 7
KERNEL_DOMAIN = "finn.kernels"


def _mixed_model():
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
    pool = helper.make_node("MaxPool", ["mt_out"], ["out"], kernel_shape=[1, 1], name="pool0")
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
    model.set_initializer(
        "thresholds",
        np.sort(
            np.random.RandomState(2).randint(0, 100, size=(MH, NUM_STEPS)).astype(np.float32),
            axis=1,
        ),
    )
    return model


def test_step_convert_to_hw_kernel_claims_pattern():
    from finn.builder.build_dataflow_config import DataflowBuildConfig
    from finn.builder.build_dataflow_steps import step_convert_to_hw

    model = _mixed_model()
    # Minimal config: standalone_thresholds True so the classic thresholding path is also
    # exercised on any remainder (there is none here — the kernel absorbed it).
    cfg = DataflowBuildConfig(
        output_dir="/tmp/seam_a_step_convert",
        synth_clk_period_ns=10.0,
        generate_outputs=[],
        standalone_thresholds=True,
    )
    model = step_convert_to_hw(model, cfg)

    op_types = [n.op_type for n in model.graph.node]
    kernel_nodes = [n for n in model.graph.node if n.domain == KERNEL_DOMAIN]
    assert len(kernel_nodes) == 1
    assert kernel_nodes[0].op_type == "MVAU"
    assert "MaxPool" in op_types
    assert "MatMul" not in op_types and "MultiThreshold" not in op_types
