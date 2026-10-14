# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import TensorRange
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.transformation.streamline.absorb as absorb


class _Unchanged(Transformation):
    def apply(self, model):
        return (model, False)


@pytest.mark.streamline
def test_the_multithresholds_new_output_keeps_its_range(monkeypatch):
    """Absorbing an NCHW Transpose into the MultiThreshold after it gives the
    MultiThreshold a new output, which states the datatype and range of the old one (UINT2
    over [0, 2], two thresholds): the writer states them itself, ahead of the
    InferDataTypes the transformation runs after (taken out here)."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 3, 3, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4, 3, 3])
    nodes = [
        helper.make_node("Transpose", ["inp"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "MultiThreshold",
            ["nchw", "thresholds"],
            ["outp"],
            domain="qonnx.custom_op.general",
            out_dtype="UINT2",
        ),
    ]
    graph = helper.make_graph(nodes, "transpose_mt", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("thresholds", np.array([[0.0, 1.0]], dtype=np.float32))
    model = model.transform(InferShapes()).transform(InferDataTypes())
    assert model.get_tensor_range("outp") == TensorRange(0, 2)

    monkeypatch.setattr(absorb, "InferDataTypes", _Unchanged)
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    (mt,) = [node for node in model.graph.node if node.op_type == "MultiThreshold"]
    assert model.get_tensor_datatype(mt.output[0]) == DataType["UINT2"]
    assert model.get_tensor_range(mt.output[0]) == TensorRange(0, 2)
