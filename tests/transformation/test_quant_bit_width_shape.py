# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The activation handlers read a Quant's bit width as Brevitas writes it, shape (1,),
without numpy's deprecated int() of a one-element array (an error from numpy 2.4)."""
import pytest

import numpy as np
import warnings
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
)

pytestmark = pytest.mark.transform

QONNX = "qonnx.custom_op.general"


@pytest.mark.parametrize("shape", [(), (1,)])
def test_activation_bit_width_shape(shape):
    f32 = np.float32
    params = [
        ("s", np.full(shape, 0.5, f32)),
        ("z", np.zeros(shape, f32)),
        ("b", np.full(shape, 4.0, f32)),
    ]
    nodes = [
        helper.make_node("Relu", ["x"], ["a"]),
        helper.make_node(
            "Quant",
            ["a", "s", "z", "b"],
            ["y"],
            domain=QONNX,
            signed=0,
            narrow=0,
            rounding_mode="ROUND",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[numpy_helper.from_array(a, n) for n, a in params],
    )
    opsets = [helper.make_opsetid("", 13), helper.make_opsetid(QONNX, 1)]
    model = ModelWrapper(qonnx_make_model(graph, opset_imports=opsets)).transform(InferShapes())
    x = np.asarray([[-1.0, 0.3, 2.2, 9.0]], dtype=np.float32)
    expected = execute_onnx(model, {"x": x})["y"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        model = model.transform(ConvertQuantActToMultiThreshold())
    # qonnx's own conversions are qonnx's to fix
    finn_src = finn.__path__[0]
    assert not [w for w in caught if w.filename.startswith(finn_src)]
    assert "MultiThreshold" in [n.op_type for n in model.graph.node]
    assert np.array_equal(execute_onnx(model, {"x": x})["y"], expected)
