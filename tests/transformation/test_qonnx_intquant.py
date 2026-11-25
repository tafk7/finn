# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

QONNX_DOMAIN = "qonnx.custom_op.general"


def quantizer(op_type, inp, out, prefix, signed, rounding_mode="ROUND"):
    return helper.make_node(
        op_type,
        [inp, prefix + "_scale", prefix + "_zeropt", prefix + "_bitwidth"],
        [out],
        domain=QONNX_DOMAIN,
        signed=signed,
        narrow=0,
        rounding_mode=rounding_mode,
    )


def initializer(name, value):
    return helper.make_tensor(name, TensorProto.FLOAT, np.shape(value), np.ravel(value))


def make_weight_relu_model(op_type, act_bitwidth=3.0, weight_zeropt=0.0):
    """A weight quantizer before a MatMul and an activation quantizer after a Relu,
    both of type op_type."""
    weights = [[0.3, -0.7, 1.2], [0.9, 0.1, -1.1], [-0.4, 0.6, 0.2], [0.5, -0.2, 0.8]]
    nodes = [
        quantizer(op_type, "w", "wq", "w", signed=1),
        helper.make_node("MatMul", ["x", "wq"], ["m"]),
        helper.make_node("Relu", ["m"], ["r"]),
        quantizer(op_type, "r", "y", "a", signed=0),
    ]
    inits = [
        initializer("w", weights),
        initializer("w_scale", 0.25),
        initializer("w_zeropt", weight_zeropt),
        initializer("w_bitwidth", 4.0),
        initializer("a_scale", 0.5),
        initializer("a_zeropt", 0.0),
        initializer("a_bitwidth", act_bitwidth),
    ]
    graph = helper.make_graph(
        nodes,
        "weight_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=inits,
    )
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(model).transform(InferShapes())


def execute(model, x):
    return oxe.execute_onnx(model, {"x": x})["y"]


@pytest.mark.transform
def test_intquant_converts_as_quant():
    rng = np.random.default_rng(0)
    inputs = [
        np.array([[0.7, -1.3, 2.1, 0.4]], dtype=np.float32),
        *rng.uniform(-3, 3, size=(16, 1, 4)).astype(np.float32),
    ]
    converted = {}
    for op_type in ["Quant", "IntQuant"]:
        model = make_weight_relu_model(op_type)
        converted[op_type] = model.transform(ConvertQONNXtoFINN())
        ops = [n.op_type for n in converted[op_type].graph.node]
        assert ops == ["MatMul", "Mul", "MultiThreshold", "Mul"]
        mt = converted[op_type].get_nodes_by_op_type("MultiThreshold")[0]
        out_dtype = helper.get_attribute_value(
            next(a for a in mt.attribute if a.name == "out_dtype")
        )
        assert out_dtype == b"UINT3"
        for x in inputs:
            assert np.array_equal(execute(converted[op_type], x), execute(model, x))
    for x in inputs:
        assert np.array_equal(execute(converted["IntQuant"], x), execute(converted["Quant"], x))
    for a, b in zip(converted["Quant"].graph.initializer, converted["IntQuant"].graph.initializer):
        assert np.array_equal(
            converted["Quant"].get_initializer(a.name),
            converted["IntQuant"].get_initializer(b.name),
        )


@pytest.mark.transform
@pytest.mark.parametrize("op_type", ["Quant", "IntQuant"])
def test_wide_intquant_left_as_wide_quant(op_type):
    # The default filter leaves an activation quantizer wider than 8 bits in the graph
    model = make_weight_relu_model(op_type, act_bitwidth=16.0)
    with pytest.warns(UserWarning, match="was not converted to a MultiThreshold node"):
        converted = model.transform(ConvertQONNXtoFINN())
    ops = [n.op_type for n in converted.graph.node]
    assert ops == ["MatMul", "Mul", "Relu", op_type]
    x = np.array([[0.7, -1.3, 2.1, 0.4]], dtype=np.float32)
    assert np.array_equal(execute(converted, x), execute(model, x))


def make_identity_model(rounding_mode, scale, zeropt, bitwidth):
    """A signed IntQuant on its own, on a [1, 73] input."""
    graph = helper.make_graph(
        [quantizer("IntQuant", "x", "y", "a", signed=1, rounding_mode=rounding_mode)],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 73])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 73])],
        initializer=[
            initializer("a_scale", scale),
            initializer("a_zeropt", zeropt),
            initializer("a_bitwidth", bitwidth),
        ],
    )
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(model).transform(InferShapes())


@pytest.mark.transform
@pytest.mark.parametrize(
    "rounding_mode, scale, zeropt, bitwidth, reason",
    [
        ("FLOOR", 1.0, 0.0, 4.0, "its rounding mode FLOOR is not ROUND"),
        ("ROUND", -1.0, 0.0, 4.0, "its scale is not static, positive and finite"),
        ("ROUND", 0.0, 0.0, 4.0, "its scale is not static, positive and finite"),
        ("ROUND", 1.0, 1.0, 4.0, "its zero point is not a static scalar 0"),
        ("ROUND", 1.0, np.zeros(73), 4.0, "its zero point is not a static scalar 0"),
        ("ROUND", 1.0, 0.0, 1.0, "its bit width 1.0 is not an integer of at least 2"),
    ],
    ids=["floor", "negative-scale", "zero-scale", "zero-point", "zero-point-vector", "one-bit"],
)
def test_intquant_declined_left_in_place(rounding_mode, scale, zeropt, bitwidth, reason):
    # The conversion would not give these quantizers' answers, so they stay in the
    # graph and QONNX executes them, as before IntQuant was converted at all
    model = make_identity_model(rounding_mode, scale, zeropt, bitwidth)
    with pytest.warns(UserWarning, match="was not converted to a MultiThreshold node, because"):
        converted = model.transform(ConvertQONNXtoFINN())
    with pytest.warns(UserWarning, match=reason):
        model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["IntQuant"]
    x = np.arange(-9.0, 9.25, 0.25, dtype=np.float32)[np.newaxis, :]
    # at scale 0, QONNX's answer for the input 0 is NaN (0 / 0)
    assert np.array_equal(execute(converted, x), execute(model, x), equal_nan=True)


@pytest.mark.transform
def test_intquant_weights_with_zero_point_left_in_place():
    model = make_weight_relu_model("IntQuant", weight_zeropt=1.0)
    with pytest.warns(UserWarning, match="was not folded into its weights, because its zero point"):
        converted = model.transform(ConvertQONNXtoFINN())
    ops = [n.op_type for n in converted.graph.node]
    assert ops == ["IntQuant", "MatMul", "MultiThreshold", "Mul"]
    x = np.array([[0.7, -1.3, 2.1, 0.4]], dtype=np.float32)
    assert np.array_equal(execute(converted, x), execute(model, x))
