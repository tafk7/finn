# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import qonnx.core.onnx_exec as qoxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


def make_quant_model(rounding_mode, scale, bitwidth=4, signed=1, weights=False, zeropt=0.0):
    values = np.array([[-9.0, -2.0, -0.5, 0.0, 0.25, 0.5, 0.75, 2.0, 9.0]], dtype=np.float32)
    node = helper.make_node(
        "Quant",
        ["w" if weights else "x", "scale", "zeropt", "bitwidth"],
        ["wq" if weights else "y"],
        name="quantizer",
        domain="qonnx.custom_op.general",
        signed=signed,
        narrow=0,
        rounding_mode=rounding_mode,
    )
    nodes = [node]
    inits = [
        helper.make_tensor("scale", TensorProto.FLOAT, [1], [scale]),
        helper.make_tensor("zeropt", TensorProto.FLOAT, [], [zeropt]),
        helper.make_tensor("bitwidth", TensorProto.FLOAT, [], [float(bitwidth)]),
    ]
    if weights:
        weight_values = np.diag(values.ravel())
        inits.append(
            helper.make_tensor("w", TensorProto.FLOAT, weight_values.shape, weight_values.ravel())
        )
        nodes.append(helper.make_node("MatMul", ["x", "wq"], ["y"]))
        values = np.ones_like(values)
    graph = helper.make_graph(
        nodes,
        "quant_conversion",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(values.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(values.shape))],
        initializer=inits,
    )
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(model).transform(InferShapes()), values


@pytest.mark.transform
@pytest.mark.parametrize(
    "rounding_mode, scale, bitwidth, signed, reason",
    [
        ("FLOOR", 1.0, 4, 1, "rounding mode FLOOR"),
        ("CEIL", 1.0, 4, 1, "rounding mode CEIL"),
        ("DOWN", 1.0, 4, 1, "rounding mode DOWN"),
        ("UP", 1.0, 4, 1, "rounding mode UP"),
        ("HALF_UP", 1.0, 4, 1, "rounding mode HALF_UP"),
        ("HALF_DOWN", 1.0, 4, 1, "rounding mode HALF_DOWN"),
        ("ROUND", -1.0, 4, 1, "scale is not static, positive and finite"),
        ("ROUND", 0.0, 4, 1, "scale is not static, positive and finite"),
        ("ROUND", np.inf, 4, 1, "scale is not static, positive and finite"),
        ("ROUND", -np.inf, 4, 1, "scale is not static, positive and finite"),
        ("ROUND", np.nan, 4, 1, "scale is not static, positive and finite"),
        ("ROUND", 1.0, 1, 0, "bit width 1.0 is not an integer of at least 2"),
    ],
    ids=[
        "floor",
        "ceil",
        "round-to-zero",
        "round-away-from-zero",
        "half-up",
        "half-down",
        "negative-scale",
        "zero-scale",
        "infinite-scale",
        "negative-infinite-scale",
        "nan-scale",
        "one-bit",
    ],
)
def test_quant_declined_left_in_place(rounding_mode, scale, bitwidth, signed, reason):
    model, values = make_quant_model(rounding_mode, scale, bitwidth, signed)
    with pytest.warns(
        UserWarning,
        match='The Quant node with name: "quantizer" was not converted to a MultiThreshold node, '
        f"because its {reason}",
    ):
        converted = model.transform(ConvertQONNXtoFINN())
    (quant,) = converted.get_nodes_by_op_type("Quant")
    assert quant.name == "quantizer"
    assert not converted.get_nodes_by_op_type("MultiThreshold")
    # QONNX returns NaN for 0 / 0 at zero scale, and for nonfinite scales.
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = qoxe.execute_onnx(model, {"x": values})["y"]
        actual = oxe.execute_onnx(converted, {"x": values})["y"]
    np.testing.assert_array_equal(actual, expected)
    if rounding_mode == "FLOOR":
        assert actual[0, 2] == -1.0  # input -0.5
    elif scale == -1.0:
        assert actual[0, 7] == 2.0  # input 2
    elif bitwidth == 1:
        assert actual[0, 3] == 0.0  # input 0, unsigned


@pytest.mark.transform
@pytest.mark.parametrize("rounding_mode", ["ROUND", "HALF_EVEN"])
def test_quant_round_positive_scale_keeps_thresholds(rounding_mode):
    model, values = make_quant_model(rounding_mode, 1.0)
    converted = model.transform(ConvertQONNXtoFINN())
    assert not converted.get_nodes_by_op_type("Quant")
    (mt,) = converted.get_nodes_by_op_type("MultiThreshold")
    # Existing signed 4-bit ROUND thresholds at scale 1, including ties to even.
    expected = np.array(
        [
            [
                -7.499999523162842,
                -6.5,
                -5.499999523162842,
                -4.5,
                -3.499999761581421,
                -2.5,
                -1.4999998807907104,
                -0.5,
                0.5000000596046448,
                1.5,
                2.500000238418579,
                3.5,
                4.500000476837158,
                5.5,
                6.500000476837158,
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(converted.get_initializer(mt.input[1]), expected)
    np.testing.assert_array_equal(
        oxe.execute_onnx(converted, {"x": values})["y"],
        qoxe.execute_onnx(model, {"x": values})["y"],
    )


@pytest.mark.transform
def test_quant_floor_weights_still_fold():
    model, values = make_quant_model("FLOOR", 0.5, weights=True)
    converted = model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["MatMul", "Mul"]
    np.testing.assert_array_equal(
        oxe.execute_onnx(converted, {"x": values})["y"],
        qoxe.execute_onnx(model, {"x": values})["y"],
    )


@pytest.mark.transform
@pytest.mark.parametrize("weights", [False, True], ids=["activation", "weights"])
@pytest.mark.parametrize("rounding_mode, scale", [("ROUND", 1.0), ("FLOOR", -1.0)])
def test_quant_nonzero_zero_point_still_raises(rounding_mode, scale, weights):
    model, _ = make_quant_model(rounding_mode, scale, weights=weights, zeropt=1.0)
    with pytest.raises(ValueError, match="Only Quant nodes with zero-point == 0"):
        model.transform(ConvertQONNXtoFINN())
