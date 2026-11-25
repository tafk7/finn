# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

INPUTS = np.array([[-2.0, -0.1, 0.0, 0.3, 0.6, 2.0]], dtype=np.float32)


def make_bipolar_act_model(act_op_type, scale, **act_attrs):
    """An activation act_op_type with act_attrs followed by a BipolarQuant with scale."""
    nodes = [
        helper.make_node(act_op_type, ["x"], ["act"], **act_attrs),
        helper.make_node("BipolarQuant", ["act", "scale"], ["y"], domain="qonnx.custom_op.general"),
    ]
    graph = helper.make_graph(
        nodes,
        "bipolar_act",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(INPUTS.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(INPUTS.shape))],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [], [scale])],
    )
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(model).transform(InferShapes())


def execute(model, x):
    return oxe.execute_onnx(model, {"x": x})["y"]


@pytest.mark.transform
@pytest.mark.parametrize("x, in_band", [(-1e-8, True), (-2.9e-8, True), (-4.5e-8, False)])
def test_bipolar_quant_after_selu_tiny_negative(x, in_band):
    # Selu keeps its input's sign, so the converted graph gives -1 for every
    # negative input. In the band -2**-25 <= x < 0, float32 exp(x) rounds to 1 and
    # the QONNX graph's Selu gives 0, so its BipolarQuant gives 1 there: a known
    # difference, which depends on the runtime's exp. Just outside, both give -1.
    x = np.full(INPUTS.shape, x, dtype=np.float32)
    model = make_bipolar_act_model("Selu", 1.0)
    converted = model.transform(ConvertQONNXtoFINN())
    assert np.array_equal(execute(converted, x), np.full(x.shape, -1.0, dtype=np.float32))
    if not in_band:
        assert np.array_equal(execute(model, x), execute(converted, x))


@pytest.mark.transform
@pytest.mark.parametrize(
    "scale, out_dtype",
    [
        (1.0, "BIPOLAR"),
        (0.5, "FLOAT32"),
        (2.0, "INT3"),
        (-1.0, "BIPOLAR"),
        (-0.5, "FLOAT32"),
        (0.0, "BINARY"),
        # outputs -2**53 and 2**53: INT54 stops at 2**53 - 1
        (2.0**53, "INT55"),
    ],
)
def test_bipolar_quant_after_selu(scale, out_dtype):
    model = make_bipolar_act_model("Selu", scale)
    converted = model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["MultiThreshold"]
    mt = getCustomOp(converted.graph.node[0])
    assert np.array_equal(converted.get_initializer(converted.graph.node[0].input[1]), [[0.0]])
    assert mt.get_nodeattr("out_scale") == 2 * scale
    assert mt.get_nodeattr("out_bias") == -scale
    assert mt.get_nodeattr("out_dtype") == out_dtype
    assert converted.get_tensor_datatype("y") == DataType[out_dtype]
    expected = np.where(INPUTS >= 0, scale, -scale).astype(np.float32)
    assert np.array_equal(execute(model, INPUTS), expected)
    assert np.array_equal(execute(converted, INPUTS), expected)
    x = np.random.default_rng(0).uniform(-4, 4, size=(1, 6)).astype(np.float32)
    assert np.array_equal(execute(converted, x), execute(model, x))


@pytest.mark.transform
@pytest.mark.parametrize(
    "act_attrs",
    [
        {},
        {"alpha": 1.67326319217681884765625, "gamma": 1.05070102214813232421875},
        {"alpha": 0.5, "gamma": 2.0},
    ],
    ids=["absent", "onnx-defaults", "other-positive"],
)
def test_bipolar_quant_after_selu_attributes(act_attrs):
    model = make_bipolar_act_model("Selu", 1.0, **act_attrs)
    converted = model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["MultiThreshold"]
    x = np.random.default_rng(0).uniform(-4, 4, size=(1, 6)).astype(np.float32)
    for inputs in [INPUTS, x]:
        assert np.array_equal(execute(converted, inputs), execute(model, inputs))


@pytest.mark.transform
@pytest.mark.parametrize(
    "act_attrs",
    [{"alpha": 0.0}, {"gamma": -1.0}, {"gamma": 0.0}],
    ids=["alpha-zero", "gamma-negative", "gamma-zero"],
)
def test_bipolar_quant_after_selu_attributes_refused(act_attrs):
    # Such a Selu does not keep its input's sign: with alpha 0 it gives 0 for every
    # negative input, with gamma -1 the opposite sign
    model = make_bipolar_act_model("Selu", 1.0, **act_attrs)
    with pytest.raises(ValueError, match="follows a Selu with alpha .* and gamma"):
        model.transform(ConvertQONNXtoFINN())


@pytest.mark.transform
@pytest.mark.parametrize(
    "scale",
    [2.0**63, 1e38, 2e38, float(np.finfo(np.float32).max)],
    ids=["2**63", "1e38", "2e38", "float32-max"],
)
def test_bipolar_quant_after_selu_large_scale_left_in_place(scale):
    # Outputs past the 64-bit integer types, or a MultiThreshold scale (twice the
    # scale) past float32: the BipolarQuant stays, and QONNX executes it
    model = make_bipolar_act_model("Selu", scale)
    with pytest.warns(UserWarning, match="was not converted to a MultiThreshold node, because"):
        converted = model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["Selu", "BipolarQuant"]
    assert np.array_equal(execute(converted, INPUTS), execute(model, INPUTS))


@pytest.mark.transform
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_bipolar_quant_after_relu_refused(scale):
    # A Relu's output is never negative, so the BipolarQuant gives its scale for
    # every input
    model = make_bipolar_act_model("Relu", scale)
    expected = np.full(INPUTS.shape, scale, dtype=np.float32)
    assert np.array_equal(execute(model, INPUTS), expected)
    with pytest.raises(ValueError, match="follows a Relu and has a nonzero scale"):
        model.transform(ConvertQONNXtoFINN())


@pytest.mark.transform
def test_bipolar_quant_after_relu_zero_scale():
    # At scale 0 the pair gives 0 for every input, which the Relu conversion computes
    model = make_bipolar_act_model("Relu", 0.0)
    converted = model.transform(ConvertQONNXtoFINN())
    assert [n.op_type for n in converted.graph.node] == ["MultiThreshold"]
    expected = np.zeros(INPUTS.shape, dtype=np.float32)
    x = np.random.default_rng(0).uniform(-4, 4, size=(1, 6)).astype(np.float32)
    for inputs in [INPUTS, x]:
        assert np.array_equal(execute(model, inputs), expected)
        assert np.array_equal(execute(converted, inputs), expected)
