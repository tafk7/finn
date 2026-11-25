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
BITWIDTH = 4


def make_quant_model(op_type, signed, narrow, scale, width, relu=False):
    """An activation quantizer rounding half to even (ROUND) on a [1, width]
    input, with a Relu in front of it if relu is set."""
    scale = np.asarray(scale, dtype=np.float32)
    nodes = [
        helper.make_node(
            op_type,
            ["q_in", "scale", "zeropt", "bitwidth"],
            ["y"],
            domain=QONNX_DOMAIN,
            signed=signed,
            narrow=narrow,
            rounding_mode="ROUND",
        )
    ]
    if relu:
        nodes.insert(0, helper.make_node("Relu", ["x"], ["q_in"]))
    else:
        nodes[0].input[0] = "x"
    inits = [
        helper.make_tensor("scale", TensorProto.FLOAT, scale.shape, scale.ravel()),
        helper.make_tensor("zeropt", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("bitwidth", TensorProto.FLOAT, [], [float(BITWIDTH)]),
    ]
    graph = helper.make_graph(
        nodes,
        "round_ties",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width])],
        initializer=inits,
    )
    model = qonnx_make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return ModelWrapper(model).transform(InferShapes())


def execute(model, x):
    return oxe.execute_onnx(model, {"x": x})["y"]


def convert(model):
    converted = model.transform(ConvertQONNXtoFINN())
    ops = [n.op_type for n in converted.graph.node]
    assert ops[0] == "MultiThreshold" and not {"Quant", "IntQuant"} & set(ops)
    return converted


def least_code(signed, narrow):
    if not signed:
        return 0
    return -(2 ** (BITWIDTH - 1)) + narrow


def max_code(signed, narrow):
    if not signed:
        return 2**BITWIDTH - 1 - narrow
    return 2 ** (BITWIDTH - 1) - 1


def float32_neighbourhood(values, steps=4):
    """The float32 values nearest to each of values, steps either side."""
    out = []
    for value in values:
        center = np.float32(value)
        out.append(center)
        below, above = center, center
        for _ in range(steps):
            below = np.nextafter(below, np.float32(-np.inf))
            above = np.nextafter(above, np.float32(np.inf))
            out.extend([below, above])
    return np.unique(np.array(out, dtype=np.float32))


def boundary_inputs(signed, narrow, scale, thresholds):
    """Inputs around every step of the quantizer: the mathematical boundaries
    (code - 1/2) * scale and the converted thresholds, with their float32
    neighbours."""
    codes = np.arange(least_code(signed, narrow) + 1, max_code(signed, narrow) + 1)
    boundaries = (codes - 0.5) * np.float64(np.float32(scale))
    return float32_neighbourhood(np.concatenate([boundaries, np.ravel(thresholds)]))


def convert_and_compare(op_type, signed, narrow, scale, relu=False):
    """Convert the quantizer and return (inputs, QONNX outputs, converted outputs)
    over the neighbourhood of every quantizer step."""
    model = make_quant_model(op_type, signed, narrow, scale, 1, relu)
    converted = convert(model)
    (mt,) = converted.get_nodes_by_op_type("MultiThreshold")
    thresholds = converted.get_initializer(mt.input[1])
    x = boundary_inputs(signed, narrow, scale, thresholds)[np.newaxis, :]
    model = make_quant_model(op_type, signed, narrow, scale, x.shape[1], relu)
    converted = convert(model)
    return x, execute(model, x), execute(converted, x)


@pytest.mark.transform
@pytest.mark.parametrize("op_type", ["Quant", "IntQuant"])
@pytest.mark.parametrize("signed", [0, 1])
@pytest.mark.parametrize("narrow", [0, 1])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_round_ties_identity(op_type, signed, narrow, scale):
    x, expected, produced = convert_and_compare(op_type, signed, narrow, scale)
    mismatch = np.flatnonzero(expected != produced)
    assert mismatch.size == 0, [
        (float(x.flat[i]), float(expected.flat[i]), float(produced.flat[i])) for i in mismatch
    ]


@pytest.mark.transform
@pytest.mark.parametrize("op_type", ["Quant", "IntQuant"])
@pytest.mark.parametrize("scale", [1.0, 0.5, 0.1])
def test_round_ties_relu(op_type, scale):
    x, expected, produced = convert_and_compare(op_type, 0, 0, scale, relu=True)
    mismatch = np.flatnonzero(expected != produced)
    assert mismatch.size == 0, [
        (float(x.flat[i]), float(expected.flat[i]), float(produced.flat[i])) for i in mismatch
    ]


@pytest.mark.transform
@pytest.mark.parametrize("op_type", ["Quant", "IntQuant"])
def test_round_ties_per_channel(op_type):
    scale = np.array([1.0, 0.5, 0.1], dtype=np.float32)
    rows = [boundary_inputs(1, 0, s, []) for s in scale]
    x = np.zeros((max(len(r) for r in rows), len(scale)), dtype=np.float32)
    for channel, row in enumerate(rows):
        x[:, channel] = np.resize(row, x.shape[0])
    model = make_quant_model(op_type, 1, 0, scale, len(scale))
    model.set_tensor_shape("x", list(x.shape))
    model.set_tensor_shape("y", list(x.shape))
    converted = convert(model)
    assert np.array_equal(execute(converted, x), execute(model, x))


@pytest.mark.transform
@pytest.mark.parametrize("op_type", ["Quant", "IntQuant"])
@pytest.mark.parametrize(
    "signed, narrow, scale, x, code, half_up_code",
    [
        # ties into odd codes at a power-of-two scale round to the even code
        (1, 0, 1.0, 0.5, 0, 1),
        (1, 0, 1.0, 2.5, 2, 3),
        (1, 0, 1.0, -7.5, -8, -7),
        (1, 0, 1.0, -1.5, -2, -1),
        (0, 0, 1.0, 6.5, 6, 7),
        (1, 0, 0.5, 0.25, 0, 1),
        # signed narrow: the step into code 1 is the MultiThreshold's eighth
        (1, 1, 1.0, 0.5, 0, 1),
        # float32(0.1): the quantizer's float32 division sends this input, the
        # float32 nearest -0.65, to -6.5000005 and so to code -7, below the half step
        (1, 0, 0.1, -0.6500000357627869, -7, -6),
    ],
)
def test_round_ties_named(op_type, signed, narrow, scale, x, code, half_up_code):
    x = np.array([[x]], dtype=np.float32)
    scale = np.float32(scale)
    model = make_quant_model(op_type, signed, narrow, scale, 1)
    converted = convert(model)
    # half_up_code is what thresholds on the half steps gave
    assert half_up_code != code
    assert execute(model, x)[0, 0] == np.float32(code) * scale
    assert execute(converted, x)[0, 0] == np.float32(code) * scale
