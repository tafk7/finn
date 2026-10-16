# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A tensor's range, from one kernel to the next: each KernelOp's output states its range as
part of its element (the kernel's statement), inference writes it with the datatype
(``finn.core.graph.facts.state``), and the next kernel reads it (``edge_tensor``).

The graph: x (3, 4) INT8 -> Thresholding ``activate`` (two thresholds a channel, bias -1:
INT2 over [-1, 1]) -> levels -> MatMul ``project`` (ternary weights) -> y.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import TensorRange
from qonnx.util.basic import qonnx_make_model
from support.models import DOMAIN, TARGET
from support.refusals import refused

from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor, initializer, reading
from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.network import network
from finn.dataflow.datatypes import DatatypeError
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.matmul import column_range, operand_range
from finn.kernels.values.semantics import IntegerTensorValue
from finn.transformation.kernels import InferKernelTensors
from finn.util.draws import drawn_inputs

INT2, INT8 = DataType["INT2"], DataType["INT8"]
THRESHOLDS = np.array([[-3 + c, 2 + c] for c in range(4)], dtype=np.float32)
# Ternary weights: [-1, 1] activations reach [-3, 3] with them, INT2's own [-2, 1] would
# reach [-6, 5].
WEIGHTS = np.array([[1, -1, 1, 0], [1, 0, -1, 1], [1, -1, 0, 1], [0, 1, -1, -1]], dtype=np.float32)


def ranged_model() -> ModelWrapper:
    """x INT8 -> Thresholding ``activate`` (bias -1) -> levels -> MatMul ``project`` -> y."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node(
            "Thresholding", ["x", "t"], ["levels"], name="activate", domain=DOMAIN, bias=-1
        ),
        helper.make_node("MatMul", ["levels", "w"], ["y"], name="project", domain=DOMAIN),
    ]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "ranged", [x], [y]),
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
        )
    )
    model.set_initializer("t", THRESHOLDS)
    model.set_initializer("w", WEIGHTS)
    model.set_tensor_datatype("x", INT8)
    model.set_tensor_datatype("t", INT8)
    model.set_tensor_datatype("w", INT8)
    write_target(model, TARGET)
    return model.transform(InferKernelTensors())


def test_each_output_states_its_range_and_the_next_kernel_reads_it() -> None:
    model = ranged_model()
    levels = ScalarEncoding(INT2, TensorRange(-1, 1))
    # The Thresholding states its counts plus bias, [-1, 1]; inference writes it with INT2.
    assert model.get_tensor_datatype("levels") == INT2
    assert model.get_tensor_range("levels") == TensorRange(-1, 1)
    assert edge_tensor(model, "levels", "test").element == levels
    # The MatMul reads it: its result range is the columns' over [-1, 1], not over INT2's
    # [-2, 1], and its result type the range's smallest encoding.
    matmul: Any = kernel_op(model, model.graph.node[1]).bound()
    assert matmul.x_tensor.element == levels
    weights = IntegerTensorValue.of(tuple(tuple(int(v) for v in row) for row in WEIGHTS))
    assert matmul.result_range == column_range(levels, weights) == (-3, 3)
    assert column_range(ScalarEncoding(INT2), weights) == (-6, 5)
    assert (model.get_tensor_datatype("y"), model.get_tensor_range("y")) == (
        DataType["INT3"],
        TensorRange(-3, 3),
    )
    # Every output on the graph is its producer's statement: the network binds.
    assert not network(model, model.graph.node).stale


def test_inference_states_each_output_again_with_its_range() -> None:
    """A datatype set alone clears its range; inference states both again."""
    model = ranged_model()
    model.set_tensor_datatype("levels", INT2)
    assert model.get_tensor_range("levels") == TensorRange(-2, 1)
    model = model.transform(InferKernelTensors())
    assert model.get_tensor_range("levels") == TensorRange(-1, 1)
    assert model.get_tensor_range("y") == TensorRange(-3, 3)


def test_an_output_whose_range_is_not_its_producers_is_stale() -> None:
    """The graph's annotation of an output is its producer's statement, its range too:
    the datatype alone, stated without its range, is stale, named by the network."""
    model = ranged_model()
    model.set_tensor_datatype("y", DataType["INT3"])
    with refused(
        "tensor-stale",
        error=KernelOpError,
        owner="stale_outputs",
        node="project",
        tensor="y",
        message=r"the graph states y \[3, 4\] INT3, its",
    ):
        network(model, model.graph.node)


def test_a_stored_range_its_datatype_does_not_hold_is_refused() -> None:
    model = ranged_model()
    # INT2 over [-1, 1], its datatype rewritten underneath it to UINT2: [-1, 1] is not UINT2's.
    (annotation,) = [a for a in model.graph.quantization_annotation if a.tensor_name == "levels"]
    for entry in annotation.quant_parameter_tensor_names:
        if entry.key == "finn_datatype":
            entry.value = "UINT2"
    with refused(
        "range-invalid",
        error=KernelOpError,
        owner="element",
        node="activate",
        tensor="levels",
        message=r"\[-1, 1\] is not a range of UINT2 values",
    ):
        edge_tensor(model, "levels", "activate")


def test_an_operand_range_holds_zero_and_is_an_ordinary_integers() -> None:
    """Every partial sum lies within the result's range when each operand's range holds 0
    (``operand_range``), whatever range its producer states."""
    assert operand_range(ScalarEncoding(DataType["INT4"], TensorRange(1, 3))) == (0, 3)
    assert operand_range(ScalarEncoding(DataType["INT4"], TensorRange(-3, -1))) == (-3, 0)
    assert operand_range(ScalarEncoding(INT2)) == (-2, 1)
    with refused("integer-not-ordinary", error=DatatypeError, message="^BIPOLAR "):
        operand_range(ScalarEncoding(DataType["BIPOLAR"]))
    with refused("integer-not-ordinary", error=DatatypeError, message="^FLOAT32 "):
        operand_range(ScalarEncoding(DataType["FLOAT32"]))


def test_a_graph_input_is_drawn_from_its_stated_range() -> None:
    """The hardware is built for an input's stated range (a graph input's is read as stated),
    so the harness's draws keep to it, its extremes included."""
    model = ranged_model()
    model.set_tensor_datatype("x", INT8, TensorRange(-3, 5))
    for draw in drawn_inputs(model, 0):
        assert -3 <= int(draw["x"].min()) and int(draw["x"].max()) <= 5
    extremes = drawn_inputs(model, 0)[0]["x"]
    assert (int(extremes.min()), int(extremes.max())) == (-3, 5)


def test_a_bind_reads_each_tensor_from_one_index_as_the_graph_states_it() -> None:
    """Inside ``reading`` (a network's bind) every tensor reads as it does outside, an
    initializer as one array shared and read-only; another model is read directly."""
    model, other = ranged_model(), ranged_model()
    tensors = ["x", "t", "levels", "w", "y"]
    outside = {tensor: edge_tensor(model, tensor, "test") for tensor in tensors}
    with reading(model):
        assert {tensor: edge_tensor(model, tensor, "test") for tensor in tensors} == outside
        weights = initializer(model, "w")
        assert weights is not None and weights is initializer(model, "w")
        assert not weights.flags.writeable
        np.testing.assert_array_equal(weights, WEIGHTS)
        assert initializer(model, "levels") is None
        assert initializer(other, "w") is not initializer(other, "w")
    assert initializer(model, "w") is not weights
