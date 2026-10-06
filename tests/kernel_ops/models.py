# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small ONNX models of KernelOps, and of the source graphs conversion reads."""

from __future__ import annotations

from typing import Any

import numpy as np
from kernels import test_design as chain
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import write_target
from finn.transformation.kernels import InferKernelTensors, resolve_target

DOMAIN = "finn.custom_op.kernels"
INT3 = DataType["INT3"]
ROWS, K, N = 3, 4, 4
WEIGHTS = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(N)] for k in range(K)])
X = np.array([[[(5 * r + 3 * k) % 8 - 4 for k in range(K)] for r in range(ROWS)]])
TARGET = resolve_target("xczu3eg-sbva484-1-e", 5.0)  # Ultra96: DSP48E2, no shell


def matmul_model(
    *,
    stored: bool = True,
    weights: Any = WEIGHTS,
    annotate: tuple[str, ...] = ("x", "w"),
    x_shape: list[int] | None = None,
    target: bool = True,
    infer: bool = True,
) -> ModelWrapper:
    """x (1, 3, 4) -> MatMul ``first`` (domain ``finn.custom_op.kernels``) with w -> y.

    ``stored``: w an initializer (the node's own), else a graph input. ``infer``: y
    stated by ``InferKernelTensors``, as the node root reads it (D6); without it, only
    the node's facts are read.
    """
    weights = np.asarray(weights, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape or [1, ROWS, K])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, list(weights.shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="first", domain=DOMAIN)
    inputs = [x] if stored else [x, w]
    graph = helper.make_graph([node], "matmul", inputs, [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
        )
    )
    if stored:
        model.set_initializer("w", weights)
    for name in annotate:
        model.set_tensor_datatype(name, INT3)
    if target:
        write_target(model, TARGET)
    return model.transform(InferKernelTensors()) if infer else model


THRESHOLDS = np.array([[-9 + c, 1 - c, 8 + 2 * c] for c in range(N)])
H = DataType["INT8"]


def thresholding_model(
    *,
    thresholds: Any = THRESHOLDS,
    stored: bool = True,
    bias: int = 0,
    annotate: tuple[str, ...] = ("x", "t"),
    infer: bool = True,
) -> ModelWrapper:
    """x (3, 4) INT8 -> Thresholding ``activate`` with t (C, N) -> y; ``infer`` as for
    ``matmul_model``."""
    thresholds = np.asarray(thresholds, dtype=np.float32)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, N])
    t = helper.make_tensor_value_info("t", TensorProto.FLOAT, list(thresholds.shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    node = helper.make_node(
        "Thresholding", ["x", "t"], ["y"], name="activate", domain=DOMAIN, bias=bias
    )
    graph = helper.make_graph([node], "thresholding", [x] if stored else [x, t], [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
        )
    )
    if stored:
        model.set_initializer("t", thresholds)
    for name in annotate:
        model.set_tensor_datatype(name, H)
    write_target(model, TARGET)
    return model.transform(InferKernelTensors()) if infer else model


def chain_source(*, annotate_input: bool = True, second_weights: bool = True) -> ModelWrapper:
    """test_design's Chain as an ONNX model, before conversion: x -> MatMul ``first`` (w1)
    -> hidden -> MultiThreshold ``activate`` -> levels -> MatMul ``second`` (w2) -> y.
    Only x's shape is known (a fresh graph); w2 is a graph input unless ``second_weights``.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [chain.ROWS, chain.INPUTS])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["hidden"], name="first"),
        helper.make_node(
            "MultiThreshold",
            ["hidden", "thresholds"],
            ["levels"],
            name="activate",
            domain="qonnx.custom_op.general",
            out_dtype="UINT2",
            out_bias=0.0,
        ),
        helper.make_node("MatMul", ["levels", "w2"], ["y"], name="second"),
    ]
    w2 = helper.make_tensor_value_info("w2", TensorProto.FLOAT, [chain.HIDDEN, chain.OUTPUTS])
    inputs = [x] if second_weights else [x, w2]
    graph = helper.make_graph(nodes, "chain", inputs, [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    model.set_initializer("w1", np.array(chain.W1, dtype=np.float32))
    if second_weights:
        model.set_initializer("w2", np.array(chain.W2, dtype=np.float32))
    model.set_initializer("thresholds", np.array(chain.THRESHOLDS[0], dtype=np.float32))
    if annotate_input:
        model.set_tensor_datatype("x", chain.A)
    for name in ("w1", "w2"):
        model.set_tensor_datatype(name, chain.W)
    model.set_tensor_datatype("thresholds", chain.H)
    return model


def lift(model: ModelWrapper, tensor: str) -> None:
    """Make an initializer a graph input (a stored node's weights become an edge)."""
    values = model.get_initializer(tensor)
    model.graph.initializer.remove(next(i for i in model.graph.initializer if i.name == tensor))
    info = model.get_tensor_valueinfo(tensor)
    if info is not None:
        model.graph.value_info.remove(info)
    model.graph.input.append(
        helper.make_tensor_value_info(tensor, TensorProto.FLOAT, list(values.shape))
    )


__all__ = [
    "DOMAIN",
    "chain_source",
    "H",
    "INT3",
    "THRESHOLDS",
    "WEIGHTS",
    "X",
    "lift",
    "matmul_model",
    "thresholding_model",
]
