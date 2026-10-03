# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small ONNX models of KernelOps, and of the source graphs conversion reads."""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import write_target
from finn.kernels.target import DspBlock

DOMAIN = "finn.custom_op.kernels"
INT3 = DataType["INT3"]
ROWS, K, N = 3, 4, 4
WEIGHTS = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(N)] for k in range(K)])
X = np.array([[[(5 * r + 3 * k) % 8 - 4 for k in range(K)] for r in range(ROWS)]])


def matmul_model(
    *,
    stored: bool = True,
    weights: Any = WEIGHTS,
    annotate: tuple[str, ...] = ("x", "w"),
    x_shape: list[int] | None = None,
    target: bool = True,
) -> ModelWrapper:
    """x (1, 3, 4) -> MatMul ``first`` (domain ``finn.custom_op.kernels``) with w -> y.

    ``stored``: w an initializer (the node's own), else a graph input.
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
        write_target(model, DspBlock.DSP48E2, 5.0)
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


__all__ = ["DOMAIN", "INT3", "WEIGHTS", "X", "lift", "matmul_model"]
