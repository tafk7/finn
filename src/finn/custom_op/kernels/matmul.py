# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: ONNX ``MatMul`` semantics, Y = A @ B, bound to ``MatMulKernel``.

The graph decides the node root: weights that are an
initializer are the node's own, the weight channel's known value, which its
``source`` stores (``StoredMatMulNode``), keyed by their value summary's
digest; weights on any other tensor arrive on a channel like any edge
(``StreamedMatMulNode``). A's leading axes are rows; B is the (k, n) matrix
ONNX stores. The output is A's leading axes and n.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    Shapes,
    admitted,
    datatype,
    rows,
    shape,
)
from finn.custom_op.kernels.cache import Facts
from finn.custom_op.kernels.roots import StoredMatMulNode, StreamedMatMulNode
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.matmul import MatMulKernel


class MatMul(KernelOp):
    """ONNX MatMul, Y = A @ B, bound to ``MatMulKernel``."""

    op_type = "MatMul"
    op_version = MatMulKernel.version
    roots = (StoredMatMulNode, StreamedMatMulNode)
    member = "matmul"
    ports = ("x", "w")

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        a, b = self.onnx_node.input
        m, k = rows(shape(model, a, label))
        weights_shape = shape(model, b, label)
        if len(weights_shape) != 2:
            raise KernelOpError(f"{label}: the weights {b} are {weights_shape}, not (k, n)")
        k_b, n = weights_shape
        if k != k_b:
            raise KernelOpError(f"{label}: {a} has {k} columns and {b} {k_b} rows")
        activation, weights_dtype = datatype(model, a, label), datatype(model, b, label)
        platform = self.target().platform
        common: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=activation,
            weights_dtype=weights_dtype,
            platform=platform,
            x_tensor=Tensor((m, k), ScalarEncoding(activation)),
        )
        key = (
            self.op_type,
            self.op_version,
            m,
            n,
            k,
            activation.name,
            weights_dtype.name,
            platform,
        )
        if model.get_initializer(b) is None:
            return Facts(StreamedMatMulNode, (*key, None), lambda: common)
        digest = admitted(model, b, weights_dtype, label)

        def formals() -> dict[str, object]:
            values = model.get_initializer(b)
            weights = tuple(tuple(int(value) for value in row) for row in values)
            return {**common, "weights": weights}

        return Facts(StoredMatMulNode, (*key, digest), formals, ("w",))

    def output_tensors(self) -> Shapes:
        result = self.view("y_tensor")
        leading = shape(self.model(), self.onnx_node.input[0], self.label)[:-1]
        return {self.onnx_node.output[0]: ((*leading, result.shape[-1]), result.element.dtype)}

    def owned_channels(self) -> dict[str, Channel]:
        if self.facts().root is not StoredMatMulNode:
            return {}
        base: Any = self.base()
        return {
            self.onnx_node.input[1]: Channel(
                tensor=self.view("w_tensor"),
                contents=base.matmul.weight_values,
                platform=self.target().platform,
            )
        }

    def place(self, channels: Mapping[str, Channel]) -> tuple[Kernel, dict[str, str]]:
        facts = self.facts()
        formals: dict[str, Any] = facts.formals()
        del formals["x_tensor"]
        a, b = self.onnx_node.input
        kernel = MatMulKernel(
            **formals,
            x_channel=channels[a],
            w_channel=channels[b],
            y_channel=channels[self.onnx_node.output[0]],
        )
        return kernel, {"x": a, "w": b}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        a, b = self.onnx_node.input
        context[self.onnx_node.output[0]] = np.matmul(context[a], context[b]).astype(np.float32)


__all__ = ["MatMul"]
