# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: ONNX ``MatMul`` semantics, Y = A @ B, bound to ``MatMulKernel``.

The graph decides which weights the node owns: weights that are an initializer
are the node's own (``Facts.owned``), the weight channel's known value
(``Facts.values``), which its ``source`` stores, keyed by their value summary's
digest, and the channel's tensor states their range; weights on any other
tensor arrive on a channel like any edge. One node root serves both: the weight
channel's contents are supplied only when the node owns them. MatMul consumes
the weights from the channel. A's leading axes are rows; B is the (k, n)
matrix ONNX stores. The output is A's leading axes and n.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    Shapes,
    admitted,
    datatype,
    integer_tensor,
    rows,
    shape,
)
from finn.custom_op.kernels.cache import Facts
from finn.kernels.matmul import MatMulKernel
from finn.kernels.values.semantics import IntegerTensorValue


class MatMul(KernelOp):
    """ONNX MatMul, Y = A @ B, bound to ``MatMulKernel``."""

    op_type = "MatMul"
    op_version = MatMulKernel.version
    kernel = MatMulKernel
    member = "matmul"
    formals = ("m", "n", "k", "activation_dtype", "weights_dtype", "platform")
    ports = ("x", "w")
    references = {"x": "x_channel", "w": "w_channel", "y": "y_channel"}
    parameters = ("w",)

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
        root = self.root()
        if model.get_initializer(b) is None:
            return Facts(root, MatMulKernel, (*key, None), lambda: common, self.edges)
        digest = admitted(model, b, weights_dtype, label)

        def values() -> dict[str, IntegerTensorValue]:
            stored = model.get_initializer(b)
            if stored is None:
                raise KernelOpError(f"{label}: {b} is not an initializer")
            return {"w": integer_tensor(stored)}

        return Facts(root, MatMulKernel, (*key, digest), lambda: common, self.edges, values, ("w",))

    def output_tensors(self) -> Shapes:
        result = self.view("result_tensor")
        leading = shape(self.model(), self.onnx_node.input[0], self.label)[:-1]
        return {self.onnx_node.output[0]: ((*leading, result.shape[-1]), result.element.dtype)}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        a, b = self.onnx_node.input
        context[self.onnx_node.output[0]] = np.matmul(context[a], context[b]).astype(np.float32)


__all__ = ["MatMul"]
