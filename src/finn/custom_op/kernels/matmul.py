# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The MatMul KernelOp: ONNX ``MatMul`` semantics, Y = A @ B, bound to ``MatMulKernel``.

H-006's reading rule decides the node root from the graph: weights that are an
initializer are the node's own, stored by its memory (``StoredMatMulNode``,
``memory`` pinned ``memstream``) and keyed by their value summary's digest;
weights on any other tensor arrive on a stream like any edge
(``StreamedMatMulNode``, ``memory`` pinned ``none``). A's leading axes are
rows; B is the (k, n) matrix ONNX stores. The output is A's leading axes and n.
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
    rows,
    shape,
)
from finn.custom_op.kernels.cache import Facts
from finn.custom_op.kernels.roots import StoredMatMulNode, StreamedMatMulNode
from finn.dataflow.tensor import ScalarEncoding, Tensor
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
        dsp, period = self.target()
        common: dict[str, object] = dict(
            m=m,
            n=n,
            k=k,
            activation_dtype=activation,
            weights_dtype=weights_dtype,
            target_dsp=dsp,
            target_period_ns=period,
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
            dsp.name,
            period,
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

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        a, b = self.onnx_node.input
        context[self.onnx_node.output[0]] = np.matmul(context[a], context[b]).astype(np.float32)


__all__ = ["MatMul"]
