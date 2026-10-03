# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``.

Integer ``MultiThreshold`` with ``out_scale`` 1, channels on the input's last
axis. The thresholds are an initializer of shape (C, N) the kernel holds (its
value owner): one set, admitted by its value summary, its digest in the key,
its annotation the threshold datatype. ``bias`` is a semantic attribute, part
of the operation. The output keeps the input's shape and takes the kernel's
result type, a fact-level derived of the table and the bias.
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
from finn.custom_op.kernels.roots import ThresholdingNode
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.streams import Stream
from finn.kernels.thresholding import ThresholdingAxiKernel


class Thresholding(KernelOp):
    """Integer MultiThreshold, Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``."""

    op_type = "Thresholding"
    op_version = ThresholdingAxiKernel.version
    roots = (ThresholdingNode,)
    member = "activate"
    ports = ("x", None)
    semantic = {"bias": ("i", True, 0)}

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        x, thresholds = self.onnx_node.input
        m, channels = rows(shape(model, x, label))
        table = model.get_initializer(thresholds)
        if table is None:
            raise KernelOpError(f"{label}: the thresholds {thresholds} must be an initializer")
        if table.ndim != 2 or table.shape[0] != channels:
            raise KernelOpError(
                f"{label}: the thresholds {thresholds} are {tuple(table.shape)}, not one row "
                f"for each of the {channels} channels of {x}"
            )
        input_dtype = datatype(model, x, label)
        threshold_dtype = datatype(model, thresholds, label)
        digest = admitted(model, thresholds, threshold_dtype, label)
        bias = int(self.get_nodeattr("bias"))

        def formals() -> dict[str, object]:
            values = model.get_initializer(thresholds)
            return dict(
                input_dtype=input_dtype,
                threshold_dtype=threshold_dtype,
                thresholds=(tuple(tuple(int(value) for value in row) for row in values),),
                bias=bias,
                x_tensor=Tensor((m, channels), ScalarEncoding(input_dtype)),
            )

        key = (
            self.op_type,
            self.op_version,
            m,
            channels,
            input_dtype.name,
            threshold_dtype.name,
            bias,
            digest,
        )
        return Facts(ThresholdingNode, key, formals)

    def output_tensors(self) -> Shapes:
        result = self.view("y_tensor")
        dims = shape(self.model(), self.onnx_node.input[0], self.label)
        return {self.onnx_node.output[0]: (dims, result.element.dtype)}

    def place(self, streams: Mapping[str, Stream]) -> tuple[Kernel, dict[str, str]]:
        formals: dict[str, Any] = self.facts().formals()
        del formals["x_tensor"]
        x = self.onnx_node.input[0]
        kernel = ThresholdingAxiKernel(
            **formals, input_stream=streams[x], output_stream=streams[self.onnx_node.output[0]]
        )
        return kernel, {"x": x}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        x, thresholds = self.onnx_node.input
        values, table = context[x], context[thresholds]
        counts = (values[..., :, None] >= table).sum(axis=-1)
        bias = int(self.get_nodeattr("bias"))
        context[self.onnx_node.output[0]] = (counts + bias).astype(np.float32)


__all__ = ["Thresholding"]
