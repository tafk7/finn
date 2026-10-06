# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``.

Integer ``MultiThreshold`` with ``out_scale`` 1, channels on the input's last
axis. The thresholds are an initializer of shape (C, N) the kernel holds (its
value owner): one set, admitted by its value summary, its digest in the key,
its annotation the threshold datatype. ``bias`` is a semantic attribute, part
of the operation. The output keeps the input's shape and takes the kernel's
result type, a fact-level derived of the table and the bias.

The ordered pass normalizes the thresholds first (``normalize_inputs``), against
the input's exact type: a broadcast row ``(1, N)`` becomes one row per channel,
and the values are rounded up and clipped to ``[min, max + 1]`` of the input
type (finn-dev's ``RoundAndClipThresholds``), then annotated with the smallest
type of the input's signedness that holds them (finn-dev's threshold
``minimize_weight_bit_width``, which its flow runs after the rounding). Exact for
integer inputs: ``x >= t`` and ``x >= ceil(t)`` agree, and a threshold outside
the input's range counts the same at its bound.
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
from finn.dataflow.datatypes import (
    DatatypeError,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.thresholding import ThresholdingAxiKernel


class Thresholding(KernelOp):
    """Integer MultiThreshold, Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``."""

    op_type = "Thresholding"
    op_version = ThresholdingAxiKernel.version
    kernel = ThresholdingAxiKernel
    member = "activate"
    formals = ("input_dtype", "threshold_dtype", "thresholds", "bias", "platform")
    ports = ("x", None)
    references = {"x": "input_channel", "y": "output_channel"}
    semantic = {"bias": ("i", True, 0)}

    def normalize_inputs(self) -> None:
        """The thresholds as integers against the input's exact type (module docstring)."""
        model, label = self.model(), self.label
        x, thresholds = self.onnx_node.input
        table = model.get_initializer(thresholds)
        try:
            low, high = ordinary_integer_bounds(datatype(model, x, label))
        except DatatypeError:
            return  # not an integer input: the kernel refuses it
        if table is None or table.ndim != 2:
            return  # facts refuse it, naming the shape
        _, channels = rows(shape(model, x, label))
        if table.shape[0] == 1 and channels > 1:
            table = np.tile(table, (channels, 1))
        table = np.clip(np.ceil(table), low, high + 1).astype(np.float32)
        least, most = int(table.min()), int(table.max())
        if low < 0:
            bits = max(max((-least - 1).bit_length() if least < 0 else 0, most.bit_length()) + 1, 2)
            dtype = resolve_qonnx_datatype_name(f"INT{bits}")
        else:
            dtype = resolve_qonnx_datatype_name(f"UINT{max(most.bit_length(), 1)}")
        if len(model.find_consumers(thresholds)) > 1:
            thresholds = model.make_new_valueinfo_name()
            self.onnx_node.input[1] = thresholds
        model.set_initializer(thresholds, table)
        model.set_tensor_datatype(thresholds, dtype)

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        x, thresholds = self.onnx_node.input
        _, channels = rows(shape(model, x, label))
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
        platform = self.target().platform

        def formals() -> dict[str, object]:
            values = model.get_initializer(thresholds)
            return dict(
                input_dtype=input_dtype,
                threshold_dtype=threshold_dtype,
                thresholds=(tuple(tuple(int(value) for value in row) for row in values),),
                bias=bias,
                platform=platform,
            )

        key = (
            self.op_type,
            self.op_version,
            input_dtype.name,
            threshold_dtype.name,
            bias,
            platform,
            digest,
        )
        return Facts(self.root(), ThresholdingAxiKernel, key, formals, self.edges)

    def output_tensors(self) -> Shapes:
        dims = shape(self.model(), self.onnx_node.input[0], self.label)
        return {self.onnx_node.output[0]: (dims, self.view("result_dtype"))}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        x, thresholds = self.onnx_node.input
        values, table = context[x], context[thresholds]
        counts = (values[..., :, None] >= table).sum(axis=-1)
        bias = int(self.get_nodeattr("bias"))
        context[self.onnx_node.output[0]] = (counts + bias).astype(np.float32)


__all__ = ["Thresholding"]
