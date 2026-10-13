# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: qonnx ``MultiThreshold`` as count plus bias,
Y = count(X >= T[c]) + bias, on ``ThresholdingAxiKernel``.

The thresholds must be an initializer: one row per channel (C, N), or one row for all
channels (1, N). Channels are on the input's last axis. ``bias`` is a semantic
attribute. ``out_dtype`` is not read: the output's datatype is the kernel's result type,
which holds every count plus the bias.

The graph keeps the thresholds as they are (floats; their annotation is not read). For
an integer input, the kernel's table is derived from them without changing any count
(``stored_thresholds``): rounded up, clipped to [min, max + 1] of the input type, and
given the narrowest type that holds them. For a signed input, that type also holds one
value below the smallest threshold, unless the smallest is the input type's minimum: the
RTL clamps an input wider than the thresholds to their type, and a clamped input must
not reach the smallest threshold (``threshold-saturation``).

The pattern refuses ``out_scale`` other than 1 (``threshold-scale``) and a non-integer
``out_bias`` (``threshold-bias``), both with ``ExtractMultiThresholdScaleBias`` as a
hint; thresholds that are not an initializer (``threshold-dynamic``); and, when the
layout is not NHWC, channels on axis 1 of an input of rank above 2 (``layout-unproven``)
or an input whose shape is not known (``fact-unstated``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from qonnx.analysis.tensor_value_summary import (
    UnsupportedTensorValueError,
    initializer_value_summary,
)

from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import datatype, integer_tensor, known_shape, rows, shape
from finn.core.space import Finding, Rejected
from finn.custom_op.kernels.base import (
    KernelOp,
    Match,
    Shapes,
    Wiring,
    node_attributes,
    refused,
    unstated,
)
from finn.custom_op.kernels.cache import Facts
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.semantics import IntegerTensorValue

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

# The transform that moves a scale and a bias out of a MultiThreshold, named as a hint
# in refusals.
LOWERING = "ExtractMultiThresholdScaleBias"


def stored_thresholds(
    stored: npt.NDArray[Any], input_dtype: QONNXDataType
) -> tuple[IntegerTensorValue, QONNXDataType]:
    """The kernel's table for the thresholds ``stored`` on an input of ``input_dtype``,
    one set of them, and its type (see the module docstring). For an input that is not
    an integer type, the thresholds as they are (truncated to integers), in the input's
    type, which the kernel refuses."""
    try:
        low, high = ordinary_integer_bounds(input_dtype)
    except DatatypeError:
        return integer_tensor(np.trunc(stored)[None]), input_dtype
    # float64 holds every integer threshold of an input up to 2**53, where float32
    # rounds past 2**24.
    table = np.clip(np.ceil(np.asarray(stored, dtype=np.float64)), low, high + 1)
    least, most = int(table.min()), int(table.max())
    if low < 0:
        floor = max(low, least - 1)  # one value below the smallest (module docstring)
        bits = max(max((-floor - 1).bit_length() if floor < 0 else 0, most.bit_length()) + 1, 2)
        dtype = resolve_qonnx_datatype_name(f"INT{bits}")
    else:
        dtype = resolve_qonnx_datatype_name(f"UINT{max(most.bit_length(), 1)}")
    return integer_tensor(table[None]), dtype


class Thresholding(KernelOp):
    """Integer MultiThreshold, Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``."""

    op_type = "Thresholding"
    op_version = ThresholdingAxiKernel.version
    kernel = ThresholdingAxiKernel
    member = "activate"
    formals = ("threshold_dtype", "thresholds", "bias", "platform")
    ports = ("x", None)
    wiring = {
        "x": Wiring("input_tensor", "input_end"),
        "y": Wiring("output_tensor", "output_end"),
    }
    semantic = {"bias": ("i", True, 0)}
    anchor = ("qonnx.custom_op.general", "MultiThreshold")

    @classmethod
    def match(cls, model: ModelWrapper, node: NodeProto) -> Match | Rejected:
        owner = cls.op_type
        attributes = node_attributes(node)
        findings: list[Finding] = []
        scale = float(attributes.get("out_scale", 1.0))
        if scale != 1.0:
            message = f"out_scale {scale} is not 1: a Thresholding is count + bias"
            findings.append(refused(owner, "threshold-scale", message, hint=LOWERING))
        bias = float(attributes.get("out_bias", 0.0))
        if bias != int(bias):
            message = f"out_bias {bias} is not an integer"
            findings.append(refused(owner, "threshold-bias", message, hint=LOWERING))
        x, thresholds = node.input[0], node.input[1]
        if model.get_initializer(thresholds) is None:
            message = f"the thresholds {thresholds} are not an initializer, which the kernel holds"
            findings.append(refused(owner, "threshold-dynamic", message, tensor=thresholds))
        layout = attributes.get("data_layout", b"NCHW")
        layout = layout.decode() if isinstance(layout, bytes) else layout
        if layout != "NHWC":
            dims = known_shape(model, x)
            if dims is None:
                message = f"the shape of {x} is not known: its channel axis is not either"
                findings.append(unstated(owner, message))
            elif len(dims) > 2:
                message = f"channels on axis 1 of the {len(dims)}-D {x} (data_layout {layout})"
                findings.append(refused(owner, "layout-unproven", message, layout=layout))
        if findings:
            return Rejected(tuple(findings))
        return Match((node,), {"bias": int(bias)})

    def bias(self) -> int:
        """The ``bias`` attribute, added to every count."""
        bias = self.get_nodeattr("bias")
        if not isinstance(bias, int):
            raise KernelOpError(f"{self.label}: bias = {bias!r} is not an integer")
        return bias

    def facts(self) -> Facts:
        model, label = self.model(), self.label
        x, thresholds = self.onnx_node.input
        _, channels = rows(shape(model, x, label))
        stored = model.get_initializer(thresholds)
        if stored is None:
            raise KernelOpError(
                f"{label}: the thresholds {thresholds} are not an initializer: not a "
                "Thresholding (its pattern refuses them, threshold-dynamic)"
            )
        if stored.ndim != 2 or stored.shape[0] not in (1, channels):
            raise KernelOpError(
                f"{label}: the thresholds {thresholds} are {tuple(stored.shape)}, neither one "
                f"row for every channel of {x} nor one for each of its {channels} channels"
            )
        if stored.size == 0:
            raise KernelOpError(f"{label}: {thresholds} is empty: it holds no thresholds")
        if np.isnan(stored).any():
            raise KernelOpError(f"{label}: {thresholds} holds NaN, which no input compares to")
        try:
            summary = initializer_value_summary(model, thresholds)
        except UnsupportedTensorValueError as error:
            raise KernelOpError(f"{label}: {thresholds}: {error}") from error
        assert summary is not None  # an initializer
        input_dtype = datatype(model, x, label)
        bias = self.bias()
        platform = self.target().platform

        def formals() -> dict[str, object]:
            table, threshold_dtype = stored_thresholds(stored, input_dtype)
            return dict(
                threshold_dtype=threshold_dtype,
                thresholds=table,
                bias=bias,
                platform=platform,
            )

        key = (
            self.op_type,
            self.op_version,
            input_dtype.name,
            bias,
            platform,
            summary.content_digest,
        )
        return Facts(self.root(), key, formals, self.input_edges, self.output_edges)

    def output_tensors(self) -> Shapes:
        dims = shape(self.model(), self.onnx_node.input[0], self.label)
        return {self.onnx_node.output[0]: (dims, self.view("result_dtype"))}

    def execute_node(self, context: dict[str, Any], graph: Any) -> None:
        """The reference: count(X >= T[c]) + bias, in X's element type."""
        x, thresholds = self.onnx_node.input
        values, table = context[x], context[thresholds]
        counts = (values[..., :, None] >= table).sum(axis=-1)
        bias = self.bias()
        context[self.onnx_node.output[0]] = (counts + bias).astype(np.asarray(values).dtype)


__all__ = ["Thresholding"]
