# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``.

Integer ``MultiThreshold`` with ``out_scale`` 1, channels on the input's last
axis. The thresholds are an initializer the kernel holds (its value owner): one
set, a row for each channel (C, N) or one row for every channel (1, N), which
the kernel binds as the graph states it (a shared row is ``ThresholdingAxiKernel``'s
C = 1). ``bias`` is a semantic attribute, part of the operation. The output keeps
the input's shape and takes the kernel's result type, a fact-level derived of the
table and the bias.

The table the kernel holds: the graph keeps the thresholds as streamlining wrote
them (floats; their annotation is not read), and the facts derive the kernel's
integer table from them and the input's exact type (``stored_thresholds``):
the values rounded up and clipped to ``[min, max + 1]`` of the input type, typed
by the smallest type of the input's signedness that holds them and, unless the
least is the input's minimum, one value below it. Every count is the graph's: for
an integer input ``x >= t`` and ``x >= ceil(t)`` agree, and a threshold outside the
input's range counts the same at its bound, so ``execute_node`` counts on the
stored floats. The value below: thresholding_axi saturates an input wider than its
thresholds to their type, so an input below the type's minimum compares as that
minimum, which a threshold there would count (``ThresholdingAxiKernel``'s
``threshold-saturation``; found by the harness's parity check,
``tests/kernel_ops/test_parity.py``). The binding is keyed by the stored table's
digest and the input's type, and the table is derived on a miss only, as MatMul's
weights' stored element is derived from their values (``edge_tensor``).

Its typing rule: ``out_dtype`` is not read. The output's annotation is the kernel's
result type, a fact-level derived of the thresholds and the bias, which holds every
count plus the bias; ``out_dtype`` changes no value ``MultiThreshold`` computes.

Its pattern: a qonnx ``MultiThreshold`` that is count + bias, with its thresholds a
fact and its channels on the input's last axis. Each condition it fails is a
finding: ``threshold-scale`` (``out_scale`` not 1) and ``threshold-bias``
(``out_bias`` not an integer), each naming streamlining's
``ExtractMultiThresholdScaleBias`` as a hint; ``threshold-dynamic`` (the thresholds
not an initializer); ``layout-unproven`` (channels on axis 1 of an input of rank
above 2, ``data_layout`` not NHWC: no Transpose is inserted, the layout is parked);
and ``fact-unstated`` when the input's shape, so its channel axis, is not known.
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
from finn.core.graph.facts import datatype, known_shape, rows, shape
from finn.core.space import Finding, Rejected
from finn.custom_op.kernels.base import KernelOp, Match, Shapes, node_attributes, refused, unstated
from finn.custom_op.kernels.cache import Facts
from finn.dataflow.datatypes import (
    DatatypeError,
    QONNXDataType,
    ordinary_integer_bounds,
    resolve_qonnx_datatype_name,
)
from finn.kernels.thresholding import ThresholdingAxiKernel

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

# A Thresholding is count + bias; streamlining's ExtractMultiThresholdScaleBias moves a
# scale and a bias out of a MultiThreshold.
LOWERING = "ExtractMultiThresholdScaleBias"


def stored_thresholds(
    stored: npt.NDArray[Any], input_dtype: QONNXDataType
) -> tuple[tuple[tuple[int, ...], ...], QONNXDataType]:
    """The table the kernel holds for the thresholds ``stored`` on an input of
    ``input_dtype``, and its type (the module docstring). The thresholds of an input
    that is not an integer type are taken as they are, in the input's type, which the
    kernel refuses."""
    try:
        low, high = ordinary_integer_bounds(input_dtype)
    except DatatypeError:
        return tuple(tuple(int(value) for value in row) for row in stored), input_dtype
    # float64 holds every integer threshold of an input up to 2**53, where float32
    # rounds past 2**24.
    table = np.clip(np.ceil(np.asarray(stored, dtype=np.float64)), low, high + 1)
    least, most = int(table.min()), int(table.max())
    if low < 0:
        floor = max(low, least - 1)  # a value below every threshold an input can fall short of
        bits = max(max((-floor - 1).bit_length() if floor < 0 else 0, most.bit_length()) + 1, 2)
        dtype = resolve_qonnx_datatype_name(f"INT{bits}")
    else:
        dtype = resolve_qonnx_datatype_name(f"UINT{max(most.bit_length(), 1)}")
    return tuple(tuple(int(value) for value in row) for row in table), dtype


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
                input_dtype=input_dtype,
                threshold_dtype=threshold_dtype,
                thresholds=(table,),
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
        x, thresholds = self.onnx_node.input
        values, table = context[x], context[thresholds]
        counts = (values[..., :, None] >= table).sum(axis=-1)
        bias = self.bias()
        context[self.onnx_node.output[0]] = (counts + bias).astype(np.asarray(values).dtype)


__all__ = ["Thresholding"]
