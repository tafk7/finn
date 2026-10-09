# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Thresholding KernelOp: Y = count(X >= T[c]) + bias, bound to ``ThresholdingAxiKernel``.

Integer ``MultiThreshold`` with ``out_scale`` 1, channels on the input's last
axis. The thresholds are an initializer the kernel holds (its value owner): one
set, a row for each channel (C, N) or one row for every channel (1, N), which
the kernel binds as the graph states it (a shared row is ``ThresholdingAxiKernel``'s
C = 1); admitted by its value summary, its digest in the key, its annotation
the threshold datatype. ``bias`` is a semantic attribute, part of the
operation. The output keeps the input's shape and takes the kernel's result
type, a fact-level derived of the table and the bias.

Input normalization: the ordered pass normalizes the thresholds first
(``normalize_inputs``), against the input's exact type: the values are rounded up
and clipped to ``[min, max + 1]`` of the input type (finn-dev's
``RoundAndClipThresholds``), then annotated with the smallest type of the input's
signedness that holds them (finn-dev's threshold ``minimize_weight_bit_width``,
which its flow runs after the rounding), and one value below the least of them
unless that is the input's minimum. They are stored in float64, which holds them
exactly. It keeps every count ``MultiThreshold`` computes: it rewrites the
thresholds only for an integer input type, where ``x >= t`` and ``x >= ceil(t)``
agree and a threshold outside the input's range counts the same at its bound. The
value below: thresholding_axi saturates an input wider than its thresholds to their
type, so an input below the type's minimum compares as that minimum, which a
threshold there would count (``ThresholdingAxiKernel``'s ``threshold-saturation``;
found by the harness's parity check, ``tests/kernel_ops/test_parity.py``).

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

from finn.core.space import Finding, Rejected
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    Match,
    Shapes,
    admitted,
    datatype,
    known_shape,
    node_attributes,
    refused,
    rows,
    shape,
    unstated,
)
from finn.custom_op.kernels.cache import Facts
from finn.dataflow.datatypes import (
    DatatypeError,
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

    def normalize_inputs(self) -> None:
        """The thresholds as integers against the input's exact type, every count
        unchanged (the module docstring's input normalization): rounded up, clipped to
        ``[min, max + 1]`` of the input's type, annotated with the narrowest type of the
        input's signedness that holds them and, unless the least is the input's minimum,
        a value below it. The thresholds of an input that is not an integer type are left
        as they are."""
        model, label = self.model(), self.label
        x, thresholds = self.onnx_node.input
        table = model.get_initializer(thresholds)
        try:
            low, high = ordinary_integer_bounds(datatype(model, x, label))
        except DatatypeError:
            return  # not an integer input: the kernel refuses it
        if table is None or table.ndim != 2:
            return  # facts refuse it, naming the shape
        # float64: ONNX gives a MultiThreshold's thresholds no type, and float64 holds
        # every integer threshold of an input up to 2**53, where float32 rounds past 2**24.
        table = np.clip(np.ceil(np.asarray(table, dtype=np.float64)), low, high + 1)
        least, most = int(table.min()), int(table.max())
        if low < 0:
            # A value below every threshold an input can fall short of (module docstring).
            floor = max(low, least - 1)
            bits = max(max((-floor - 1).bit_length() if floor < 0 else 0, most.bit_length()) + 1, 2)
            dtype = resolve_qonnx_datatype_name(f"INT{bits}")
        else:
            dtype = resolve_qonnx_datatype_name(f"UINT{max(most.bit_length(), 1)}")
        if len(model.find_consumers(thresholds)) > 1:
            thresholds = model.make_new_valueinfo_name()
            self.onnx_node.input[1] = thresholds
        model.set_initializer(thresholds, table)
        model.set_tensor_datatype(thresholds, dtype)

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
        table = model.get_initializer(thresholds)
        if table is None:
            raise KernelOpError(
                f"{label}: the thresholds {thresholds} are not an initializer: not a "
                "Thresholding (its pattern refuses them, threshold-dynamic)"
            )
        if table.ndim != 2 or table.shape[0] not in (1, channels):
            raise KernelOpError(
                f"{label}: the thresholds {thresholds} are {tuple(table.shape)}, neither one row "
                f"for every channel of {x} nor one for each of its {channels} channels"
            )
        input_dtype = datatype(model, x, label)
        threshold_dtype = datatype(model, thresholds, label)
        digest = admitted(model, thresholds, threshold_dtype, label)
        bias = self.bias()
        platform = self.target().platform

        def formals() -> dict[str, object]:
            values = model.get_initializer(thresholds)
            if values is None:
                raise KernelOpError(f"{label}: {thresholds} is not an initializer")
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
