# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX operators to KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` states the build target it is given (``finn.platform.resolve_target``
resolves it) once, in the one place ``read_target(model)`` reads it, the model's
``finn.platform`` metadata, imports the domain at its ``opset_version`` when the
model does not import it yet (inserting a node never raises a model's import), and
rewrites each node it can bind:

- ``MatMul`` (the ONNX operator) into a ``MatMul`` KernelOp;
- ``MultiThreshold`` with ``out_scale`` 1, an integral ``out_bias`` and its
  channels on the input's last axis (``NHWC``, or a 2-D input) into a
  ``Thresholding`` with that ``bias``.

Other nodes are left alone. The nodes keep their names, inputs and outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from onnx import NodeProto, helper
from qonnx.transformation.base import Transformation

import finn.custom_op.kernels as domain
from finn.custom_op.kernels.base import write_target
from finn.custom_op.partition.kernel_partitions import KERNEL_OPS_DOMAIN
from finn.kernels.target import Target

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def _attributes(node: NodeProto) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _thresholding(model: ModelWrapper, node: NodeProto) -> NodeProto | None:
    """A Thresholding for an integer MultiThreshold over the input's last axis, if it is one."""
    attributes = _attributes(node)
    if float(attributes.get("out_scale", 1.0)) != 1.0:
        return None
    bias = float(attributes.get("out_bias", 0.0))
    if bias != int(bias):
        return None
    layout = attributes.get("data_layout", b"NCHW")
    layout = layout.decode() if isinstance(layout, bytes) else layout
    dims = model.get_tensor_shape(node.input[0])
    if layout != "NHWC" and (dims is None or len(dims) > 2):
        return None  # channels on axis 1 of a wider tensor, or a tensor not known yet
    return helper.make_node(
        "Thresholding",
        list(node.input),
        list(node.output),
        name=node.name,
        domain=KERNEL_OPS_DOMAIN,
        bias=int(bias),
    )


class ToKernelOps(Transformation):
    """Each node a KernelOp binds rewritten as one, the target stated in the model."""

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.target = target

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        write_target(model, self.target)
        if KERNEL_OPS_DOMAIN not in model.get_opset_imports():
            model.set_opset_import(KERNEL_OPS_DOMAIN, domain.opset_version)
        graph = model.graph
        for index, node in enumerate(list(graph.node)):
            new: NodeProto | None
            if node.op_type == "MatMul" and node.domain == "":
                new = helper.make_node(
                    "MatMul",
                    list(node.input),
                    list(node.output),
                    name=node.name,
                    domain=KERNEL_OPS_DOMAIN,
                )
            elif node.op_type == "MultiThreshold":
                new = _thresholding(model, node)
                if new is None:
                    continue
            else:
                continue
            graph.node.remove(node)
            graph.node.insert(index, new)
        return model, False


__all__ = ["ToKernelOps"]
