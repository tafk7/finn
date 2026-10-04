# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion: ONNX operators to KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` states the build target once (a ``Target``, from
``finn.kernels.target.resolve_target``: the part, the clock period and the
platform's capabilities), in the one place ``target(model)`` reads it, the
model's ``finn.platform`` metadata (a model stating phase 1's untyped target
keys is refused), imports the domain at its ``opset_version`` when the model does not
import it yet (inserting a node never raises a model's import), and rewrites
each node it can bind:

- ``MatMul`` (the ONNX operator) into a ``MatMul`` KernelOp;
- ``MultiThreshold`` with ``out_scale`` 1, an integral ``out_bias`` and its
  channels on the input's last axis (``NHWC``, or a 2-D input) into a
  ``Thresholding`` with that ``bias``.

Other nodes are left alone. The nodes keep their names, inputs and outputs.
"""

from __future__ import annotations

from typing import Any

from onnx import helper
from qonnx.transformation.base import Transformation

import finn.custom_op.kernels as domain
from finn.custom_op.kernels.base import refuse_phase1_target, write_target
from finn.kernels.target import Target

DOMAIN = domain.__name__


def _attributes(node: Any) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _thresholding(model: Any, node: Any) -> Any | None:
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
        domain=DOMAIN,
        bias=int(bias),
    )


class ToKernelOps(Transformation):  # type: ignore[misc]
    """Each node a KernelOp binds rewritten as one, the target stated in the model."""

    def __init__(self, target: Target) -> None:
        super().__init__()
        self.target = target

    def apply(self, model: Any) -> tuple[Any, bool]:
        refuse_phase1_target(model)
        write_target(model, self.target)
        if DOMAIN not in model.get_opset_imports():
            model.set_opset_import(DOMAIN, domain.opset_version)
        graph = model.graph
        for index, node in enumerate(list(graph.node)):
            new: Any
            if node.op_type == "MatMul" and node.domain == "":
                new = helper.make_node(
                    "MatMul", list(node.input), list(node.output), name=node.name, domain=DOMAIN
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
