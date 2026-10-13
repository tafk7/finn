# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Shape and datatype inference in graph order (``InferKernelTensors``).

qonnx's inference passes ask every node at once, but a KernelOp can answer only once
its inputs are known. This pass visits the nodes in order instead (``infer_node``):

- a KernelOp computes its outputs' shapes and datatypes from its kernel. It overwrites
  what was there, so a change upstream (weights replaced, or made a graph input) reaches
  the nodes after it on the next run;
- another custom op uses its shape stand-in and its own datatype rule;
- a standard op uses ONNX's per-node shape inference, given its initializers' values (a
  Reshape's target shape), and qonnx's datatype rule. An output whose shape ONNX cannot
  infer keeps the one it had.

The KernelOps' qonnx hooks answer from their kernels too, so qonnx's passes agree once
this one has run. ``ToKernelOps`` runs the same step on each node as it converts; this
pass re-runs it after a later change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx.shape_inference as shape_inference
from onnx import NodeProto, defs, helper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import infer_node_datatype

from finn.core.graph.errors import KernelOpError
from finn.custom_op.kernels.base import KernelOp

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.core.reporting.findings import Finding


def _standard(model: ModelWrapper, node: NodeProto) -> None:
    """ONNX's shape inference for one standard node, its outputs' shapes written.

    An initializer input is given with its value (a Reshape's target shape); an output
    whose inferred shape is not fully known keeps the shape it had.
    """
    opsets = model.get_opset_imports()
    schema = defs.get_schema(node.op_type, opsets.get(node.domain, 13), node.domain)
    stored = {item.name: item for item in model.graph.initializer}
    types, values = {}, {}
    for name in node.input:
        if name in stored:
            values[name] = stored[name]
            info = helper.make_tensor_value_info(
                name, stored[name].data_type, list(stored[name].dims)
            )
        else:
            known = model.get_tensor_valueinfo(name)
            if known is None:
                raise KernelOpError(f"{node.name or node.op_type}: {name} has no shape yet")
            info = known
        types[name] = info.type
    for name, proto in shape_inference.infer_node_outputs(schema, node, types, values).items():
        tensor = proto.tensor_type
        dims = [dim.dim_value if dim.HasField("dim_value") else 0 for dim in tensor.shape.dim]
        if tensor.HasField("shape") and all(dims):
            model.set_tensor_shape(name, dims)


def infer_node(model: ModelWrapper, node: NodeProto, admit: bool = False) -> tuple[Finding, ...]:
    """Infer ``node``'s outputs from its inputs (see the module docstring).

    With ``admit``, as conversion calls it, a KernelOp the kernels refuse
    (``KernelOp.admission``) returns their refusals instead, its outputs left unset.
    """
    if not is_custom_op(node.domain):
        _standard(model, node)
        infer_node_datatype(model, node, False)
        return ()
    op = model.get_customop_wrapper(node)
    if not isinstance(op, KernelOp):
        standin = op.make_shape_compatible_op(model)
        _standard(model, standin)
        op.infer_node_datatype(model)
        return ()
    if admit:
        refused = op.admission()
        if refused:
            return refused
    for name, (dims, dtype) in op.infer_output_tensors(model).items():
        model.set_tensor_shape(name, list(dims))
        model.set_tensor_datatype(name, dtype)
    return ()


class InferKernelTensors(Transformation):
    """One pass in graph order, ``infer_node`` on each node; see the module docstring."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:
            infer_node(model, node)
        return model, False


__all__ = ["InferKernelTensors", "infer_node"]
