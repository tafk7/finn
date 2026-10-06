# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ordered inference: every tensor's shape and datatype, node by node, in graph order.

A KernelOp binds from its inputs' shapes and annotations, so qonnx's
whole-graph passes cannot run on a graph whose KernelOps' inputs are not known
yet: they ask every node at once. ``InferKernelTensors`` visits the nodes in
graph order instead:

- a KernelOp first normalizes its value inputs against its inputs' exact types
  (``normalize_inputs``: a Thresholding's thresholds as integers), then answers
  its outputs from its kernel's fact-level views, in its node root bound on its
  inputs (the facts, the input channels' tensors and an owned initializer's
  value: MatMul's result range is its weights' columns'; the whole root reads the
  outputs this pass states), and the pass writes them. An output's annotation is
  its producer's statement: every run states it again from the node's facts and
  replaces what was there, narrower or wider, so a change of the facts (weights
  replaced or lifted to a graph input) reaches the nodes that follow on the next
  run. An input that conflicts with the node (unannotated, or an initializer whose
  values its annotation does not hold) is refused by the node's facts;
- another custom op runs its shape stand-in through ONNX's per-node inference,
  then its own datatype hook;
- a standard op goes through ONNX's per-node inference, its initializers given
  with their values (a Reshape's target shape), and qonnx's datatype rule; an
  output whose shape the inference leaves unknown keeps the one it had.

The KernelOps' own qonnx hooks answer from the same views, so qonnx's passes
agree once this one has run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx.shape_inference as shape_inference
from onnx import NodeProto, defs, helper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import infer_node_datatype

from finn.custom_op.kernels.base import KernelOp, KernelOpError

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


def _standard(model: ModelWrapper, node: NodeProto) -> None:
    """ONNX's inference for one standard node, its outputs' shapes written.

    An initializer input is given with its value (a Reshape's target shape is
    one), as stored; an output whose inferred shape is not fully known keeps
    the shape it had.
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


class InferKernelTensors(Transformation):
    """One pass in graph order; see the module docstring."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:
            if not is_custom_op(node.domain):
                _standard(model, node)
                infer_node_datatype(model, node, False)
                continue
            op = model.get_customop_wrapper(node)
            if not isinstance(op, KernelOp):
                standin = op.make_shape_compatible_op(model)
                _standard(model, standin)
                op.infer_node_datatype(model)
                continue
            op.normalize_inputs()
            for name, (dims, dtype) in op.infer_output_tensors(model).items():
                model.set_tensor_shape(name, list(dims))
                model.set_tensor_datatype(name, dtype)
        return model, False


__all__ = ["InferKernelTensors"]
