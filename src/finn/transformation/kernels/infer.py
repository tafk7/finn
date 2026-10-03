# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ordered inference: every tensor's shape and datatype, node by node, in graph order.

A KernelOp binds from its inputs' shapes and annotations, so qonnx's
whole-graph passes cannot run on a graph whose KernelOps' inputs are not known
yet: they ask every node at once. ``InferKernelTensors`` visits the nodes in
graph order instead:

- a KernelOp answers its outputs from its node root's fact-level views, and the
  pass writes them; a stated annotation narrower than the exact type is
  refused, a wider one replaced (FLOAT32 is no statement);
- another custom op runs its shape stand-in through ONNX's per-node inference,
  then its own datatype hook;
- a standard op goes through ONNX's per-node inference and qonnx's datatype rule.

The KernelOps' own qonnx hooks answer from the same views, so qonnx's passes
agree once this one has run.
"""

from __future__ import annotations

from typing import Any

import onnx.shape_inference as shape_inference
from onnx import TensorProto, defs, helper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import _infer_node_datatype

from finn.custom_op.kernels.base import KernelOp, KernelOpError, annotated, datatype
from finn.dataflow.datatypes import DatatypeError, QONNXDataType, ordinary_integer_bounds


def _admits(stated: QONNXDataType, exact: QONNXDataType) -> bool:
    """Whether a stated annotation holds every value of the exact type."""
    try:
        low, high = ordinary_integer_bounds(stated)
    except DatatypeError:
        return False
    need_low, need_high = ordinary_integer_bounds(exact)
    return low <= need_low and need_high <= high


def _standard(model: Any, node: Any) -> None:
    """ONNX's inference for one standard node, its outputs' shapes written."""
    opsets = model.get_opset_imports()
    schema = defs.get_schema(node.op_type, opsets.get(node.domain, 13), node.domain)
    types = {}
    for name in node.input:
        info = model.get_tensor_valueinfo(name)
        if info is None:
            initializer = model.get_initializer(name)
            if initializer is None:
                raise KernelOpError(f"{node.name or node.op_type}: {name} has no shape yet")
            info = helper.make_tensor_value_info(name, TensorProto.FLOAT, list(initializer.shape))
        types[name] = info.type
    for name, proto in shape_inference.infer_node_outputs(schema, node, types).items():
        model.set_tensor_shape(name, [dim.dim_value for dim in proto.tensor_type.shape.dim])


class InferKernelTensors(Transformation):  # type: ignore[misc]
    """One pass in graph order; see the module docstring."""

    def apply(self, model: Any) -> tuple[Any, bool]:
        for node in model.graph.node:
            if not is_custom_op(node.domain):
                _standard(model, node)
                _infer_node_datatype(model, node, False)
                continue
            op = model.get_customop_wrapper(node)
            if not isinstance(op, KernelOp):
                standin = op.make_shape_compatible_op(model)
                _standard(model, standin)
                op.infer_node_datatype(model)
                continue
            for name, (dims, dtype) in op.infer_output_tensors(model).items():
                if annotated(model, name):
                    stated = datatype(model, name, op.label)
                    if stated.name != "FLOAT32" and not _admits(stated, dtype):
                        raise KernelOpError(
                            f"{op.label}: {name} is annotated {stated.name}, narrower than "
                            f"{dtype.name}"
                        )
                model.set_tensor_shape(name, list(dims))
                model.set_tensor_datatype(name, dtype)
        return model, False


__all__ = ["InferKernelTensors"]
