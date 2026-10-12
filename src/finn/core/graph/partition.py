# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's partition: its node, its body, and the digest of its design.

A kernel partition is a model whose nodes are all KernelOps (``KERNEL_OPS_DOMAIN``), or
the ``StreamingDataflowPartition`` node (``PARTITION_DOMAIN``) whose ``model`` attribute
names such a body. The kernel path cuts its model once, so its parent graph holds one
such node, and ``partition_body`` opens the body through it.

``design_digest`` is the digest of a model's design: the model serialized
deterministically, with ``finn.outputs`` cleared and the orders that carry no meaning
made canonical. It covers the choices, the facts, the weights and the target, and
nothing a build made, so an artifact states the digest of the design it was made from.

It imports qonnx only, so the builder and the flow read a partition without the kernel
stack.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, MutableSequence
from typing import Any

from onnx import ModelProto, NodeProto
from qonnx.core.metadata import clear
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.core.graph.annotations.outputs import OUTPUTS

KERNEL_OPS_DOMAIN = "finn.custom_op.kernels"
"""The KernelOps' ONNX domain: the name of the package that registers them."""

PARTITION_DOMAIN = "finn.custom_op.partition"
"""The domain of the parent graph's partition node: the name of the package that
registers it."""

PARTITION_OP = "StreamingDataflowPartition"
"""The op type of the parent graph's node whose body is a partition."""


def _sort(field: MutableSequence[Any], key: Callable[[Any], Any]) -> None:
    """Order a repeated field of messages by ``key``, in place."""
    ordered = [copy.deepcopy(item) for item in sorted(field, key=key)]
    del field[:]
    field.extend(ordered)


def design_digest(model: ModelWrapper) -> str:
    """The digest of ``model``'s design: its ``ModelProto`` with ``finn.outputs`` cleared,
    the orders that carry no meaning made canonical (the model's and the graph's
    metadata entries, each node's attributes, the initializers, the value infos, the
    tensors' annotations and their entries), serialized deterministically, as
    ``sha256:<hex>``. It covers the choices (node attributes, ``finn.channel`` on the
    tensors), the facts, the weights and the target, and is the same for the model
    saved and read back. The model is not changed."""
    proto = ModelProto()
    proto.CopyFrom(model.model)
    graph = proto.graph
    clear(graph.metadata_props, OUTPUTS)
    _sort(graph.metadata_props, lambda prop: prop.key)
    _sort(proto.metadata_props, lambda prop: prop.key)
    for node in graph.node:
        _sort(node.attribute, lambda attribute: attribute.name)
    _sort(graph.initializer, lambda tensor: tensor.name)
    _sort(graph.value_info, lambda info: info.name)
    for annotation in graph.quantization_annotation:
        _sort(annotation.quant_parameter_tensor_names, lambda prop: prop.key)
    _sort(graph.quantization_annotation, lambda annotation: annotation.tensor_name)
    return "sha256:" + hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest()


def is_partition(node: NodeProto) -> bool:
    """Whether ``node`` is a partition node (StreamingDataflowPartition): its body is a
    model of its own."""
    return bool(node.domain == PARTITION_DOMAIN and node.op_type == PARTITION_OP)


def is_kernel_partition(model: ModelWrapper) -> bool:
    """Whether a partition model is a model of KernelOps: it has nodes, all KernelOps."""
    return bool(model.graph.node) and all(
        node.domain == KERNEL_OPS_DOMAIN for node in model.graph.node
    )


def body_file(node: NodeProto) -> str:
    """The file of a partition node's body: its ``model`` attribute, refused
    (``TypeError``) when that is no path."""
    path = getCustomOp(node).get_nodeattr("model")
    if not isinstance(path, str):
        raise TypeError(f"{node.name}: its model attribute is {path!r}, not a path")
    return path


def kernel_partition_body(node: NodeProto) -> ModelWrapper | None:
    """The body of a StreamingDataflowPartition node whose body is a partition of
    KernelOps, opened through the node; None for any other node."""
    if not is_partition(node):
        return None
    body = ModelWrapper(body_file(node))
    return body if is_kernel_partition(body) else None


def kernel_partition_nodes(model: ModelWrapper) -> list[NodeProto]:
    """The model's partition nodes (``PARTITION_DOMAIN``'s StreamingDataflowPartition)
    whose body is a partition of KernelOps."""
    return [node for node in model.graph.node if kernel_partition_body(node) is not None]


def partition_body(model: ModelWrapper) -> tuple[NodeProto, ModelWrapper, str]:
    """The kernel path's parent graph's one partition node of KernelOps, its body opened
    through the node, and the body's file. A graph with none, or more than one, is
    refused: the kernel path cuts its KernelOps once."""
    found = kernel_partition_nodes(model)
    if len(found) != 1:
        raise ValueError(
            f"the model holds {len(found)} partitions of KernelOps; the kernel path's "
            "parent graph holds one (step_create_dataflow_partition cuts once)"
        )
    (node,) = found
    path = body_file(node)
    return node, ModelWrapper(path), path


__all__ = [
    "KERNEL_OPS_DOMAIN",
    "PARTITION_DOMAIN",
    "PARTITION_OP",
    "body_file",
    "design_digest",
    "is_kernel_partition",
    "is_partition",
    "kernel_partition_body",
    "kernel_partition_nodes",
    "partition_body",
]
