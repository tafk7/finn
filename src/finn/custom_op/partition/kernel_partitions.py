# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel partitions: what the flow reads about a partition of KernelOps, and what is
built of it.

A kernel partition is a partition model whose nodes are all KernelOps (the
``finn.custom_op.kernels`` domain, ``KERNEL_OPS_DOMAIN``), or the
StreamingDataflowPartition node (of this package's domain, ``PARTITION_DOMAIN``) whose
body is one. The kernel path cuts its model once (``finn.transformation.kernels.cut``):
the parent graph holds one such node, and its body is opened through it
(``partition_body``).

What a build made of a partition that a later step consumes is one typed namespace of
graph metadata (``qonnx.core.metadata``), ``finn.outputs`` (``OUTPUTS``), and no flat
key: on the partition's body, its packaged IP (directory, VLNV, interface names;
``PackagePartition``) and the digest of the design it was made from (``made_from``);
on the build's model, the parent graph, the bitfile the shell's integration made and
its hardware handoff, with the digest of the IP's design and the shell options the
bitfile was built with. The run's own record (its reports, the tools' projects and
transcripts) is the build's manifest's, not the model's
(``finn.builder.build_dataflow_manifest``; decision SR3, item 4). A partition's
boundary is not stored: what reads it (the integration export and the driver it
describes, the interface description, the testbench) derives it from the configured
root (``finn.custom_op.kernels.shell.configured_root``).

An artifact is keyed by the design it was made from, not cleared when the design
changes: ``design_digest`` is the digest of a model, serialized
deterministically with ``finn.outputs`` cleared, so it covers the choices, the facts,
the weights and the target, and nothing a build made. A reader compares the digest an
artifact states with the body's now: the pynq shell's build packages the partition
again on a mismatch, the driver and the deployment refuse a bitfile built from another
design, and the deployment a driver made from another (decision SR6, item 8).

This module depends on qonnx (and onnx, which qonnx depends on) only, below the
KernelOps and their transformations, so the builder and the flow import it without
loading the kernel stack.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Callable, MutableSequence
from typing import Any

from onnx import ModelProto, NodeProto
from qonnx.core.metadata import JSON, Namespace, clear
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

KERNEL_OPS_DOMAIN = "finn.custom_op.kernels"
"""The KernelOps' ONNX domain: the name of the package that registers them."""

PARTITION_DOMAIN = "finn.custom_op.partition"
"""The domain of the parent graph's partition node: the name of the package that
registers it."""

PARTITION_OP = "StreamingDataflowPartition"
"""The op type of the parent graph's node whose body is a partition."""

OUTPUTS = Namespace("finn.outputs", version=1)
"""What a build made of a partition of KernelOps (typed graph metadata): the packaged
IP on the partition's body, the shell's integration on the build's parent graph."""


def _named(item: object) -> bool:
    return isinstance(item, str) and bool(item)


def _interface_names(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(names, list) for names in value.values())


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _options(value: object) -> bool:
    return isinstance(value, dict)


OUTPUT_IP = OUTPUTS.key("ip", str, check=_named, expect="a directory")
"""The packaged IP's directory (its ``component.xml``)."""
OUTPUT_VLNV = OUTPUTS.key("vlnv", str, check=_named, expect="a VLNV")
"""The packaged IP's VLNV."""
OUTPUT_INTERFACES = OUTPUTS.key(
    "interfaces",
    JSON,
    check=_interface_names,
    expect="interface names by kind (clk, rst, s_axis, m_axis, axilite, ...)",
)
"""The packaged IP's interface names by kind, as the stitched-IP contract lists them."""
OUTPUT_MADE_FROM = OUTPUTS.key(
    "made_from", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design the packaged IP was made from (``design_digest`` of the
body it was packaged from)."""
OUTPUT_BITFILE = OUTPUTS.key("bitfile", str, check=_named, expect="a file")
"""The bitfile the shell's integration made."""
OUTPUT_HWH = OUTPUTS.key("hwh", str, check=_named, expect="a file")
"""The bitfile's hardware handoff."""
OUTPUT_BITFILE_MADE_FROM = OUTPUTS.key(
    "bitfile_made_from", str, check=_digest, expect="a design digest (sha256:<64 hex digits>)"
)
"""The digest of the design of the partition's IP the bitfile was built from (the body's
``made_from`` when the shell's build ran)."""
OUTPUT_BITFILE_OPTIONS = OUTPUTS.key(
    "bitfile_options", JSON, check=_options, expect="the shell's build options by name"
)
"""The shell's build options the bitfile was built with (``enable_hw_debug``)."""


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
    "OUTPUTS",
    "OUTPUT_BITFILE",
    "OUTPUT_BITFILE_MADE_FROM",
    "OUTPUT_BITFILE_OPTIONS",
    "OUTPUT_HWH",
    "OUTPUT_INTERFACES",
    "OUTPUT_IP",
    "OUTPUT_MADE_FROM",
    "OUTPUT_VLNV",
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
