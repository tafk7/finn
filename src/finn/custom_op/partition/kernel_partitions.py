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

What a build made of a partition is one typed namespace of graph metadata
(``qonnx.core.metadata``), ``finn.outputs`` (``OUTPUTS``), and no flat key: on the
partition's body, its packaged IP (directory, VLNV, interface names;
``PackagePartition``); on the build's model, the parent graph, what the shell's
integration made (the integration project, the bitfile, the hardware handoff, the
reports by name, and the host runtime that runs it). A partition's boundary is not
stored: what reads it (the integration export and the driver it describes, the
interface description, the testbench) derives it from the configured root
(``finn.transformation.kernels.package.configured_root``).

This module depends on qonnx only, below the KernelOps and their transformations, so
the builder and the flow import it without loading the kernel stack.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.metadata import JSON, Namespace
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


def _reports(value: object) -> bool:
    return isinstance(value, dict) and all(_named(path) for path in value.values())


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
OUTPUT_PROJECT = OUTPUTS.key("project", str, check=_named, expect="a directory")
"""The shell's integration project (the Vivado block design's)."""
OUTPUT_BITFILE = OUTPUTS.key("bitfile", str, check=_named, expect="a file")
"""The bitfile the shell's integration made."""
OUTPUT_HWH = OUTPUTS.key("hwh", str, check=_named, expect="a file")
"""The bitfile's hardware handoff."""
OUTPUT_REPORTS = OUTPUTS.key(
    "reports", JSON, check=_reports, expect="report files by name (synthesis, timing, ...)"
)
"""The integration's reports by name."""
OUTPUT_HOST_RUNTIME = OUTPUTS.key("host_runtime", str, check=_named, expect="a host runtime")
"""What runs the bitfile on the host (``finn.platform.ShellRow.host_runtime``)."""


def is_kernel_partition(model: Any) -> bool:
    """Whether a partition model is a model of KernelOps: it has nodes, all KernelOps."""
    return bool(model.graph.node) and all(
        node.domain == KERNEL_OPS_DOMAIN for node in model.graph.node
    )


def _body_file(node: Any) -> str:
    path = getCustomOp(node).get_nodeattr("model")
    if not isinstance(path, str):
        raise TypeError(f"{node.name}: its model attribute is {path!r}, not a path")
    return path


def kernel_partition_body(node: Any) -> ModelWrapper | None:
    """The body of a StreamingDataflowPartition node whose body is a partition of
    KernelOps, opened through the node; None for any other node."""
    if node.op_type != PARTITION_OP or node.domain != PARTITION_DOMAIN:
        return None
    body = ModelWrapper(_body_file(node))
    return body if is_kernel_partition(body) else None


def kernel_partition_nodes(model: Any) -> list[Any]:
    """The model's partition nodes (``PARTITION_DOMAIN``'s StreamingDataflowPartition)
    whose body is a partition of KernelOps."""
    return [node for node in model.graph.node if kernel_partition_body(node) is not None]


def partition_body(model: Any) -> tuple[Any, ModelWrapper, str]:
    """The kernel path's parent graph's one partition node of KernelOps, its body opened
    through the node, and the body's file. A graph with none, or more than one, is
    refused: the kernel path cuts its KernelOps once."""
    found = kernel_partition_nodes(model)
    if len(found) != 1:
        raise ValueError(
            f"the model holds {len(found)} partitions of KernelOps; the kernel path's "
            "parent graph holds one (step_kernel_partition cuts once)"
        )
    (node,) = found
    path = _body_file(node)
    return node, ModelWrapper(path), path


__all__ = [
    "KERNEL_OPS_DOMAIN",
    "OUTPUTS",
    "OUTPUT_BITFILE",
    "OUTPUT_HOST_RUNTIME",
    "OUTPUT_HWH",
    "OUTPUT_INTERFACES",
    "OUTPUT_IP",
    "OUTPUT_PROJECT",
    "OUTPUT_REPORTS",
    "OUTPUT_VLNV",
    "PARTITION_DOMAIN",
    "PARTITION_OP",
    "is_kernel_partition",
    "kernel_partition_body",
    "kernel_partition_nodes",
    "partition_body",
]
