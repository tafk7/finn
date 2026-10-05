# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel partitions: what the flow reads about a partition of KernelOps.

A kernel partition is a partition model whose nodes are all KernelOps (the
``finn.custom_op.kernels`` domain, ``KERNEL_OPS_DOMAIN``), or the
StreamingDataflowPartition node whose body is one. Its boundary facts are typed
metadata on the partition model, the ``finn.partition`` namespace (``PARTITION``):
per boundary port, in port order, its tensor, shape, datatype, lanes, beats,
element bits and TDATA width. PackagePartition writes them
(``finn.transformation.kernels.package``); InsertIODMA and the driver read them
here (``partition_facts``, ``kernel_partition_ports``) instead of asking a first or
last HW node. ``beats`` counts the whole tensor (one inference), a repetition the
boundary keeps included.

This module depends on qonnx only, so the flow imports it at the top without
loading the kernel stack.
"""

from __future__ import annotations

from qonnx.core.metadata import JSON, Namespace
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import Any

KERNEL_OPS_DOMAIN = "finn.custom_op.kernels"
"""The KernelOps' ONNX domain: the name of the package that registers them."""

PARTITION = Namespace("finn.partition", version=1)
"""A partition model's boundary facts (typed graph metadata, ``qonnx.core.metadata``).
They describe that partition only, so a body does not inherit them."""

PORT_FACTS = ("port", "tensor", "shape", "datatype", "lanes", "beats", "element_bits", "tdata")
"""One boundary port's facts, the fields of each object in ``inputs`` and ``outputs``."""


def _port_facts(value: object) -> bool:
    """A list of port facts: objects with exactly ``PORT_FACTS``, the port, tensor and
    datatype named, the shape a list of positive ints, the counts and widths positive."""

    def positive(item: object) -> bool:
        return type(item) is int and item > 0

    return isinstance(value, list) and all(
        isinstance(port, dict)
        and tuple(sorted(port)) == tuple(sorted(PORT_FACTS))
        and all(
            isinstance(port[name], str) and port[name] for name in ("port", "tensor", "datatype")
        )
        and isinstance(port["shape"], list)
        and all(positive(dim) for dim in port["shape"])
        and all(positive(port[name]) for name in ("lanes", "beats", "element_bits", "tdata"))
        for port in value
    )


_PORTS = f"a list of port facts (objects of {', '.join(PORT_FACTS)})"
PARTITION_INPUTS = PARTITION.key("inputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary inputs' facts, in port order (``s_axis_<i>``)."""
PARTITION_OUTPUTS = PARTITION.key("outputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary outputs' facts, in port order (``m_axis_<j>``)."""


def is_kernel_partition(model: Any) -> bool:
    """Whether a partition model is a model of KernelOps: it has nodes, all KernelOps."""
    return bool(model.graph.node) and all(
        node.domain == KERNEL_OPS_DOMAIN for node in model.graph.node
    )


def partition_facts(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A packaged partition model's boundary facts, inputs and outputs; a model without
    them is refused (PackagePartition writes them)."""
    inputs, outputs = model.get(PARTITION_INPUTS), model.get(PARTITION_OUTPUTS)
    if inputs is None or outputs is None:
        raise ValueError(
            "the partition model states no boundary facts (finn.partition); run PackagePartition"
        )
    return inputs, outputs


def kernel_partition_ports(node: Any) -> dict[str, dict[str, Any]] | None:
    """For a StreamingDataflowPartition of KernelOps (its body packaged), each boundary
    port's facts by the partition node's tensor that the port carries; None for any
    other node, so it also tells a partition of KernelOps apart. The body is loaded
    once; its ports are in its graph's order, which is the partition node's."""
    if node.op_type != "StreamingDataflowPartition":
        return None
    body = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
    if not is_kernel_partition(body):
        return None
    inputs, outputs = partition_facts(body)
    return dict(zip(node.input, inputs, strict=True)) | dict(zip(node.output, outputs, strict=True))


__all__ = [
    "KERNEL_OPS_DOMAIN",
    "PARTITION",
    "PARTITION_INPUTS",
    "PARTITION_OUTPUTS",
    "PORT_FACTS",
    "is_kernel_partition",
    "kernel_partition_ports",
    "partition_facts",
]
