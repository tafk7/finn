# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel partitions: what the flow reads about a partition of KernelOps, and what is
built of it.

A kernel partition is a partition model whose nodes are all KernelOps (the
``finn.custom_op.kernels`` domain, ``KERNEL_OPS_DOMAIN``), or the
StreamingDataflowPartition node whose body is one. The kernel path cuts its model once:
the parent graph holds one such node, and its body is opened through it
(``partition_body``).

Two typed namespaces of graph metadata (``qonnx.core.metadata``), and no flat key:

- ``finn.partition`` (``PARTITION``), on the partition's body: its boundary, the ends'
  facts. Per boundary port, in port order: its tensor and shape; the stream at the
  partition's free side (the channel's element, its value range, lanes, beats and
  TDATA width); and the end the shell places there, if it places one (its kind,
  direction, memory width, words a frame, whether a width converter sits between,
  the call's constant, frames a call, AXI-Lite buses: ``finn.kernels.ends.EndContract``).
  The kernel path's cut writes them (``finn.transformation.kernels.package.
  write_boundary_facts``). ``beats`` counts the whole tensor (one inference), a
  repetition the boundary keeps included. The element is the channel's: an end that
  moves bytes (``IODMA_hls``) frames it in its own container, and the facts say what
  the bytes are.
- ``finn.outputs`` (``OUTPUTS``), what a build made of it: on the partition's body,
  its packaged IP (directory, VLNV, interface names; ``PackagePartition``); on the
  build's model, the parent graph, what the shell's integration made (the integration
  project, the bitfile, the hardware handoff, the reports by name, and the host runtime
  that runs it).

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

PARTITION_OP = "StreamingDataflowPartition"
"""The op type of the parent graph's node whose body is a partition."""

PARTITION = Namespace("finn.partition", version=1)
"""A partition model's boundary, the ends' facts (typed graph metadata). They describe
that partition only, so a body does not inherit them."""

PORT_FACTS = ("port", "tensor", "shape", "element", "range", "lanes", "beats", "tdata", "end")
"""One boundary port's facts, the fields of each object in ``inputs`` and ``outputs``."""

END_FACTS = (
    "kind",
    "direction",
    "memory_width",
    "words",
    "converter",
    "call_cycles",
    "frames_per_call",
    "control_buses",
)
"""The facts of the end on a boundary port (``end``; ``None`` where the shell places none)."""


def _positive(item: object) -> bool:
    return type(item) is int and item > 0


def _named(item: object) -> bool:
    return isinstance(item, str) and bool(item)


def _end_facts(end: object) -> bool:
    """An end's facts: exactly ``END_FACTS``, named kind and direction (``in`` or
    ``out``), positive widths, counts and frames a call, a boolean converter, and
    no fewer than zero call cycles and buses."""
    return (
        isinstance(end, dict)
        and tuple(sorted(end)) == tuple(sorted(END_FACTS))
        and _named(end["kind"])
        and end["direction"] in ("in", "out")
        and all(_positive(end[name]) for name in ("memory_width", "words", "frames_per_call"))
        and type(end["converter"]) is bool
        and all(
            type(end[name]) is int and end[name] >= 0 for name in ("call_cycles", "control_buses")
        )
    )


def _port_facts(value: object) -> bool:
    """A list of port facts: objects with exactly ``PORT_FACTS``, the port, tensor and
    element named, the shape a list of positive ints, the counts and widths positive,
    the range two ints (or ``None``), the end an end's facts (or ``None``)."""

    def value_range(item: object) -> bool:
        return item is None or (
            isinstance(item, list)
            and len(item) == 2
            and all(type(bound) is int for bound in item)
            and item[0] <= item[1]
        )

    return isinstance(value, list) and all(
        isinstance(port, dict)
        and tuple(sorted(port)) == tuple(sorted(PORT_FACTS))
        and all(_named(port[name]) for name in ("port", "tensor", "element"))
        and isinstance(port["shape"], list)
        and all(_positive(dim) for dim in port["shape"])
        and all(_positive(port[name]) for name in ("lanes", "beats", "tdata"))
        and value_range(port["range"])
        and (port["end"] is None or _end_facts(port["end"]))
        for port in value
    )


_PORTS = f"a list of port facts (objects of {', '.join(PORT_FACTS)})"
PARTITION_INPUTS = PARTITION.key("inputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary inputs' facts, in port order (``s_axis_<i>``)."""
PARTITION_OUTPUTS = PARTITION.key("outputs", JSON, check=_port_facts, expect=_PORTS)
"""The boundary outputs' facts, in port order (``m_axis_<j>``)."""


OUTPUTS = Namespace("finn.outputs", version=1)
"""What a build made of a partition of KernelOps (typed graph metadata): the packaged
IP on the partition's body, the shell's integration on the build's parent graph."""


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


def kernel_partition_nodes(model: Any) -> list[Any]:
    """The model's StreamingDataflowPartition nodes whose body is a partition of KernelOps."""
    return [
        node
        for node in model.graph.node
        if node.op_type == PARTITION_OP and is_kernel_partition(ModelWrapper(_body_file(node)))
    ]


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


def partition_facts(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A partition body's boundary facts, inputs and outputs; a model without them is
    refused (the kernel path's cut writes them)."""
    inputs, outputs = model.get(PARTITION_INPUTS), model.get(PARTITION_OUTPUTS)
    if inputs is None or outputs is None:
        raise ValueError(
            "the partition model states no boundary facts (finn.partition); the kernel "
            "path's cut writes them (write_boundary_facts)"
        )
    return inputs, outputs


__all__ = [
    "END_FACTS",
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
    "PARTITION",
    "PARTITION_INPUTS",
    "PARTITION_OP",
    "PARTITION_OUTPUTS",
    "PORT_FACTS",
    "is_kernel_partition",
    "kernel_partition_nodes",
    "partition_body",
    "partition_facts",
]
