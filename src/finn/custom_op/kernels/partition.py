# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition root: one Kernel of a set of KernelOp nodes, its streams the graph's tensors.

``partition_root(model, nodes)`` builds, and changes no graph:

- **streams**, one per ONNX tensor, in node order: a node's graph inputs, the
  parameter streams it owns (named after the initializer, the kernel's view),
  its outputs. A stream's tensor is the graph's value_info and annotation (D6),
  its platform the model's target's, a MatMul's weight operand a
  ``BufferedStream``. Only the subgraph's ONNX
  inputs and outputs are boundaries, named by the shell's convention
  ``s_axis_<i>`` and ``m_axis_<i>`` (D4): a stream refuses a boundary no port
  names (``stream-boundary``);
- **kernels**, one per node, from its facts, the graph's pins as keywords;
- **replay**: each node's kernel choices, then the edge choices (an edge's
  adapter selector is forced, never persisted); an edge choice the current
  graph refuses is stale, dropped and reported, and the forced case applies
  again;
- **owners**: each member's node and attribute prefix, how a choice made in the
  root goes back to the node that persists it (D8): a kernel's on its node, an
  edge's on its consumer, a parameter stream's on its value owner.

Members are named as the graph: streams by tensor and kernels by node
(``\\W`` as ``_``); a node and a tensor of one name are refused, not renamed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from finn.core.space import composite, design_space, inspection
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    committed,
    datatype,
    rows,
    shape,
    target,
    value_type,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.streams import BufferedStream, Stream


class Partition(Kernel):
    """A partition's hardware: one stream per ONNX tensor, one kernel per node."""

    id = "finn.custom_op.kernels.partition"
    version = 1


@dataclass(frozen=True)
class PartitionRoot:
    """The configured root; each member's owning node and attribute prefix; the edge
    choices replay dropped as stale; each boundary tensor's port."""

    point: Any
    owners: Mapping[str, tuple[str, str]]
    dropped: tuple[str, ...]
    boundary: tuple[tuple[str, str], ...]


def member(name: str) -> str:
    """A graph name as a member name."""
    return re.sub(r"\W", "_", name)


def _typed(point: Any, choices: Mapping[str, object]) -> dict[str, object]:
    kinds = {item.key: value_type(item) for item in inspection.decisions(point)}
    return {
        key: bool(value) if kinds.get(key) == "bool" else value for key, value in choices.items()
    }


def partition_root(model: Any, nodes: Iterable[Any], *, name: str = "partition") -> PartitionRoot:
    """The root of ``nodes``, KernelOp nodes of ``model``; see the module docstring."""
    nodes = list(nodes)
    ops = [model.get_customop_wrapper(node) for node in nodes]
    for node, op in zip(nodes, ops):
        if not isinstance(op, KernelOp):
            raise KernelOpError(f"{node.name}: a partition root places KernelOps only")
    produced = {tensor for node in nodes for tensor in node.output}
    inside = {id(node) for node in nodes}
    used_outside = {
        tensor for node in model.graph.node if id(node) not in inside for tensor in node.input
    } | {output.name for output in model.graph.output}
    owned = {tensor for op in ops for tensor in op.owned_streams()}
    inputs = [
        tensor
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced and tensor not in owned and model.get_initializer(tensor) is None
    ]
    outputs = [tensor for node in nodes for tensor in node.output if tensor in used_outside]
    ports = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    ports |= {tensor: f"m_axis_{index}" for index, tensor in enumerate(outputs)}
    weights = {node.input[1] for node, op in zip(nodes, ops) if op.op_type == "MatMul"}
    platform = target(model).platform

    members: dict[str, object] = {}
    streams: dict[str, Stream] = {}

    def add(tensor: str, stream: Stream) -> None:
        streams[tensor] = members[member(tensor)] = stream

    def declare(tensor: str, label: str) -> None:
        if tensor in streams:
            return
        dims = rows(shape(model, tensor, label))
        carried = Tensor(dims, ScalarEncoding(datatype(model, tensor, label)))
        kind = BufferedStream if tensor in weights else Stream
        port: dict[str, Any] = {"port": ports[tensor]} if tensor in ports else {}
        add(tensor, kind(tensor=carried, platform=platform, **port))

    # Streams in node order: a node's graph inputs, the parameter streams it owns, its outputs.
    for node, op in zip(nodes, ops):
        for tensor in node.input:
            if tensor in inputs or tensor in produced:
                declare(tensor, op.label)
        for tensor, stream in op.owned_streams().items():
            add(tensor, stream)
        for tensor in node.output:
            declare(tensor, op.label)

    owners: dict[str, tuple[str, str]] = {}
    kernel_choices: dict[str, object] = {}
    edge_choices: dict[str, object] = {}
    for node, op in zip(nodes, ops):
        kernel = member(node.name)
        if kernel in members:
            raise KernelOpError(f"{node.name}: a node and a tensor are both named {kernel}")
        members[kernel], by_port = op.place(streams)
        owners[kernel] = (node.name, "")
        for port, tensor in by_port.items():
            owners[member(tensor)] = (node.name, f"{port}.")
        for attribute, value in op.choices().items():
            head, _, rest = attribute.partition(".")
            if head in by_port:
                edge_choices[f"{member(by_port[head])}.{rest}"] = value
            else:
                kernel_choices[f"{kernel}.{attribute}"] = value

    root: Any = composite(name, members, base=Partition)
    point = design_space(root())
    if kernel_choices:
        replayed = committed(point, _typed(point, kernel_choices))
        if isinstance(replayed, dict):
            raise KernelOpError(
                f"{name}: refused choices: "
                + "; ".join(f"{key}: {why}" for key, why in sorted(replayed.items())),
                tuple(sorted(replayed)),
            )
        point = replayed
    dropped: list[str] = []
    if edge_choices:
        edges = _typed(point, edge_choices)
        together = committed(point, edges)
        if isinstance(together, dict):
            for key, value in edges.items():
                alone = committed(point, {key: value})
                if isinstance(alone, dict):
                    dropped.append(key)
                else:
                    point = alone
        else:
            point = together
    return PartitionRoot(point, owners, tuple(dropped), tuple(ports.items()))


def save_partition_choices(
    model: Any, root: PartitionRoot, choices: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    """Persist choices made on purpose in a partition root, each on its owning node (D8)."""
    per_node: dict[str, dict[str, object]] = {}
    for key, value in choices.items():
        head, _, rest = key.partition(".")
        if head not in root.owners:
            raise KernelOpError(f"{key}: no node of the partition owns {head}")
        node, prefix = root.owners[head]
        per_node.setdefault(node, {})[prefix + rest] = value
    by_name = {node.name: node for node in model.graph.node}
    for node, values in per_node.items():
        model.get_customop_wrapper(by_name[node]).save(values)
    return per_node


__all__ = ["Partition", "PartitionRoot", "member", "partition_root", "save_partition_choices"]
