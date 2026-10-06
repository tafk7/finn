# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition root: one Kernel of a set of KernelOp nodes, its channels the graph's tensors.

``partition_root(model, nodes)`` builds, and changes no graph:

- **channels**, one per ONNX tensor, in node order: a node's graph inputs, the
  parameter channels it owns (named after the initializer, the kernel's view),
  its outputs. A channel's tensor is the graph's value_info and annotation
  its platform the model's target's. Only the subgraph's ONNX inputs and
  outputs are boundaries, named by the shell's convention ``s_axis_<i>`` and
  ``m_axis_<i>``: a channel refuses a boundary no port names
  (``channel-boundary``);
- **kernels**, one per node, from its facts, the graph's pins as keywords;
- **replay**: each node's kernel choices, then the edge choices (an edge's
  adapter selector is forced, never persisted); an edge choice the current
  graph refuses is stale, dropped and reported, and the forced case applies
  again;
- **owners**: each member's node and attribute prefix, how a choice made in the
  root goes back to the node that persists it: a kernel's on its node, an
  edge's on its consumer, a parameter channel's on its value owner. An output
  boundary consumed by no KernelOp (a graph output) is its producer's, under
  its output port; one a KernelOp outside the partition consumes is that
  node's, applied in its own partition as an input boundary, so here its
  ``transport`` is pinned ``direct``: one FIFO per edge, on the consumer's side.
  A choice a node holds under an output port whose channel a KernelOp now
  consumes is stale, dropped and reported.

Members are named as the graph: channels by tensor and kernels by node
(``\\W`` as ``_``); two members of one name (a node and a tensor, or two nodes) are
refused, not renamed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from onnx import NodeProto

from finn.core.space import Space, composite, design_space
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    committed,
    datatype,
    read_target,
    refusal,
    rows,
    shape,
    typed_choices,
)
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

S = TypeVar("S", bound=Space)

KERNEL_OPS = "finn.custom_op.kernels"


class Partition(Kernel):
    """A partition's hardware: one channel per ONNX tensor, one kernel per node."""

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


def _boundary(model: ModelWrapper, nodes: list[NodeProto], owned: set[str]) -> dict[str, str]:
    """The port of each boundary tensor, inputs then outputs, in node order: an input
    no node produces (and not a parameter: owned, or an initializer the op reads as a
    fact) is ``s_axis_<i>``; an output read outside ``nodes`` is ``m_axis_<j>``."""
    produced = {tensor for node in nodes for tensor in node.output}
    inside = {id(node) for node in nodes}
    used_outside = {
        tensor for node in model.graph.node if id(node) not in inside for tensor in node.input
    } | {output.name for output in model.graph.output}
    inputs = [
        tensor
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced and tensor not in owned and model.get_initializer(tensor) is None
    ]
    outputs = [tensor for node in nodes for tensor in node.output if tensor in used_outside]
    ports = {tensor: f"s_axis_{index}" for index, tensor in enumerate(inputs)}
    return ports | {tensor: f"m_axis_{index}" for index, tensor in enumerate(outputs)}


def _handed_on(model: ModelWrapper, nodes: list[NodeProto]) -> set[str]:
    """The outputs of ``nodes`` a KernelOp outside them consumes: their transport is that
    consumer's, chosen in its own partition."""
    inside = {id(node) for node in nodes}
    consumed = {
        tensor
        for node in model.graph.node
        if node.domain == KERNEL_OPS and id(node) not in inside
        for tensor in node.input
    }
    return {tensor for node in nodes for tensor in node.output if tensor in consumed}


def _channels(
    model: ModelWrapper,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    owned: list[dict[str, Channel]],
    ports: Mapping[str, str],
    handed_on: set[str],
) -> dict[str, Channel]:
    """The partition's channels by tensor, in node order: a node's inputs on an edge or
    the boundary, the parameter channels it owns, its outputs. An output handed on to a
    KernelOp outside is pinned ``direct``: its FIFO, if any, is the consumer's."""
    platform = read_target(model).platform
    parameters = {tensor for channels in owned for tensor in channels}
    channels: dict[str, Channel] = {}

    def declare(tensor: str, label: str) -> None:
        if tensor in channels:
            return
        dims = rows(shape(model, tensor, label))
        carried = Tensor(dims, ScalarEncoding(datatype(model, tensor, label)))
        port: dict[str, Any] = {"port": ports[tensor]} if tensor in ports else {}
        if tensor in handed_on:
            port["transport"] = "direct"
        channels[tensor] = Channel(tensor=carried, platform=platform, **port)

    for node, op, parameter_channels in zip(nodes, ops, owned):
        for tensor in node.input:
            if tensor not in parameters and model.get_initializer(tensor) is None:
                declare(tensor, op.label)
        channels |= parameter_channels
        for tensor in node.output:
            declare(tensor, op.label)
    return channels


@dataclass
class _Placed:
    """The kernels placed on the channels, by member; each member's owner (node,
    attribute prefix); the nodes' choices split by what declares them, and those stale
    before replay (an output's, now an edge another KernelOp owns)."""

    kernels: dict[str, Kernel]
    owners: dict[str, tuple[str, str]]
    kernel_choices: dict[str, object]
    edge_choices: dict[str, object]
    stale: list[str]


def _place(
    nodes: list[NodeProto],
    ops: list[KernelOp],
    channels: Mapping[str, Channel],
    produced: set[str],
) -> _Placed:
    """Each node's kernel on ``channels``, and its choices as root keys: a kernel's under
    the kernel's member, an input or owned channel's under the channel's, and an output's
    under the channel's where its node is the producer that owns it (``produced``: graph
    outputs no KernelOp consumes)."""
    placed = _Placed({}, {}, {}, {}, [])
    channel_members = {member(tensor) for tensor in channels}
    for node, op in zip(nodes, ops):
        kernel = member(node.name)
        if kernel in channel_members or kernel in placed.kernels:
            other = "a tensor" if kernel in channel_members else "another node"
            raise KernelOpError(f"{node.name}: a node and {other} are both named {kernel}")
        placed.kernels[kernel], by_port = op.place(channels)
        by_port |= {
            port: tensor for port, tensor in zip(op.outputs, node.output) if tensor in produced
        }
        placed.owners[kernel] = (node.name, "")
        for port, tensor in by_port.items():
            placed.owners[member(tensor)] = (node.name, f"{port}.")
        for attribute, value in op.choices().items():
            head, _, rest = attribute.partition(".")
            if head in by_port:
                placed.edge_choices[f"{member(by_port[head])}.{rest}"] = value
            elif head in op.outputs:
                tensor = node.output[op.outputs.index(head)]
                placed.stale.append(f"{member(tensor)}.{rest}")
            else:
                placed.kernel_choices[f"{kernel}.{attribute}"] = value
    return placed


def _replay_edges(point: S, choices: Mapping[str, object]) -> tuple[S, tuple[str, ...]]:
    """``choices`` replayed on ``point``: together when the graph accepts them all,
    otherwise one by one, each refused one dropped as stale (its forced case applies)."""
    edges = typed_choices([point], choices)
    together = committed(point, edges)
    if not isinstance(together, dict):
        return together, ()
    dropped: list[str] = []
    for key, value in edges.items():
        alone = committed(point, {key: value})
        if isinstance(alone, dict):
            dropped.append(key)
        else:
            point = alone
    return point, tuple(dropped)


def partition_root(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = "partition"
) -> PartitionRoot:
    """The root of ``nodes``, KernelOp nodes of ``model``; see the module docstring."""
    nodes = list(nodes)
    ops = [model.get_customop_wrapper(node) for node in nodes]
    for node, op in zip(nodes, ops):
        if not isinstance(op, KernelOp):
            raise KernelOpError(f"{node.name}: a partition root places KernelOps only")
    owned = [op.owned_channels() for op in ops]
    ports = _boundary(model, nodes, {tensor for channels in owned for tensor in channels})
    handed_on = _handed_on(model, nodes)
    channels = _channels(model, nodes, ops, owned, ports, handed_on)
    outputs = {tensor for node in nodes for tensor in node.output if tensor in ports}
    placed = _place(nodes, ops, channels, outputs - handed_on)

    members = {member(tensor): channel for tensor, channel in channels.items()} | placed.kernels
    root: Any = composite(name, members, base=Partition)
    point = design_space(root())
    if placed.kernel_choices:
        replayed = committed(point, typed_choices([point], placed.kernel_choices))
        if isinstance(replayed, dict):
            raise refusal(name, replayed)
        point = replayed
    dropped: tuple[str, ...] = ()
    if placed.edge_choices:
        point, dropped = _replay_edges(point, placed.edge_choices)
    return PartitionRoot(point, placed.owners, (*placed.stale, *dropped), tuple(ports.items()))


def save_partition_choices(
    model: ModelWrapper, root: PartitionRoot, choices: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    """Persist choices made on purpose in a partition root, each on its owning node."""
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
