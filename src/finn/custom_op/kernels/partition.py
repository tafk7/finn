# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Partition: one Kernel of a set of KernelOp nodes, its channels the graph's tensors.

``partition(model, nodes)`` builds a set of nodes' Partition, and changes no
graph; the shell root (``finn.custom_op.kernels.shell``) places it and replays
its nodes' choices:

- **channels**, one per ONNX tensor, in node order, each declared from the
  graph before any kernel is placed: a node's graph inputs, the parameter
  channels it owns (named after the initializer), its outputs. A channel's
  tensor is the graph's value_info and annotation; a parameter channel's is
  the initializer's over its values' range, and its contents the node's value
  of it (``Facts.values``), which its source stores; every channel's platform
  is the model's target's. Only the subgraph's ONNX inputs and outputs are
  boundaries, named by the shell's convention ``s_axis_<i>`` and
  ``m_axis_<j>``. A **boundary channel is not the Partition's**: it is a
  reference input of the class (``Param()``, named as the channel's member),
  which the shell root declares and supplies, so that what crosses the boundary
  is the shell's to place; the channels between its nodes are its own;
- **kernels**, one per node: its op's placement (``KernelOp.place``, the one its
  node root is generated from) with literal formals, on these channels;
- **owners**: each member's node and attribute prefix, how a choice made in the
  root goes back to the node that persists it: a kernel's on its node, an
  edge's on its consumer, a parameter channel's on its value owner. An output
  boundary (a graph output) is its producer's, under its output port. A choice a
  node holds under an output port whose channel a KernelOp now consumes is stale
  (``stale``);
- the nodes' **choices**, by member key, for the root to replay;
- **reuse**: the class, and so the compiled model of a root that places it, is kept
  by what it is built from, by value (``PartitionKey``): the name; the target's
  platform; each channel as declared (the tensor it carries, rows and annotation,
  its port); each node's name, op class, node-root class,
  facts key (its formals and owned values by value, as the bind cache keys them),
  owned parameter ports and tensors. A call on the same facts reuses the class, and
  reads an owned value only to build it. ``PARTITIONS`` keeps the 16 most recently
  used.

Members are named as the graph: channels by tensor and kernels by node
(``\\W`` as ``_``); two members of one name (a node and a tensor, or two nodes) are
refused, not renamed. Owners and choices are keyed by member, as the Partition
names them; a root that places it names them by its own paths.

``nodes`` are every KernelOp of ``model``: the cut decides which KernelOps go
together (``CutKernelPartition``), so a KernelOp of the model outside them, a second
partition of KernelOps, is refused by name. Several partitions are to be designed as
one shell root (when CNV or Alveo needs them), not as neighbouring roots.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from onnx import NodeProto

from finn.core.space import Param, composite
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    edge_tensor,
    kernel_op,
    read_target,
)
from finn.custom_op.kernels.cache import Facts, LeastRecentlyUsed
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.ends import EndOffer
from finn.kernels.target import Platform
from finn.kernels.values.semantics import IntegerTensorValue

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

KERNEL_OPS = "finn.custom_op.kernels"


class Partition(Kernel):
    """A partition's hardware: one kernel per node, one channel per ONNX tensor between
    them; a boundary channel is a reference input, the root's."""

    id = "finn.custom_op.kernels.partition"
    version = 1


@dataclass(frozen=True)
class Declared:
    """A channel as the partition declares it: the tensor it carries, rows and
    annotation (a parameter channel's over its values' range), and its boundary port."""

    tensor: Tensor
    port: str | None = None

    def channel(
        self,
        platform: Platform,
        contents: IntegerTensorValue | None = None,
        end_offer: tuple[EndOffer, ...] = (),
    ) -> Channel:
        """The channel on the target's ``platform``, carrying ``contents`` (an owned
        parameter's value) when given, its free side offered the ends ``end_offer``
        when any."""
        settings: dict[str, Any] = {"tensor": self.tensor}
        if contents is not None:
            settings["contents"] = contents
        if end_offer:
            settings["end_offer"] = end_offer
        if self.port is not None:
            settings["port"] = self.port
        return Channel(platform=platform, **settings)


@dataclass(frozen=True)
class Placement:
    """What placing a node's kernel reads: the node's name (its member), its op class,
    node-root class and facts key (its formals and owned values by value), the parameter
    ports it owns, and its tensors."""

    node: str
    op: type[KernelOp]
    root: type[Kernel]
    facts: tuple[Hashable, ...]
    owned: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class PartitionKey:
    """What a partition's class is built from, by value: its name, the target's platform,
    its channels as declared, in order, and its nodes' placements, in order."""

    name: str
    platform: Platform
    channels: tuple[tuple[str, Declared], ...]
    kernels: tuple[Placement, ...]


PARTITIONS: LeastRecentlyUsed[type[Partition]] = LeastRecentlyUsed(16)
"""The process's partition classes by ``PartitionKey``: the 16 most recently used."""


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


def _second_partition(model: ModelWrapper, nodes: list[NodeProto]) -> list[str]:
    """The KernelOps of ``model`` outside ``nodes``: a second partition of KernelOps."""
    inside = {id(node) for node in nodes}
    return [
        node.name
        for node in model.graph.node
        if node.domain == KERNEL_OPS and id(node) not in inside
    ]


def _channels(
    model: ModelWrapper,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
) -> dict[str, Declared]:
    """The partition's channels by tensor, in node order: a node's inputs on an edge or
    the boundary, the parameter channels it owns (the initializer's tensor; its value,
    the contents, is bound when the class is built), its outputs."""
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    channels: dict[str, Declared] = {}

    def declare(tensor: str, label: str) -> None:
        if tensor not in channels:
            carried = edge_tensor(model, tensor, label)
            channels[tensor] = Declared(carried, ports.get(tensor))

    for node, op, tensors in zip(nodes, ops, owned):
        for tensor in node.input:
            if tensor not in parameters and model.get_initializer(tensor) is None:
                declare(tensor, op.label)
        for tensor in tensors.values():
            channels[tensor] = Declared(edge_tensor(model, tensor, op.label))
        for tensor in node.output:
            declare(tensor, op.label)
    return channels


@dataclass
class _Owners:
    """Each member's owner (node, attribute prefix); the nodes' choices split by what
    declares them, and those stale before replay (an output's, now an edge another
    KernelOp owns)."""

    owners: dict[str, tuple[str, str]]
    kernel_choices: dict[str, object]
    edge_choices: dict[str, object]
    stale: list[str]


def _owners(
    nodes: list[NodeProto], ops: list[KernelOp], channels: Iterable[str], produced: set[str]
) -> _Owners:
    """Each node's kernel member and channels' owners, and its choices as root keys: a
    kernel's under the kernel's member, an input or owned channel's under the channel's,
    and an output's under the channel's where its node is the producer that owns it
    (``produced``: the output boundaries)."""
    found = _Owners({}, {}, {}, [])
    channel_members = {member(tensor) for tensor in channels}
    kernels: set[str] = set()
    for node, op in zip(nodes, ops):
        kernel = member(node.name)
        if kernel in channel_members or kernel in kernels:
            other = "a tensor" if kernel in channel_members else "another node"
            raise KernelOpError(f"{node.name}: a node and {other} are both named {kernel}")
        kernels.add(kernel)
        by_port = op.inputs() | {
            port: tensor for port, tensor in zip(op.outputs, node.output) if tensor in produced
        }
        found.owners[kernel] = (node.name, "")
        for port, tensor in by_port.items():
            found.owners[member(tensor)] = (node.name, f"{port}.")
        for attribute, value in op.choices().items():
            head, _, rest = attribute.partition(".")
            if head in by_port:
                found.edge_choices[f"{member(by_port[head])}.{rest}"] = value
            elif head in op.outputs:
                tensor = node.output[op.outputs.index(head)]
                found.stale.append(f"{member(tensor)}.{rest}")
            else:
                found.kernel_choices[f"{kernel}.{attribute}"] = value
    return found


def _composite(
    name: str,
    platform: Platform,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    facts: list[Facts],
    declared: Mapping[str, Declared],
    owned: list[dict[str, str]],
) -> type[Partition]:
    """The partition's class: each boundary channel a reference input named as its member,
    the others its own as ``declared`` (an owned parameter's with its node's value as
    contents), and each node's kernel placed on them from its ``facts``."""
    contents: dict[str, IntegerTensorValue] = {}
    for each, tensors in zip(facts, owned):
        if tensors:
            values = each.values()
            contents |= {tensor: values[port] for port, tensor in tensors.items()}
    # A reference input stands for the channel the root supplies: kernels bind it as
    # they bind a channel of their own.
    channels: dict[str, Channel] = {
        tensor: Param() if each.port is not None else each.channel(platform, contents.get(tensor))
        for tensor, each in declared.items()
    }
    kernels = {
        member(node.name): op.place(each, channels) for node, op, each in zip(nodes, ops, facts)
    }
    members = {member(tensor): channel for tensor, channel in channels.items()} | kernels
    references = {member(tensor): Channel for tensor, each in declared.items() if each.port}
    return composite(name, members, base=Partition, annotations=references)


@dataclass(frozen=True)
class Partitioned:
    """A set of nodes' Partition (``partition``): its class (``space``); its channels as
    declared, in node order; its boundary, each tensor's port, inputs then outputs; its
    kernels' members, in node order; each member's owner (node, attribute prefix); the
    nodes' choices by member key, for a root to replay; and the keys stale before replay,
    each with why. Members are the Partition's: a boundary channel's is the name of its
    reference input."""

    space: type[Partition]
    platform: Platform
    channels: tuple[tuple[str, Declared], ...]
    boundary: tuple[tuple[str, str], ...]
    kernels: tuple[str, ...]
    owners: Mapping[str, tuple[str, str]]
    choices: Mapping[str, object]
    stale: Mapping[str, str]


def partition(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = "partition"
) -> Partitioned:
    """The Partition of ``nodes``, KernelOp nodes of ``model``; see the module docstring."""
    nodes = list(nodes)
    outside = _second_partition(model, nodes)
    if outside:
        raise KernelOpError(
            f"{', '.join(outside)}: KernelOps outside the partition {name!r}, a second "
            "partition of KernelOps, refused until several partitions are designed as one "
            "shell root"
        )
    ops = [kernel_op(model, node) for node in nodes]
    facts = [op.facts() for op in ops]
    owned = [op.owned(each) for op, each in zip(ops, facts)]
    ports = _boundary(model, nodes, {tensor for tensors in owned for tensor in tensors.values()})
    declared = _channels(model, nodes, ops, owned, ports)
    outputs = {tensor for node in nodes for tensor in node.output if tensor in ports}
    found = _owners(nodes, ops, declared, outputs)

    platform = read_target(model).platform
    key = PartitionKey(
        name,
        platform,
        tuple(declared.items()),
        tuple(
            Placement(
                node.name,
                type(op),
                each.root,
                each.key,
                each.owned,
                tuple(node.input),
                tuple(node.output),
            )
            for node, op, each in zip(nodes, ops, facts)
        ),
    )
    space = PARTITIONS.get(
        key, lambda: _composite(name, platform, nodes, ops, facts, declared, owned)
    )
    return Partitioned(
        space,
        platform,
        tuple(declared.items()),
        tuple(ports.items()),
        tuple(member(node.name) for node in nodes),
        found.owners,
        found.kernel_choices | found.edge_choices,
        dict.fromkeys(found.stale, "held under an output port a KernelOp now consumes"),
    )


__all__ = [
    "PARTITIONS",
    "Partitioned",
    "Declared",
    "Partition",
    "PartitionKey",
    "Placement",
    "member",
    "partition",
]
