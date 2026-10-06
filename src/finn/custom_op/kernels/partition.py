# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition root: one Kernel of a set of KernelOp nodes, its channels the graph's tensors.

``partition_root(model, nodes)`` builds, and changes no graph:

- **channels**, one per ONNX tensor, in node order, each declared from the
  graph before any kernel is placed: a node's graph inputs, the parameter
  channels it owns (named after the initializer), its outputs. A channel's
  tensor is the graph's value_info and annotation; a parameter channel's is
  the initializer's over its values' range, and its contents the node's value
  of it (``Facts.values``), which its source stores; every channel's platform
  is the model's target's. Only the subgraph's ONNX inputs and outputs are
  boundaries, named by the shell's convention ``s_axis_<i>`` and
  ``m_axis_<i>``: a channel refuses a boundary no port names
  (``channel-boundary``);
- **kernels**, one per node: its op's placement (``KernelOp.place``, the one its
  node root is generated from) with literal formals, on these channels;
- **replay**: every node's choices, kernel and edge alike, together; a choice
  the current facts refuse or make inapplicable (a key nested under a selector
  a fact change un-forced: ``compute.packed.pe`` once ``compute`` is open) is
  stale: dropped and reported with why (``dropped``), and what it chose is open
  again, or forced (an edge's adapter selectors are forced, never persisted).
  Nothing is written: ``persist`` writes a configured point's choices back,
  each node's whole, which clears the dropped ones;
- **owners**: each member's node and attribute prefix, how a choice made in the
  root goes back to the node that persists it: a kernel's on its node, an
  edge's on its consumer, a parameter channel's on its value owner. An output
  boundary consumed by no KernelOp (a graph output) is its producer's, under
  its output port; one a KernelOp outside the partition consumes is that
  node's, applied in its own partition as an input boundary, so here its
  ``transport`` is pinned ``direct``: one FIFO per edge, on the consumer's side.
  A choice a node holds under an output port whose channel a KernelOp now
  consumes is stale, dropped and reported;
- **reuse**: the partition's class, and so its compiled model, is kept by what it
  is built from, by value (``PartitionKey``): the name; the target's platform;
  each channel as declared (the tensor it carries, rows and annotation, its port,
  pinned ``direct`` or not); each node's name, op class, node-root class, facts
  key (its formals and owned values by value, as the bind cache keys them), owned
  parameter ports and tensors. A call on the same facts reuses the class, never a
  point, and reads an owned value only to build the class: the
  choices are replayed on a fresh design space every call. ``PARTITIONS`` keeps the
  16 most recently used.

Members are named as the graph: channels by tensor and kernels by node
(``\\W`` as ``_``); two members of one name (a node and a tensor, or two nodes) are
refused, not renamed.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from onnx import NodeProto

from finn.core.space import Space, composite, design_space
from finn.custom_op.kernels.base import (
    KernelOp,
    KernelOpError,
    committed,
    edge_tensor,
    kernel_op,
    read_target,
    typed_choices,
)
from finn.custom_op.kernels.cache import Facts, LeastRecentlyUsed
from finn.dataflow.tensor import Tensor
from finn.kernels.base import Kernel
from finn.kernels.channels import Channel
from finn.kernels.configure import chosen
from finn.kernels.target import Platform
from finn.kernels.values.semantics import IntegerTensorValue

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
    """The configured root; each member's owning node and attribute prefix; the choices
    replay dropped as stale, each with why; each boundary tensor's port; its members
    (channels, then kernels), whose cost a design space exploration reads."""

    point: Any
    owners: Mapping[str, tuple[str, str]]
    dropped: Mapping[str, str]
    boundary: tuple[tuple[str, str], ...]
    members: tuple[str, ...]


@dataclass(frozen=True)
class Declared:
    """A channel as the partition declares it: the tensor it carries, rows and
    annotation (a parameter channel's over its values' range), its boundary port, and
    whether it is pinned ``direct`` (an output handed on)."""

    tensor: Tensor
    port: str | None = None
    direct: bool = False

    def channel(self, platform: Platform, contents: IntegerTensorValue | None = None) -> Channel:
        """The channel on the target's ``platform``, carrying ``contents`` (an owned
        parameter's value) when given."""
        settings: dict[str, Any] = {"tensor": self.tensor}
        if contents is not None:
            settings["contents"] = contents
        if self.port is not None:
            settings["port"] = self.port
        if self.direct:
            settings["transport"] = "direct"
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
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
    handed_on: set[str],
) -> dict[str, Declared]:
    """The partition's channels by tensor, in node order: a node's inputs on an edge or
    the boundary, the parameter channels it owns (the initializer's tensor; its value,
    the contents, is bound when the class is built), its outputs. An output handed on to
    a KernelOp outside is pinned ``direct``: its FIFO, if any, is the consumer's."""
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    channels: dict[str, Declared] = {}

    def declare(tensor: str, label: str) -> None:
        if tensor not in channels:
            carried = edge_tensor(model, tensor, label)
            channels[tensor] = Declared(carried, ports.get(tensor), tensor in handed_on)

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
    (``produced``: graph outputs no KernelOp consumes)."""
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
    """The partition's class: its channels as ``declared``, an owned parameter's with its
    node's value as contents, and each node's kernel placed on them from its ``facts``."""
    contents: dict[str, IntegerTensorValue] = {}
    for each, tensors in zip(facts, owned):
        if tensors:
            values = each.values()
            contents |= {tensor: values[port] for port, tensor in tensors.items()}
    channels = {
        tensor: each.channel(platform, contents.get(tensor)) for tensor, each in declared.items()
    }
    kernels = {
        member(node.name): op.place(each, channels) for node, op, each in zip(nodes, ops, facts)
    }
    members = {member(tensor): channel for tensor, channel in channels.items()} | kernels
    return composite(name, members, base=Partition)


def _replay(point: S, choices: Mapping[str, object]) -> tuple[S, dict[str, str]]:
    """``choices`` replayed on ``point``: together when the facts accept them all;
    otherwise every refused or inapplicable one is dropped as stale, with why, and the
    rest replayed again, until they are accepted (what a dropped one chose is open
    again, or forced). A refusal of the batch that names none of its keys is resolved
    one choice at a time."""
    remaining = typed_choices([point], choices)
    dropped: dict[str, str] = {}
    while remaining:
        together = committed(point, remaining)
        if not isinstance(together, dict):
            return together, dropped
        named = {key: why for key, why in together.items() if key in remaining}
        if not named:
            for key, value in remaining.items():
                alone = committed(point, {key: value})
                if isinstance(alone, dict):
                    dropped[key] = "; ".join(alone.values())
                else:
                    point = alone
            return point, dropped
        dropped |= named
        remaining = {key: value for key, value in remaining.items() if key not in named}
    return point, dropped


def partition_root(
    model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = "partition"
) -> PartitionRoot:
    """The root of ``nodes``, KernelOp nodes of ``model``; see the module docstring."""
    nodes = list(nodes)
    ops = [kernel_op(model, node) for node in nodes]
    facts = [op.facts() for op in ops]
    owned = [op.owned(each) for op, each in zip(ops, facts)]
    ports = _boundary(model, nodes, {tensor for tensors in owned for tensor in tensors.values()})
    handed_on = _handed_on(model, nodes)
    declared = _channels(model, nodes, ops, owned, ports, handed_on)
    outputs = {tensor for node in nodes for tensor in node.output if tensor in ports}
    found = _owners(nodes, ops, declared, outputs - handed_on)

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
    root: Any = PARTITIONS.get(
        key, lambda: _composite(name, platform, nodes, ops, facts, declared, owned)
    )
    point, dropped = _replay(design_space(root()), found.kernel_choices | found.edge_choices)
    stale = dict.fromkeys(found.stale, "held under an output port a KernelOp now consumes")
    # The members as ``_composite`` declares them: channels, then kernels.
    members = (*(member(tensor) for tensor in declared), *(member(node.name) for node in nodes))
    return PartitionRoot(point, found.owners, stale | dropped, tuple(ports.items()), members)


def persist(model: ModelWrapper, root: PartitionRoot, point: Any) -> dict[str, dict[str, object]]:
    """Write ``point``'s choices, a configured point of ``root``'s class, back on the nodes
    that own them, each node's whole: every Decision ``point`` commits (a forced one is
    never committed) on its owner, and a choice a node holds that ``point`` does not
    commit (one replay dropped as stale) cleared. Returns what each node now holds."""
    per_node: dict[str, dict[str, object]] = {node: {} for node, _ in root.owners.values()}
    for key, value in chosen(point).items():
        head, _, rest = key.partition(".")
        if head not in root.owners:
            raise KernelOpError(f"{key}: no node of the partition owns {head}")
        node, prefix = root.owners[head]
        per_node[node][prefix + rest] = value
    by_name = {node.name: node for node in model.graph.node}
    for node, values in per_node.items():
        op = kernel_op(model, by_name[node])
        held = op.choices()
        if values or held:
            op.save(dict.fromkeys(held) | values)
    return per_node


__all__ = [
    "PARTITIONS",
    "Declared",
    "Partition",
    "PartitionKey",
    "PartitionRoot",
    "Placement",
    "member",
    "partition_root",
    "persist",
]
