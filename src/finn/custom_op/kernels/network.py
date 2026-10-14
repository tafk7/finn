# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network of KernelOp nodes a graph binds, and its choices written back.

``network(model, nodes)`` binds a ``finn.kernels.network.Network`` straight from the
graph, for any selection of KernelOp nodes (a partition's body, a slice of it, one node),
and changes no graph. Its shell is the model's target's (``read_target``): the
target's board's row of that shell (``finn.platform.shells.shell_row``) states the ends
it offers and its budgets.

- **kernels**, one per node, named as the node: its kernel bound from its facts by the
  bind cache, one binding per node (``KernelOp.bound``);
- **channels**, one per ONNX tensor, named as the tensor, in node order: a node's graph
  inputs, the parameter channels it owns (named after the initializer), its outputs.
  Each is a ``Wire`` of the facts a ``Channel`` takes: its tensor (the graph's,
  ``edge_tensor``, at the graph's rank; a parameter channel's, the tensor and contents
  its kernel states for it, ``Fact(node, wiring.carried)``, ``Fact(node,
  wiring.stored)``), its ends (``Interface(node, wiring.end)``, from the op's
  ``wiring``), and the target's platform. Only the selection's ONNX inputs and outputs
  are boundaries, named by the shell's convention ``s_axis_<i>`` and ``m_axis_<j>``, each
  offered the ends the row offers (``ShellRow.ends``; ``Channel.end``: one offered is
  forced, so nothing is persisted). A channel has one consumer, so a tensor read more
  than once is refused by name before anything is bound (``tensor-fan-out``): read by
  two of the nodes (an edge, or a parameter both own), or read by one and at an output
  boundary too (a graph output, or read outside the nodes). Fan-out waits for a channel
  that forks;
- **its order**: members are declared input boundary channels, the other channels, the
  kernels, output boundary channels (``Reshape_0_out0``, ..., ``MatMul_0``, ...), the
  module's instance order; each member's node in the module is its name with ``\\W`` as
  ``_``. A node and a tensor, or two nodes, of one name are refused: a member is named
  once, as ``kernel_choices.json`` names it;
- **its stored choices**, replayed as each member is bound (``Network``'s ``stored``):
  every node's attributes and every channel's ``finn.channel`` on its tensor (in a
  partition's body only, ``refuse_outside_body``). A stored choice the current facts
  refuse is dropped as stale, with why (``Partition.stale``); a channel whose ends wait
  holds its choices until it is bound. With every KernelOp of the model selected, choices
  a tensor states that is no channel of it are stale too. Nothing is written: ``persist``
  writes a network's choices back, each owner's whole;
- **its outputs**: each node's output is its kernel's product. Where the graph's
  annotation of an output (its shape's element count and last axis, its datatype and
  its range) differs from what its producer states, the graph is stale, refused by name
  (``tensor-stale``: "run InferKernelTensors"), before any channel's own
  ``channel-tensor`` is read;
- **its module** is the partition's IP, the cut's (``PARTITION``): its members'
  netlists, named as the graph's nodes and tensors, with the boundary's pins, named
  after the partition (its stem ``finn_<name>``) and identified as the partition's IP
  (``PRODUCER``), since an end is integrated beside the IP, not in it.

``baseline_choices`` gives every choice a body leaves open its baseline and persists it
(the cut's, and ``BaselineChoices``' for a body made by hand); ``save_channels`` states
channels' choices on their tensors, checked by binding the network, which alone holds
both of a channel's ends. The shell's admission and resources over a network are
``finn.custom_op.kernels.shell``'s.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from onnx import NodeProto
from qonnx.core.metadata import MetadataError

from finn.core.graph.annotations.channel import (
    CHANNEL,
    CHANNEL_KEYS,
    channel_choices,
    refuse_outside_body,
)
from finn.core.graph.annotations.platform import read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available, QueryResult, Space
from finn.custom_op.kernels.base import KernelOp, kernel_op, refusal, typed, why
from finn.custom_op.kernels.cache import Facts
from finn.kernels.artifacts.module import Composed, ProducerIdentity
from finn.kernels.channels import Channel
from finn.kernels.explore import Seam
from finn.kernels.network import Fact, Interface, Key, Network, NetworkError, Wire, dotted, wire
from finn.platform.capabilities import Resources
from finn.platform.shells import ShellRow, shell_row

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

PARTITION = "partition"
"""The partition's name: the cut's node and body file, and the IP made from it, which
is its network's module (``finn_partition``)."""

PRODUCER = ProducerIdentity("finn.custom_op.kernels.partition", "1")
"""What derives a partition's module: identified as the partition's, so the IP's name and
digest are the partition's."""


def member(name: str) -> str:
    """A graph name as its member's node in the module (``\\W`` as ``_``): its instances'
    prefix, and the name a shell's resources and reports give it."""
    return re.sub(r"\W", "_", name)


class Partition(Network):
    """A network bound from a graph's nodes, in its shell (see the module docstring): its
    ``name`` (its module's), its shell's ``row``, its ``boundary`` (each boundary tensor
    and its port, inputs then outputs), the boundary channels offered ends (``ends``), the
    stored choices dropped as stale where it was bound, by key, with why (``stale``), and
    whether it holds every KernelOp of its model (``whole``). Immutable, as a network: a
    change returns another, which keeps these."""

    __slots__ = ("_memo", "boundary", "ends", "name", "row", "stale", "whole")

    name: str
    row: ShellRow
    boundary: tuple[tuple[str, str], ...]
    ends: tuple[str, ...]
    stale: Mapping[str, str]
    whole: bool
    _memo: dict[str, object]

    def __init__(
        self,
        members: Mapping[str, Space | Wire],
        *,
        name: str,
        row: ShellRow,
        boundary: tuple[tuple[str, str], ...],
        stored: Mapping[Key, object],
        stray: Mapping[str, str],
        whole: bool,
    ) -> None:
        super().__init__(members, stored=stored, nodes={each: member(each) for each in members})
        self.name, self.row, self.boundary, self.whole = name, row, boundary, whole
        self.ends = tuple(tensor for tensor, _ in boundary) if row.ends else ()
        self.stale = {dotted(key): why(found) for key, found in self.dropped.items()} | dict(stray)
        self._memo = {}

    def _changed(self) -> None:
        self._memo = {}

    def _read(self, question: str, answer: Callable[[], Any]) -> Any:
        """``answer()``, read once of this network (it is immutable)."""
        if question not in self._memo:
            self._memo[question] = answer()
        return self._memo[question]

    def module_of(self) -> QueryResult[Composed]:
        """Its module, the partition's IP (``finn_<name>``), as far as its members
        decide it."""
        found: QueryResult[Composed] = self._read(
            "module", lambda: self.module(PRODUCER, f"finn_{self.name.lower()}")
        )
        return found

    def own_resources(self) -> QueryResult[Resources]:
        """What the shell uses: its members', its ends' and its static region's sum
        (``shell_resources``)."""
        # The shell imports the network, so its resources are read through it here.
        from finn.custom_op.kernels.shell import _resources, _together  # noqa: PLC0415

        found = _resources(self)
        return Available(found.value.total) if isinstance(found, Available) else _together([found])


# -- binding the network from the graph ----------------------------------------------------


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


def _fan_out(
    model: ModelWrapper,
    nodes: list[NodeProto],
    owned: list[dict[str, str]],
    ports: Mapping[str, str],
) -> list[Finding]:
    """Each tensor whose channel would have more than one consumer, in node order: read
    by two of ``nodes`` (an input that is not an initializer, or a parameter a node owns),
    or by one and at its output port too (``m_axis_<j>``)."""
    readers: dict[str, list[str]] = {}
    for node, tensors in zip(nodes, owned):
        for tensor in node.input:
            if tensor in tensors.values() or model.get_initializer(tensor) is None:
                readers.setdefault(tensor, []).append(node.name)
    found = []
    for tensor, names in readers.items():
        port = ports.get(tensor, "")
        if port.startswith("m_axis_"):
            message = f"{tensor} is read by {', '.join(names)} and leaves the partition at {port}"
        elif len(names) > 1:
            message = f"{tensor} is read by {', '.join(names)}"
        else:
            continue
        details = (("tensor", tensor),)
        found.append(Finding(FindingKind.REJECTION, "tensor-fan-out", "fan_out", message, details))
    return found


def _tensors(model: ModelWrapper, nodes: list[NodeProto], owned: list[dict[str, str]]) -> list[str]:
    """Each tensor a channel carries, in node order: a node's inputs on an edge or the
    boundary (an initializer it reads as a fact carries none), the parameters it owns,
    its outputs."""
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    found: dict[str, None] = {}
    for node, tensors in zip(nodes, owned):
        for tensor in node.input:
            if tensor not in parameters and model.get_initializer(tensor) is None:
                found.setdefault(tensor)
        for tensor in tensors.values():
            found.setdefault(tensor)
        for tensor in node.output:
            found.setdefault(tensor)
    return list(found)


def _named(nodes: list[NodeProto], tensors: Iterable[str]) -> None:
    """Refuse two members of one name: a node and a tensor, or two nodes."""
    kinds: dict[str, str] = dict.fromkeys(tensors, "tensor")
    for node in nodes:
        if node.name in kinds:
            other = "another node" if kinds[node.name] == "node" else "a tensor"
            raise KernelOpError(
                f"{node.name}: a node and {other} are both named {node.name}", (node.name,)
            )
        kinds[node.name] = "node"


def _stored(
    model: ModelWrapper,
    nodes: list[NodeProto],
    ops: list[KernelOp],
    tensors: list[str],
    stating: Mapping[str, Mapping[str, object]],
    whole: bool,
) -> tuple[dict[Key, object], dict[str, str]]:
    """The choices the network replays, by key: each node's attributes, each channel's as
    its tensor states them (or as ``stating`` gives a tensor's); and, where ``whole``, those
    of a tensor that states some and is no channel of it, by dotted key, each with why."""
    found: dict[Key, object] = {}
    for node, op in zip(nodes, ops):
        found |= {(node.name, key): value for key, value in op.stored().items()}
    stated = model.tensors_stating(CHANNEL)
    refuse_outside_body(model, stated)
    for tensor in tensors:
        if tensor in stating:
            values: Mapping[str, object] = stating[tensor]
        elif tensor in stated:
            values = channel_choices(model, tensor)
        else:
            continue
        found |= {(tensor, key): value for key, value in typed(Channel, values).items()}
    channels = set(tensors)
    stray = {
        dotted((tensor, key)): "stated on a tensor that is no channel of the partition"
        for tensor in stated
        if whole and tensor not in channels
        for key in channel_choices(model, tensor)
    }
    return found, stray


def _stale(
    model: ModelWrapper, network: Network, nodes: list[NodeProto], ops: list[KernelOp]
) -> None:
    """Refuse a graph whose annotation of an output differs from what its producer
    states (its element count, its last axis, or its element: the datatype and its
    range), by name: the graph is stale (``tensor-stale``)."""
    findings: list[Finding] = []
    for node, op in zip(nodes, ops):
        kernel = network.member(node.name)
        for port, tensor in zip(op.outputs, node.output):
            stated = kernel.query(getattr(type(kernel), op.wiring[port].tensor))
            if not isinstance(stated, Available):
                continue  # what it waits on, or why it refuses, is its channel's to name
            graph = edge_tensor(model, tensor, op.label)
            produced = stated.value
            if (
                graph.size == produced.size
                and graph.shape[-1] == produced.shape[-1]
                and graph.element == produced.element
            ):
                continue
            findings.append(
                Finding(
                    FindingKind.REJECTION,
                    "tensor-stale",
                    "stale_outputs",
                    f"the graph states {tensor} {list(graph.shape)} {graph.element}, "
                    f"its producer {node.name} {list(produced.shape)} {produced.element}: "
                    "run InferKernelTensors",
                    (("tensor", tensor), ("node", node.name)),
                )
            )
    if findings:
        raise KernelOpError("; ".join(map(str, findings)), findings=tuple(findings))


def network(model: ModelWrapper, nodes: Iterable[NodeProto], *, name: str = PARTITION) -> Partition:
    """The network of ``nodes``, KernelOp nodes of ``model``, its module the partition
    ``name``'s IP, in the shell of the model's target: its row's ends offered on the
    boundary channels; see the module docstring. A tensor read more than once is refused
    (``tensor-fan-out``), and so is a stale output (``tensor-stale``)."""
    return _network(model, list(nodes), name, {})


def _network(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    stating: Mapping[str, Mapping[str, object]],
) -> Partition:
    """``network``, each tensor of ``stating`` replayed with the channel choices it gives
    in place of those the tensor states."""
    ops = [kernel_op(model, node) for node in nodes]
    facts: list[Facts] = [op.facts() for op in ops]
    owned = [op.owned(each) for op, each in zip(ops, facts)]
    ports = _boundary(model, nodes, {tensor for tensors in owned for tensor in tensors.values()})
    fanned = _fan_out(model, nodes, owned, ports)
    if fanned:
        raise KernelOpError(
            f"{name}: a channel has one consumer, and fan-out is not designed yet (a channel "
            "that forks): " + "; ".join(map(str, fanned)),
            findings=fanned,
        )
    tensors = _tensors(model, nodes, owned)
    _named(nodes, tensors)
    inside = {id(node) for node in nodes}
    whole = all(id(node) in inside for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN)
    stored, stray = _stored(model, nodes, ops, tensors, stating, whole)
    target = read_target(model)
    row = shell_row(target.shell, target.board)
    settings: dict[str, dict[str, object]] = {tensor: {} for tensor in tensors}
    readers: dict[str, str] = {}
    for node, op, held in zip(nodes, ops, owned):
        sides = [("consumer", port, tensor) for port, tensor in op.inputs().items()]
        sides += [("producer", port, tensor) for port, tensor in zip(op.outputs, node.output)]
        for side, port, tensor in sides:
            wiring = op.wiring[port]
            settings[tensor][side] = Interface(node.name, wiring.end)
            readers.setdefault(tensor, op.label)
            if port in held:
                settings[tensor]["tensor"] = Fact(node.name, str(wiring.carried))
                settings[tensor]["contents"] = Fact(node.name, str(wiring.stored))
    wires: dict[str, Wire] = {}
    for tensor in tensors:
        facts_of = settings[tensor]
        if "tensor" not in facts_of:
            facts_of["tensor"] = edge_tensor(model, tensor, readers[tensor])
        boundary_port = ports.get(tensor)
        if boundary_port is not None:
            facts_of["port"] = boundary_port
            if row.ends:
                facts_of["end_offer"] = row.ends
        wires[tensor] = wire(platform=target.platform, **facts_of)

    def on(side: str) -> dict[str, Wire]:
        """The channels at the ports ``<side><i>`` (``""``: no port, none on the boundary)."""
        return {
            tensor: each
            for tensor, each in wires.items()
            if ports.get(tensor, "").rstrip("0123456789") == side
        }

    kernels: dict[str, Space] = {node.name: op.bound() for node, op in zip(nodes, ops)}
    members: dict[str, Space | Wire] = {**on("s_axis_"), **on(""), **kernels, **on("m_axis_")}
    try:
        bound = Partition(
            members,
            name=name,
            row=row,
            boundary=tuple(ports.items()),
            stored=stored,
            stray=stray,
            whole=whole,
        )
    except NetworkError as error:
        raise KernelOpError(f"{name}: {error}") from error
    _stale(model, bound, nodes, ops)
    return bound


# -- writing choices back -----------------------------------------------------------------


def _encoded(tensor: str, values: Mapping[str, object]) -> None:
    """Refuse a channel choice ``CHANNEL`` does not declare or admit, by name."""
    unknown = sorted(set(values) - set(CHANNEL_KEYS))
    if unknown:
        raise KernelOpError(
            f"{tensor}: {unknown} are not a channel's choices ({CHANNEL.name})", tuple(unknown)
        )
    for key, value in values.items():
        try:
            CHANNEL_KEYS[key].encode(value)
        except MetadataError as error:
            raise KernelOpError(f"{tensor}: {error}", (key,)) from error


def _write_channel(model: ModelWrapper, tensor: str, choices: Mapping[str, object]) -> None:
    """State ``choices`` on ``tensor``'s channel, its whole: what it stated before is
    cleared."""
    model.clear(CHANNEL, tensor=tensor)
    for key, value in sorted(choices.items()):
        model.set(CHANNEL_KEYS[key], value, tensor=tensor)


def save_channels(model: ModelWrapper, choices: Mapping[str, Mapping[str, object | None]]) -> None:
    """Commit ``choices`` (made on purpose), by tensor, over those each tensor's channel
    states (``CHANNEL``), in a partition's body, ``model``: replayed with every other
    choice of the network of its KernelOps, which alone holds both of a channel's ends,
    and written only if replay keeps each of them; ``None`` clears a choice. A key
    ``CHANNEL`` does not declare, a value it does not admit, a tensor that is no channel
    of the network, a channel whose ends wait on a kernel's open choices (its choices
    cannot be checked yet) and a model that is not a partition's body are refused, by
    name. Public API on purpose, though only tests call it: the checked way to state a
    channel's choices by hand."""
    refuse_outside_body(model, list(choices))
    merged: dict[str, dict[str, object]] = {}
    for tensor, given in choices.items():
        _encoded(tensor, {key: value for key, value in given.items() if value is not None})
        merged[tensor] = {
            key: value
            for key, value in {**channel_choices(model, tensor), **given}.items()
            if value is not None
        }
    bound = _network(model, list(model.graph.node), PARTITION, merged)
    for tensor in merged:
        if tensor not in bound.channels:
            raise KernelOpError(
                f"{tensor}: no channel of the partition (a stream between its kernels, or "
                "on its boundary, or a parameter a kernel owns)",
                (tensor,),
            )
        waiting = bound.unbound(tensor)
        if waiting is not None:
            raise KernelOpError(
                f"{tensor}: its channel's ends wait on {waiting.member}'s open choices ("
                + "; ".join(map(str, waiting.result.findings))
                + f"), so its choices cannot be checked: save {waiting.member}'s first",
                (tensor,),
            )
        refused = {
            key: why(found) for (owner, key), found in bound.dropped.items() if owner == tensor
        }
        if refused:
            raise refusal(tensor, refused)
    for tensor, values in merged.items():
        _write_channel(model, tensor, values)


def persist(model: ModelWrapper, network: Partition) -> dict[str, dict[str, object]]:
    """Write ``network``'s choices, a network of ``model``'s nodes (``network``), back
    where they are stated, each owner's whole: every Decision it commits (a forced one
    is never committed), and the choices an unbound channel holds, on its owner, a
    kernel's node or a channel's tensor; a choice an owner holds that the network does
    not (one replay dropped as stale) cleared, and, where it holds every KernelOp of the
    model, so are the channel choices of a tensor that is no channel of it. Every value is
    checked and encoded before any is written, so a write cannot fail halfway. Returns
    what each owner now holds, by graph name."""
    held_by: dict[str, dict[str, object]] = {name: {} for name in network.members}
    for (owner, key), value in network.chosen().items():
        held_by[owner][key] = value
    nodes = {node.name: node for node in model.graph.node}
    stated = model.tensors_stating(CHANNEL)
    writes: list[Callable[[], None]] = []
    for name, values in held_by.items():
        if name in network.kernels:
            op = kernel_op(model, nodes[name])
            held = op.choices()
            if values or held:
                writes.append(_kernel_write(op, op.saved(dict.fromkeys(held) | values)))
        elif values or name in stated:
            _encoded(name, values)
            writes.append(_channel_write(model, name, values))
    stray = [tensor for tensor in stated if network.whole and tensor not in held_by]
    for write in writes:
        write()
    for tensor in stray:
        model.clear(CHANNEL, tensor=tensor)
    return held_by


def _kernel_write(op: KernelOp, merged: Mapping[str, int | str]) -> Callable[[], None]:
    return lambda: op.write(merged)


def _channel_write(
    model: ModelWrapper, tensor: str, values: Mapping[str, object]
) -> Callable[[], None]:
    return lambda: _write_channel(model, tensor, values)


def baseline_choices(model: ModelWrapper) -> dict[str, str]:
    """Every choice of a partition's body, ``model``, at its baseline where the body
    leaves it open, and persisted, each owner's whole (``persist``): the network of its
    KernelOps bound, their stored choices replayed in context, and every open choice
    with cases that is not required given its first viable case the seam accepts, in
    rank order (``Seam.baseline``). The channels' choices are created here (a channel has
    both its ends only in the network), and the nodes' own are re-checked: one replay
    drops as stale is re-baselined. Returns the dropped choices, by key, with why, as
    replay reports them. A required choice (a FIFO's depth) stays open."""
    bound = network(model, model.graph.node)
    baselined = Seam(read_target(model).platform).baseline(bound)
    persist(model, baselined)
    return dict(bound.stale)


__all__ = [
    "PARTITION",
    "PRODUCER",
    "Partition",
    "baseline_choices",
    "member",
    "network",
    "persist",
    "save_channels",
]
