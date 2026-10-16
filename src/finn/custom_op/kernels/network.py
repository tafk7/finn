# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The network of KernelOp nodes a graph binds, and its choices written back.

``network(model, nodes, ends=...)`` binds a ``finn.kernels.network.Network`` straight
from the graph, for any selection of KernelOp nodes (a partition's body, a slice of it, one
node), and changes no graph: it reads the graph once, its tensors from one index
(``finn.core.graph.facts.reading``). The ends offered on its boundary (``ends``, each an
``finn.kernels.ends.EndOffer``; none by default) are a fact its caller gives: a shell's
row offers them, and ``finn.custom_op.kernels.shell`` binds a partition's network in the
shell of the model's target (``in_shell``).

- **kernels**, one per node, named as the node: its kernel's facts, read once
  (``KernelOp.facts``), configured through the build's member cache;
- **channels**, one per ONNX tensor, named as the tensor, in node order: a node's graph
  inputs, the parameter channels it owns (named after the initializer), its outputs.
  Each is a ``Wire`` of the facts a ``Channel`` takes: its tensor (the graph's,
  ``edge_tensor``, at the graph's rank; a parameter channel's, the tensor and contents
  its kernel states for it, ``Fact(node, wiring.carried)``, ``Fact(node,
  wiring.stored)``), its ends (``Interface(node, wiring.end)``, from the op's
  ``wiring``), and the target's platform. Only the selection's ONNX inputs and outputs
  are boundaries, named by the shell's convention ``s_axis_<i>`` and ``m_axis_<j>``, each
  offered the ends given (``ends``; ``Channel.end``: one offered is forced, so nothing is
  persisted). A channel has one consumer, so a tensor read more
  than once is refused by name before anything is bound (``tensor-fan-out``): read by
  two of the nodes (an edge, or a parameter both own), or read by one and at an output
  boundary too (a graph output, or read outside the nodes). Fan-out waits for a channel
  that forks;
- **its order**: members are declared input boundary channels, the other channels, the
  kernels, output boundary channels (``Reshape_0_out0``, ..., ``MatMul_0``, ...), the
  module's instance order; each member's node in the module is its name with ``\\W`` as
  ``_``. A node and a tensor, or two nodes, of one name are refused: a member is named
  once, as the choices log ``kernel_choices.json`` names it;
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

**Kept per build:** a network bound is kept in the build's member cache
(``finn.kernels.members``), keyed by the model's ``design_digest`` (all but
``finn.outputs``, which a bind does not read), the orders the digest makes canonical and a
bind reads (each node's attributes, the metadata's and the annotations' keys), the target
the model states (a body opened inside its parent inherits it), the end offers given, what
the network is made as (``Partition``, or a shell's ``InShell`` with its row), the nodes
selected (by position), the name and the channel choices given. A refusal is never kept.
In the verifying mode a hit is bound again with no cache open and compared.

A partition's resources (``Partition.own_resources``) are its members'. The shell's
admission and resources over a network (its members', its ends' and its static region's),
and the body's choices stated by hand (``save_channels``) or given their baselines
(``baseline_choices``), each on the network in its shell, are
``finn.custom_op.kernels.shell``'s.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

from onnx import NodeProto
from qonnx.core.metadata import SEPARATOR, MetadataError

from finn.core.graph.annotations.channel import (
    CHANNEL,
    CHANNEL_KEYS,
    channel_choices,
    refuse_outside_body,
)
from finn.core.graph.annotations.outputs import OUTPUTS
from finn.core.graph.annotations.platform import PLATFORM, read_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.facts import edge_tensor, initializer, reading
from finn.core.graph.partition import KERNEL_OPS_DOMAIN, design_digest
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available, QueryResult, Rejected, Space, Unresolved
from finn.custom_op.kernels.base import KernelOp, kernel_op, typed, why
from finn.kernels.artifacts.module import Composed, ProducerIdentity
from finn.kernels.base import RESOURCES
from finn.kernels.channels import Channel
from finn.kernels.ends import EndOffer
from finn.kernels.members import Configured, Facts, current, observed, uncached
from finn.kernels.members import typed as key_part
from finn.kernels.network import Fact, Interface, Key, Network, NetworkError, Wire, dotted, wire
from finn.kernels.utilization import total
from finn.platform.capabilities import Resources

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
    """A network bound from a graph's nodes (see the module docstring): its ``name`` (its
    module's), its ``boundary`` (each boundary tensor and its port, inputs then outputs),
    the boundary channels offered ends (``ends``), the stored choices dropped as stale
    where it was bound, by key, with why (``stale``), and whether it holds every KernelOp
    of its model (``whole``). Immutable, as a network: a change returns another, which
    keeps these."""

    __slots__ = ("_memo", "boundary", "ends", "name", "stale", "whole")

    name: str
    boundary: tuple[tuple[str, str], ...]
    ends: tuple[str, ...]
    stale: Mapping[str, str]
    whole: bool
    _memo: dict[str, object]

    def __init__(
        self,
        members: Mapping[str, Space | Facts | Wire],
        *,
        name: str,
        ends: tuple[EndOffer, ...],
        boundary: tuple[tuple[str, str], ...],
        stored: Mapping[Key, object],
        stray: Mapping[str, str],
        whole: bool,
    ) -> None:
        super().__init__(members, stored=stored, nodes={each: member(each) for each in members})
        self.name, self.boundary, self.whole = name, boundary, whole
        self.ends = tuple(tensor for tensor, _ in boundary) if ends else ()
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
        """What its members use, their own statements' sum; or what they wait on, or why
        one is refused. A shell adds its ends' and its static region's
        (``finn.custom_op.kernels.shell``)."""

        def read() -> QueryResult[Resources]:
            used = [self.query(name, RESOURCES) for name in self.members]
            missing = [each for each in used if not isinstance(each, Available)]
            if missing:
                return _together(missing)
            return Available(total(each.value for each in used if isinstance(each, Available)))

        found: QueryResult[Resources] = self._read("members' resources", read)
        return found


def _together(results: Iterable[QueryResult[Any]]) -> Rejected | Unresolved:
    """Results that are not available, as one: refused where one is, else waiting."""
    held = list(results)
    findings = tuple(
        dict.fromkeys(finding for each in held for finding in getattr(each, "findings", ()))
    )
    if any(isinstance(each, Rejected) for each in held):
        return Rejected(findings)
    return Unresolved(findings)


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
        if tensor not in produced and tensor not in owned and initializer(model, tensor) is None
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
            if tensor in tensors.values() or initializer(model, tensor) is None:
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
            if tensor not in parameters and initializer(model, tensor) is None:
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
            raise KernelOpError.refused(
                "name-shared",
                "unique_names",
                f"{node.name}: a node and {other} are both named {node.name}",
                (node.name,),
                node=node.name,
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


def network(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str = PARTITION,
    ends: tuple[EndOffer, ...] = (),
) -> Partition:
    """The network of ``nodes``, KernelOp nodes of ``model``, its module the partition
    ``name``'s IP, on the platform of the model's target: ``ends`` offered on the boundary
    channels (none by default; a shell's row offers them, ``in_shell``); see the module
    docstring. A tensor read more than once is refused (``tensor-fan-out``), and so is a
    stale output (``tensor-stale``)."""
    return _network(model, list(nodes), name, {}, ends, Partition)


P = TypeVar("P", bound=Partition)


def _network(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    stating: Mapping[str, Mapping[str, object]],
    ends: tuple[EndOffer, ...],
    kind: Callable[..., P],
) -> P:
    """``network``, made by ``kind`` (``Partition``, or a shell's network over it), each
    tensor of ``stating`` replayed with the channel choices it gives in place of those the
    tensor states: kept in the build's member cache (the module docstring)."""
    cache = current()
    found = None if cache is None else _key(model, nodes, name, stating, ends, kind)
    if cache is None or found is None:
        return _read(model, nodes, name, stating, ends, kind)

    def again() -> P:
        with uncached():
            return _read(model, nodes, name, stating, ends, kind)

    return cast(
        P,
        cache.get(
            found,
            lambda: _read(model, nodes, name, stating, ends, kind),
            again,
            _observed,
            "network",
        ),
    )


def _made_as(kind: Callable[..., Partition]) -> Hashable:
    """What a network is made as, for its key: its class, and what a ``partial`` gives it
    (a shell's row)."""
    if isinstance(kind, partial):
        return (kind.func, key_part(kind.args), key_part(kind.keywords))
    return kind


def _key(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    stating: Mapping[str, Mapping[str, object]],
    ends: tuple[EndOffer, ...],
    kind: Callable[..., Partition],
) -> Hashable | None:
    """The network's key in the build's cache (the module docstring), or None where a
    node is not the model's or its target is not stated as one (the bind names why)."""
    graph = model.graph
    positions = {id(node): index for index, node in enumerate(graph.node)}
    selected = tuple(positions.get(id(node)) for node in nodes)
    if None in selected:
        return None
    try:
        stated = model.namespace(PLATFORM)
    except MetadataError:
        return None
    outputs = OUTPUTS.name + SEPARATOR
    orders = (
        tuple(tuple(attribute.name for attribute in node.attribute) for node in graph.node),
        tuple(prop.key for prop in graph.metadata_props if not prop.key.startswith(outputs)),
        tuple(prop.key for prop in model.model.metadata_props),
        tuple(
            (item.tensor_name, tuple(prop.key for prop in item.quant_parameter_tensor_names))
            for item in graph.quantization_annotation
        ),
    )
    found = (
        "network",
        design_digest(model),
        orders,
        key_part(stated),
        key_part(ends),
        _made_as(kind),
        selected,
        name,
        key_part({tensor: dict(values) for tensor, values in stating.items()}),
    )
    try:
        hash(found)
    except TypeError:
        return None
    return found


def _observed(bound: Partition) -> object:
    """What is read of a network bound, to compare in the verifying mode: its class, its
    ends and shell (a shell's row, its resources), its members, their order, wiring and
    nodes, each member's point as the member cache compares one, what replay dropped and
    why each unbound channel waits."""
    points = {
        each: observed(Configured(bound.member(each)))
        for each in bound.members
        if bound.unbound(each) is None
    }
    return (
        type(bound),
        (bound.name, bound.boundary, bound.ends, dict(bound.stale), bound.whole),
        (getattr(bound, "row", None), bound.own_resources()),
        (bound.order, bound.members, {each: bound.node(each) for each in bound.members}),
        {each: bound.wiring(each) for each in bound.channels},
        {each: bound.facts(each) for each in points if each in bound.channels},
        dict(bound.dropped),
        {each: bound.unbound(each) for each in bound.members if each not in points},
        points,
    )


def _read(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    stating: Mapping[str, Mapping[str, object]],
    ends: tuple[EndOffer, ...],
    kind: Callable[..., P],
) -> P:
    """The network bound, the graph's tensors read from one index."""
    with reading(model):
        return _bound(model, nodes, name, stating, ends, kind)


def _bound(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    stating: Mapping[str, Mapping[str, object]],
    ends: tuple[EndOffer, ...],
    kind: Callable[..., P],
) -> P:
    ops = [kernel_op(model, node) for node in nodes]
    owned = [op.owned() for op in ops]
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
    platform = read_target(model).platform
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
            if ends:
                facts_of["end_offer"] = ends
        wires[tensor] = wire(platform=platform, **facts_of)

    def on(side: str) -> dict[str, Wire]:
        """The channels at the ports ``<side><i>`` (``""``: no port, none on the boundary)."""
        return {
            tensor: each
            for tensor, each in wires.items()
            if ports.get(tensor, "").rstrip("0123456789") == side
        }

    kernels: dict[str, Facts] = {node.name: op.facts() for node, op in zip(nodes, ops)}
    members: dict[str, Facts | Wire] = {**on("s_axis_"), **on(""), **kernels, **on("m_axis_")}
    try:
        bound = kind(
            members,
            name=name,
            ends=ends,
            boundary=tuple(ports.items()),
            stored=stored,
            stray=stray,
            whole=whole,
        )
    except NetworkError as error:
        raise KernelOpError.refused(
            "network-invalid", "Partition", f"{name}: {error}", node=name
        ) from error
    _stale(model, bound, nodes, ops)
    return bound


# -- writing choices back -----------------------------------------------------------------


def _encoded(tensor: str, values: Mapping[str, object]) -> None:
    """Refuse a channel choice ``CHANNEL`` does not declare or admit, by name."""
    unknown = sorted(set(values) - set(CHANNEL_KEYS))
    if unknown:
        raise KernelOpError.refused(
            "channel-choice-unknown",
            CHANNEL.name,
            f"{tensor}: {unknown} are not a channel's choices ({CHANNEL.name})",
            tuple(unknown),
            tensor=tensor,
        )
    for key, value in values.items():
        try:
            CHANNEL_KEYS[key].encode(value)
        except MetadataError as error:
            raise KernelOpError.refused(
                "channel-choice-invalid",
                CHANNEL.name,
                f"{tensor}: {error}",
                (key,),
                tensor=tensor,
            ) from error


def _write_channel(model: ModelWrapper, tensor: str, choices: Mapping[str, object]) -> None:
    """State ``choices`` on ``tensor``'s channel, its whole: what it stated before is
    cleared."""
    model.clear(CHANNEL, tensor=tensor)
    for key, value in sorted(choices.items()):
        model.set(CHANNEL_KEYS[key], value, tensor=tensor)


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


__all__ = [
    "PARTITION",
    "PRODUCER",
    "Partition",
    "member",
    "network",
    "persist",
]
