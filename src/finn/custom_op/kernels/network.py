# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The partition of KernelOp nodes a graph binds, one design space, and its choices
written back.

``network(model, nodes, movers=...)`` binds a ``Partition`` straight from the graph, for any
selection of KernelOp nodes (a partition's body, a slice of it, one node), and changes no
graph: it reads the graph once, its tensors from one index
(``finn.core.graph.facts.reading``), into a declaration (``declaration``), which the
composer (``finn.kernels.composer``) makes one Space class of, and binds its stored choices
on one design space of that class (``finn.kernels.partition.partition``). The movers offered
on its boundary (``movers``, each a ``finn.kernels.movers.MoverOffer``; none by default) are
a fact its caller gives: a shell's row offers them, and ``finn.custom_op.kernels.shell``
binds a partition in the shell of the model's target (``in_shell``).

**The declaration** (``declaration``):

- **kernels**, one per node, named as the node: its kernel's facts, read once
  (``KernelOp.facts``);
- **channels**, one per ONNX tensor, named as the tensor, in node order: a node's graph
  inputs, the parameter channels it owns (named after the initializer), its outputs. Each
  has its two sides, a kernel's port (``Port``: the op's ``wiring``, an input on the
  consumer's side, an output on the producer's) or, where no port of the selection is on
  it, the channel's source (a parameter channel, whose kernel states its value) or the
  boundary (``Side.boundary``: its AXIS port, offered the movers given); its tensor (the
  graph's, ``edge_tensor``, at the graph's rank; a parameter channel's, the tensor and
  contents its kernel states for it, ``Stated``); and the target's platform. Only the
  selection's ONNX inputs and outputs are boundaries, named by the shell's convention
  ``s_axis_<i>`` and ``m_axis_<j>``. An edge of a slice (a selection short of every
  KernelOp) whose node beyond it is a KernelOp is no boundary: that node is a **frozen
  neighbour**, and the edge channel's side there is its port's side at its stored
  choices, as a fact (``Side.neighbour``: an upstream neighbour's stream, label and word
  width, a downstream one's stream and label), the edge channel the slice's member. Only
  an accepted value is a fact: a neighbour whose stored choices or kernel are refused, or
  whose side is not known at them, is refused by name (``neighbour-unaccepted``); another
  neighbour is another design space of the slice. An edge at a graph input or output, or
  at a node that is no KernelOp, stays a boundary. A channel has one consumer, so a
  tensor read more than once is refused by name before anything is declared
  (``tensor-fan-out``): read by two of the nodes (an edge, or a parameter both own), or
  read by one and at an output boundary too (a graph output, or read outside the nodes);
  and at a frozen neighbour, read by anything else beside the channel's two sides, as the
  whole partition reads it. Fan-out waits for a channel that forks. A node and a tensor,
  or two nodes, of one name are refused (``name-shared``):
  a member is named once, as the choices log ``kernel_choices.json`` names it; and so is
  what the composer refuses (``PartitionRefused``: its findings, as a ``KernelOpError``);
- **its order**: the channels are declared input boundary channels, the other channels,
  output boundary channels, which the composer ranks (each after the members it reads); its
  module places them input boundary channels, the other channels, the kernels, output
  boundary channels (``Reshape_0_out0``, ..., ``MatMul_0``, ...), its instance order; each
  member's node in the module is its member name (``member_name``: ``\\W`` as ``_``).

**Its stored choices** (bind is assignment): every node's attributes and every channel's
``finn.channel`` on its tensor (in a partition's body only, ``refuse_outside_body``), bound
with the engine's ``bind`` on the design space of the graph's facts, never refused there. A
stored value outside its domain over the current facts (or a key that is no choice) is
dropped as stale, with why (``Partition.stale``). A stored set that breaks a rule binds, and
the partition reads refused: the member's validity names the rule, and so do its readers
(``configured_network``, ``verify_partition``, and the seam's step, ``Seam.refusals``). A
channel whose choices wait on an open kernel choice holds them, waiting
(``inspection.Waiting``). Nothing is filled in. With every KernelOp of the model selected,
choices a tensor states that is no channel of it are stale too. Nothing is written:
``persist``, the one writer of choices, writes a partition's choices back, each owner's
whole.

**Its outputs**: each node's output is its kernel's product. Where the graph's annotation of
an output (its shape's element count and last axis, its datatype and its range) differs from
what its producer states, the graph is stale, refused by name (``tensor-stale``: "run
InferKernelTensors"), before any channel's own ``channel-tensor`` is read.

**Its module** is the partition's IP, the cut's (``PARTITION``): its members' netlists,
named as the graph's nodes and tensors, with the boundary's pins, named after the partition
(its stem ``finn_<name>``) and identified as the partition's IP (``PRODUCER``, which
``Partition.module_of`` is given), since a mover is integrated beside the IP, not in it. Its
views are read out of the design space, composed from its members' (the netlist with each
channel's outer hops from the kernels' contract views, FIFO sizing's paces, resources),
never bound into it.

A partition's resources (``Partition.own_resources``) are its members'. A partition in its
shell (``finn.custom_op.kernels.shell.in_shell``) holds the shell's budgets as rules of its
own (its shell member, ``finn.kernels.budgets``); the shell's resources over a partition
(its members', its movers' and its static region's), and the cut's choices
(``make_cut_choices``) on the partition in its shell, are ``finn.custom_op.kernels.shell``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar, cast

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
from finn.core.graph.facts import edge_tensor, initializer, reading
from finn.core.graph.partition import KERNEL_OPS_DOMAIN
from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import Available, Rejected, Space, Unresolved, inspection
from finn.custom_op.kernels.base import KernelOp, kernel_op, typed
from finn.dataflow.streams import Stream
from finn.kernels.artifacts.module import ProducerIdentity
from finn.kernels.channels import Channel
from finn.kernels.composer import (
    ChannelDeclaration,
    Key,
    PartitionDeclaration,
    PartitionRefused,
    Port,
    Stated,
)
from finn.kernels.explore import dotted
from finn.kernels.movers import MoverOffer
from finn.kernels.partition import Partition, partition
from finn.kernels.sides import Side

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

PARTITION = "partition"
"""The partition's name: the cut's node and body file, and the IP made from it, which
is its module (``finn_partition``)."""

PRODUCER = ProducerIdentity("finn.custom_op.kernels.partition", "1")
"""What derives a partition's module: identified as the partition's, so the IP's name and
digest are the partition's."""


def stale_text(key: str, found: Iterable[Finding]) -> str:
    """A stale choice as a line, its key (dotted, ``first.pe``) named once:
    ``first.pe = 3: not one of 1, 2, 4 (domain-membership)``, the value
    where the findings hold it, each finding's message and its code (a message that is its
    code, its code only)."""
    findings = tuple(found)
    values = [dict(each.details)["value"] for each in findings if "value" in dict(each.details)]
    named = f"{key} = {values[0]}" if values else key
    said = []
    for finding in findings:
        message = finding.message
        if values:
            message = message.removeprefix(f"{values[0]!r}: ")
        said.append(message if message == finding.code else f"{message} ({finding.code})")
    return f"{named}: " + "; ".join(said)


# -- reading the graph --------------------------------------------------------------------


def _boundary(
    model: ModelWrapper, nodes: list[NodeProto], owned: set[str], frozen: Collection[str]
) -> dict[str, str]:
    """The port of each boundary tensor, inputs then outputs, in node order: an input
    no node produces (and not a parameter: owned, or an initializer the op reads as a
    fact) is ``s_axis_<i>``; an output read outside ``nodes`` is ``m_axis_<j>``; a tensor
    at a frozen neighbour (``frozen``) is neither."""
    produced = {tensor for node in nodes for tensor in node.output}
    inside = {id(node) for node in nodes}
    used_outside = {
        tensor for node in model.graph.node if id(node) not in inside for tensor in node.input
    } | {output.name for output in model.graph.output}
    inputs = [
        tensor
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced
        and tensor not in owned
        and tensor not in frozen
        and initializer(model, tensor) is None
    ]
    outputs = [
        tensor
        for node in nodes
        for tensor in node.output
        if tensor in used_outside and tensor not in frozen
    ]
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


def _beyond(
    model: ModelWrapper, nodes: list[NodeProto], owned: set[str]
) -> tuple[dict[str, tuple[str, NodeProto]], dict[str, Finding]]:
    """Each tensor at the selection's edge whose node beyond it is a KernelOp, in node
    order: the side that node is on, ``producer`` (an input it produces) or ``consumer``
    (an output it alone reads), and the node; and each such tensor that something else
    reads too (a node on either side, or the graph's outputs), refused as the whole
    partition refuses its fan-out (``tensor-fan-out``). A tensor at a graph input or
    output, or at a node that is no KernelOp, is none: it stays a boundary."""
    inside = {id(node) for node in nodes}
    produced = {tensor for node in nodes for tensor in node.output}
    graph_outputs = {output.name for output in model.graph.output}

    def readers(tensor: str, within: bool) -> list[NodeProto]:
        return [
            node
            for node in model.graph.node
            if (id(node) in inside) == within and tensor in node.input
        ]

    frozen: dict[str, tuple[str, NodeProto]] = {}
    fanned: dict[str, Finding] = {}
    edges = [
        (tensor, "producer")
        for tensor in dict.fromkeys(t for node in nodes for t in node.input)
        if tensor not in produced and tensor not in owned and initializer(model, tensor) is None
    ]
    edges += [(tensor, "consumer") for node in nodes for tensor in node.output]
    for tensor, side in edges:
        outside = readers(tensor, within=False)
        if side == "producer":
            found = model.find_producer(tensor)
            beyond = None if found is None or found.domain != KERNEL_OPS_DOMAIN else found
            others = outside
        else:
            kernels = [node for node in outside if node.domain == KERNEL_OPS_DOMAIN]
            beyond = kernels[0] if kernels else None
            others = [node for node in outside if node is not beyond]
        if beyond is None:
            continue
        within = readers(tensor, within=True)
        alone = len(within) == (1 if side == "producer" else 0)
        if alone and not others and tensor not in graph_outputs:
            frozen[tensor] = (side, beyond)
            continue
        names = [node.name for node in (*within, *outside)]
        output = " and is a graph output" if tensor in graph_outputs else ""
        fanned[tensor] = Finding(
            FindingKind.REJECTION,
            "tensor-fan-out",
            "fan_out",
            f"{tensor} is read by {', '.join(names)}{output}",
            (("tensor", tensor),),
        )
    return frozen, fanned


def _neighbour(model: ModelWrapper, tensor: str, side: str, node: NodeProto) -> Side:
    """The side a frozen neighbour, KernelOp ``node`` beyond the selection on ``side`` of
    ``tensor``, gives the edge channel, as a fact: its port's ``<p>_side`` at its stored
    choices (``KernelOp.point``; its facts alone, where it stores none), as a frozen
    neighbour's (``Side.neighbour``: an upstream one's stream, label and word width, a
    downstream one's stream and label), its label below the node's name. Only an
    accepted value is a fact: refused (``neighbour-unaccepted``, naming the neighbour, its
    port and why) where its stored choices are refused, its kernel is refused, or its side
    is not known there (it waits on an open choice, or is refused)."""
    op = kernel_op(model, node)
    port = (
        op.outputs[list(node.output).index(tensor)]
        if side == "producer"
        else op.ports[list(node.input).index(tensor)]
    )
    named = "" if port is None else op.wiring[port].contract.removesuffix("_contract")

    def unaccepted(why: str, causes: Iterable[Finding] = ()) -> KernelOpError:
        message = (
            f"{tensor}: its frozen neighbour {node.name} ({op.op_type}), at its port "
            f"{named or '(none)'}, gives no accepted value: {why}"
        )
        details = (("tensor", tensor), ("node", node.name), ("port", named))
        finding = Finding(
            FindingKind.REJECTION, "neighbour-unaccepted", tensor, message, details, tuple(causes)
        )
        return KernelOpError(message, findings=(finding,))

    if port is None:
        raise unaccepted("it reads the tensor as an argument, at no port")
    try:
        point = op.point()
    except KernelOpError as error:
        raise unaccepted("its stored choices are refused", error.findings) from error
    validity = inspection.validity(point)
    if isinstance(validity.result, Rejected):
        raise unaccepted("its kernel is refused", validity.result.findings)
    answer = point.query(getattr(type(point), f"{named}_side"))
    if not isinstance(answer, Available) or not isinstance(answer.value, Side):
        why = (
            "its side waits on an open choice"
            if isinstance(answer, Unresolved)
            else (
                "its side is refused" if isinstance(answer, Rejected) else "its port is not placed"
            )
        )
        raise unaccepted(why, getattr(answer, "findings", ()))
    found: Side = answer.value
    stated = Side.neighbour(cast(Stream, found.stream), found.label, found.width)
    return stated.under(node.name)


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
    whole: bool,
) -> tuple[dict[Key, object], dict[str, tuple[Finding, ...]]]:
    """The choices the partition binds, by key: each node's attributes, each channel's as
    its tensor states them; and, where ``whole``, those of a tensor that states some and is
    no channel of it, by dotted key, each with its finding (``channel-choice-stray``)."""
    found: dict[Key, object] = {}
    for node, op in zip(nodes, ops):
        found |= {(node.name, key): value for key, value in op.stored().items()}
    stated = model.tensors_stating(CHANNEL)
    refuse_outside_body(model, stated)
    for tensor in tensors:
        if tensor in stated:
            values = channel_choices(model, tensor)
            found |= {(tensor, key): value for key, value in typed(Channel, values).items()}
    channels = set(tensors)
    stray: dict[str, tuple[Finding, ...]] = {
        dotted((tensor, key)): (
            Finding(
                FindingKind.REJECTION,
                "channel-choice-stray",
                key,
                "stated on a tensor that is no channel of the partition",
                (("tensor", tensor),),
            ),
        )
        for tensor in stated
        if whole and tensor not in channels
        for key in channel_choices(model, tensor)
    }
    return found, stray


def _stale(
    model: ModelWrapper, bound: Partition, nodes: list[NodeProto], ops: list[KernelOp]
) -> None:
    """Refuse a graph whose annotation of an output differs from what its producer
    states (its element count, its last axis, or its element: the datatype and its
    range), by name: the graph is stale (``tensor-stale``)."""
    findings: list[Finding] = []
    for node, op in zip(nodes, ops):
        kernel = bound.member(node.name)
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


class _Read:
    """What the graph states of a selection: its declaration, its module's order, its
    boundary's ports, its KernelOps, the choices it stores and those stray, and whether it
    holds every KernelOp of the model. A record read once, never a value a Space holds."""

    __slots__ = ("declared", "ops", "order", "ports", "stored", "stray", "whole")

    def __init__(
        self,
        declared: PartitionDeclaration,
        order: tuple[str, ...],
        ports: dict[str, str],
        ops: list[KernelOp],
        stored: dict[Key, object],
        stray: dict[str, tuple[Finding, ...]],
        whole: bool,
    ) -> None:
        self.declared, self.order, self.ports, self.ops = declared, order, ports, ops
        self.stored, self.stray, self.whole = stored, stray, whole


def _graph(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    movers: tuple[MoverOffer, ...],
    shell: Mapping[str, object] | None = None,
) -> _Read:
    """The selection ``nodes`` of ``model`` as the graph states it (``_Read``; the module
    docstring), in a shell where ``shell`` states its row's facts
    (``PartitionDeclaration.shell``), refused where a tensor is read more than once
    (``tensor-fan-out``) or two members share a name (``name-shared``)."""
    ops = [kernel_op(model, node) for node in nodes]
    owned = [op.owned() for op in ops]
    parameters = {tensor for tensors in owned for tensor in tensors.values()}
    beyond, crossed = _beyond(model, nodes, parameters)
    ports = _boundary(model, nodes, parameters, beyond)
    fanned = [
        each
        for each in _fan_out(model, nodes, owned, ports)
        if dict(each.details)["tensor"] not in crossed
    ]
    fanned += crossed.values()
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
    stored, stray = _stored(model, nodes, ops, tensors, whole)
    platform = read_target(model).platform
    sides: dict[str, dict[str, Port | Side]] = {tensor: {} for tensor in tensors}
    facts: dict[str, dict[str, object]] = {tensor: {} for tensor in tensors}
    readers: dict[str, str] = {}
    for node, op, held in zip(nodes, ops, owned):
        wired = [("consumer", port, tensor) for port, tensor in op.inputs().items()]
        wired += [("producer", port, tensor) for port, tensor in zip(op.outputs, node.output)]
        for side, port, tensor in wired:
            wiring = op.wiring[port]
            sides[tensor][side] = Port(node.name, wiring.contract.removesuffix("_contract"))
            readers.setdefault(tensor, op.label)
            if port in held:
                facts[tensor]["tensor"] = Stated(node.name, str(wiring.carried))
                facts[tensor]["contents"] = Stated(node.name, str(wiring.stored))
    for tensor, (side, node) in beyond.items():
        sides[tensor][side] = _neighbour(model, tensor, side, node)
    channels: dict[str, ChannelDeclaration] = {}
    for tensor in tensors:
        given = facts[tensor]
        if "tensor" not in given:
            given["tensor"] = edge_tensor(model, tensor, readers[tensor])
        given["platform"] = platform
        ends = sides[tensor]
        if "contents" in given and "producer" not in ends:
            ends["producer"] = Side.source()
        for each in ("producer", "consumer"):
            if each in ends:
                continue
            if tensor not in ports:
                raise KernelOpError.refused(
                    "channel-boundary",
                    tensor,
                    f"{tensor}: a boundary of its partition, but no port names it: only an ONNX "
                    "input or output of a partition crosses its boundary",
                    tensor=tensor,
                )
            ends[each] = Side.boundary(ports[tensor], movers)
        channels[tensor] = ChannelDeclaration(ends["producer"], ends["consumer"], given)

    def on(prefix: str) -> list[str]:
        """The channels at the ports ``<prefix><i>`` (``""``: no port, none on the
        boundary)."""
        return [
            tensor for tensor in tensors if ports.get(tensor, "").rstrip("0123456789") == prefix
        ]

    kernels = {node.name: op.facts() for node, op in zip(nodes, ops)}
    declared_order = [*on("s_axis_"), *on(""), *on("m_axis_")]
    declared = PartitionDeclaration(
        kernels, {tensor: channels[tensor] for tensor in declared_order}, shell
    )
    order = (*on("s_axis_"), *on(""), *kernels, *on("m_axis_"))
    return _Read(declared, order, ports, ops, stored, stray, whole)


def declaration(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    movers: tuple[MoverOffer, ...] = (),
    shell: Mapping[str, object] | None = None,
) -> PartitionDeclaration:
    """The partition of ``nodes``, KernelOp nodes of ``model``, as the graph declares it
    (the module docstring): each node's kernel by its facts, in graph order, and each
    channel by its tensor with its sides and facts, its boundary offered ``movers``; in a
    shell where ``shell`` states its row's facts."""
    with reading(model):
        return _graph(model, list(nodes), PARTITION, movers, shell).declared


def network(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str = PARTITION,
    movers: tuple[MoverOffer, ...] = (),
) -> Partition:
    """The partition of ``nodes``, KernelOp nodes of ``model``, its module the partition
    ``name``'s IP, on the platform of the model's target: ``movers`` offered on the boundary
    channels (none by default; a shell's row offers them, ``in_shell``); see the module
    docstring. A tensor read more than once is refused (``tensor-fan-out``), and so is a
    stale output (``tensor-stale``)."""
    return _network(model, list(nodes), name, movers, Partition)


P = TypeVar("P", bound=Partition)


def _network(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    movers: tuple[MoverOffer, ...],
    kind: Callable[..., P],
    shell: Mapping[str, object] | None = None,
) -> P:
    """``network``, made by ``kind`` (``Partition``, or a shell's partition over it), in a
    shell where ``shell`` states its row's facts, the graph's tensors read from one
    index."""
    with reading(model):
        read = _graph(model, nodes, name, movers, shell)
        try:
            made = partition(
                read.declared,
                read.stored,
                name=name,
                order=read.order,
                owners={node.name: type(op) for node, op in zip(nodes, read.ops)},
                ports=tuple(read.ports.items()),
                movers=movers,
                stray=read.stray,
                whole=read.whole,
                kind=kind,
            )
        except PartitionRefused as error:
            message = f"{name}: " + "; ".join(map(str, error.findings))
            raise KernelOpError(message, findings=error.findings) from error
        _stale(model, made, nodes, read.ops)
        return made


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


def persist(model: ModelWrapper, partition: Partition) -> dict[str, dict[str, object]]:
    """Write ``partition``'s choices, a partition of ``model``'s nodes (``network``), back
    where they are stated, each owner's whole: every choice its design space stores,
    whatever its state (set, waiting or dormant), split by member, on its owner, a kernel's
    node or a channel's tensor; a choice an owner holds that the partition does not (one the
    bind dropped as stale) cleared, and, where it holds every KernelOp of the model, so are
    the channel choices of a tensor that is no channel of it.

    The one writer of a partition's choices: a route to one is ``Hold`` in a step after the
    cut, this function for the compiler (``persist_kernel`` beside it for conversion, one
    node's kernel alone), and ``support.configure.configured`` for a test. Channel
    choices are a partition's body's only: written on a graph that is not (host nodes
    beside the KernelOps) they are refused, by name, before anything is written
    (``channel-choices-outside-body``). Every value is checked and encoded before any is
    written, so a write cannot fail halfway; none is set again on a fresh point, since the
    partition has bound each (its choices are checked as they are set).
    Returns what each owner now holds, by graph name."""
    names = list(partition.members)
    kernels = set(partition.kernels)
    held_by: dict[str, dict[str, object]] = {name: {} for name in names}
    for (owner, key), value in partition.chosen(names).items():
        held_by[owner][key] = value
    refuse_outside_body(
        model, [name for name, values in held_by.items() if values and name not in kernels]
    )
    nodes = {node.name: node for node in model.graph.node}
    stated = model.tensors_stating(CHANNEL)
    writes: list[Callable[[], None]] = []
    for name, values in held_by.items():
        if name in kernels:
            op = kernel_op(model, nodes[name])
            if values or op.choices():
                writes.append(_kernel_write(op, op._saved(values)))
        elif values or name in stated:
            _encoded(name, values)
            writes.append(_channel_write(model, name, values))
    stray = [tensor for tensor in stated if partition.whole and tensor not in held_by]
    for write in writes:
        write()
    for tensor in stray:
        model.clear(CHANNEL, tensor=tensor)
    return held_by


def persist_kernel(model: ModelWrapper, node: NodeProto, point: Space) -> dict[str, object]:
    """Write ``point``, the KernelOp ``node``'s kernel alone, on the node, its whole: every
    choice it stores, by key, whatever its state (``inspection.states``: set, waiting or
    dormant). Conversion's write (``finn.transformation.kernels.convert``): one node, no
    partition, nothing on a tensor. As ``persist``, every value is checked and encoded
    before any is written. Returns what the node now holds."""
    held = {
        key: state.value
        for key, state in inspection.states(point)
        if isinstance(state, (inspection.Set, inspection.Waiting, inspection.Dormant))
    }
    op = kernel_op(model, node)
    op._write(op._saved(held))
    return held


def _kernel_write(op: KernelOp, merged: Mapping[str, int | str]) -> Callable[[], None]:
    return lambda: op._write(merged)


def _channel_write(
    model: ModelWrapper, tensor: str, values: Mapping[str, object]
) -> Callable[[], None]:
    return lambda: _write_channel(model, tensor, values)


__all__ = [
    "PARTITION",
    "PRODUCER",
    "Partition",
    "declaration",
    "network",
    "persist",
    "persist_kernel",
    "stale_text",
]
