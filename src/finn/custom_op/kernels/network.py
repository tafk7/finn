# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The core of KernelOp nodes a graph binds, one design space, and its choices written back.

``network(model, nodes, name=...)`` binds a core straight from the graph, for any
selection of KernelOp nodes (a DataflowCore's body, a slice of it, one node), and changes no
graph: it reads the graph once, its tensors from one index
(``finn.core.graph.facts.reading``), into a declaration (``declaration``), which the
composer (``finn.kernels.composer``) makes one Space class of, a ``DataflowCoreKernel``, and
binds its stored choices on one design space of that class
(``finn.kernels.dataflow_core.bound``: a ``Bound``). Its name (``name``), which its module's
name leaves out, is its caller's: a node's own name (``finn.custom_op.partition.DataflowCore``),
which every step holding only its body is given. ``bound_from`` binds them on another design
space made of the declaration: a body's core in its shell, which
``finn.custom_op.kernels.shell`` makes (``shell_point``), its row offering a mover on each edge
(``offer``, a ``finn.kernels.movers.MoverOffer``; None by default). The mover offered states
each boundary side it is offered on (``MoverOffer.states``: row-major, one pass a frame);
where none is, the build states the side its statement for the side's tensor
(``finn.core.graph.annotations.boundary``, ``finn.boundary``: the build's ``boundary``, by port,
which the cut writes by tensor), and where it states none the side is unstated. A build statement on
a boundary offered a mover is refused by name (``boundary-stated-twice``): the build refuses it
before its first step, so only a model annotated by hand reaches this.

**The declaration** (``declaration``):

- **kernels**, one per node, named as the node: its kernel's facts, read once
  (``KernelOp.facts``), and its policies' owner, its KernelOp class;
- **channels**, one per ONNX tensor, named as the tensor, in node order: a node's graph
  inputs, the parameter channels it owns (named after the initializer), its outputs. Each
  has its two sides, a kernel's port (``Port``: the op's ``wiring``, an input on the
  consumer's side, an output on the producer's) or, where no port of the selection is on
  it, the channel's source (a parameter channel, whose kernel states its value) or the
  boundary (``Side.boundary``: its AXIS port, stated by the mover offered, or else by the
  build's statement for its tensor, unstated where neither is); its tensor (the graph's,
  ``edge_tensor``, at the graph's rank; a parameter channel's, the tensor and contents its
  kernel states for it, ``Stated``); and the target's platform. Only the
  selection's ONNX inputs and outputs are boundaries, named by the shell's convention
  ``s_axis_<i>`` and ``m_axis_<j>``. An edge of a slice (a selection short of every
  KernelOp) whose node beyond it is a KernelOp is no boundary: that node is a **frozen
  neighbour**, and the edge channel's side there is its port's side at its stored
  choices, as a fact (``Side.neighbour``: an upstream neighbour's output side, its stream,
  label and word width, a downstream one's input side, its stream and label), the edge
  channel the slice's member; its port's stream pins there, which the edge channel's hop
  is wired to, and its pace, which sizing reads, are facts of the slice's class too
  (``DataflowCoreDeclaration.frozen``). Only an accepted value is a fact: a neighbour
  whose stored choices or kernel are refused, or whose side is not known at them, is
  refused by name (``neighbour-unaccepted``); another neighbour is another design space of
  the slice. An edge at a graph input or output, or at a node that is no KernelOp, stays a
  boundary. A channel has one consumer, so a
  tensor read more than once is refused by name before anything is declared
  (``tensor-fan-out``): read by two of the nodes (an edge, or a parameter both own), or
  read by one and at an output boundary too (a graph output, or read outside the nodes);
  and at a frozen neighbour, read by anything else beside the channel's two sides, as the
  whole core reads it. Fan-out waits for a channel that forks. A node and a tensor,
  or two nodes, of one name are refused (``name-shared``):
  a member is named once, as the choices log ``kernel_choices.json`` names it; and so is
  what the composer refuses (``DataflowCoreRefused``: its findings, as a ``KernelOpError``);
- **its order**: the channels are declared input boundary channels, the other channels,
  output boundary channels, which the composer ranks (each after the members it reads); its
  module places them input boundary channels, the other channels, the kernels, output
  boundary channels (``Reshape_0_out0``, ..., ``MatMul_0``, ...), its instance order, which
  the declaration states with its name (``DataflowCoreDeclaration.order``,
  ``name``); each member's node in the module is its member name (``member_name``: ``\\W``
  as ``_``).

**Its stored choices** (bind is assignment): every node's attributes and every channel's
``finn.channel`` on its tensor (in a DataflowCore's body only, ``refuse_outside_body``),
bound with the engine's ``bind`` on the design space of the graph's facts, never refused
there. A stored value outside its domain over the current facts (or a key that is no
choice) is dropped as stale, with why (``Bound.stale``). A stored set that breaks a rule
binds, and the core reads refused: the member's validity names the rule, and so do its
readers (``configured_network``, ``verify_partition``, and the seam's step,
``Seam.refusals``). A channel whose choices wait on an open kernel choice holds them,
waiting (``inspection.Waiting``). Nothing is filled in. With every KernelOp of the model
selected, choices a tensor states that is no channel of it are stale too. Nothing is
written: ``persist``, the one writer of choices, writes a core's choices back, each owner's
whole.

**Its outputs**: each node's output is its kernel's product. Where the graph's annotation of
an output (its shape's element count and last axis, its datatype and its range) differs from
what its producer states, the graph is stale, refused by name (``tensor-stale``: "run
InferKernelTensors"), before any channel's own ``channel-tensor`` is read.

**Its module** is the DataflowCore's IP (a slice with a frozen neighbour has none,
``slice-not-emitted``), as its class answers it
(``finn.kernels.dataflow_core.DataflowCoreKernel.module``): its members' netlists, named as
the graph's nodes and tensors, with the boundary's pins, named after its name (its stem
``finn_<name>``) and derived by its class, since a mover is integrated beside the IP, not in
it. Its views are read out of the design space, composed from its members' (the netlist
with each channel's outer hops from the kernels' stream pins views, FIFO sizing's paces,
resources), never bound into it.

A core's resources are its members', as its class answers them (``Kernel.resources``). A
core in its shell (``finn.custom_op.kernels.shell.shell_point``, ``finn.kernels.shell``) is a
member of the shell's design space, beside its movers and the shell's rules; the shell's
resources (its members', its movers' and its static region's), and the cut's choices
(``make_cut_choices``) on the core in its shell, are ``finn.custom_op.kernels.shell``'s.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

from onnx import NodeProto
from qonnx.core.metadata import MetadataError

from finn.core.graph.annotations.boundary import read_boundary
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
from finn.core.search import Key, dotted
from finn.core.space import Available, Rejected, Space, Unresolved, inspection
from finn.custom_op.kernels.base import KernelOp, kernel_op, typed
from finn.dataflow.streams import Stream
from finn.kernels.artifacts.abi import Direction
from finn.kernels.base import Kernel, PortPins
from finn.kernels.channels import Channel, Neighbours
from finn.kernels.composer import (
    ChannelDeclaration,
    DataflowCoreDeclaration,
    DataflowCoreRefused,
    Port,
    Stated,
    compose,
)
from finn.kernels.dataflow_core import Bound, DataflowCoreKernel, bound
from finn.kernels.explore import CoreComposition
from finn.kernels.laws import StatedRate, TimedPace
from finn.kernels.movers import MoverOffer
from finn.kernels.port import exposed_pins, exposed_side, stream_port
from finn.kernels.sides import Side

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

K = TypeVar("K", bound=Kernel)


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


def partition_ports(model: ModelWrapper) -> dict[str, str]:
    """The port of each boundary tensor of the partition of every KernelOp of ``model``,
    inputs then outputs (``s_axis_<i>``, ``m_axis_<j>``; the module docstring): the ports the
    cut's partition has, read before the cut or on its body alike."""
    nodes = [node for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN]
    owned = {tensor for node in nodes for tensor in kernel_op(model, node).owned().values()}
    return _boundary(model, nodes, owned, ())


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


def _neighbour(
    model: ModelWrapper, tensor: str, side: str, node: NodeProto
) -> tuple[Side, PortPins | None, TimedPace | StatedRate | None]:
    """The side a frozen neighbour, KernelOp ``node`` beyond the selection on ``side`` of
    ``tensor``, gives the edge channel, as a fact: its port's ``side`` at its stored
    choices (``KernelOp.point``; its facts alone, where it stores none), as a frozen
    neighbour's (``Side.neighbour``: an upstream one's output side, its stream, label and
    word width, a downstream one's input side, its stream and label), its label below the
    node's name; that port's stream pins there (``stream_pins``), which its hop is wired to;
    and its pace in cycles there (``timed_pace``: its kernel's stated rate's where that names
    it, else its pace at its kernel's interval, where it owns its pacing), which sizing
    reads. Only an accepted value is a fact: refused (``neighbour-unaccepted``, naming the
    neighbour, its port and why) where its stored choices are refused, its kernel
    is refused, or its side or its pace in cycles is not known there (it waits on an open
    choice, or is refused); a pace answered None is no pace."""
    op = kernel_op(model, node)
    port = (
        op.outputs[list(node.output).index(tensor)]
        if side == "producer"
        else op.ports[list(node.input).index(tensor)]
    )
    named = "" if port is None else op.wiring[port].port

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
    declared = stream_port(type(point), named)
    answer = point.query(declared.side)
    if not isinstance(answer, Available) or not isinstance(answer.value, Side):
        why = (
            "its side waits on an open choice"
            if isinstance(answer, Unresolved)
            else (
                "its side is refused" if isinstance(answer, Rejected) else "its port is not placed"
            )
        )
        raise unaccepted(why, getattr(answer, "findings", ()))
    found: Side = exposed_side(type(point), named, answer.value)
    stated = Side.neighbour(
        cast(Direction, found.direction), cast(Stream, found.stream), found.label, found.width
    )
    pins = point.query(declared.stream_pins)
    given = pins.value if isinstance(pins, Available) else None
    if isinstance(given, PortPins):
        given = exposed_pins(point, named, given)
    paced = point.query(declared.timed_pace)
    if isinstance(paced, (Rejected, Unresolved)):
        why = "waits on an open choice" if isinstance(paced, Unresolved) else "is refused"
        raise unaccepted(f"its pace in cycles {why}", paced.findings)
    pace = paced.value if isinstance(paced, Available) else None
    return (
        stated.under(node.name),
        given.under(node.name) if isinstance(given, PortPins) else None,
        pace if isinstance(pace, (TimedPace, StatedRate)) else None,
    )


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
    model: ModelWrapper, core: CoreComposition, nodes: list[NodeProto], ops: list[KernelOp]
) -> None:
    """Refuse a graph whose annotation of an output differs from what its producer
    states (its element count, its last axis, or its element: the datatype and its
    range), by name: the graph is stale (``tensor-stale``)."""
    findings: list[Finding] = []
    for node, op in zip(nodes, ops):
        kernel = core.member(node.name)
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
    """What the graph states of a selection: its declaration (its name, its order,
    its kernels' policies' owners and the ports beside its edge channels at frozen
    neighbours among it), its KernelOps, and the choices it stores and those stray. A record
    read once, never a value a Space holds."""

    __slots__ = ("declared", "ops", "stored", "stray")

    def __init__(
        self,
        declared: DataflowCoreDeclaration,
        ops: list[KernelOp],
        stored: dict[Key, object],
        stray: dict[str, tuple[Finding, ...]],
    ) -> None:
        self.declared, self.ops = declared, ops
        self.stored, self.stray = stored, stray


def _graph(
    model: ModelWrapper,
    nodes: list[NodeProto],
    name: str,
    offer: MoverOffer | None = None,
) -> _Read:
    """The selection ``nodes`` of ``model`` as the graph states it (``_Read``; the module
    docstring), named ``name``, each boundary side stated by the mover ``offer`` where one
    is offered, refused where a tensor is read more than once (``tensor-fan-out``) or two
    members share a name (``name-shared``)."""
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
    built = read_boundary(model)
    # The mover offered states each boundary side; none offered, the build's statement
    # for the side's tensor where it states one, else the side is unstated.
    twice = [tensor for tensor in ports if tensor in built] if offer is not None else []
    if offer is not None and twice:
        named = ", ".join(f"{ports[tensor]} ({tensor})" for tensor in twice)
        raise KernelOpError.refused(
            "boundary-stated-twice",
            name,
            f"{name}: the model's boundary statements state {named}, and the boundary is "
            f"offered an {offer.kind} mover, which states each side it is offered on: "
            "a build statement would be a second source",
            tensor=twice[0],
        )
    sides: dict[str, dict[str, Port | Side]] = {tensor: {} for tensor in tensors}
    facts: dict[str, dict[str, object]] = {tensor: {} for tensor in tensors}
    readers: dict[str, str] = {}
    for node, op, held in zip(nodes, ops, owned):
        wired = [("consumer", port, tensor) for port, tensor in op.inputs().items()]
        wired += [("producer", port, tensor) for port, tensor in zip(op.outputs, node.output)]
        for side, port, tensor in wired:
            wiring = op.wiring[port]
            sides[tensor][side] = Port(node.name, wiring.port)
            readers.setdefault(tensor, op.label)
            if port in held:
                facts[tensor]["tensor"] = Stated(node.name, str(wiring.carried))
                facts[tensor]["contents"] = Stated(node.name, str(wiring.stored))
    frozen: dict[str, Neighbours] = {}
    for tensor, (side, node) in beyond.items():
        neighbour, pins, pace = _neighbour(model, tensor, side, node)
        sides[tensor][side] = neighbour
        frozen[tensor] = (
            Neighbours(producer=pins, producer_pace=pace)
            if side == "producer"
            else Neighbours(consumer=pins, consumer_pace=pace)
        )
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
            outside = offer.states() if offer is not None else built.get(tensor)
            ends[each] = Side.boundary(ports[tensor], outside)
        channels[tensor] = ChannelDeclaration(ends["producer"], ends["consumer"], given)

    def on(prefix: str) -> list[str]:
        """The channels at the ports ``<prefix><i>`` (``""``: no port, none on the
        boundary)."""
        return [
            tensor for tensor in tensors if ports.get(tensor, "").rstrip("0123456789") == prefix
        ]

    kernels = {node.name: op.facts() for node, op in zip(nodes, ops)}
    declared_order = [*on("s_axis_"), *on(""), *on("m_axis_")]
    declared = DataflowCoreDeclaration(
        kernels,
        {tensor: channels[tensor] for tensor in declared_order},
        name=name,
        order=(*on("s_axis_"), *on(""), *kernels, *on("m_axis_")),
        owners={node.name: type(op) for node, op in zip(nodes, ops)},
        frozen=frozen,
    )
    return _Read(declared, ops, stored, stray)


def declaration(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str,
    offer: MoverOffer | None = None,
) -> DataflowCoreDeclaration:
    """The core of ``nodes``, KernelOp nodes of ``model``, as the graph declares it (the
    module docstring), named ``name``: each node's kernel by its facts, in graph
    order, and each channel by its tensor with its sides and facts, its boundary stated by
    the mover ``offer`` where one is offered."""
    with reading(model):
        return _graph(model, list(nodes), name, offer).declared


def composed(declared: DataflowCoreDeclaration, made: Callable[[DataflowCoreDeclaration], K]) -> K:
    """The design space ``made`` makes of ``declared`` (the core's class's, ``compose``, or
    its shell's), nothing chosen; a refusal of the composer's raised as the graph's, named by
    the core's name."""
    try:
        return made(declared)
    except DataflowCoreRefused as error:
        message = f"{declared.name}: " + "; ".join(map(str, error.findings))
        raise KernelOpError(message, findings=error.findings) from error


def core(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str,
) -> DataflowCoreKernel:
    """The core of ``nodes``, KernelOp nodes of ``model``, as ``network`` makes it, its
    design space with nothing chosen: no stored choice read (``network`` binds them)."""
    return composed(declaration(model, nodes, name=name), compose)


def network(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str,
) -> Bound[DataflowCoreKernel]:
    """The core of ``nodes``, KernelOp nodes of ``model``, named ``name`` (its
    node's), on the platform of the model's target, its stored choices bound (``Bound``);
    see the module docstring. A tensor read more than once is refused
    (``tensor-fan-out``), and so is a stale output (``tensor-stale``)."""
    return bound_from(model, nodes, name=name, made=compose)


def bound_from(
    model: ModelWrapper,
    nodes: Iterable[NodeProto],
    *,
    name: str,
    made: Callable[[DataflowCoreDeclaration], K],
    offer: MoverOffer | None = None,
) -> Bound[K]:
    """As ``network``, the stored choices bound on the design space ``made`` makes of the
    declaration (its boundary stated by the mover ``offer`` where one is offered): a body's
    core in its shell (``finn.custom_op.kernels.shell.shell_point``)."""
    selected = list(nodes)
    with reading(model):
        read = _graph(model, selected, name, offer)
        found = bound(composed(read.declared, made), read.stored, stray=read.stray)
        _stale(model, CoreComposition(found.point), selected, read.ops)
        return found


def bound_over(
    bodies: Sequence[tuple[ModelWrapper, str]],
    *,
    made: Callable[[Sequence[DataflowCoreDeclaration]], K],
    offer: MoverOffer | None = None,
) -> Bound[K]:
    """Several cores in one design space: each body ``(model, name)``'s KernelOp nodes
    declared as ``network`` declares them, named ``name`` (its node's), its boundary stated
    by the mover ``offer`` where one is offered; the design space ``made`` makes of their
    declarations, in order (a build's cores in their shell,
    ``finn.custom_op.kernels.shell.build_shell``), every body's stored choices bound on it.
    A body's staleness is its own point's to refuse (``bound_from``); a refusal of the
    composer's is raised as the graph's, named by the cores' names."""
    reads: list[_Read] = []
    for model, name in bodies:
        with reading(model):
            reads.append(_graph(model, list(model.graph.node), name, offer))
    declared = [read.declared for read in reads]
    try:
        point = made(declared)
    except DataflowCoreRefused as error:
        names = ", ".join(each.name for each in declared)
        message = f"{names}: " + "; ".join(map(str, error.findings))
        raise KernelOpError(message, findings=error.findings) from error
    stored = {key: value for read in reads for key, value in read.stored.items()}
    stray = {key: value for read in reads for key, value in read.stray.items()}
    return bound(point, stored, stray=stray)


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


def persist(model: ModelWrapper, point: Kernel) -> dict[str, dict[str, object]]:
    """Write ``point``'s choices, a design space of the core of ``model``'s nodes
    (``network``) or of its shell's (``bound_from``), back where they are stated, each
    owner's whole: every choice it stores, whatever its state (set, waiting or dormant),
    split by member, on its owner, a kernel's node or a channel's tensor; a choice an owner
    holds that the core does not (one the bind dropped as stale) cleared, and, where it
    holds every KernelOp of the model, so are the channel choices of a tensor that is no
    channel of it.

    The one writer of a core's choices: a route to one is a hold in step_explore's
    strategies, this function for the compiler (``persist_kernel`` beside it for conversion,
    one node's kernel alone, and ``finn.custom_op.partition.DataflowCore.persist`` for a
    node's body), and ``support.configure.configured`` for a test. Channel choices are a
    DataflowCore's body's only: written on a graph that is not (host nodes beside the
    KernelOps) they are refused, by name, before anything is written
    (``channel-choices-outside-body``). Every value is checked and encoded before any is
    written, so a write cannot fail halfway; none is set again on a fresh point, since the
    core has bound each (its choices are checked as they are set).
    Returns what each owner now holds, by graph name."""
    core = CoreComposition(point)
    names = list(core.members)
    kernels = set(core.kernels)
    whole = kernels >= {node.name for node in model.graph.node if node.domain == KERNEL_OPS_DOMAIN}
    held_by: dict[str, dict[str, object]] = {name: {} for name in names}
    for (owner, key), value in core.chosen(names).items():
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
    stray = [tensor for tensor in stated if whole and tensor not in held_by]
    for write in writes:
        write()
    for tensor in stray:
        model.clear(CHANNEL, tensor=tensor)
    return held_by


def persist_kernel(model: ModelWrapper, node: NodeProto, point: Space) -> dict[str, object]:
    """Write ``point``, the KernelOp ``node``'s kernel alone, on the node, its whole: every
    choice it stores, by key, whatever its state (``inspection.states``: set, waiting or
    dormant). Conversion's write (``finn.transformation.kernels.convert``): one node, no
    core, nothing on a tensor. As ``persist``, every value is checked and encoded
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
    "bound_from",
    "bound_over",
    "composed",
    "core",
    "declaration",
    "network",
    "partition_ports",
    "persist",
    "persist_kernel",
    "stale_text",
]
