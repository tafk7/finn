# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels: one tensor from one producer to one consumer, a kernel with children between
two ports.

A ``Channel`` is one producer-to-consumer edge, a member of a network
(``finn.kernels.network``) or of a kernel with children, carrying one ``tensor`` given as
a fact. The streams of the ports on it are facts too, ``producer`` and ``consumer``: each
the ``Stream`` a member presents at one of its ports (a kernel's ``x_stream``), the fact
it is bound to its side, or absent: the network's boundary. Beside each, the network gives
its port's label (``producer_label``, ``consumer_label``: ``matmul.compute.packed.x``),
and the producing port's word width (``producer_width``, its ``x_data_width``): a stream
given without them is refused (``channel-side``). The channel's tensor is never derived
from a port.

Each side's stream must traverse the tensor (``channel-tensor``), and the steps between
their beat sequences (``finn.dataflow.steps``) must be ones a chain of FinnLib modules
carries out (``channel-plan``); each hop between them connects directly
(``compatible``: their sequences, and a boundary output's word against the AXIS word that
carries it, ``channel-padding``). A channel runs, in order:

    source -> output adapter -> transport -> input adapter -> consumer

The steps split at the transport: their output side (width conversions) is the
``output_adapter``, the ``vpc`` chain placed where they convert a width, their input side
(a reorder, replay included, and markers) the ``adapter``, the fixed chain
(``finn.kernels.adapters``) its steps take, picked by them (``input_chain``): ``none``
where nothing happens after the transport, refused where no FinnLib chain carries them
out (``adapter-plan``). Its input generators share one memory style, the channel's choice
(``adapter_ram_style``), ranked by the most bits one of them holds a frame
(``adapter_ram_preference``), in UltraRAM only where the platform has it
(``adapter_uram_present``). The ``transport`` selector is ``direct``, which places
nothing, or ``fifo``; the FIFO owns its ``depth`` (the cut's policy requires it: an
exploration sizes it, ``finn.kernels.cut``) and its ``ram_style``, and carries data words
only (``fifo-markers``). Its keys are the channel's, by its name (``("x", "transport")``).

A channel with a port on one side only is a boundary: its ``port`` names the module's
AXIS port (``in0_V``), refused where absent (``channel-boundary``), whose contract the
channel builds from its facts (``boundary``). The boundary presents what its inner side
presents, by one rule: an input without the replay or window its receiver realizes
(``unreplayed``), an output as produced, neither with markers. Where a
shell offers movers (``mover_offer``, ``finn.kernels.movers``), the channel places its
``mover`` on that free side; it binds no RTL in the module and exports its contract
(``MOVER``) for the shell's budget check and resources.

A channel whose tensor has a known value (``contents``) places its ``source`` (a
memory, ``MemStreamKernel``) as its only producer (``channel-users``), storing one period
in the order its consumer reads it; with several sets, the source's set port is the
channel's port (``set``, served by its source: ``set_stream``), which the index tensor's
channel consumes.

The source and the output adapter are placed where their condition holds, not chosen,
and each keeps the label it is placed at (``SOURCE_LABEL``, ``OUTPUT_ADAPTER_LABEL``),
so the names emitted from them do not depend on its node.

As a kernel with children (a ``KernelSpace`` binding no module), its members are its
source, its adapters' stages, its FIFO (``ChannelFifo``) and its mover, in the order the
stream runs through them. It checks its stages by their streams (``staged``, each stage's
own derived values) and wires them by their contracts (``stages``, a view). Its answers
are the base's views over theirs, but for two it states itself, calling the base's: its
netlist places its source and its output adapter at their labels and adds the hops between
its own ends (its source, its stages, its boundary; ``NETLIST``), and a channel of wires
takes no cycles of its own (0). Its hops to the ports beside it need those ports'
contracts, which its network holds: the network wires them (``outer_hops``, from the ports
it read, ``Neighbours``) and places them around the channel's netlist. It exports its
boundary's AXIS bus (``BOUNDARY``) and its mover's contract (``MOVER``: ``placed_mover``,
from the mover's own derived ``contract``). It is no module of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Select,
    ValueSemantics,
    View,
    constraint,
    derived,
    domain,
    reject,
    view,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.steps import Steps, Unrealizable
from finn.dataflow.steps import steps as between
from finn.dataflow.streams import Stream, Streams, misfit
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, period, unreplayed
from finn.kernels.adapters import (
    ADAPTERS,
    OUTPUT_ADAPTER,
    InputAdapter,
    RealizedStage,
    Stage,
    StageStreams,
    realize,
)
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import CLOCK, RESET, Bus, Endpoint
from finn.kernels.artifacts.module import Fragment, Link, LinkEnd, Marker
from finn.kernels.base import (
    BOUNDARY,
    BUFFERING,
    CYCLES,
    NETLIST,
    NETLIST_SEMANTICS,
    RESOURCES,
    KernelSpace,
    PortContract,
)
from finn.kernels.declaration import declared_reads
from finn.kernels.fifo import FifoKernel
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.movers import MOVER, IodmaMover, MoverContract, MoverOffer
from finn.kernels.port import STREAM, served
from finn.kernels.transport import (
    AxisBeat,
    Mismatch,
    StreamContract,
    compatibility,
    lane_permutation,
    marker_bit,
    marker_pairs,
)
from finn.kernels.utilization import memory_preference, memory_styles, uram_constraint
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.platform.capabilities import Platform

# -- the labels a channel's placements keep --------------------------------------------------
#
# Each was the node of a one-case selector's case. As a placement its node is the member's
# name (``source``, ``output_adapter``), but the names emitted from it (its instances, the
# nets on its pins, its set port's) keep these labels: the channel's netlist places it there.

SOURCE_LABEL = "source.memstream"
"""Where a channel's netlist places its source: its producer's owner (``source_label``), and
the label of the set port it serves (``source.memstream.set``)."""

OUTPUT_ADAPTER_LABEL = "output_adapter.vpc"
"""Where a channel's netlist places its output adapter, its chain's stages below it
(``output_adapter.vpc.vpc``: a chain labels its stages by its chain, then the stage)."""

_PLACED = {"source": SOURCE_LABEL, "output_adapter": OUTPUT_ADAPTER_LABEL}
"""Each placement's node, and the label the channel's netlist places it at."""

BOUNDARY_CONTRACT: ValueSemantics[StreamContract | None] = ValueSemantics(
    StreamContract,
    "boundary contract",
    lambda value: value is None or isinstance(value, StreamContract),
    lambda left, right: left == right,
    lambda value: value,
)
"""The AXIS port a channel's module presents for its side without a port; None where both
sides have one."""


def boundary_contract(
    name: str, element: ScalarEncoding, sequence: BeatSequence, endpoint: Endpoint
) -> StreamContract:
    """The AXIS port a composed module presents for one of its own channels."""
    if len(sequence.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    form = sequence.form
    beat = AxisBeat(name, element.dtype, form.lanes, endpoint=endpoint, last=bool(sequence.markers))
    transport = beat.native(clock=CLOCK, reset=RESET)
    markers = ((transport.markers[0].signal, sequence.markers[0]),) if sequence.markers else ()
    return StreamContract(transport, element, form, sequence.repetition, markers)


def _link_end(instance: str | None, contract: StreamContract) -> LinkEnd:
    transport = contract.transport
    return LinkEnd(instance, transport.data, transport.data_width, transport.valid, transport.ready)


def wired(
    source: str | None, produced: StreamContract, sink: str | None, consumed: StreamContract
) -> Link:
    """One hop that connects directly, as wires: each sink lane from its source lane
    (``lane_permutation``), and each marker the sink requires from the source bit that
    guarantees it, or tied high when it closes every beat (``marker_pairs``)."""
    markers: list[Marker] = []
    for offered, required in marker_pairs(produced, consumed):
        signal, bit = (None, None) if offered is None else marker_bit(offered)
        pin, position = marker_bit(required)
        markers.append((signal, bit, pin, position))
    return Link(
        _link_end(source, produced),
        _link_end(sink, consumed),
        produced.element.bits,
        lane_permutation(produced, consumed),
        tuple(markers),
    )


@dataclass(frozen=True)
class Neighbours:
    """The ports beside a channel, as its network reads them to bind it (each
    ``Interface``'s contract, labelled from the network: ``matmul.compute.packed.x``): the
    producing kernel's and the consuming kernel's, each None where that side is the
    channel's own, its source (a known value) or its boundary."""

    producer: PortContract | None = None
    consumer: PortContract | None = None


@dataclass(frozen=True)
class OuterHops:
    """A channel's hops to the ports beside it (``outer_hops``), each None where that side
    is its own: ``producer`` from the producing kernel's port (to the channel's first
    stage, or to its consumer where it has none), ``consumer`` to the consuming kernel's
    port (from its last stage, or from its source or boundary where it has none). Its
    other hops, both ends its own, are its netlist's."""

    producer: Link | None = None
    consumer: Link | None = None

    def around(self, fragment: Fragment) -> Fragment:
        """``fragment``, the channel's netlist, with these hops around its links: the
        producer's first, the consumer's last, in the order the stream runs."""
        before = () if self.producer is None else (self.producer,)
        after = () if self.consumer is None else (self.consumer,)
        return Fragment(fragment.instances, (*before, *fragment.links, *after), fragment.exports)


_End = tuple[str | None, StreamContract]
"""One end of a hop: its owner, relative to the channel (``None``: the module's own pins),
and the contract it presents."""


def _kernel_of(port: PortContract, owner: str) -> str | Rejected:
    """The node of the kernel whose port ``port`` is, beside the channel: the node above
    its label (``matmul.compute.packed.x``, at any depth below the network:
    ``^matmul.compute.packed``). Refused as ``channel-port``, owned by ``owner``."""
    kernel = port.label.rpartition(".")[0]
    if not kernel:
        return reject(
            "channel-port",
            f"{port.label} presents a contract, but is no kernel's port",
            owner=owner,
        )
    return "^" + kernel


def _own_source(channel: Channel) -> _End:
    """The producing end a channel holds itself: its source, below it at its label, or
    its boundary, the module's own pins."""
    if channel.valued:
        offered: PortContract = channel.source.output_contract
        return (channel.source_label, offered.contract)
    return _own_boundary(channel)


def _own_boundary(channel: Channel) -> _End:
    """A channel's boundary, the module's own pins, as one end of a hop; a ``ValueError``
    where it has a port on each side, and so none."""
    found: StreamContract | None = channel.boundary
    if found is None:
        raise ValueError("a channel with a port on each side has no boundary")
    return (None, found)


def outer_hops(channel: Channel, neighbours: Neighbours, *, owner: str) -> OuterHops | Rejected:
    """``channel``'s hops to the ports beside it (``neighbours``, as its network reads
    them), as wires (``wired``), of a channel whose netlist is answered.

    A port belongs to its kernel beside the channel (``_kernel_of``); a stage and the
    source sit below the channel at their labels; a boundary side is the module's own
    pins (``None``). Refused where a port is no kernel's (``channel-port``), or where a
    hop's wires do not form (``channel-link``), each finding owned by ``owner``: the
    channel's name in its network.
    """
    owners: list[str | None] = []
    for port in (neighbours.producer, neighbours.consumer):
        node = None if port is None else _kernel_of(port, owner)
        if isinstance(node, Rejected):
            return node
        owners.append(node)
    producing: _End | None = None
    if neighbours.producer is not None:
        producing = (owners[0], neighbours.producer.contract)
    consuming: _End | None = None
    if neighbours.consumer is not None:
        consuming = (owners[1], neighbours.consumer.contract)
    stages: tuple[Stage, ...] = channel.stages
    # What each port meets of the channel: its first stage or its last, or, with none,
    # the other side: the other port, or the channel's own source or boundary.
    if stages:
        into: _End = (stages[0].label, stages[0].input)
        out_of: _End = (stages[-1].label, stages[-1].output)
    else:
        into = consuming or _own_boundary(channel)
        out_of = producing or _own_source(channel)
    try:
        produced = None if producing is None else wired(*producing, *into)
        consumed = None
        # A channel of no stage between two ports is one hop, the producer's.
        if consuming is not None and (stages or producing is None):
            consumed = wired(*out_of, *consuming)
    except ArtifactError as error:
        return reject("channel-link", str(error), owner=owner)
    return OuterHops(produced, consumed)


def _below(holder: str, stages: tuple[StageStreams, ...]) -> tuple[StageStreams, ...]:
    """``stages``, each labelled below ``holder``, the channel's member that holds them."""
    return tuple(replace(stage, label=f"{holder}.{stage.label}") for stage in stages)


def _below_stages(holder: str, stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    """``stages``, each labelled below ``holder``, as ``_below``."""
    return tuple(replace(stage, label=f"{holder}.{stage.label}") for stage in stages)


class ChannelFifo(KernelSpace):
    """An identity stage: what arrives at it, presented on both sides of a native FIFO.

    Its netlist, buffering and resources are its FIFO's, by the base. It exports no
    cycles: a FIFO presents what arrives, whose beats are counted where they are made
    (the channel's producer, its source, its output adapter), so its channel's cycles
    are its other members'. It presents no port: its channel checks it by its FIFO's
    streams (``stage_streams``) and wires it hop by hop (``stages``: its FIFO's module,
    read from the FIFO's netlist, and its ports' contracts).
    """

    id = "finn.channel.fifo"

    tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    platform: Platform = Param()
    # Known by membership only: the cut's policy is required (finn.kernels.cut), and a
    # FIFO is chosen by sizing it (SizeFifos).
    depth: int = Decision(
        domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32, ordered=True),
    )

    @derived
    def carried(self) -> BeatSequence:
        """What its FIFO carries: what arrives, its data words only (a marker closing every
        beat, which it may carry, is tied high after it)."""
        arriving = self.arriving
        return BeatSequence(arriving.form, arriving.repetition)

    buffer = FifoKernel(
        input_tensor=tensor,
        arriving=carried,
        depth=depth,
        platform=platform,
    )

    @constraint
    def carries_data(self) -> bool | Rejected:
        """What arrives carries no marker a FIFO would have to carry (one closing every
        beat is tied high after it)."""
        if any(not rule.constant for rule in self.arriving.markers):
            return reject(
                "fifo-markers",
                "what arrives carries markers, and a FIFO carries data words only",
            )
        return True

    @derived
    def stage_streams(self) -> tuple[StageStreams, ...]:
        """Its FIFO's streams and output word width, its derived values, labelled as its
        ``Stage`` is."""
        buffer = self.buffer
        stage = StageStreams(
            "fifo.buffer", buffer.input_stream, buffer.output_stream, buffer.output_data_width
        )
        return (stage,)

    @view
    @declared_reads("buffer.netlist", "buffer.input_contract", "buffer.output_contract")
    def stages(self) -> tuple[Stage, ...]:
        buffer = self.buffer
        return (
            Stage(
                library_module(buffer),
                buffer.input_contract.contract,
                buffer.output_contract.contract,
                "fifo.buffer",
            ),
        )

    exports = {
        NETLIST: KernelSpace.netlist,
        BUFFERING: KernelSpace.buffering,
        RESOURCES: KernelSpace.resources,
    }


@dataclass(frozen=True)
class ChannelSides:
    """A channel's producing and consuming sides: each side's owner (its port's label from
    the network, ``matmul.compute.packed.x``; the channel's source label, ``SOURCE_LABEL``,
    for a known value; None on the boundary) and its stream (on the boundary, what the
    module's AXIS port presents: ``Channel.boundary``)."""

    source_owner: str | None
    source: Stream
    sink_owner: str | None
    sink: Stream


class Channel(KernelSpace):
    """One tensor between the two streams it takes: one producer, one consumer.

    ``producer`` and ``consumer`` are the streams of the ports on it (a kernel's
    ``x_stream``), facts its network gives (``Interface``); the fact a stream is bound to
    is its side. Each comes with its port's label (``producer_label``,
    ``consumer_label``: ``matmul.compute.packed.x``, as the network reads it), and the
    producer with its port's word width (``producer_width``, its ``x_data_width``). A side
    without a port is the boundary, presented as the AXIS port ``port`` (``boundary``),
    where the ``mover`` sits where the shell offers one (``mover_offer``). A channel with a
    known value places its ``source`` as its producer; with several sets, the source reads the set
    indices (``index_tensor``) from the channel whose consumer is ``set_stream``, the one
    port it presents (``set``, its source's). ``platform`` is the target's: its source,
    its adapters' stages and its FIFO read it. Its members are declared in the order the
    stream runs through them, so its netlist places them in that order.
    """

    id = "finn.channel"

    tensor: Tensor = Param()
    # False admits no adapter: the two sides must connect directly.
    adaptable: bool = Param(default=True)
    port: str | None = Param(default=None)
    producer: Stream | None = Param(default=None, semantics=STREAM)
    consumer: Stream | None = Param(default=None, semantics=STREAM)
    producer_label: str | None = Param(default=None)
    consumer_label: str | None = Param(default=None)
    producer_width: int | None = Param(default=None)
    contents: IntegerTensor | None = Param(default=None, semantics=INTEGER_TENSOR)
    sets: int = Param(default=1)
    index_tensor: Tensor | None = Param(default=None)
    platform: Platform = Param()
    mover_offer: tuple[MoverOffer, ...] | None = Param(default=None)

    @derived
    def valued(self) -> bool:
        """Whether the channel carries a known value, which its source drives: without
        ``contents`` its source is not in play."""
        return self.contents is not None

    @derived
    def offered(self) -> bool:
        """Whether a shell offers movers on its free side: without the offers its mover is
        not in play."""
        return self.mover_offer is not None

    @derived
    def attached(self) -> tuple[Stream, ...]:
        """The streams given, the producer's first."""
        found: list[Stream] = []
        if self.producer is not None:
            found.append(self.producer)
        if self.consumer is not None:
            found.append(self.consumer)
        return tuple(found)

    @derived
    def consumed(self) -> Stream | Rejected:
        """The consuming port's stream: the order a source stores the value in."""
        if self.consumer is None:
            return reject("channel-source", "a known value streams to exactly one consumer")
        return self.consumer

    @derived
    def source_form(self) -> Traversal:
        """One period of the value in the order its consumer reads it."""
        return period(self.consumed.sequence.form)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def source_dtype(self) -> QONNXDataType:
        return self.tensor.element.dtype

    # Placed only where ``valued`` holds: its contents are given there.
    source = MemStreamKernel(
        when=valued,
        dtype=source_dtype,
        form=source_form,
        contents=cast("IntegerTensor", contents),
        sets=sets,
        set_tensor=index_tensor,
        platform=platform,
    )

    @derived(when=valued)
    def source_label(self) -> str:
        """Its source's label (``SOURCE_LABEL``): its producer's owner."""
        return SOURCE_LABEL

    # With several sets, its source's set port, a port of the channel labelled below the
    # source's label (``source.memstream.set``): the index tensor's channel's consumer. A
    # channel without a known value has no source, so no set port: nothing reads the
    # indices.
    set = served(source, "set", label=SOURCE_LABEL)

    @derived
    def sides(self) -> ChannelSides | Rejected:
        """The producing and consuming sides, without the stages between them: each the
        stream bound to it and its port's label; with a known value, the producer is the
        channel's source (its ``output_stream``); a side without a port is the boundary,
        what the module's AXIS port presents (``boundary``)."""
        if not self.attached and not self.valued:
            return reject("channel-unused", "no present port is on this channel")
        producer: tuple[str | None, Stream] | None = None
        consumer: tuple[str | None, Stream] | None = None
        # A side is given whole: a stream without its port's label (or a producer's
        # without its word width) would read as the boundary's.
        if self.producer is not None:
            if self.producer_label is None:
                return reject("channel-side", "a producer's stream comes with its port's label")
            if self.producer_width is None:
                return reject("channel-side", "a producer's stream comes with its word width")
            producer = (self.producer_label, self.producer)
        if self.consumer is not None:
            if self.consumer_label is None:
                return reject("channel-side", "a consumer's stream comes with its port's label")
            consumer = (self.consumer_label, self.consumer)
        if self.valued:
            if producer is not None:
                return reject(
                    "channel-users",
                    "a channel with a value has its source as its only producer, "
                    f"not {producer[0]}",
                )
            produced: Stream = self.source.output_stream
            producer = (self.source_label, produced)
        if producer is not None and consumer is not None:
            return ChannelSides(*producer, *consumer)
        if self.port is None:
            return reject(
                "channel-boundary",
                "a boundary of its network, but no port names it: only an ONNX input or output "
                "of a partition crosses its boundary",
            )
        # A side without a port is the boundary. Seen from inside, the module's input is
        # the AXIS target and its output the initiator.
        try:
            if consumer is not None:
                inside = self._boundary(consumer[1], Endpoint.TARGET)
                return ChannelSides(None, _stream(inside), *consumer)
            if producer is not None:
                inside = self._boundary(producer[1], Endpoint.INITIATOR)
                return ChannelSides(*producer, None, _stream(inside))
        except ValueError as error:
            return reject("channel-boundary", str(error))
        return reject("channel-unused", "no present port is on this channel")

    def _boundary(self, inside: Stream, endpoint: Endpoint) -> StreamContract:
        """The AXIS port the module presents for a side without a port, from what the
        other side presents (``inside``): an input without the replay or window its
        receiver realizes (``unreplayed``), an output as produced, neither with markers."""
        form = inside.sequence.form
        if endpoint is Endpoint.TARGET:
            form = unreplayed(form)
        # Read past ``sides``' test of the port: given here.
        port = cast(str, self.port)
        return boundary_contract(port, self.tensor.element, BeatSequence(form), endpoint)

    @derived(semantics=BOUNDARY_CONTRACT)
    def boundary(self) -> StreamContract | None:
        """The AXIS port the module presents for the side without a port
        (``boundary_contract``: ``port``, the tensor's element, and what the other side
        presents); None where both sides have a port."""
        found = self.sides
        if found.source_owner is None:
            return self._boundary(found.sink, Endpoint.TARGET)
        if found.sink_owner is None:
            return self._boundary(found.source, Endpoint.INITIATOR)
        return None

    @derived
    def free_side(self) -> StreamContract | Rejected:
        """The side without a port, as the module presents it (``boundary``): where a
        shell's mover sits."""
        found: StreamContract | None = self.boundary
        if found is None:
            return reject(
                "channel-mover",
                "a mover sits on a boundary's free side, and both sides have a port",
            )
        return found

    # Placed only where ``offered`` holds: its offers are given there.
    mover = IodmaMover(
        when=offered, offers=cast("tuple[MoverOffer, ...]", mover_offer), side=free_side
    )
    mover_contract = View(mover.contract)

    @derived
    def placed_mover(self) -> tuple[MoverContract, ...]:
        """Its mover's contract (the mover's own derived value), or none without a
        mover."""
        if not self.offered:
            return ()
        contract: MoverContract = self.mover.contract
        return (contract,)

    @derived
    def streams(self) -> Streams:
        """The two sides' streams: element and beat sequence."""
        found = self.sides
        return Streams(found.source, found.sink)

    @constraint
    def well_formed(self) -> bool | Rejected:
        """Each side's stream traverses this channel's tensor
        (``finn.dataflow.streams.misfit``)."""
        found = self.sides
        why = misfit(self.tensor, self.streams, (found.source_owner, found.sink_owner))
        return True if why is None else reject("channel-tensor", why)

    @derived
    def steps(self) -> Steps | Rejected:
        """What must happen between the source's beat sequence and the sink's."""
        streams = self.streams
        try:
            return between(streams.source.sequence, streams.sink.sequence)
        except Unrealizable as error:
            return reject("channel-plan", f"no adapter can join the streams: {error}")

    @derived
    def output_steps(self) -> Steps:
        """The steps before the transport (``Steps.output``)."""
        return self.steps.output

    @derived
    def input_steps(self) -> Steps:
        """The steps after the transport (``Steps.input``)."""
        return self.steps.input

    @derived
    def output_adapting(self) -> bool:
        return self.adaptable and bool(self.output_steps)

    @derived
    def adapting(self) -> bool:
        return self.adaptable and bool(self.input_steps)

    @constraint
    def realizable(self) -> bool | Rejected:
        found = self.steps
        if found and not self.adaptable:
            return reject(
                "channel-plan",
                f"the streams need {found.describe()}, and this channel admits no adapter",
            )
        return True

    output_adapter = OUTPUT_ADAPTER(
        when=output_adapting, tensor=tensor, steps=output_steps, platform=platform
    )
    output_adapter_stages = View(output_adapter.stages)

    @derived
    def arriving(self) -> BeatSequence:
        """What arrives at the transport: what the output adapter's last stage presents
        (the last ``sink`` of its realization), or without one what the producer
        presents."""
        if not self.output_adapting:
            return self.sides.source.sequence
        realization: tuple[RealizedStage, ...] = self.output_adapter.realization
        return realization[-1].sink

    # ``direct`` places nothing: the hop across the transport connects directly.
    transport: ChannelFifo | None = Select(
        direct=None,
        fifo=ChannelFifo(tensor=tensor, arriving=arriving, platform=platform),
    )

    @derived
    def input_chain(self) -> str | Rejected:
        """The chain its input adapter places (an ``ADAPTERS`` key): the kinds of the
        stages that carry out the steps after the transport (``realize``), joined, or
        ``none`` where it does not adapt. Refused where the steps are (``channel-plan``),
        or where no FinnLib chain carries them out (``adapter-plan``)."""
        if not self.adapting:
            return "none"
        try:
            realization = realize(self.input_steps)
        except Unrealizable as error:
            return reject("adapter-plan", f"no FinnLib chain carries out the plan: {error}")
        return "_".join(stage.kind for stage in realization)

    @derived
    def adapter_placed(self) -> bool:
        """Whether its input adapter places a chain: ``input_chain`` names one."""
        return self.input_chain != "none"

    # The one memory style of its input adapter's generators, present where it places a
    # chain; given to every chain, which gives it to each of its input generators.
    adapter_ram_style: str = Decision(domain=memory_styles(), when=adapter_placed)
    adapter_uram_present = uram_constraint(
        platform, adapter_ram_style, ("ultra",), when=adapter_placed
    )

    adapter: InputAdapter | None = Select(
        by=input_chain,
        none=None,
        **ADAPTERS,
        tensor=tensor,
        steps=input_steps,
        platform=platform,
        ram_style=adapter_ram_style,
    )
    adapter_stages = View(adapter.stages)  # type: ignore[union-attr]

    @view(when=adapter_placed)
    def adapter_ram_preference(self) -> tuple[str, ...]:
        """Its adapter's memory styles in its preference for the most bits one of its
        input generators holds a frame (the picked chain's ``frame_bits``,
        ``memory_preference``): the ranking a first valid policy may follow."""
        adapter = self.adapter
        bits: int = 0 if adapter is None else adapter.frame_bits
        return memory_preference(bits)

    @derived
    def staged(self) -> tuple[StageStreams, ...]:
        """Each stage's streams and output word width (each holder's ``stage_streams``,
        its stage kernels' derived values), in the order the stream runs through them: the
        output adapter's, the transport's FIFO, if any, then the input adapter's, each
        labelled as its ``Stage`` is in ``stages``."""
        found: list[StageStreams] = []
        if self.output_adapting:
            found += _below("output_adapter", self.output_adapter.stage_streams)
        fifo = self.transport
        if fifo is not None:
            found += _below("transport", fifo.stage_streams)
        adapter = self.adapter
        if adapter is not None:
            found += _below("adapter", adapter.stage_streams)
        return tuple(found)

    @view
    def stages(self) -> tuple[Stage, ...]:
        """Its stages' modules and ports' contracts (each holder's ``stages``), in the order
        the stream runs through them, labelled below the channel: the product its netlist
        and its network's ``outer_hops`` read to wire them."""
        found: list[Stage] = []
        if self.output_adapting:
            found += _below_stages("output_adapter", self.output_adapter.stages)
        fifo = self.transport
        if fifo is not None:
            found += _below_stages("transport", fifo.stages)
        adapter = self.adapter
        if adapter is not None:
            found += _below_stages("adapter", adapter.stages)
        return tuple(found)

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop's two streams, source through every stage to sink, connect directly
        (``compatibility``); and on a boundary output, the last producer's word (a stage's,
        the producing port's, ``producer_width``, or the source's) is no wider than the
        AXIS word that carries it (``channel-padding``).

        An element mismatch is ``well_formed``'s (``channel-tensor``: each side
        against the tensor); ``compatibility`` does not compare elements.
        """
        sides = self.sides
        found: list[Mismatch] = []
        current = sides.source
        width: int | None = None
        for stage in self.staged:
            found += compatibility(current, stage.input)
            current, width = stage.output, stage.width
        found += compatibility(current, sides.sink)
        boundary: StreamContract | None = self.boundary
        if sides.sink_owner is None and boundary is not None:
            if width is None and sides.source_owner is not None:
                width = self.source.output_data_width if self.valued else self.producer_width
            if width is not None and width > boundary.transport.data_width:
                found.append(
                    Mismatch(
                        "channel-padding",
                        "a child's padding is wider than the top word that must carry it",
                    )
                )
        return _refusal(found)

    # -- its answers: the base's, with its own hops, and none of a channel of wires' cycles --

    @view(semantics=NETLIST_SEMANTICS)
    @declared_reads(
        "netlists",
        "controls",
        "sides",
        "boundary",
        "valued",
        "source_label",
        "source.output_contract",
        "stages",
    )
    def netlist(self) -> Fragment | Rejected:
        """Its members' netlists under their nodes, in the order the stream runs through
        them (its source, its output adapter's stages, its FIFO, its input adapter's
        stages), its source and its output adapter at their labels (``SOURCE_LABEL``,
        ``OUTPUT_ADAPTER_LABEL``), and its own hops, as wires (``wired``): each hop both of
        whose ends are its own (its source, below it at its label; its stages, at theirs;
        its boundary, the module's own pins), from its children's contracts (its
        ``stages`` and its source's). Its hops to the ports beside it are its network's
        (``outer_hops``); a channel of wires between two kernels places nothing."""
        sides = self.sides
        boundary: StreamContract | None = self.boundary
        current: _End | None = None
        if self.valued:
            offered: PortContract = self.source.output_contract
            current = (self.source_label, offered.contract)
        elif sides.source_owner is None and boundary is not None:
            current = (None, boundary)
        stages: tuple[Stage, ...] = self.stages
        links: list[Link] = []
        try:
            for stage in stages:
                if current is not None:
                    links.append(wired(*current, stage.label, stage.input))
                current = (stage.label, stage.output)
            if current is not None and sides.sink_owner is None and boundary is not None:
                links.append(wired(*current, None, boundary))
        except ArtifactError as error:
            # A hop that does not connect (another element, say) is refused by the
            # channel's constraints; its wires do not exist.
            return reject("channel-link", str(error))
        return KernelSpace._netlist(self, tuple(links), _PLACED)

    @view
    @declared_reads("member_cycles")
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through its own members: the slowest's (its
        source's, its adapters' stages', its mover's), one beat a cycle at best; a FIFO
        states none, and a channel of wires takes none of its own."""
        return KernelSpace._cycles(self, default=0)

    @view
    def boundary_bus(self) -> tuple[Bus, ...]:
        """The AXIS bus the module presents for it (``boundary``), when one side has no
        port."""
        found: StreamContract | None = self.boundary
        return () if found is None else (found.transport.axis_bus(),)

    mover_export = View(placed_mover)

    exports = {
        NETLIST: netlist,
        BOUNDARY: boundary_bus,
        CYCLES: cycles,
        BUFFERING: KernelSpace.buffering,
        RESOURCES: KernelSpace.resources,
        MOVER: mover_export,
    }


def _stream(contract: StreamContract) -> Stream:
    """The stream a contract states: its element and its beat sequence."""
    return Stream(contract.element, contract.sequence)


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


__all__ = [
    "OUTPUT_ADAPTER_LABEL",
    "SOURCE_LABEL",
    "Channel",
    "ChannelFifo",
    "ChannelSides",
    "Neighbours",
    "OuterHops",
    "boundary_contract",
    "outer_hops",
    "wired",
]
