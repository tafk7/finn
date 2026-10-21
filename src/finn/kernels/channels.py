# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels: one tensor from one producer to one consumer, a kernel with children between
two ports.

A ``Channel`` is one producer-to-consumer edge, a member of a network
(``finn.kernels.network``) or of a kernel with children, carrying one ``tensor`` given as
a fact. The contracts of the ports on it are facts too, ``producer`` and ``consumer``: each
the ``PortContract`` a member presents for one of its ports (a kernel's ``x_contract``),
labelled from the network (``Interface``: ``matmul.compute.packed.x``), or absent: the
network's boundary. The channel's tensor is never derived from a port.

Each side's stream must traverse the tensor (``channel-tensor``), and the plan between
their beat sequences (``finn.dataflow.plan``) must be one a chain of FinnLib modules
carries out (``channel-plan``). A channel runs, in order:

    source -> output adapter -> transport -> input adapter -> consumer

The plan splits at the transport: its output side (width conversions) is the
``output_adapter`` Decision, its input side (a reorder, replay included, and markers) the
``adapter`` Decision, each over fixed chains (``finn.kernels.adapters``). The
``transport`` Decision is ``direct`` or ``fifo``; the FIFO owns its ``depth`` (required:
an exploration sizes it) and its ``ram_style``, and carries data words only
(``fifo-markers``). Its keys are the channel's, by its name (``("x", "transport")``).

A channel with a port on one side only is a boundary: its ``port`` names the module's
AXIS port (``in0_V``), refused where absent (``channel-boundary``). The boundary presents
what its inner side presents, by one rule: an input without the replay or window its
receiver realizes (``unreplayed``), an output as produced, neither with markers. Where a
shell offers movers (``mover_offer``, ``finn.kernels.movers``), the ``mover`` Decision places one
on that free side; it binds no RTL in the module and exports its contract (``MOVER``) for
the shell's budget check and resources.

A channel whose tensor has a known value (``contents``) has a ``source`` (``SOURCES``: a
memory) as its only producer (``channel-users``), storing one period in the order its
consumer reads it; with several sets, the source's set port is the channel's port
(``set``, served by its source: ``set_contract``), which the index tensor's channel
consumes.

As a kernel with children (a ``KernelSpace`` binding no module), its members are its
source, its adapters' stages, its FIFO (``ChannelFifo``) and its mover, in the order the
stream runs through them. Its answers are the base's views over theirs, but for two it
states itself, calling the base's: its netlist adds its hops (``NETLIST``), and a channel
of wires takes no cycles of its own (0). It exports its boundary's AXIS bus
(``BOUNDARY``) and its mover's contract (``MOVER``: ``placed_mover``, from the mover's own
derived ``contract``). It is no module of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    View,
    constraint,
    derived,
    domain,
    reject,
    selected,
    supplied,
    view,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.plan import Plan, Unrealizable, plan
from finn.dataflow.streams import Stream, Streams, misfit
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, period, unreplayed
from finn.kernels.adapters import ADAPTERS, OUTPUT_ADAPTERS, Stage, StreamAdapter
from finn.kernels.artifacts import ArtifactError
from finn.kernels.artifacts.abi import CLOCK, RESET, Bus, Endpoint
from finn.kernels.artifacts.module import Fragment, Link, LinkEnd, Marker
from finn.kernels.base import (
    BOUNDARY,
    BUFFERING,
    CYCLES,
    NETLIST,
    NETLIST_SEMANTICS,
    PORT_CONTRACT,
    RESOURCES,
    KernelSpace,
    PortContract,
    declared_reads,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.library import library_module
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.movers import MOVER, MOVERS, IodmaMover, MoverContract, MoverOffer
from finn.kernels.port import served
from finn.kernels.transport import (
    AxisBeat,
    Mismatch,
    StreamContract,
    compatibility,
    lane_permutation,
    marker_bit,
    marker_pairs,
)
from finn.kernels.values.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.platform.capabilities import Platform

SOURCES: dict[str, type[MemStreamKernel]] = {"memstream": MemStreamKernel}
"""The kernels that can drive a channel with its known value, by case: a memory."""


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


class _Direct(Space):
    """No stage: the hop across the transport connects directly. It places nothing, so
    it exports nothing its channel sums."""

    @view
    def stages(self) -> tuple[Stage, ...]:
        return ()


class ChannelFifo(KernelSpace):
    """An identity stage: what arrives at it, presented on both sides of a native FIFO.

    Its netlist, buffering and resources are its FIFO's, by the base. It exports no
    cycles: a FIFO presents what arrives, whose beats are counted where they are made
    (the channel's producer, its source, its output adapter), so its channel's cycles
    are its other members'. It presents no port: its channel wires it hop by hop
    (``stages``: its FIFO's module, read from the FIFO's netlist, and its ports'
    contracts).
    """

    id = "finn.channel.fifo"
    version = 1

    tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    platform: Platform = Param()

    @derived
    def carried(self) -> BeatSequence:
        """What its FIFO carries: what arrives, its data words only (a marker closing every
        beat, which it may carry, is tied high after it)."""
        arriving = self.arriving
        return BeatSequence(arriving.form, arriving.repetition)

    buffer = FifoKernel(
        input_tensor=tensor,
        arriving=carried,
        # Required: no policy gives a depth; a FIFO is chosen by sizing it (SizeFifos).
        depth=Decision(
            domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32, ordered=True),
            required=True,
        ),
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
    the network, ``matmul.compute.packed.x``; the channel's source label, ``source.<case>``,
    for a known value; None on the boundary) and its contract."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract


class Channel(KernelSpace):
    """One tensor between the ports whose contracts it takes: one producer, one consumer.

    ``producer`` and ``consumer`` are the contracts of the ports on it, facts its network
    gives (``Interface``). A side without a port is the boundary, presented as the AXIS
    port ``port``, where the ``mover`` the shell offers (``mover_offer``) sits. A channel
    with a known value has its ``source`` as its producer; with several sets, the
    source reads the set indices (``index_tensor``) from the channel whose consumer is
    ``set_contract``, the one port it presents (``set``, its source's). ``platform`` is the
    target's: its source, its adapters' stages and its FIFO read it. Its members are
    declared in the order the stream runs through them, so its netlist places them in that
    order.
    """

    id = "finn.channel"
    version = 1

    tensor: Tensor = Param()
    # False admits no adapter: the two sides must connect directly.
    adaptable: bool = Param(default=True)
    port: str = Param(required=False)
    producer: PortContract = Param(required=False, semantics=PORT_CONTRACT)
    consumer: PortContract = Param(required=False, semantics=PORT_CONTRACT)
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    sets: int = Param(default=1)
    index_tensor: Tensor = Param(required=False)
    platform: Platform = Param()
    mover_offer: tuple[MoverOffer, ...] = Param(required=False)
    # Whether the channel carries a known value, which its source drives: where nothing
    # supplies ``contents``, the source never applies and its candidates are not compiled.
    valued = supplied(contents)
    # Whether a shell offers movers on its free side: where nothing supplies the offers,
    # the mover never applies and its candidates are not compiled.
    offered = supplied(mover_offer)

    @derived
    def attached(self) -> tuple[PortContract, ...]:
        """The port contracts given, the producer's first."""
        found: list[PortContract] = []
        if self.present(Channel.producer):
            found.append(self.producer)
        if self.present(Channel.consumer):
            found.append(self.consumer)
        return tuple(found)

    @derived
    def consumed(self) -> StreamContract | Rejected:
        """The consuming port's contract: the order a source stores the value in."""
        consumers = [
            given.contract
            for given in self.attached
            if given.contract.transport.endpoint is not Endpoint.INITIATOR
        ]
        if len(consumers) != 1:
            return reject("channel-source", "a known value streams to exactly one consumer")
        return consumers[0]

    @derived
    def source_form(self) -> Traversal:
        """One period of the value in the order its consumer reads it."""
        return period(self.consumed.form)

    @derived(semantics=QONNX_DATATYPE_VALUE_SEMANTICS)
    def source_dtype(self) -> QONNXDataType:
        return self.tensor.element.dtype

    source: MemStreamKernel = Decision(
        SOURCES,
        when=valued,
        dtype=source_dtype,
        form=source_form,
        contents=contents,
        sets=sets,
        set_tensor=index_tensor,
        platform=platform,
    )
    source_case = selected(source)

    @derived
    def source_label(self) -> str:
        return f"source.{self.source_case}"

    # With several sets, its source's set port, a port of the channel labelled below it
    # (``source.memstream.set``): the index tensor's channel's consumer. A channel without
    # a known value has no source, so no set port: nothing reads the indices.
    set = served(source, "set")

    @derived
    def source_contract(self) -> StreamContract:
        offered: PortContract = self.source.output_contract
        return offered.contract

    @derived
    def sides(self) -> ChannelSides | Rejected:
        """The producing and consuming sides, without the stages between them: with a
        known value, the producer is the channel's source."""
        producers: list[tuple[str | None, StreamContract]] = []
        consumers: list[tuple[str | None, StreamContract]] = []
        for given in self.attached:
            contract = given.contract
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((given.label, contract))
        if self.valued:
            if producers:
                return reject(
                    "channel-users",
                    "a channel with a value has its source as its only producer, "
                    f"not {producers[0][0]}",
                )
            producers.append((self.source_label, self.source_contract))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(given.label for given in self.attached)
            return reject(
                "channel-users",
                f"a channel has at most one producer and one consumer; its ports are {named}",
            )
        if not producers and not consumers:
            return reject("channel-unused", "no present port is on this channel")
        if not (producers and consumers) and not self.present(Channel.port):
            return reject(
                "channel-boundary",
                "a boundary of its network, but no port names it: only an ONNX input or output "
                "of a partition crosses its boundary",
            )
        try:
            # A side without a port is the boundary. Seen from inside, the
            # module's input is the AXIS target and its output the initiator.
            if not producers:
                inside = consumers[0][1]
                producers.append((None, self._boundary(inside, Endpoint.TARGET)))
            if not consumers:
                inside = producers[0][1]
                consumers.append((None, self._boundary(inside, Endpoint.INITIATOR)))
        except ValueError as error:
            return reject("channel-boundary", str(error))
        return ChannelSides(*producers[0], *consumers[0])

    def _boundary(self, inside: StreamContract, endpoint: Endpoint) -> StreamContract:
        form = unreplayed(inside.form) if endpoint is Endpoint.TARGET else inside.form
        return boundary_contract(self.port, self.tensor.element, BeatSequence(form), endpoint)

    @derived
    def free_side(self) -> StreamContract | Rejected:
        """The side without a port, as the module presents it: where a shell's mover
        sits."""
        found = self.sides
        if found.source_owner is None:
            return found.source
        if found.sink_owner is None:
            return found.sink
        return reject(
            "channel-mover", "a mover sits on a boundary's free side, and both sides have a port"
        )

    mover: IodmaMover = Decision(MOVERS, when=offered, offers=mover_offer, side=free_side)
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
        """The logical part of the two sides' contracts: element and beat sequence."""
        found = self.sides
        return Streams(
            Stream(found.source.element, found.source.sequence),
            Stream(found.sink.element, found.sink.sequence),
        )

    @constraint
    def well_formed(self) -> bool | Rejected:
        """Each side's stream traverses this channel's tensor
        (``finn.dataflow.streams.misfit``)."""
        found = self.sides
        why = misfit(self.tensor, self.streams, (found.source_owner, found.sink_owner))
        return True if why is None else reject("channel-tensor", why)

    @derived
    def plan(self) -> Plan | Rejected:
        """What must happen between the source's beat sequence and the sink's."""
        streams = self.streams
        try:
            return plan(streams.source.sequence, streams.sink.sequence)
        except Unrealizable as error:
            return reject("channel-plan", f"no adapter can join the streams: {error}")

    @derived
    def output_plan(self) -> Plan:
        """The plan's side before the transport (``Plan.output``)."""
        return self.plan.output

    @derived
    def input_plan(self) -> Plan:
        """The plan's side after the transport (``Plan.input``)."""
        return self.plan.input

    @derived
    def output_adapting(self) -> bool:
        return self.adaptable and bool(self.output_plan)

    @derived
    def adapting(self) -> bool:
        return self.adaptable and bool(self.input_plan)

    @constraint
    def realizable(self) -> bool | Rejected:
        found = self.plan
        if found and not self.adaptable:
            return reject(
                "channel-plan",
                f"the streams need {found.describe()}, and this channel admits no adapter",
            )
        return True

    output_adapter: StreamAdapter = Decision(
        OUTPUT_ADAPTERS,
        when=output_adapting,
        tensor=tensor,
        plan=output_plan,
        platform=platform,
    )
    output_adapter_stages = View(output_adapter.stages)

    @derived
    def output_adapted(self) -> tuple[Stage, ...]:
        """The output adapter's stages (its ``stages``), in order, labelled below the
        channel; none when the plan converts no width before the transport."""
        if not self.output_adapting:
            return ()
        stages: tuple[Stage, ...] = self.output_adapter.stages
        return tuple(replace(stage, label=f"output_adapter.{stage.label}") for stage in stages)

    @derived
    def adapted(self) -> tuple[Stage, ...]:
        """The input adapter's stages (its ``stages``), in order, labelled below the
        channel; none when nothing is left to do after the transport."""
        if not self.adapting:
            return ()
        stages: tuple[Stage, ...] = self.adapter.stages
        return tuple(replace(stage, label=f"adapter.{stage.label}") for stage in stages)

    @derived
    def arriving(self) -> BeatSequence:
        """What arrives at the transport: the beat sequence after the output adapter."""
        adapted = self.output_adapted
        output = adapted[-1].output if adapted else None
        return self.sides.source.sequence if output is None else output.sequence

    transport: _Direct | ChannelFifo = Decision(
        {
            "direct": _Direct,
            "fifo": ChannelFifo(tensor=tensor, arriving=arriving, platform=platform),
        }
    )
    transport_stages = View(transport.stages)

    adapter: StreamAdapter = Decision(
        ADAPTERS, when=adapting, tensor=tensor, plan=input_plan, platform=platform
    )
    adapter_stages = View(adapter.stages)

    @derived
    def staged(self) -> tuple[Stage, ...]:
        """The output adapter's stages, the transport's FIFO, if any, then the input
        adapter's stages: each its child's ``stages``, labelled below the channel."""
        transport: tuple[Stage, ...] = self.transport.stages
        transported = tuple(replace(stage, label=f"transport.{stage.label}") for stage in transport)
        return (*self.output_adapted, *transported, *self.adapted)

    # The stages, the product its holder reads; ``compatible`` and ``hops`` read the value.
    stages = View(staged)

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop, source through every stage to sink, connects directly.

        An element mismatch is ``well_formed``'s (``channel-tensor``: each side
        against the tensor); ``compatibility`` does not compare elements.
        """
        sides = self.sides
        found: list[Mismatch] = []
        current, current_top = sides.source, sides.source_owner is None
        for stage in self.staged:
            found += compatibility(
                current, stage.input, source_is_top=current_top, sink_is_top=False
            )
            current, current_top = stage.output, False
        found += compatibility(
            current, sides.sink, source_is_top=current_top, sink_is_top=sides.sink_owner is None
        )
        if sides.sink_owner is None and not current_top:
            if current.transport.data_width > sides.sink.transport.data_width:
                found.append(
                    Mismatch(
                        "channel-padding",
                        "a child's padding is wider than the top word that must carry it",
                    )
                )
        return _refusal(found)

    @derived
    def hops(self) -> tuple[Link, ...] | Rejected:
        """Each hop, source through every stage to sink, as wires.

        A side belongs to its kernel beside the channel: its label is a
        kernel's port (``matmul.compute.packed.x``, at any depth below the
        network), its kernel the node above it (``^matmul.compute.packed``).
        A stage and the source sit below the channel at their labels; a
        boundary side is the module's own pins (``None``).
        """
        sides = self.sides
        owners: list[str | None] = []
        for side, node in enumerate((sides.source_owner, sides.sink_owner)):
            if side == 0 and self.valued:
                owners.append(node)  # the source, below the channel at its label
                continue
            kernel = None if node is None else node.rpartition(".")[0]
            if kernel == "":
                return reject(
                    "channel-port", f"{node} presents a contract, but is no kernel's port"
                )
            owners.append(None if kernel is None else "^" + kernel)
        links: list[Link] = []
        owner, current = owners[0], sides.source
        try:
            for stage in self.staged:
                links.append(wired(owner, current, stage.label, stage.input))
                owner, current = stage.label, stage.output
            links.append(wired(owner, current, owners[1], sides.sink))
        except ArtifactError as error:
            # A hop that does not connect (another element, say) is refused by the
            # channel's constraints; its wires do not exist.
            return reject("channel-link", str(error))
        return tuple(links)

    # -- its answers: the base's, with its hops, and none of a channel of wires' cycles ----

    @view(semantics=NETLIST_SEMANTICS)
    @declared_reads("netlists", "controls", "hops")
    def netlist(self) -> Fragment | Rejected:
        """Its members' netlists under their nodes, in the order the stream runs through
        them (its source, its output adapter's stages, its FIFO, its input adapter's
        stages), and its hops; a channel of wires places none."""
        return KernelSpace._netlist(self, self.hops)

    @view
    @declared_reads("member_cycles")
    def cycles(self) -> int | Rejected:
        """The clock cycles a frame takes through its own members: the slowest's (its
        source's, its adapters' stages', its mover's), one beat a cycle at best; a FIFO
        states none, and a channel of wires takes none of its own."""
        return KernelSpace._cycles(self, default=0)

    @view
    def boundary_bus(self) -> tuple[Bus, ...]:
        """The AXIS bus the module presents for it, when one side has no port."""
        found = self.sides
        sides = ((found.source_owner, found.source), (found.sink_owner, found.sink))
        return tuple(contract.transport.axis_bus() for owner, contract in sides if owner is None)

    mover_export = View(placed_mover)

    exports = {
        NETLIST: netlist,
        BOUNDARY: boundary_bus,
        CYCLES: cycles,
        BUFFERING: KernelSpace.buffering,
        RESOURCES: KernelSpace.resources,
        MOVER: mover_export,
    }


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


__all__ = [
    "SOURCES",
    "Channel",
    "ChannelFifo",
    "ChannelSides",
    "boundary_contract",
    "wired",
]
