# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Channels: one tensor from one producer to one consumer, an ordinary Space kernels reference.

A ``Channel`` is one real producer-to-consumer edge, a node of its own,
declared in the kernel with children (or the root) that owns the edge, beside
the kernels it joins, carrying one ``tensor`` its owner supplies: stated, or
read from a kernel's fact-level view (``MatMulKernel.weight_tensor``). A kernel
has one reference input per channel it sits on (``output_channel: Channel =
Param()``), bound to the ``channel`` of one of its ports
(``finn.kernels.port``), which exports its contract under ``PORT``; a kernel
with children passes the reference down to the child that uses it, so the
channel's user is always a leaf's port, at any depth. Each end presents its
own traversal of the tensor in that contract, reading only the channel's
``tensor``: a tensor derived from an end is a dependency cycle, but a kernel's
fact-level view (its facts and value choices, no port) is not one. The
channel sees the ports that reference it through ``users = Users(PORT)``,
each with only the contract it presents on this channel, so a port's refusal
stays on its own channel. The contract's transport endpoint says whether the
kernel produces into the channel (initiator) or consumes from it (target).

Its ``ends`` are the contracts' logical part (``finn.dataflow.ends``): each
must traverse the tensor (``well_formed``, ``channel-tensor``), and the
``plan`` between their beat sequences (``finn.dataflow.plan``) must be one a
chain of steps carries out (``channel-plan``). A channel runs, in order:

    source -> output adapter -> transport -> input adapter -> consumer

The plan splits at the transport. Its output side (width conversions) opens
the ``output_adapter`` Decision over nodes, its input side (a reorder, replay
included, and markers) the ``adapter`` Decision; the candidates of each are
fixed chains of FinnLib modules (``finn.kernels.adapters``), each refusing a
plan it does not carry out, so at most one survives a side. A channel whose
``adaptable`` input is False admits no adapter and refuses any plan.

A channel with a user on one side only is a boundary of the root that
declares it. Its ``port`` input names the top-level AXIS port (``in0_V``): an
ABI name is design data of the channel, independent of its node name, which
is its identity and the prefix of its persisted decision keys. A boundary
whose ``port`` is not supplied is refused (``channel-boundary``): a root names
a port only for what may cross it, an ONNX input or output of a partition.
The boundary presents what its internal end presents, by one rule: an input
boundary without the replay its receiver realizes (``unreplayed``), an output
boundary as produced, neither with markers and both as a single pass.

Every channel, boundaries included, owns a ``transport`` Decision over two
nodes between its adapters: ``direct`` or ``fifo``. The FIFO candidate owns its
``depth``, a Decision known by its domain's membership only, which a design
space exploration proposes, and the FIFO's ``ram_style``; it is an identity
stage presenting what arrives at it, inside the root that declares the channel
(on a boundary, between the root's pins and its user). A FIFO carries data
words only: where what arrives carries markers (a producer whose own output
carries them; a width conversion drops them), the FIFO refuses itself
(``fifo-markers``). The adapters' and transport's stages are checked on both
sides, on the channel's ``platform``. Their keys are the channel's
(``x.output_adapter``, ``x.transport``, ``x.adapter``), so they belong to
whoever owns the edge.

A channel whose tensor has a known value (``contents``, one operand per set;
with several ``sets``, its ``index`` channel selects one) carries a
``source`` Decision over the kernels that can drive it with that value
(``SOURCES``: a memory). It applies only when the value is known
(``valued``), and it has no ``none`` case: one viable candidate is forced. The
source is the channel's producer end, placed by the channel (``staged``), its
leaf below the channel at ``source.<case>``; it stores one period of the value
in the order the channel's consumer reads it. The invariant: **a channel with
a value has its source as its only producer** (``channel-users``).

Beside ``compatible``, each checked hop is resolved into wires (``wired``),
and the channel exports its netlist under ``NETLIST``: its stages' leaves at
their labels below it (``output_adapter.vpc.vpc``, ``transport.fifo.buffer``,
``adapter.input_gen.input_gen``) and its hops. A user's end belongs to the
kernel whose ``Port`` it is, beside the channel (``^compute.packed``); a
boundary end is the root's own pins. Under ``BOUNDARY`` it exports the AXIS bus
its boundary side presents, if any.

A channel refuses itself as a kernel does, through its ``admission``
(``well_formed``, ``realizable``, ``compatible``). It exports its cost, which
waits on no admission and reads no memory style: under ``CYCLES`` the clock
cycles a frame takes through its own stages (the most beats its source's or an
adapter's port carries a frame; a FIFO presents what arrives, so it adds none,
and a channel of wires takes none of its own), under ``BUFFERING`` the bits its
stages hold (an adapter's ``input_gen`` buffers, a FIFO's depth).

The Space class refers to itself (``index``) and, through its source's port, is
referred to by ``finn.kernels.port`` and ``finn.kernels.memstream``, which
this module imports: those annotate with ``channels.Channel`` and the Space
engine resolves the reference when it collects the Space class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    Users,
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
from finn.dataflow.ends import End, Ends, misfit
from finn.dataflow.plan import Plan, Unrealizable, plan
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, period, unreplayed
from finn.kernels.adapters import ADAPTERS, OUTPUT_ADAPTERS, Stage, StreamAdapter
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import BuildError, Fragment, Leaf, Link, LinkEnd, Marker
from finn.kernels.base import BOUNDARY, BUFFERING, CLOCK, CYCLES, NETLIST, PORT, RESET
from finn.kernels.fifo import FifoKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.target import Platform
from finn.kernels.transport import (
    AxisBeat,
    Level,
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

SOURCES: dict[str, type[Space] | Space] = {"memstream": MemStreamKernel}
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


def _end(instance: str | None, contract: StreamContract) -> LinkEnd:
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
        _end(source, produced),
        _end(sink, consumed),
        produced.element.bits,
        lane_permutation(produced, consumed),
        tuple(markers),
    )


class _Direct(Space):
    """No stage: the hop across the transport connects directly."""

    @view
    def stages(self) -> tuple[Stage, ...]:
        return ()

    @view
    def held_bits(self) -> int:
        return 0


class ChannelFifo(Space):
    """An identity stage: what arrives at it, presented on both sides of a native FIFO."""

    tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    platform: Platform = Param()

    @derived
    def word_bits(self) -> int:
        return self.arriving.form.lanes * self.tensor.element.bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(
            domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32, ordered=True)
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

    # The candidate's refusal, which the transport Decision's viability reads.
    admission = ConstraintGroup(carries_data)

    @view
    def held_bits(self) -> int:
        """Its depth of words."""
        return self.buffer.depth * self.word_bits

    @view
    def stages(self) -> tuple[Stage, ...]:
        element, arriving, buffer = self.tensor.element, self.arriving, self.buffer
        module = buffer.module
        assert isinstance(module, Leaf)
        return (
            Stage(
                module,
                StreamContract(buffer.input.transport, element, arriving.form, arriving.repetition),
                StreamContract(
                    buffer.output.transport, element, arriving.form, arriving.repetition
                ),
                "fifo.buffer",
            ),
        )


@dataclass(frozen=True)
class ChannelEnds:
    """A channel's two ends: each end's owner (the user's port node, None at the root's
    boundary) and its contract."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract


class Channel(Space):
    """One tensor between the kernels that reference it: one producer, one consumer.

    ``users`` holds the port each present user presents on this channel,
    located by the user's name and the input it references this channel
    through. A side without a user is the root's boundary, presented as the
    AXIS port ``port``. A channel with a known value has its ``source`` as its
    producer. ``platform`` is the target's, stated by whoever declares the
    channel: its source, its adapter's stages and its FIFO read it.
    """

    tensor: Tensor = Param()
    # False admits no adapter: the ends must connect directly.
    adaptable: bool = Param(default=True)
    port: str = Param(required=False)
    users = Users(PORT)
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    sets: int = Param(default=1)
    index: Channel = Param(required=False)
    platform: Platform = Param()
    # Whether the channel carries a known value, which its source drives: where nothing
    # supplies ``contents``, the source never applies and its candidates are not compiled.
    valued = supplied(contents)

    @derived
    def consumed(self) -> StreamContract | Rejected:
        """The consuming user's contract: the order a source stores the value in."""
        consumers = [
            end.value
            for end in self.users
            if end.value.transport.endpoint is not Endpoint.INITIATOR
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
        set_channel=index,
        platform=platform,
        staged=True,
    )
    source_case = selected(source)
    source_module = View(source.module)

    @derived
    def source_label(self) -> str:
        return f"source.{self.source_case}"

    @derived
    def source_contract(self) -> StreamContract:
        contract: StreamContract = self.source.output.contract
        return contract

    @derived
    def endpoints(self) -> ChannelEnds | Rejected:
        """The producing and consuming ends, without the stages between them: with a
        known value, the producer is the channel's source."""
        producers: list[tuple[str | None, StreamContract]] = []
        consumers: list[tuple[str | None, StreamContract]] = []
        for end in self.users:
            contract = end.value
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((end.node, contract))
        if self.valued:
            if producers:
                return reject(
                    "channel-users",
                    "a channel with a value has its source as its only producer, "
                    f"not {producers[0][0]}",
                )
            producers.append((self.source_label, self.source_contract))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(f"{end.node}.{end.member}" for end in self.users)
            return reject(
                "channel-users",
                f"a channel has at most one producer and one consumer; referenced by {named}",
            )
        if not producers and not consumers:
            return reject("channel-unused", "no present kernel references this channel")
        if not (producers and consumers) and not self.present(Channel.port):
            return reject(
                "channel-boundary",
                "a boundary of its root, but no port names it: only an ONNX input or output "
                "of a partition crosses its boundary",
            )
        try:
            # A side without a user is the boundary. Seen from inside, the
            # root's input is the AXIS target and its output the initiator.
            if not producers:
                inside = consumers[0][1]
                producers.append((None, self._boundary(inside, Endpoint.TARGET)))
            if not consumers:
                inside = producers[0][1]
                consumers.append((None, self._boundary(inside, Endpoint.INITIATOR)))
        except ValueError as error:
            return reject("channel-boundary", str(error))
        return ChannelEnds(*producers[0], *consumers[0])

    def _boundary(self, inside: StreamContract, endpoint: Endpoint) -> StreamContract:
        form = unreplayed(inside.form) if endpoint is Endpoint.TARGET else inside.form
        return boundary_contract(self.port, self.tensor.element, BeatSequence(form), endpoint)

    @derived
    def ends(self) -> Ends:
        """The logical part of the two contracts: element and beat sequence."""
        found = self.endpoints
        return Ends(
            End(found.source_owner, found.source.element, found.source.sequence),
            End(found.sink_owner, found.sink.element, found.sink.sequence),
        )

    @constraint
    def well_formed(self) -> bool | Rejected:
        """Each end traverses this channel's tensor (``finn.dataflow.ends.misfit``)."""
        why = misfit(self.tensor, self.ends)
        return True if why is None else reject("channel-tensor", why)

    @derived
    def plan(self) -> Plan | Rejected:
        """What must happen between the source's beat sequence and the sink's."""
        ends = self.ends
        try:
            return plan(ends.source.sequence, ends.sink.sequence)
        except Unrealizable as error:
            return reject("channel-plan", f"no adapter can join the ends: {error}")

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
                f"the ends need {found.describe()}, and this channel admits no adapter",
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
    output_adapter_beats = View(output_adapter.beats)
    output_adapter_held = View(output_adapter.held_bits)

    adapter: StreamAdapter = Decision(
        ADAPTERS, when=adapting, tensor=tensor, plan=input_plan, platform=platform
    )
    adapter_stages = View(adapter.stages)
    adapter_beats = View(adapter.beats)
    adapter_held = View(adapter.held_bits)

    @derived
    def output_adapted(self) -> tuple[Stage, ...]:
        """The output adapter's stages, in order, labelled below the channel; none when
        the plan converts no width before the transport."""
        if not self.output_adapting:
            return ()
        return tuple(
            replace(stage, label=f"output_adapter.{stage.label}")
            for stage in self.output_adapter_stages
        )

    @derived
    def adapted(self) -> tuple[Stage, ...]:
        """The input adapter's stages, in order, labelled below the channel; none when
        nothing is left to do after the transport."""
        if not self.adapting:
            return ()
        return tuple(
            replace(stage, label=f"adapter.{stage.label}") for stage in self.adapter_stages
        )

    @derived
    def arriving(self) -> BeatSequence:
        """What arrives at the transport: the beat sequence after the output adapter."""
        adapted = self.output_adapted
        output = adapted[-1].output if adapted else None
        return self.endpoints.source.sequence if output is None else output.sequence

    transport: _Direct | ChannelFifo = Decision(
        {
            "direct": _Direct,
            "fifo": ChannelFifo(tensor=tensor, arriving=arriving, platform=platform),
        }
    )
    transport_stages = View(transport.stages)

    transport_held = View(transport.held_bits)

    @view
    def stages(self) -> tuple[Stage, ...]:
        """The output adapter's stages, the transport's FIFO, if any, then the input
        adapter's stages."""
        transported = tuple(
            replace(stage, label=f"transport.{stage.label}") for stage in self.transport_stages
        )
        return (*self.output_adapted, *transported, *self.adapted)

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop, source through every stage to sink, connects directly.

        An element mismatch is ``well_formed``'s (``channel-tensor``: each end
        against the tensor); ``compatibility`` does not compare elements.
        """
        ends = self.endpoints
        found: list[Mismatch] = []
        current, current_top = ends.source, ends.source_owner is None
        for stage in self.stages:
            found += compatibility(
                current, stage.input, source_is_top=current_top, sink_is_top=False
            )
            current, current_top = stage.output, False
        found += compatibility(
            current, ends.sink, source_is_top=current_top, sink_is_top=ends.sink_owner is None
        )
        if ends.sink_owner is None and not current_top:
            if current.transport.data_width > ends.sink.transport.data_width:
                found.append(
                    Mismatch(
                        Level.PHYSICAL,
                        "channel-padding",
                        "a child's padding is wider than the top word that must carry it",
                    )
                )
        return _refusal(found)

    @derived
    def hops(self) -> tuple[Link, ...] | Rejected:
        """Each hop, source through every stage to sink, as wires.

        A user's end belongs to its kernel beside the channel: the user is a
        kernel's ``Port`` (``compute.packed.x``, at any depth below the
        channel's owner), its kernel the node above it (``^compute.packed``).
        A stage and the source sit below the channel at their labels; a
        boundary end is the root's own pins (``None``).
        """
        ends = self.endpoints
        owners: list[str | None] = []
        for side, node in enumerate((ends.source_owner, ends.sink_owner)):
            if side == 0 and self.valued:
                owners.append(node)  # the source, below the channel at its label
                continue
            kernel = None if node is None else node.rpartition(".")[0]
            if kernel == "":
                return reject("channel-user", f"{node} presents an end, but is no kernel's port")
            owners.append(None if kernel is None else "^" + kernel)
        links: list[Link] = []
        owner, current = owners[0], ends.source
        try:
            for stage in self.stages:
                links.append(wired(owner, current, stage.label, stage.input))
                owner, current = stage.label, stage.output
            links.append(wired(owner, current, owners[1], ends.sink))
        except BuildError as error:
            # A hop that does not connect (another element, say) is refused by the
            # channel's constraints; its wires do not exist.
            return reject("channel-link", str(error))
        return tuple(links)

    @view(requires=(well_formed, realizable, compatible))
    def netlist(self) -> Fragment:
        """Its source's and stages' leaves below it, and its hops."""
        stages = tuple((stage.label, stage.module) for stage in self.stages)
        if self.valued:
            module = self.source_module
            assert isinstance(module, Leaf)
            stages = ((self.source_label, module), *stages)
        return Fragment(stages, self.hops)

    admission = ConstraintGroup(well_formed, realizable, compatible)

    # -- what a design space exploration ranks by ------------------------------------------

    @derived
    def frame_cycles(self) -> int:
        """The clock cycles a frame takes through its own stages: the most beats its
        source's port or an adapter's carries a frame (one beat a cycle at best)."""
        found = [0]
        if self.valued:
            found.append(self.source_contract.form.beats)
        if self.output_adapting:
            found.append(self.output_adapter_beats)
        if self.adapting:
            found.append(self.adapter_beats)
        return max(found)

    @derived
    def held_bits(self) -> int:
        """The bits its stages hold between its ends: the adapters' buffers, a FIFO's
        depth."""
        held: int = self.transport_held
        if self.output_adapting:
            held += self.output_adapter_held
        if self.adapting:
            held += self.adapter_held
        return held

    # Cost, not validity: read apart from the channel's admission.
    cycles = View(frame_cycles)
    buffering = View(held_bits)

    @view(requires=(well_formed, realizable, compatible))
    def boundary_bus(self) -> tuple[Bus, ...]:
        """The AXIS bus the root presents for it, when one side has no user."""
        ends = self.endpoints
        sides = ((ends.source_owner, ends.source), (ends.sink_owner, ends.sink))
        return tuple(contract.transport.axis_bus() for owner, contract in sides if owner is None)

    exports = {NETLIST: netlist, BOUNDARY: boundary_bus, CYCLES: cycles, BUFFERING: buffering}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


__all__ = [
    "SOURCES",
    "Channel",
    "ChannelEnds",
    "ChannelFifo",
    "boundary_contract",
    "wired",
]
