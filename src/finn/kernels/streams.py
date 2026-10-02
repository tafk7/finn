# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces that kernels reference.

A ``Stream`` is the physical form of the logical stream
(``finn.dataflow.stream``): a node of its own, declared in the composite
beside the kernels it joins, carrying one ``tensor`` supplied by the
composite. A kernel has one reference input per stream it sits on
(``output_stream: Stream = Param()``) and exports, under ``PORT``, one
contract per input: ``exports = {PORT: {output_stream: output_port}}``. Each
end presents its own traversal of the tensor in that contract, reading only
the stream's ``tensor``. The stream sees the kernels that reference it through
``users = Users(PORT)``, each with only the port it presents on this stream,
so a port's refusal stays on its own stream. The contract's transport
endpoint says whether the kernel produces into the stream (initiator) or
consumes from it (target). The one-producer-one-consumer rule, compatibility
and the AXIS boundary belong to this family, not to the engine; the tensor
each end must traverse and the plan are the logical stream's.

A stream with a user on one side only is a boundary of its composite. Its
``port`` input names the top-level AXIS port (``in0_V``): an ABI name is
design data of the stream, independent of the stream's node name, which is
its identity and the prefix of its persisted decision keys. The boundary
presents what its internal end presents, by one rule: an input boundary
without the replay its receiver realizes (``unreplayed``), an output boundary
as produced, neither with markers and both as a single pass.

The logical stream derives the ``plan`` between the two ends' beat
sequences and refuses one it cannot carry out (``stream-plan``); ``ends`` is
the contracts' logical part. A non-empty plan opens the stream's ``adapter``
Decision over nodes, whose candidates are fixed chains of FinnLib modules
(``finn.kernels.adapters``); each refuses a plan it does not carry out, so at
most one survives. A stream whose ``adaptable`` input is False admits no
adapter and refuses any plan. The adapter's modules are the stream's stages,
each checked on both of its sides.

A ``BufferedStream`` owns a ``transport`` Decision over two nodes, ``direct``
and ``fifo``, after its adapter. The FIFO candidate owns its ``depth`` and the
FIFO's ``ram_style``; it is an identity stage presenting what arrives at it.
Each stream exports its checked ``Connection`` under ``CONNECTION``; its
composite wires them (``finn.kernels.composite.netlist``), each stage as
``u_<stream>_<stage>``.

Beside ``compatible``, each checked hop is resolved into wires (``wired``:
lanes, valid, ready, marker bits), and the stream exports its netlist under
``NETLIST``: its stages' leaves at their labels below it
(``adapter.input_gen.input_gen``, ``transport.fifo.buffer``) and its hops. A
user's end belongs to the kernel whose ``Port`` it is, beside the stream
(``^compute.packed``, at any depth below the stream's composite); a boundary
end is the root's own pins. Under ``BOUNDARY`` it exports the AXIS bus its
boundary side presents, if any.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    Users,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    domain,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import Endpoint as LinkEnd
from finn.kernels.artifacts.module import Fragment, Link
from finn.kernels.base import NETLIST, PORT
from finn.kernels.fifo import FifoKernel
from finn.kernels.transport import (
    AxiStream,
    Level,
    Mismatch,
    StreamContract,
    compatibility,
    lane_permutation,
    marker_bit,
    marker_pairs,
)
from finn.dataflow.stream import End, Ends
from finn.dataflow.stream import Stream as LogicalStream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, unreplayed
from finn.kernels.adapters import ADAPTERS, Stage, StreamAdapter


# The composed module's clocking pins: its interface convention, not a routing rule.
CLOCK, CLOCK2X, RESET = "ap_clk", "ap_clk2x", "ap_rst_n"

BOUNDARY = ViewKey("boundary", default_semantics(tuple))
"""A stream's AXIS bus on its composite's boundary: one, or none when both of its ends
are its composite's children."""

ADAPTER_RAM_STYLES = "*.adapter.*.ram_style"
"""The keys (``fnmatch``) of every adapter stage's memory choice, an ``input_gen``'s."""


def boundary_contract(
    name: str, element: ScalarEncoding, sequence: BeatSequence, endpoint: Endpoint
) -> StreamContract:
    """The AXIS port a composed module presents for one of its own streams."""
    if len(sequence.markers) > 1:
        raise ValueError("an AXIS boundary carries at most one marker")
    form = sequence.form
    stream = AxiStream(
        name, element.dtype, form.lanes, endpoint=endpoint, last=bool(sequence.markers)
    )
    transport = stream.native(clock=CLOCK, reset=RESET)
    markers = {transport.markers[0].signal: sequence.markers[0]} if sequence.markers else {}
    return StreamContract(transport, element, form, sequence.repetition, markers)


def _end(instance: str | None, contract: StreamContract) -> LinkEnd:
    transport = contract.transport
    return LinkEnd(instance, transport.data, transport.data_width, transport.valid, transport.ready)


def wired(
    source: str | None, produced: StreamContract, sink: str | None, consumed: StreamContract
) -> Link:
    """One hop that connects directly, as wires: each sink lane from its source lane
    (``lane_permutation``), and each marker the sink requires from the source bit that
    guarantees it (``marker_pairs``)."""
    markers = []
    for offered, required in marker_pairs(produced, consumed):
        (signal, bit), (pin, position) = marker_bit(offered), marker_bit(required)
        markers.append((signal, bit, pin, position))
    return Link(
        _end(source, produced),
        _end(sink, consumed),
        produced.element.bits,
        lane_permutation(produced, consumed),
        tuple(markers),
    )


class _Direct(Space):
    @view
    def stage(self) -> Stage:
        return Stage()


class StreamFifo(Space):
    """An identity adapter: what arrives at it, presented on both sides of a native FIFO."""

    tensor: Tensor = Param()
    arriving: BeatSequence = Param()

    @derived
    def word_bits(self) -> int:
        return self.arriving.form.lanes * self.tensor.element.bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view
    def stage(self) -> Stage:
        element, arriving, buffer = self.tensor.element, self.arriving, self.buffer
        return Stage(
            buffer.build_requirements,
            StreamContract(buffer.input.transport, element, arriving.form, arriving.repetition),
            StreamContract(buffer.output.transport, element, arriving.form, arriving.repetition),
            "fifo",
            module=buffer.module,
            label="fifo.buffer",
        )


@dataclass(frozen=True)
class Connection:
    """One checked stream; an owner of None is the composed module itself.

    ``source_input`` and ``sink_input`` name the reference input each owner
    presents its end through (``y_stream``); empty at a boundary.
    """

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract
    stages: tuple[Stage, ...] = ()
    source_input: str = ""
    sink_input: str = ""


CONNECTION_SEMANTICS = default_semantics(Connection)
CONNECTION = ViewKey("connection", CONNECTION_SEMANTICS)


class Stream(LogicalStream):
    """A relation between the kernels that reference it: one producer, one consumer.

    ``users`` holds the port each present user presents on this stream,
    located by the user's name and the input it references this stream
    through. A side without a user is the composite's boundary, presented as
    the AXIS port ``port``: an input boundary without the replay its receiver
    realizes, an output boundary as produced, neither with markers and both
    as a single pass.
    """

    port: str = Param(required=False)
    users = Users(PORT)

    @derived
    def endpoints(self) -> Connection | Rejected:
        """The producing and consuming ends, without the stages between them."""
        producers: list[tuple[str | None, StreamContract, str]] = []
        consumers: list[tuple[str | None, StreamContract, str]] = []
        for end in self.users:
            contract = end.value
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((end.node, contract, end.member))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(f"{end.node}.{end.member}" for end in self.users)
            return reject(
                "stream-users",
                f"a stream has at most one producer and one consumer; referenced by {named}",
            )
        if not producers and not consumers:
            return reject("stream-unused", "no present kernel references this stream")
        try:
            # A side without a user is the boundary. Seen from inside, the
            # composite's input is the AXIS target and its output the initiator.
            if not producers:
                inside = consumers[0][1]
                producers.append((None, self._boundary(inside, Endpoint.TARGET), ""))
            if not consumers:
                inside = producers[0][1]
                consumers.append((None, self._boundary(inside, Endpoint.INITIATOR), ""))
        except ValueError as error:
            return reject("stream-boundary", str(error))
        (source_owner, source, source_input) = producers[0]
        (sink_owner, sink, sink_input) = consumers[0]
        return Connection(
            source_owner, source, sink_owner, sink, source_input=source_input, sink_input=sink_input
        )

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

    adapter: StreamAdapter = Decision(
        ADAPTERS,
        when=LogicalStream.adapting,
        tensor=LogicalStream.tensor,
        plan=LogicalStream.plan,
    )
    adapter_stages = View(adapter.stages)

    @derived
    def adapted(self) -> tuple[Stage, ...]:
        """The adapter's stages, in order, labelled below the stream; none when the ends
        connect directly."""
        if not self.adapting:
            return ()
        return tuple(
            replace(stage, label=f"adapter.{stage.label}") for stage in self.adapter_stages
        )

    @derived
    def arriving(self) -> BeatSequence:
        """What arrives after the adapter: the beat sequence a transport stage receives."""
        adapted = self.adapted
        output = adapted[-1].output if adapted else None
        return self.endpoints.source.sequence if output is None else output.sequence

    @view
    def stages(self) -> tuple[Stage, ...]:
        return self.adapted

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop, source through every stage to sink, connects directly.

        An element mismatch is ``well_formed``'s (``stream-tensor``: each end
        against the tensor), so it is not reported here a second time.
        """
        ends = self.endpoints
        found: list[Mismatch] = []
        current, current_top = ends.source, ends.source_owner is None
        for stage in self.stages:
            assert stage.input is not None and stage.output is not None
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
                        "stream-padding",
                        "a child's padding is wider than the top word that must carry it",
                    )
                )
        return _refusal([item for item in found if item.code != "stream-element"])

    @derived
    def link(self) -> Connection:
        return replace(self.endpoints, stages=self.stages)

    connection = View(
        link, requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible)
    )

    @derived
    def hops(self) -> tuple[Link, ...] | Rejected:
        """Each hop, source through every stage to sink, as wires.

        A user's end belongs to its kernel beside the stream: the user is a
        kernel's ``Port`` (``compute.packed.x``, at any depth below the
        stream's composite), its kernel the node above it (``^compute.packed``).
        A stage sits below the stream at its label; a boundary end is the
        root's own pins (``None``).
        """
        ends = self.endpoints
        owners: list[str | None] = []
        for node in (ends.source_owner, ends.sink_owner):
            kernel = None if node is None else node.rpartition(".")[0]
            if kernel == "":
                return reject("stream-user", f"{node} presents an end, but is no kernel's port")
            owners.append(None if kernel is None else "^" + kernel)
        links: list[Link] = []
        owner, current = owners[0], ends.source
        for stage in self.stages:
            assert stage.input is not None and stage.output is not None
            links.append(wired(owner, current, stage.label, stage.input))
            owner, current = stage.label, stage.output
        links.append(wired(owner, current, owners[1], ends.sink))
        return tuple(links)

    @view(requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible))
    def netlist(self) -> Fragment:
        """Its stages' leaves below it, and its hops."""
        stages = tuple(
            (stage.label, stage.module) for stage in self.stages if stage.module is not None
        )
        return Fragment(stages, self.hops)

    @view(requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible))
    def boundary_bus(self) -> tuple[Bus, ...]:
        """The AXIS bus its composite presents for it, when one side has no user."""
        ends = self.endpoints
        sides = ((ends.source_owner, ends.source), (ends.sink_owner, ends.sink))
        return tuple(contract.transport.axis_bus() for owner, contract in sides if owner is None)

    @view
    def boundary(self) -> StreamContract | Rejected:
        """The port its composite presents for it: the side without a user."""
        ends = self.endpoints
        if ends.source_owner is None:
            return ends.source
        if ends.sink_owner is None:
            return ends.sink
        return reject("stream-internal", "both ends of this stream are its composite's children")

    exports = {CONNECTION: connection, NETLIST: netlist, BOUNDARY: boundary_bus}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


class BufferedStream(Stream):
    """A stream whose ``transport`` after its adapter is direct or a FIFO."""

    transport: _Direct | StreamFifo = Decision(
        {
            "direct": _Direct,
            "fifo": StreamFifo(tensor=Stream.tensor, arriving=Stream.arriving),
        }
    )
    transport_stage = View(transport.stage)

    @view
    def stages(self) -> tuple[Stage, ...]:
        fifo = self.transport_stage
        if fifo.requirements is None:
            return self.adapted
        return (*self.adapted, replace(fifo, label=f"transport.{fifo.label}"))


__all__ = [
    "ADAPTER_RAM_STYLES",
    "BOUNDARY",
    "BufferedStream",
    "CLOCK",
    "CLOCK2X",
    "CONNECTION",
    "CONNECTION_SEMANTICS",
    "Connection",
    "RESET",
    "Stream",
    "StreamFifo",
    "boundary_contract",
    "wired",
]
