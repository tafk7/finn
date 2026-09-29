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
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import PORT
from finn.kernels.fifo import FifoKernel
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.contract import (
    STREAM_CONTRACT,
    Mismatch,
    StreamContract,
    compatibility,
)
from finn.dataflow.stream import ENDS, End, Ends
from finn.dataflow.stream import Stream as LogicalStream
from finn.dataflow.tensor import TENSOR, ScalarEncoding, Tensor
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence, unreplayed
from finn.kernels.adapters import ADAPTERS, STAGE_SEMANTICS, STAGES, Stage, StreamAdapter


# The composed module's clocking pins: its interface convention, not a routing rule.
CLOCK, CLOCK2X, RESET = "ap_clk", "ap_clk2x", "ap_rst_n"

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


class _Direct(Space):
    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        return Stage()


class StreamFifo(Space):
    """An identity adapter: what arrives at it, presented on both sides of a native FIFO."""

    tensor: Tensor = Param(semantics=TENSOR)
    arriving: BeatSequence = Param(semantics=BEAT_SEQUENCE)

    @derived
    def word_bits(self) -> int:
        return self.arriving.form.lanes * self.tensor.element.bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
    )

    @view(semantics=STAGE_SEMANTICS)
    def stage(self) -> Stage:
        element, arriving, buffer = self.tensor.element, self.arriving, self.buffer
        return Stage(
            buffer.build_requirements,
            StreamContract(buffer.input.transport, element, arriving.form, arriving.repetition),
            StreamContract(buffer.output.transport, element, arriving.form, arriving.repetition),
            "fifo",
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

    @derived(semantics=CONNECTION_SEMANTICS)
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

    @derived(semantics=ENDS)
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

    @derived(semantics=STAGES)
    def adapted(self) -> tuple[Stage, ...]:
        """The adapter's stages, in order; none when the ends connect directly."""
        return self.adapter_stages if self.adapting else ()

    @derived(semantics=BEAT_SEQUENCE)
    def arriving(self) -> BeatSequence:
        """What arrives after the adapter: the beat sequence a transport stage receives."""
        adapted = self.adapted
        output = adapted[-1].output if adapted else None
        return self.endpoints.source.sequence if output is None else output.sequence

    @view(semantics=STAGES)
    def stages(self) -> tuple[Stage, ...]:
        return self.adapted

    @constraint
    def compatible(self) -> bool | Rejected:
        """Each hop, source through every stage to sink, connects directly."""
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
        return _refusal(found)

    @derived(semantics=CONNECTION_SEMANTICS)
    def link(self) -> Connection:
        return replace(self.endpoints, stages=self.stages)

    connection = View(
        link, requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible)
    )

    @view(semantics=STREAM_CONTRACT)
    def boundary(self) -> StreamContract | Rejected:
        """The port its composite presents for it: the side without a user."""
        ends = self.endpoints
        if ends.source_owner is None:
            return ends.source
        if ends.sink_owner is None:
            return ends.sink
        return reject("stream-internal", "both ends of this stream are its composite's children")

    exports = {CONNECTION: connection}


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

    @view(semantics=STAGES)
    def stages(self) -> tuple[Stage, ...]:
        fifo = self.transport_stage
        return (*self.adapted, *((fifo,) if fifo.requirements is not None else ()))


__all__ = [
    "ADAPTER_RAM_STYLES",
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
]
