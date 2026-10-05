# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Streams as ordinary Spaces that kernels reference.

A ``Stream`` is the physical form of the logical stream
(``finn.dataflow.stream``): one real producer-to-consumer edge, a node of its
own, declared in the kernel with children (or the root) that owns the edge,
beside the kernels it joins, carrying one ``tensor`` its owner supplies: stated,
or read from a kernel's fact-level view (``MatMulKernel.weight_tensor``). A
kernel has one reference input per stream it sits on (``output_stream:
Stream = Param()``), bound to the ``stream`` of one of its ports
(``finn.kernels.port``), which exports its contract under ``PORT``; a kernel
with children passes the reference down to the child that uses it, so the
stream's user is always a leaf's port, at any depth. Each end presents its own
traversal of the tensor in that contract, reading only the stream's
``tensor``. The stream sees the ports that reference it through ``users =
Users(PORT)``, each with only the contract it presents on this stream, so a
port's refusal stays on its own stream. The contract's transport
endpoint says whether the kernel produces into the stream (initiator) or
consumes from it (target). The one-producer-one-consumer rule, compatibility
and the AXIS boundary belong to this family, not to the engine; the tensor
each end must traverse and the plan are the logical stream's.

A stream with a user on one side only is a boundary of the root that declares
it. Its ``port`` input names the top-level AXIS port (``in0_V``): an ABI name is
design data of the stream, independent of the stream's node name, which is
its identity and the prefix of its persisted decision keys. A boundary whose
``port`` is not supplied is refused (``stream-boundary``): a root names a port
only for what may cross it, an ONNX input or output of a partition, so an
unnamed boundary is a stream that lost one of its users. The boundary
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
each checked on both of its sides, on the stream's ``platform`` (an
``input_gen``'s ``ultra`` memory requires its UltraRAM).

A ``BufferedStream`` owns a ``transport`` Decision over two nodes, ``direct``
and ``fifo``, after its adapter. The FIFO candidate owns its ``depth`` and the
FIFO's ``ram_style`` (on the stream's ``platform``); it is an identity stage
presenting what arrives at it.
The adapter and transport choices are keyed under the stream
(``x.adapter``, ``w.transport``), so they belong to whoever owns the edge.
Its ``netlist`` view is accepted when the stream is: its ends, plan and every
hop checked.

A stream whose tensor has a known value (``contents``, one operand per set; with
several ``sets``, its ``index`` stream selects one) carries a ``source``
Decision over the kernels that can drive it with that value (``SOURCES``: a
memory now). It applies only when the value is known (``valued``, whether
``contents`` is supplied: a stream whose declaration supplies none compiles no
source), and it has no ``none`` case: each candidate refuses on its own facts and on the
``platform``'s capabilities, and one viable candidate is forced. The source is
the stream's producer end, placed by the stream (``staged``), its leaf below
the stream at ``source.<case>``; it stores one period of the value in the
order the stream's consumer reads it. The invariant: **a stream with a value
has its source as its only producer** (``stream-users``). Its keys are the
stream's (``w.source``, ``w.source.memstream.ram_style``), so they belong to
whoever owns the edge, for an initializer the node that consumes it.

Beside ``compatible``, each checked hop is resolved into wires (``wired``:
lanes, valid, ready, marker bits), and the stream exports its netlist under
``NETLIST``: its stages' leaves at their labels below it
(``adapter.input_gen.input_gen``, ``transport.fifo.buffer``) and its hops. A
user's end belongs to the kernel whose ``Port`` it is, beside the stream
(``^compute.packed``, at any depth below the stream's owner); a boundary
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
    constraint,
    derived,
    domain,
    reject,
    selected,
    supplied,
    view,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.stream import End, Ends
from finn.dataflow.stream import Stream as LogicalStream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Traversal, period, unreplayed
from finn.kernels.adapters import ADAPTERS, Stage, StreamAdapter
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.artifacts.module import BuildError, Fragment, Leaf, Link, LinkEnd
from finn.kernels.base import BOUNDARY, CLOCK, NETLIST, PORT, RESET
from finn.kernels.datatypes.semantics import (
    INTEGER_TENSOR,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    IntegerTensor,
)
from finn.kernels.fifo import FifoKernel
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.target import Platform
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

SOURCES: dict[str, type[Space] | Space] = {"memstream": MemStreamKernel}
"""The kernels that can drive a stream with its known value: a memory; later a fetcher
from memory-mapped memory, a loop's memory, a source reloadable over AXI-Lite."""

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
    platform: Platform = Param()

    @derived
    def word_bits(self) -> int:
        return self.arriving.form.lanes * self.tensor.element.bits

    buffer = FifoKernel(
        word_bits=word_bits,
        depth=Decision(domain=domain(accepts=lambda *, candidate: 2 <= candidate < 2**32)),
        platform=platform,
    )

    @view
    def stage(self) -> Stage:
        element, arriving, buffer = self.tensor.element, self.arriving, self.buffer
        module = buffer.module
        assert isinstance(module, Leaf)
        return Stage(
            module,
            StreamContract(buffer.input.transport, element, arriving.form, arriving.repetition),
            StreamContract(buffer.output.transport, element, arriving.form, arriving.repetition),
            "fifo.buffer",
        )


@dataclass(frozen=True)
class StreamEnds:
    """A stream's two ends: each end's owner (the user's port node, None at the root's
    boundary) and its contract."""

    source_owner: str | None
    source: StreamContract
    sink_owner: str | None
    sink: StreamContract


class Stream(LogicalStream):
    """A relation between the kernels that reference it: one producer, one consumer.

    ``users`` holds the port each present user presents on this stream,
    located by the user's name and the input it references this stream
    through. A side without a user is the root's boundary, presented as
    the AXIS port ``port``: an input boundary without the replay its receiver
    realizes, an output boundary as produced, neither with markers and both
    as a single pass. A stream with a known value has its ``source`` as its
    producer. ``platform`` is the target's, stated by whoever declares the stream:
    its source, its adapter's stages and its FIFO read it.
    """

    port: str = Param(required=False)
    users = Users(PORT)
    contents: IntegerTensor = Param(semantics=INTEGER_TENSOR, required=False)
    sets: int = Param(default=1)
    index: LogicalStream = Param(required=False)
    platform: Platform = Param()
    # Whether the stream carries a known value, which its source drives: where nothing
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
            return reject("stream-source", "a known value streams to exactly one consumer")
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
        set_stream=index,
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
    def endpoints(self) -> StreamEnds | Rejected:
        """The producing and consuming ends, without the stages between them: with a
        known value, the producer is the stream's source."""
        producers: list[tuple[str | None, StreamContract]] = []
        consumers: list[tuple[str | None, StreamContract]] = []
        for end in self.users:
            contract = end.value
            producing = contract.transport.endpoint is Endpoint.INITIATOR
            (producers if producing else consumers).append((end.node, contract))
        if self.valued:
            if producers:
                return reject(
                    "stream-users",
                    "a stream with a value has its source as its only producer, "
                    f"not {producers[0][0]}",
                )
            producers.append((self.source_label, self.source_contract))
        if len(producers) > 1 or len(consumers) > 1:
            named = ", ".join(f"{end.node}.{end.member}" for end in self.users)
            return reject(
                "stream-users",
                f"a stream has at most one producer and one consumer; referenced by {named}",
            )
        if not producers and not consumers:
            return reject("stream-unused", "no present kernel references this stream")
        if not (producers and consumers) and not self.present(Stream.port):
            return reject(
                "stream-boundary",
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
            return reject("stream-boundary", str(error))
        return StreamEnds(*producers[0], *consumers[0])

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
        platform=platform,
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
    def hops(self) -> tuple[Link, ...] | Rejected:
        """Each hop, source through every stage to sink, as wires.

        A user's end belongs to its kernel beside the stream: the user is a
        kernel's ``Port`` (``compute.packed.x``, at any depth below the
        stream's owner), its kernel the node above it (``^compute.packed``).
        A stage and the source sit below the stream at their labels; a
        boundary end is the root's own pins (``None``).
        """
        ends = self.endpoints
        owners: list[str | None] = []
        for side, node in enumerate((ends.source_owner, ends.sink_owner)):
            if side == 0 and self.valued:
                owners.append(node)  # the source, below the stream at its label
                continue
            kernel = None if node is None else node.rpartition(".")[0]
            if kernel == "":
                return reject("stream-user", f"{node} presents an end, but is no kernel's port")
            owners.append(None if kernel is None else "^" + kernel)
        links: list[Link] = []
        owner, current = owners[0], ends.source
        try:
            for stage in self.stages:
                assert stage.input is not None and stage.output is not None
                links.append(wired(owner, current, stage.label, stage.input))
                owner, current = stage.label, stage.output
            links.append(wired(owner, current, owners[1], ends.sink))
        except BuildError as error:
            # A hop that does not connect (another element, say) is refused by the
            # stream's constraints; its wires do not exist.
            return reject("stream-link", str(error))
        return tuple(links)

    @view(requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible))
    def netlist(self) -> Fragment:
        """Its source's and stages' leaves below it, and its hops."""
        stages = tuple(
            (stage.label, stage.module) for stage in self.stages if stage.module is not None
        )
        if self.valued:
            module = self.source_module
            assert isinstance(module, Leaf)
            stages = ((self.source_label, module), *stages)
        return Fragment(stages, self.hops)

    @view(requires=(LogicalStream.well_formed, LogicalStream.realizable, compatible))
    def boundary_bus(self) -> tuple[Bus, ...]:
        """The AXIS bus the root presents for it, when one side has no user."""
        ends = self.endpoints
        sides = ((ends.source_owner, ends.source), (ends.sink_owner, ends.sink))
        return tuple(contract.transport.axis_bus() for owner, contract in sides if owner is None)

    exports = {NETLIST: netlist, BOUNDARY: boundary_bus}


def _refusal(found: Sequence[Mismatch]) -> bool | Rejected:
    if not found:
        return True
    return reject(found[0].code, "; ".join(f"{item.code}: {item.message}" for item in found))


class BufferedStream(Stream):
    """A stream whose ``transport`` after its adapter is direct or a FIFO."""

    transport: _Direct | StreamFifo = Decision(
        {
            "direct": _Direct,
            "fifo": StreamFifo(
                tensor=Stream.tensor, arriving=Stream.arriving, platform=Stream.platform
            ),
        }
    )
    transport_stage = View(transport.stage)

    @view
    def stages(self) -> tuple[Stage, ...]:
        fifo = self.transport_stage
        if fifo.module is None:
            return self.adapted
        return (*self.adapted, replace(fifo, label=f"transport.{fifo.label}"))


__all__ = [
    "ADAPTER_RAM_STYLES",
    "SOURCES",
    "BufferedStream",
    "Stream",
    "StreamEnds",
    "StreamFifo",
    "boundary_contract",
    "wired",
]
