# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's interfaces: one ``Port`` node each.

A kernel declares one ``Port`` per interface of its module. Every port has a
``transport`` (its ready/valid pins, required of each kind of port) and
exports its pins under ``PINS``, which the kernel's module collects in
declaration order (``finn.kernels.base``). A port its configuration leaves
``idle`` exports under ``HELD`` the pins it holds: its inputs low and its
outputs unused.

- A ``WordPort`` carries opaque words on FinnLib's native pins (``idat``,
  ``ivld``, ``irdy`` for a target, ``odat``, ``ovld``, ``ordy`` for an
  initiator, on ``clk`` and ``rst``), with any loop-completion ``markers``.
- A ``StreamPort`` sits on a stream (``stream``). It carries an ``element``,
  admits it (``admits``, an integer policy; absent when its kernel admits the
  element itself), presents its ``sequence`` of the stream's tensor, and
  derives an AXIS bus named ``name`` carrying the sequence's lanes each beat,
  with a ``TLAST`` when the sequence carries a marker; or, given ``signals``
  (data, valid, ready), those ready/valid pins carrying the lanes' bits
  exactly, without a marker. It exports its contract under ``PORT`` through
  its stream, which refuses an end whose element it does not carry. A port
  left without a stream is idle, with the pins of ``dtype`` and
  ``idle_lanes``.

A ``StreamPort``'s ``sequence`` is required. A ``ScheduledPort`` derives it
from its kernel's ``Schedule``: the indices it reads (``index``), its lane
order (``lanes``, outer first), the indices it presents after (``reduces``) or
before (``holds``), and the reduction its marker closes (``closes``). With
``reshaped`` it reads its stream's tensor as a row-major view of the shape its
indices address (a densely realized depthwise operation reads (M, K, N)
activations as (M, K * N)), and carries its stream's element. A ``GivenPort``
presents the sequence and the element (``dtype``) its kernel gives it.
"""

from __future__ import annotations

from finn.core.space import (
    Param,
    Rejected,
    Space,
    ValueSemantics,
    constraint,
    default_semantics,
    derived,
    reject,
    required,
    view,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import SCHEDULE, Affine, Index, Refused, Schedule
from finn.dataflow.stream import Stream
from finn.dataflow.tensor import SCALAR_ENCODING, ScalarEncoding
from finn.dataflow.traversal import BEAT_SEQUENCE, BeatSequence
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.base import HELD, PINS, PORT, TIEOFFS_SEMANTICS, Tieoffs
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels.physical.contract import STREAM_CONTRACT, StreamContract
from finn.kernels.physical.stream import ReadyValidStream, StreamMarker

INTEGER_POLICY: ValueSemantics[Integer | None] = ValueSemantics(
    Integer,
    "integer policy",
    lambda value: value is None or isinstance(value, Integer),
    lambda left, right: left == right,
    lambda value: value,
)
"""An integer policy a port's hardware takes; None when its kernel admits the element."""
INDICES = default_semantics(tuple)
MARKERS = default_semantics(tuple)
SIGNAL_NAMES = default_semantics(tuple)
AXI_STREAM = default_semantics(AxiStream)
TRANSPORT = default_semantics(ReadyValidStream)


class Port(Space):
    """One interface of a kernel's module: its pins, and what it holds while idle."""

    name: str = Param()
    endpoint: Endpoint = Param()
    clock: str = Param(default="ap_clk")
    reset: str = Param(default="ap_rst_n")
    transport = required(ReadyValidStream)

    @derived
    def idle(self) -> bool:
        return False

    @view(semantics=default_semantics(tuple))
    def pins(self) -> tuple[object, ...]:
        return self.transport.pins()

    @view(semantics=TIEOFFS_SEMANTICS)
    def held(self) -> Tieoffs:
        """While idle: the forward pins of a target and the ready of an initiator held low."""
        if not self.idle:
            return Tieoffs()
        transport = self.transport
        inputs: list[tuple[str, int]] = []
        unused: list[str] = []
        for signal in transport.pins():
            if signal.direction is Direction.IN:
                inputs.append((signal.name, 0))
            else:
                unused.append(signal.name)
        return Tieoffs(tuple(inputs), tuple(unused))

    exports = {PINS: pins, HELD: held}


class WordPort(Port):
    """Opaque words on FinnLib's native ready/valid pins, with optional markers."""

    bits: int = Param()
    markers: tuple[StreamMarker, ...] = Param(default=(), semantics=MARKERS)
    clock: str = Param(default="clk")
    reset: str = Param(default="rst")

    @derived(semantics=TRANSPORT)
    def transport(self) -> ReadyValidStream | Rejected:
        side = "i" if self.endpoint is Endpoint.TARGET else "o"
        if not 1 <= self.bits <= 0xFFFFFFFF:
            return reject("port-width", f"{self.name}: a word is a positive native width")
        return ReadyValidStream(
            self.name,
            self.bits,
            self.endpoint,
            f"{side}dat",
            f"{side}vld",
            f"{side}rdy",
            self.clock,
            self.reset,
            self.markers,
        )


class StreamPort(Port):
    """An AXIS interface on one stream: element admission, beat sequence and pins."""

    stream: Stream = Param(required=False)
    admits: Integer | None = Param(default=None, semantics=INTEGER_POLICY)
    # The element of a port left without a stream (every element of a GivenPort's), and its lanes.
    dtype: QONNXDataType = Param(semantics=QONNX_DATATYPE_VALUE_SEMANTICS, required=False)
    idle_lanes: int = Param(default=1)
    # Ready/valid pins (data, valid, ready) carrying the words instead of an AXIS bus.
    signals: tuple[str, ...] = Param(default=(), semantics=SIGNAL_NAMES)
    sequence = required(BeatSequence)

    @derived
    def idle(self) -> bool:
        return not self.present(StreamPort.stream)

    @derived(semantics=SCALAR_ENCODING)
    def element(self) -> ScalarEncoding | Rejected:
        """The stream's element; ``dtype`` while idle."""
        if self.idle:
            return ScalarEncoding.admit(self.dtype)
        return self.stream.tensor.element

    @constraint
    def admitted(self) -> bool | Rejected:
        """The element is one this port's hardware takes."""
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def lane_count(self) -> int:
        return self.idle_lanes if self.idle else self.sequence.form.lanes

    @derived
    def marker_count(self) -> int:
        return 0 if self.idle else len(self.sequence.markers)

    @derived(semantics=AXI_STREAM)
    def axis(self) -> AxiStream | Rejected:
        if self.marker_count > 1:
            return reject("port-markers", f"{self.name} has one TLAST; the sequence needs more")
        try:
            return AxiStream(
                self.name,
                self.element.dtype,
                self.lane_count,
                endpoint=self.endpoint,
                last=bool(self.marker_count),
            )
        except ValueError as error:
            return reject("port-lanes", f"{self.name}: {error}")

    @derived(semantics=TRANSPORT)
    def transport(self) -> ReadyValidStream | Rejected:
        if not self.signals:
            return self.axis.native(clock=self.clock, reset=self.reset)
        if len(self.signals) != 3:
            return reject("port-signals", f"{self.name}: signals are data, valid and ready")
        if self.marker_count:
            return reject("port-markers", f"{self.name}: ready/valid pins carry no marker")
        data, valid, ready = self.signals
        bits = self.lane_count * self.element.bits
        try:
            return ReadyValidStream(
                self.name, bits, self.endpoint, data, valid, ready, self.clock, self.reset
            )
        except ValueError as error:
            return reject("port-width", f"{self.name}: {error}")

    @view(semantics=default_semantics(tuple), requires=(admitted,))
    def pins(self) -> tuple[object, ...]:
        if self.signals:
            return self.transport.pins()
        return (self.axis.bus(clock=self.clock, reset=self.reset),)

    @view(semantics=STREAM_CONTRACT, requires=(admitted,))
    def contract(self) -> StreamContract:
        sequence = self.sequence
        transport = self.transport
        markers = {transport.markers[0].signal: sequence.markers[0]} if transport.markers else {}
        return StreamContract(transport, self.element, sequence.form, sequence.repetition, markers)

    exports = {PORT: {stream: contract}, PINS: pins, HELD: Port.held}


class ScheduledPort(StreamPort):
    """A stream port presenting its kernel's schedule through the indices it reads."""

    schedule: Schedule = Param(semantics=SCHEDULE)
    index: tuple[Index | Affine, ...] = Param(semantics=INDICES)
    lanes: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    reduces: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    holds: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    closes: tuple[Index, ...] = Param(default=(), semantics=INDICES)
    reshaped: bool = Param(default=False)

    @derived(semantics=BEAT_SEQUENCE)
    def sequence(self) -> BeatSequence | Rejected:
        schedule, index = self.schedule, self.index
        try:
            view = None
            if self.reshaped:
                if not all(isinstance(axis, Index) for axis in index):
                    raise Refused("a reshaped port reads plain indices")
                view = tuple(schedule.extent(axis) for axis in index)  # type: ignore[arg-type]
            form = schedule.present(
                self.stream.tensor.shape,
                index,
                lanes=self.lanes,
                reduces=self.reduces,
                holds=self.holds,
                view=view,
            )
            markers = (schedule.closing(self.closes),) if self.closes else ()
            return BeatSequence(form, markers=markers)
        except (Refused, KeyError) as error:
            return reject("port-schedule", f"{self.name}: {error}")


class GivenPort(StreamPort):
    """A stream port presenting the beat sequence and element its kernel gives it."""

    sequence: BeatSequence = Param(semantics=BEAT_SEQUENCE)

    @derived(semantics=SCALAR_ENCODING)
    def element(self) -> ScalarEncoding | Rejected:
        """The kernel's ``dtype``, placed or idle: its stream refuses another."""
        return ScalarEncoding.admit(self.dtype)


__all__ = [
    "AXI_STREAM",
    "GivenPort",
    "INDICES",
    "INTEGER_POLICY",
    "MARKERS",
    "Port",
    "SIGNAL_NAMES",
    "ScheduledPort",
    "StreamPort",
    "TRANSPORT",
    "WordPort",
]
