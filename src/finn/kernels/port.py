# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's interfaces: one ``Port`` node each.

A kernel declares one ``Port`` per interface of its module. Every port has a
``transport`` (its ready/valid pins, required of each kind of port) and
exports its pins under ``PINS``, which the kernel's module collects in
declaration order (``finn.kernels.base``), and the clock it runs on under
``CLOCKED`` (a value), which must be its kernel's. A port its configuration leaves
``idle`` exports under ``HELD`` the pins it holds: its inputs low and its
outputs unused. A port's constraints count toward its kernel's validity; its
pins, what it holds and its contract are products, refused while it is.

An ``AxiStreamPort`` takes the tensor it reads or writes (``tensor``): an input's
is a fact its kernel gives it, an output's the kernel's own statement of what it
produces. It presents what it carries of it (``presented``): its kernel's
``Schedule`` projected through the indices it reads (``index``), its lane order
(``lanes``, outer first), the indices it is presented after (``reduces``) or before
(``holds``) and the reduction its marker closes (``closes``), through a row-major
view when ``reshaped``; or, for a traversal no schedule derives, a given
``sequence``. Its element is its tensor's, unless its hardware fixes one
(``dtype``: a set port's selector), which its channel holds the tensor to;
``admits`` is the integer policy its hardware takes (``element_taken``). It is an
AXIS bus named ``name``, with a ``TLAST`` when what it presents carries a marker, its
member pins spelled as its module spells them (``spelling``); or, given ``signals`` (data,
valid, ready), those ready/valid pins, with a native loop-completion vector when
``loop_ends`` names it (``olst``: the level each bit closes), else without a marker.
``native_stream`` declares one on FinnLib's native pins (``idat``, ``ivld``,
``irdy`` for a target, ``odat``, ``ovld``, ``ordy`` for an initiator, on ``clk``
and ``rst``): a channel stage's (``fifo``, ``vpc``, ``input_gen``), presenting the
sequence it is given. An input placed with a schedule exports its read of the tensor
under ``ACCESS`` (a value, read whatever the port's validity), from which its kernel
binds its indices' extents; every placed port exports the beats it carries a frame
under ``BEATS``, its kernel's cycles, and its contract states its ``pace`` (when it
presents each beat, ``Schedule.beat_times``). Its ``contract`` is what its kernel
states as its end on a channel (``finn.kernels.base.PortEnd``). Without a tensor it is
an interface not used (a set port without sets): idle, one lane of its ``dtype``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

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
from finn.dataflow.schedule import Access, Affine, Index, Pace, Refused, Schedule, plain_index
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd
from finn.kernels.artifacts.abi import CLOCK, RESET, Direction, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import (
    ACCESS,
    BEATS,
    CLOCKED,
    HELD,
    NATIVE_CLOCKING,
    PINS,
)
from finn.kernels.transport import (
    AxisBeat,
    AxisSpelling,
    MarkerKind,
    ReadyValidStream,
    StreamContract,
    StreamMarker,
)
from finn.kernels.values.domains import Integer, admit_element
from finn.kernels.values.semantics import QONNX_DATATYPE_VALUE_SEMANTICS

T = TypeVar("T")

INTEGER_POLICY: ValueSemantics[Integer | None] = ValueSemantics(
    Integer,
    "integer policy",
    lambda value: value is None or isinstance(value, Integer),
    lambda left, right: left == right,
    lambda value: value,
)
"""An integer policy a port's hardware takes; None when its kernel admits the element."""


class Port(Space):
    """One interface of a kernel's module: its pins, and what it holds while idle."""

    name: str = Param()
    endpoint: Endpoint = Param()
    clock: str = Param(default=CLOCK)
    reset: str = Param(default=RESET)
    transport = required(ReadyValidStream)

    @derived
    def idle(self) -> bool:
        return False

    @view
    def pins(self) -> tuple[Pin, ...]:
        return self.transport.pins()

    @derived
    def clock_pin(self) -> str:
        """The clock it runs on: a value export, so its kernel checks it (``clocked``)
        whatever this port's validity."""
        return self.clock

    @view
    def held(self) -> Held:
        """While idle: the forward pins of a target and the ready of an initiator held low."""
        if not self.idle:
            return Held()
        transport = self.transport
        inputs: list[tuple[str, int]] = []
        unused: list[str] = []
        for signal in transport.pins():
            if signal.direction is Direction.IN:
                inputs.append((signal.name, 0))
            else:
                unused.append(signal.name)
        return Held(tuple(inputs), tuple(unused))

    exports = {PINS: pins, HELD: held, CLOCKED: clock_pin}


def or_none(semantics: ValueSemantics[T]) -> ValueSemantics[T | None]:
    """``semantics``, or None: a formal a port may leave unset."""
    return ValueSemantics(
        semantics.type_token,
        f"{semantics.name} or None",
        lambda value: value is None or semantics.recognizes(value),
        lambda left, right: (
            left is None
            and right is None
            or left is not None
            and right is not None
            and semantics.equal(left, right)
        ),
        lambda value: None if value is None else semantics.snapshot(value),
    )


@dataclass(frozen=True)
class LoopEnds:
    """A native loop-completion vector on ready/valid pins (FinnLib's ``olst``): its
    ``signal``, and the level each of its bits closes, ``levels[bit]``."""

    signal: str
    levels: tuple[LevelEnd, ...]


OPTIONAL_SCHEDULE = or_none(default_semantics(Schedule))
OPTIONAL_SEQUENCE = or_none(default_semantics(BeatSequence))
OPTIONAL_DTYPE = or_none(QONNX_DATATYPE_VALUE_SEMANTICS)
OPTIONAL_LOOP_ENDS = or_none(default_semantics(LoopEnds))


class AxiStreamPort(Port):
    """One AXI-Stream interface: what it presents of its tensor, its element, its pins.

    It presents either its kernel's ``schedule``, projected through the indices
    it reads (``index``), its lane order (``lanes``, outer first), the indices
    it is presented after (``reduces``) or before (``holds``) and the reduction
    its marker closes (``closes``), read through a row-major view when
    ``reshaped``; or, for a traversal no schedule derives, a given ``sequence``.
    Exactly one of the two. An input placed with a schedule exports its read of the
    tensor (``ACCESS``, a value), from which its kernel binds its indices' extents, and
    every scheduled port's contract states its pace (``Pace``: the schedule with its
    ``reduces`` and ``holds``), so a channel reads when each end presents its beats.

    Its ``tensor`` is the one it reads, a fact its kernel gives it, or the one it
    writes: a producer's (an initiator's) is its kernel's statement of its output,
    derived from the kernel's facts, choices and input tensors, never given: a
    compiler asks a kernel for its outputs before the downstream tensor exists. A
    producer's tensor binds no extent: the kernel derives it from them. Its element is
    its tensor's: a producer that knows its values states their range in it. A port
    whose hardware fixes its element (a set port's selector) states it, ``dtype``, and
    its channel holds the tensor to it. ``admits`` is the integer policy its hardware
    takes, a constraint (``element_taken``) counted in its kernel's validity. Without
    a tensor it is an interface not used: idle, one lane of its ``dtype``. It is an
    AXIS bus named ``name``, with a ``TLAST`` when what it presents carries a marker;
    or, given ``signals`` (data, valid, ready), those ready/valid pins, without a
    marker.
    """

    tensor: Tensor = Param(required=False)
    schedule: Schedule | None = Param(default=None, semantics=OPTIONAL_SCHEDULE)
    index: tuple[Index | Affine, ...] = Param(default=())
    lanes: tuple[Index, ...] = Param(default=())
    reduces: tuple[Index, ...] = Param(default=())
    holds: tuple[Index, ...] = Param(default=())
    closes: tuple[Index, ...] = Param(default=())
    reshaped: bool = Param(default=False)
    sequence: BeatSequence | None = Param(default=None, semantics=OPTIONAL_SEQUENCE)
    # The element its hardware fixes, if any; an idle port's pins carry it.
    dtype: QONNXDataType | None = Param(default=None, semantics=OPTIONAL_DTYPE)
    admits: Integer | None = Param(default=None, semantics=INTEGER_POLICY)
    # Ready/valid pins (data, valid, ready) carrying the words instead of an AXIS bus.
    signals: tuple[str, ...] = Param(default=())
    # With ``signals``: the loop-completion vector its markers ride on, if any.
    loop_ends: LoopEnds | None = Param(default=None, semantics=OPTIONAL_LOOP_ENDS)
    # How the module spells the AXIS bus's member pins (HLS's are upper case).
    spelling: AxisSpelling = Param(default=AxisSpelling.LOWER)

    @derived
    def idle(self) -> bool:
        return not self.present(AxiStreamPort.tensor)

    @derived
    def placed(self) -> bool:
        return not self.idle

    @derived
    def presented(self) -> BeatSequence | Rejected:
        """What the port presents of its tensor: its schedule's projection, or given."""
        schedule, given = self.schedule, self.sequence
        if (schedule is None) == (given is None):
            return reject(
                "port-presentation", f"{self.name}: exactly one of a schedule or a sequence"
            )
        if given is not None:
            return given
        assert schedule is not None
        index = self.index
        try:
            view = None
            if self.reshaped:
                plain = [each for each in map(plain_index, index) if each is not None]
                if len(plain) != len(index):
                    raise Refused(
                        "reshape-not-plain",
                        "AxiStreamPort.presented",
                        "a reshaped port reads plain indices",
                    )
                view = tuple(schedule.extent(axis) for axis in plain)
            form = schedule.present(
                self.tensor.shape,
                index,
                lanes=self.lanes,
                reduces=self.reduces,
                holds=self.holds,
                view=view,
            )
            markers = (schedule.closing(self.closes),) if self.closes else ()
            return BeatSequence(form, markers=markers)
        except Refused as error:
            return reject("port-schedule", f"{self.name}: {error}")

    @derived
    def binds(self) -> bool:
        """Placed, an input, and reading indices: its read of the tensor binds its kernel's
        extents. A producer's tensor is derived from them.

        Reads the indices, not the schedule: the schedule reads the extents.
        """
        return not self.idle and self.endpoint is Endpoint.TARGET and bool(self.index)

    @derived(when=binds)
    def access(self) -> Access:
        """Its read of the tensor, while it binds: a value export, so a refusal of this
        port never refuses its kernel's extents."""
        return Access(self.name, self.tensor.shape, self.index, self.reshaped)

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        """``dtype`` when its hardware fixes one, placed or idle (its channel holds the
        tensor to it); else its tensor's. A producer's is its tensor's."""
        if self.dtype is not None:
            if self.endpoint is Endpoint.INITIATOR:
                return reject("port-element", f"{self.name}: a producer's element is its tensor's")
            return admit_element(self.dtype, None)
        if self.idle:
            return reject("port-element", f"{self.name}: an idle port states its dtype")
        return self.tensor.element

    @constraint
    def element_taken(self) -> bool | Rejected:
        """The element is one this port's hardware takes."""
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def lane_count(self) -> int | Rejected:
        """The lanes a beat carries: what it presents; idle, one."""
        if not self.idle:
            return self.presented.form.lanes
        if self.lanes:
            return reject(
                "port-idle",
                f"{self.name}: an idle port carries one lane, not lanes of {self.lanes}",
            )
        return 1

    @derived
    def marker_count(self) -> int:
        return 0 if self.idle else len(self.presented.markers)

    @derived
    def axis(self) -> AxisBeat | Rejected:
        if self.marker_count > 1:
            return reject("port-markers", f"{self.name} has one TLAST; the sequence needs more")
        try:
            return AxisBeat(
                self.name,
                self.element.dtype,
                self.lane_count,
                endpoint=self.endpoint,
                last=bool(self.marker_count),
                spelling=self.spelling,
            )
        except ValueError as error:
            return reject("port-lanes", f"{self.name}: {error}")

    @derived
    def transport(self) -> ReadyValidStream | Rejected:
        if not self.signals:
            return self.axis.native(clock=self.clock, reset=self.reset)
        if len(self.signals) != 3:
            return reject("port-signals", f"{self.name}: signals are data, valid and ready")
        data, valid, ready = self.signals
        markers: tuple[StreamMarker, ...] = ()
        ends = self.loop_ends
        if ends is not None:
            unclosed = [rule.beats for rule in self.presented.markers if rule not in ends.levels]
            if not ends.levels or unclosed:
                return reject(
                    "port-markers",
                    f"{self.name}: no bit of {ends.signal} closes every {unclosed} beats"
                    if unclosed
                    else f"{self.name}: {ends.signal} closes no level",
                )
            markers = (StreamMarker(ends.signal, MarkerKind.LOOP_END, len(ends.levels)),)
        elif self.marker_count:
            return reject("port-markers", f"{self.name}: ready/valid pins carry no marker")
        bits = self.lane_count * self.element.bits
        try:
            return ReadyValidStream(
                self.name, bits, self.endpoint, data, valid, ready, self.clock, self.reset, markers
            )
        except ValueError as error:
            return reject("port-signals", f"{self.name}: {error}")

    @view
    def pins(self) -> tuple[Pin, ...]:
        if self.signals:
            return self.transport.pins()
        return (self.axis.bus(clock=self.clock, reset=self.reset),)

    @view
    def contract(self) -> StreamContract:
        presented = self.presented
        transport = self.transport
        ends = self.loop_ends
        markers: tuple[tuple[str, LevelEnd], ...]
        if ends is not None:
            # Each bit whose level it presents, by its bit (``olst[1]``).
            markers = tuple(
                (f"{ends.signal}[{bit}]", level)
                for bit, level in enumerate(ends.levels)
                if level in presented.markers
            )
        elif transport.markers:
            markers = ((transport.markers[0].signal, presented.markers[0]),)
        else:
            markers = ()
        schedule = self.schedule
        pace = None if schedule is None else Pace(schedule, self.reduces, self.holds)
        return StreamContract(
            transport, self.element, presented.form, presented.repetition, markers, pace
        )

    @view(when=placed)
    def beats(self) -> int:
        """The beats it carries a frame."""
        return self.presented.form.beats

    exports = {
        PINS: pins,
        HELD: Port.held,
        CLOCKED: Port.clock_pin,
        ACCESS: access,
        BEATS: beats,
    }


def native_stream(
    name: str, endpoint: Endpoint, tensor: Any, sequence: Any, loop_ends: Any = None
) -> AxiStreamPort:
    """A channel stage's port on FinnLib's native pins, presenting ``sequence`` of
    ``tensor``: ``idat``, ``ivld``, ``irdy`` for a target, ``odat``, ``ovld``, ``ordy``
    for an initiator, on ``clk`` and ``rst``, with the loop-completion vector
    ``loop_ends`` where given. Declare it in a kernel's body (``input = native_stream(
    "input", Endpoint.TARGET, input_tensor, arriving)``)."""
    side = "i" if endpoint is Endpoint.TARGET else "o"
    vector = {} if loop_ends is None else {"loop_ends": loop_ends}
    return AxiStreamPort(
        name=name,
        endpoint=endpoint,
        tensor=tensor,
        sequence=sequence,
        signals=(f"{side}dat", f"{side}vld", f"{side}rdy"),
        clock=NATIVE_CLOCKING.clock,
        reset=NATIVE_CLOCKING.reset,
        **vector,
    )


__all__ = [
    "AxiStreamPort",
    "INTEGER_POLICY",
    "LoopEnds",
    "Port",
    "native_stream",
]
