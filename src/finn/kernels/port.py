# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's interfaces: one ``Port`` node each.

A kernel declares one ``Port`` per interface of its module. Every port has a
``transport`` (its ready/valid pins, required of each kind of port) and
exports its pins under ``PINS``, which the kernel's module collects in
declaration order (``finn.kernels.base``), and the clock it runs on under
``CLOCKED``, which must be its kernel's. A port its configuration leaves
``idle`` exports under ``HELD`` the pins it holds: its inputs low and its
outputs unused.

- A ``WordPort`` carries opaque words on FinnLib's native pins (``idat``,
  ``ivld``, ``irdy`` for a target, ``odat``, ``ovld``, ``ordy`` for an
  initiator, on ``clk`` and ``rst``), with any loop-completion ``markers``: a
  channel stage's port (``input_gen``, ``vpc``, ``fifo``).
- An ``AxiStreamPort`` sits on a channel (``channel``) and presents what it
  carries of the channel's tensor (``presented``): its kernel's ``Schedule``
  projected through the indices it reads (``index``), its lane order
  (``lanes``, outer first), the indices it is presented after (``reduces``)
  or before (``holds``) and the reduction its marker closes (``closes``),
  through a row-major view when ``reshaped``; or, for a traversal no schedule
  derives, a given ``sequence``. Its element is its ``dtype`` when given
  (its channel refuses another; a producer must give it) and otherwise its
  channel's; ``admits`` is the
  integer policy its hardware takes. It is an AXIS bus named ``name``, with a
  ``TLAST`` when what it presents carries a marker; or, given ``signals``
  (data, valid, ready), those ready/valid pins, without a marker. Placed with
  a schedule, it exports its read of the tensor under ``ACCESS``, from which
  its kernel binds its indices' extents. Left without a channel it is idle,
  with the pins of its ``dtype`` and of the ``factors`` of its ``lanes``.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, TypeVar

import finn.kernels
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
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule
from finn.dataflow.tensor import ScalarEncoding
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Direction, Endpoint
from finn.kernels.artifacts.module import Held
from finn.kernels.base import ACCESS, CLOCKED, HELD, PINS, PORT
from finn.kernels.datatypes.domains import Integer
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.transport import AxiStream, ReadyValidStream, StreamContract, StreamMarker

if TYPE_CHECKING:
    import finn.kernels.channels

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
    clock: str = Param(default="ap_clk")
    reset: str = Param(default="ap_rst_n")
    transport = required(ReadyValidStream)

    @derived
    def idle(self) -> bool:
        return False

    @view
    def pins(self) -> tuple[object, ...]:
        return self.transport.pins()

    @view
    def clock_pin(self) -> str:
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


class WordPort(Port):
    """Opaque words on FinnLib's native ready/valid pins, with optional markers."""

    bits: int = Param()
    markers: tuple[StreamMarker, ...] = Param(default=())
    clock: str = Param(default="clk")
    reset: str = Param(default="rst")

    @derived
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


OPTIONAL_SCHEDULE = or_none(default_semantics(Schedule))
OPTIONAL_SEQUENCE = or_none(default_semantics(BeatSequence))
OPTIONAL_DTYPE = or_none(QONNX_DATATYPE_VALUE_SEMANTICS)


class AxiStreamPort(Port):
    """One stream interface: what it presents of its channel's tensor, its element, its pins.

    It presents either its kernel's ``schedule``, projected through the indices
    it reads (``index``), its lane order (``lanes``, outer first), the indices
    it is presented after (``reduces``) or before (``holds``) and the reduction
    its marker closes (``closes``), read through a row-major view when
    ``reshaped``; or, for a traversal no schedule derives, a given ``sequence``.
    Exactly one of the two. Placed with a schedule, it exports its read of the
    tensor (``ACCESS``), from which its kernel binds its indices' extents.

    Its element is ``dtype`` when given, placed or idle (its channel refuses
    another), and otherwise its channel's. A producer (an initiator) must give
    it, from its kernel's facts, choices and input elements, never from its
    own output channel: a compiler asks a kernel for its output types before
    the downstream tensor exists. A producer that knows its values states
    their ``value_range`` (minimum, maximum) with it; ``()`` is the datatype's own.
    ``admits`` is the integer policy its hardware takes. Idle (no channel), it
    carries the lanes of the ``factors`` of its ``lanes`` indices. It is an
    AXIS bus named ``name``, with a ``TLAST`` when what it presents carries a
    marker; or, given ``signals`` (data, valid, ready), those ready/valid pins,
    without a marker. ``staged``, a channel's source placed by the channel itself
    (``finn.kernels.channels``), presents its given sequence and dtype without a
    channel reference: it is not idle.
    """

    # By its full path, and not imported: channels imports this module (a channel's source
    # has a port), and the engine resolves the annotation when it collects the Space class. A
    # kernel that places a port on a channel names that Space class itself, so it is loaded.
    channel: finn.kernels.channels.Channel = Param(required=False)
    schedule: Schedule | None = Param(default=None, semantics=OPTIONAL_SCHEDULE)
    index: tuple[Index | Affine, ...] = Param(default=())
    lanes: tuple[Index, ...] = Param(default=())
    reduces: tuple[Index, ...] = Param(default=())
    holds: tuple[Index, ...] = Param(default=())
    closes: tuple[Index, ...] = Param(default=())
    reshaped: bool = Param(default=False)
    factors: dict[Index, int] = Param(default={})
    sequence: BeatSequence | None = Param(default=None, semantics=OPTIONAL_SEQUENCE)
    dtype: QONNXDataType | None = Param(default=None, semantics=OPTIONAL_DTYPE)
    value_range: tuple[int, ...] = Param(default=())
    admits: Integer | None = Param(default=None, semantics=INTEGER_POLICY)
    # Ready/valid pins (data, valid, ready) carrying the words instead of an AXIS bus.
    signals: tuple[str, ...] = Param(default=())
    staged: bool = Param(default=False)

    @derived
    def idle(self) -> bool:
        return not self.present(AxiStreamPort.channel) and not self.staged

    @derived
    def presented(self) -> BeatSequence | Rejected:
        """What the port presents of its channel's tensor: its schedule's projection, or given."""
        schedule, given = self.schedule, self.sequence
        if (schedule is None) == (given is None):
            return reject("port-presentation", f"{self.name}: a schedule or a sequence, not both")
        if given is not None:
            return given
        assert schedule is not None
        index = self.index
        try:
            view = None
            if self.reshaped:
                if not all(isinstance(axis, Index) for axis in index):
                    raise Refused("a reshaped port reads plain indices")
                view = tuple(schedule.extent(axis) for axis in index)  # type: ignore[arg-type]
            form = schedule.present(
                self.channel.tensor.shape,
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

    @derived
    def binds(self) -> bool:
        """Placed and reading indices: its read of the tensor binds its kernel's extents.

        Reads the indices, not the schedule: the schedule reads the extents.
        """
        return not self.idle and bool(self.index)

    @view(when=binds)
    def access(self) -> Access:
        return Access(self.name, self.channel.tensor.shape, self.index, self.reshaped)

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        """``dtype`` over ``value_range`` when given, placed or idle (its channel refuses
        another); else the channel's."""
        dtype, bounds = self.dtype, self.value_range
        if dtype is not None:
            if bounds and len(bounds) != 2:
                return reject("port-element", f"{self.name}: a range is (minimum, maximum)")
            return ScalarEncoding.admit(dtype, (bounds[0], bounds[1]) if bounds else None)
        if self.endpoint is Endpoint.INITIATOR:
            # From facts, choices and input elements only: an op asks before its output exists.
            return reject("port-element", f"{self.name}: a producer states its dtype")
        if self.idle:
            return reject("port-element", f"{self.name}: an idle port states its dtype")
        return self.channel.tensor.element

    @constraint
    def admitted(self) -> bool | Rejected:
        """The element is one this port's hardware takes."""
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def lane_count(self) -> int:
        if self.idle:
            factors = self.factors
            return prod(factors.get(index, 1) for index in self.lanes)
        return self.presented.form.lanes

    @derived
    def marker_count(self) -> int:
        return 0 if self.idle else len(self.presented.markers)

    @derived
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

    @derived
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

    @view(requires=(admitted,))
    def pins(self) -> tuple[object, ...]:
        if self.signals:
            return self.transport.pins()
        return (self.axis.bus(clock=self.clock, reset=self.reset),)

    @view(requires=(admitted,))
    def contract(self) -> StreamContract:
        presented = self.presented
        transport = self.transport
        markers = {transport.markers[0].signal: presented.markers[0]} if transport.markers else {}
        return StreamContract(
            transport, self.element, presented.form, presented.repetition, markers
        )

    exports = {
        PORT: {channel: contract},
        PINS: pins,
        HELD: Port.held,
        CLOCKED: Port.clock_pin,
        ACCESS: access,
    }


__all__ = [
    "AxiStreamPort",
    "INTEGER_POLICY",
    "Port",
    "WordPort",
    "or_none",
]
