# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's stream ports: each a declaration of its kernel, lowered into its members.

A kernel binding a module declares one ``AxiStreamPort`` per AXI-Stream port of its module,
in its class body (``x = AxiStreamPort(name=..., tensor=x_tensor, schedule=schedule,
...)``); a kernel declaring one binds a module (``port-unbound``). It is a plain frozen
value, not a Space: when the class is made, the kernel base lowers it
(``finn.kernels.declaration.KernelDeclaration``) into ordinary members of the kernel, named
after it (``x``):

- ``x_element``, a derived value: the element it carries, refused as ``port-element``
  (an idle port without a ``dtype``, a producer given one). It reads its tensor or its
  ``dtype``, never its schedule, so a rule of its kernel reading a port's element
  (``self.x_element.bits``) does not wait on the kernel's folding;
- ``x_stream``, a derived value while the port is placed (given its tensor): its
  ``Stream`` (``finn.dataflow.streams``), its element (``x_element``) and the beat
  sequence it presents, refused as ``port-presentation`` or ``port-schedule``;
- ``x_data_width``, a derived value while it is placed: the bits of its words
  (``data_width``, from its stream), which a channel reads of the port producing into it;
- ``x_label``, a derived value: its label, ``x``, the one its contract carries;
- ``x_side``, a derived value while it is placed: the side it gives a channel
  (``finn.kernels.sides.Side.kernel``), its stream, its label and, on an output port
  (an initiator), its word width: what a channel's ``producer`` or ``consumer`` is bound
  to;
- ``x_access``, a derived value while it binds (placed, an input, reading indices): its
  read of the tensor, from which its kernel binds its indices' extents;
- ``x_element_taken``, a constraint where it states the integer policy its hardware takes
  (``admits``): its element is one it takes;
- ``x_contract``, a view while it is placed: the ``PortContract`` it presents to a
  channel, labelled ``x``, which the channel's network reads (the port's label, its pins,
  its pace);
- ``x_placed`` and ``x_binds``, the guards of these, where they are not known when the
  class is made.

Each argument is a constant or a member of its kernel (``tensor=x_tensor``,
``schedule=schedule``, ``admits=activation_policy``), which its members read by the name
the lowering fixes and declare (``__reads__``): never a dynamic read.

What it presents of its tensor: its kernel's ``Schedule`` projected through the indices it
reads (``index``), its lane order (``lanes``, outer first), the indices it is presented
after (``reduces``) or before (``holds``) and the reduction its marker closes (``closes``),
through a row-major view when ``reshaped``; or, for a traversal no schedule derives, a
given ``sequence``. Exactly one of the two. Its tensor is the one it reads, a fact its
kernel gives it, or the one it writes: a producer's (an initiator's) is its kernel's
statement of its output, derived from the kernel's facts, choices and input tensors, never
given. Its element is its tensor's, unless its hardware fixes one (``dtype``: a set port's
selector), which its channel holds the tensor to. Without a tensor it is a port not
used: idle, one lane of its ``dtype``.

Its pins, the AXIS bus, the pins it holds idle, its transport, its beats, its pace and its
contract are functions of its declaration and its kernel's values (its stream, or idle its
element, its ``clocking``, its schedule), not members: ``pins``, ``held``, ``axis``,
``transport``, ``beats``, ``pace``, ``contract``. The kernel's products call them (its
module's pins and holds, its cycles, ``x_contract``, ``x_data_width``); a kernel's rule
that reads a placed port's word width reads ``x_data_width``. Their physical refusals
(``port-markers``, ``port-signals``, ``port-lanes``, ``port-idle``) stay in them. It is an
AXIS bus named ``name``, with a ``TLAST`` when what it presents carries a marker, its
member pins spelled as its module spells them (``spelling``); or, given ``signals`` (data,
valid, ready), those ready/valid pins, with a native loop-completion vector when
``loop_ends`` names it (``olst``: the level each bit closes), else without a marker. Its
pins run on its kernel's clock and reset (``clocking``). ``native_stream`` declares one on
FinnLib's native pins (``idat``, ``ivld``, ``irdy`` for a target, ``odat``, ``ovld``,
``ordy`` for an initiator): a channel stage's (``fifo``, ``vpc``, ``input_gen``),
presenting the sequence it is given.

A kernel with children presents a child's port as its own: ``x = served(compute, "x")``
lowers to ``x_stream`` and ``x_data_width``, the selected child's, and ``x_label``,
``x_side`` and ``x_contract``, the child's labelled below it (``compute.packed.x``).

A port's endpoint is read off its class, before any design space (``port_endpoint``: a
served port's is its child's): the composer (``finn.kernels.composer``) places an output
port on a channel's producer side and an input port on its consumer side by it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from finn.core.space import (
    DefinitionError,
    Group,
    Inapplicable,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    reject,
    space_formals,
    view,
)
from finn.core.space._nodes import NodeDecision, NodeDecl, selected, unwrap
from finn.core.space.declarations import declared_path
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Access, Affine, Index, Pace, Refused, Schedule, plain_index
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd
from finn.kernels.artifacts.abi import Clocking, Direction, Endpoint, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import PortContract, UntimedKernel
from finn.kernels.declaration import (
    KernelDeclaration,
    Product,
    declarations,
    declared_reads,
    is_member,
)
from finn.kernels.sides import Side
from finn.kernels.transport import (
    AxisBeat,
    AxisSpelling,
    MarkerKind,
    ReadyValidTransport,
    StreamContract,
    StreamMarker,
)
from finn.kernels.values.domains import Integer, admit_element

if TYPE_CHECKING:
    from finn.kernels.port_model import PortSurface


def _applies(space: Space, guard: str | None) -> bool:
    """Whether a member guarded by the member ``guard`` of ``space`` applies: always where it
    has none; else where that member holds."""
    return guard is None or bool(getattr(space, guard))


@dataclass(frozen=True)
class LoopEnds:
    """A native loop-completion vector on ready/valid pins (FinnLib's ``olst``): its
    ``signal``, and the level each of its bits closes, ``levels[bit]``."""

    signal: str
    levels: tuple[LevelEnd, ...]


@dataclass(frozen=True)
class PhysicalPort:
    """Resolved port inputs for pure physical calculations, without authoring callbacks.

    A PortModel reads these literal fields in an ordinary method. This record is
    neither a Space nor a symbolic binding/allowed-read table.
    """

    name: str
    endpoint: Endpoint
    signals: tuple[str, ...] = ()
    loop_ends: LoopEnds | None = None
    spelling: AxisSpelling = AxisSpelling.LOWER
    lanes: tuple[Index, ...] = ()
    schedule: Schedule | None = None
    reduces: tuple[Index, ...] = ()
    holds: tuple[Index, ...] = ()


# -- the declaration -------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AxiStreamPort(KernelDeclaration):
    """One AXI-Stream port of a kernel's module: what it presents of its tensor, its
    element, its pins. The module docstring says what it lowers to."""

    name: str
    endpoint: Endpoint
    tensor: Tensor | None = None
    schedule: Schedule | None = None
    index: tuple[Index | Affine, ...] = ()
    lanes: tuple[Index, ...] = ()
    reduces: tuple[Index, ...] = ()
    holds: tuple[Index, ...] = ()
    closes: tuple[Index, ...] = ()
    reshaped: bool = False
    sequence: BeatSequence | None = None
    # The element its hardware fixes, if any; an idle port's pins carry it.
    dtype: QONNXDataType | None = None
    admits: Integer | None = None
    # Ready/valid pins (data, valid, ready) carrying the words instead of an AXIS bus.
    signals: tuple[str, ...] = ()
    # With ``signals``: the loop-completion vector its markers ride on, if any.
    loop_ends: LoopEnds | None = None
    # How the module spells the AXIS bus's member pins (HLS's are upper case).
    spelling: AxisSpelling = AxisSpelling.LOWER

    # The arguments its views read (placed, and idle: an idle port's lanes too), its
    # contract (its pace), and its stream.
    VIEWED: ClassVar[tuple[str, ...]] = ("name", "endpoint", "signals", "loop_ends", "spelling")
    IDLE: ClassVar[tuple[str, ...]] = ("lanes",)
    PACED: ClassVar[tuple[str, ...]] = (*VIEWED, "schedule", "reduces", "holds")
    PRESENTED: ClassVar[tuple[str, ...]] = (
        "name",
        "schedule",
        "sequence",
        "index",
        "reshaped",
        "tensor",
        "lanes",
        "reduces",
        "holds",
        "closes",
    )

    KIND: ClassVar[str] = "port"
    module_port: ClassVar[bool] = True

    def label(self) -> str:
        return self.name

    # -- what it lowers to -------------------------------------------------------------------

    def placement(self) -> bool | None:
        """Whether it is placed: True or False when the class is made, None where it reads
        whether its tensor is given (``<name>_placed``): a formal whose default is ``None``,
        an optional tensor (``x: Tensor | None = Param(default=None)``). A formal with any
        other default is always given."""
        tensor = self.tensor
        if tensor is None:
            return False
        if isinstance(tensor, Param) and tensor.default is None:
            return None
        if is_member(tensor) and getattr(tensor, "when", None) is not None:
            raise DefinitionError.broken(
                "port-tensor-guarded",
                "AxiStreamPort",
                f"{self.name}: its tensor is a guarded member; give it a tensor that is "
                "always stated, or a formal",
                declaration=self.name,
            )
        return True

    def binding(self) -> bool | None:
        """Whether its read of its tensor binds its kernel's extents, as ``placement``:
        None where ``<name>_binds`` reads it."""
        placed = self.placement()
        if placed is False:
            return False
        if not is_member(self.endpoint) and self.endpoint is not Endpoint.TARGET:
            return False  # a producer's tensor is derived from its kernel's extents
        if is_member(self.endpoint) or is_member(self.index):
            return None
        if not self.index:
            return False
        return placed

    def placed(self, kernel: Space, name: str) -> bool:
        """Whether it is placed on ``kernel``, declared as ``name``."""
        placement = self.placement()
        if placement is None:
            found: bool = getattr(kernel, f"{name}_placed")
            return found
        return placement

    def lower(self, owner: type[UntimedKernel], name: str) -> Mapping[str, object]:
        port = self
        members: dict[str, object] = {}
        placement = self.placement()
        placed = (f"{name}_placed",) if placement is None else ()
        guard: str | None = None
        if placement is None:

            @declared_reads(*self.paths("tensor"))
            def placed_(self: Space) -> bool:
                """Whether its tensor is given: the port is idle without it."""
                return port.value(self, "tensor") is not None

            guard = f"{name}_placed"
            members[guard] = derived(placed_)
        tensor = self.paths("tensor") if placement is not False else ()

        @declared_reads(*self.paths("name", "dtype", "endpoint"), *placed, *tensor)
        def element_(self: Space) -> ScalarEncoding | Rejected:
            return port.element(self, name)

        members[f"{name}_element"] = derived(element_)
        if self.admits is not None:

            @declared_reads(f"{name}_element", *self.paths("admits"))
            def element_taken(self: Space) -> bool | Rejected:
                """The element is one this port's hardware takes."""
                policy: Integer | None = port.value(self, "admits")
                if policy is None:
                    return True
                found: ScalarEncoding = getattr(self, f"{name}_element")
                return policy.check(found.dtype)

            members[f"{name}_element_taken"] = constraint(element_taken)
        if placement is False:
            return members

        # What a member applies under, a member of the kernel read first, before any other.
        under = () if guard is None else (guard,)

        @declared_reads(*under, *self.paths(*AxiStreamPort.PRESENTED), f"{name}_element")
        def stream(self: Space) -> Stream | Rejected | Inapplicable:
            if not _applies(self, guard):
                return Inapplicable()
            return port.presented(self, name)

        members[f"{name}_stream"] = derived(stream)

        @declared_reads(*under, f"{name}_stream", *self.paths(*AxiStreamPort.VIEWED))
        def data_width_(self: Space) -> int | Rejected | Inapplicable:
            if not _applies(self, guard):
                return Inapplicable()
            return data_width(
                port.read(self, *AxiStreamPort.VIEWED), getattr(self, f"{name}_stream")
            )

        members[f"{name}_data_width"] = derived(data_width_)

        @declared_reads()
        def label_(self: Space) -> str:
            return name

        members[f"{name}_label"] = derived(label_)
        # An output port's side carries its word width: read where its endpoint may say so.
        endpoint = self.endpoint
        widths = (
            (f"{name}_data_width",) if is_member(endpoint) or endpoint is Endpoint.INITIATOR else ()
        )

        @declared_reads(*under, f"{name}_stream", f"{name}_label", *widths, *self.paths("endpoint"))
        def side_(self: Space) -> Side | Inapplicable:
            if not _applies(self, guard):
                return Inapplicable()
            stream: Stream = getattr(self, f"{name}_stream")
            label: str = getattr(self, f"{name}_label")
            if port.value(self, "endpoint") is Endpoint.INITIATOR:
                return Side.kernel(stream, label, getattr(self, f"{name}_data_width"))
            return Side.kernel(stream, label)

        members[f"{name}_side"] = derived(side_)
        binding = self.binding()
        binds = guard
        if binding is None:

            @declared_reads(*placed, *self.paths("endpoint", "index"))
            def binds_(self: Space) -> bool:
                return (
                    port.placed(self, name)
                    and port.value(self, "endpoint") is Endpoint.TARGET
                    and bool(port.value(self, "index"))
                )

            binds = f"{name}_binds"
            members[binds] = derived(binds_)
        if binding is not False:

            @declared_reads(
                *(() if binds is None else (binds,)), *self.paths("tensor", "index", "reshaped")
            )
            def access(self: Space) -> Access | Inapplicable:
                if not _applies(self, binds):
                    return Inapplicable()
                return Access(
                    name,
                    port.value(self, "tensor").shape,
                    port.value(self, "index"),
                    port.value(self, "reshaped"),
                )

            members[f"{name}_access"] = derived(access)

        @declared_reads(*under, f"{name}_stream", *self.paths(*AxiStreamPort.PACED))
        def contract_(self: Space) -> PortContract | Rejected | Inapplicable:
            if not _applies(self, guard):
                return Inapplicable()
            kernel = cast("type[UntimedKernel]", type(self))
            found = contract(
                port.read(self, *AxiStreamPort.PACED),
                getattr(self, f"{name}_stream"),
                kernel.clock_domain,
            )
            return found if isinstance(found, Rejected) else PortContract(name, found)

        members[f"{name}_contract"] = view(contract_)
        return members

    def element(self, kernel: Space, name: str) -> ScalarEncoding | Rejected:
        """Its element on ``kernel`` (``<name>_element``): ``dtype`` when its hardware fixes
        one, placed or idle (its channel holds the tensor to it); else its tensor's. A
        producer's is its tensor's; an idle port without a ``dtype`` has none."""
        label = self.value(kernel, "name")
        dtype: QONNXDataType | None = self.value(kernel, "dtype")
        if dtype is not None:
            if self.value(kernel, "endpoint") is Endpoint.INITIATOR:
                return reject("port-element", f"{label}: a producer's element is its tensor's")
            return admit_element(dtype, None)
        if not self.placed(kernel, name):
            return reject("port-element", f"{label}: an idle port states its dtype")
        found: ScalarEncoding = self.value(kernel, "tensor").element
        return found

    def presented(self, kernel: Space, name: str) -> Stream | Rejected:
        """Its stream on ``kernel`` (``<name>_stream``): what it presents of its tensor, its
        schedule's projection or given, and its element (``<name>_element``)."""
        label = self.value(kernel, "name")
        schedule, given = self.value(kernel, "schedule"), self.value(kernel, "sequence")
        if (schedule is None) == (given is None):
            return reject("port-presentation", f"{label}: exactly one of a schedule or a sequence")
        if given is not None:
            sequence: BeatSequence = given
        else:
            index = self.value(kernel, "index")
            try:
                shaped = None
                if self.value(kernel, "reshaped"):
                    plain = [each for each in map(plain_index, index) if each is not None]
                    if len(plain) != len(index):
                        raise Refused(
                            "reshape-not-plain",
                            "AxiStreamPort.presented",
                            "a reshaped port reads plain indices",
                        )
                    shaped = tuple(schedule.extent(axis) for axis in plain)
                form = schedule.present(
                    self.value(kernel, "tensor").shape,
                    index,
                    lanes=self.value(kernel, "lanes"),
                    reduces=self.value(kernel, "reduces"),
                    holds=self.value(kernel, "holds"),
                    view=shaped,
                )
                closes = self.value(kernel, "closes")
                markers = (schedule.closing(closes),) if closes else ()
                sequence = BeatSequence(form, markers=markers)
            except Refused as error:
                return reject("port-schedule", f"{label}: {error}")
        element: ScalarEncoding = getattr(kernel, f"{name}_element")
        return Stream(element, sequence)

    # -- what a kernel's module reads of it -------------------------------------------------

    def reads(self, name: str, product: Product) -> tuple[str, ...]:
        placement = self.placement()
        placed = (f"{name}_placed",) if placement is None else ()
        stream = (f"{name}_stream",) if placement is not False else ()
        idle = (
            (f"{name}_element", *self.paths(*AxiStreamPort.IDLE)) if placement is not True else ()
        )
        if product == "pins":
            return (*placed, *stream, *self.paths(*AxiStreamPort.VIEWED), *idle)
        if product == "held":
            return (*placed, *self.paths(*AxiStreamPort.VIEWED), *idle)
        if product == "beats":
            return (*placed, *stream)
        binding = self.binding()
        if binding is False:
            return ()
        guards = (f"{name}_binds",) if binding is None else placed
        return (*guards, f"{name}_access")

    def viewed(self, kernel: UntimedKernel, name: str) -> tuple[AxiStreamPort, Presenting]:
        """What its views read on ``kernel``, declared as ``name``: its arguments, and its
        stream while placed, else its element alone (``<name>_element``)."""
        if self.placed(kernel, name):
            found: Stream = getattr(kernel, f"{name}_stream")
            return self.read(kernel, *AxiStreamPort.VIEWED), found
        element: ScalarEncoding = getattr(kernel, f"{name}_element")
        return self.read(kernel, *AxiStreamPort.VIEWED, *AxiStreamPort.IDLE), element

    def pins(self, kernel: UntimedKernel, name: str) -> tuple[Pin, ...] | Rejected:
        port, presenting = self.viewed(kernel, name)
        return pins(port, presenting, type(kernel).clock_domain)

    def held(self, kernel: UntimedKernel, name: str) -> Held | Rejected:
        if self.placed(kernel, name):
            return Held()
        port, presenting = self.viewed(kernel, name)
        return held(port, presenting, type(kernel).clock_domain)

    def presents(self, kernel: UntimedKernel, name: str) -> bool:
        return True  # a port's pins: placed, or held while idle

    def access(self, kernel: UntimedKernel, name: str) -> Access | None:
        binding = self.binding()
        if binding is None:
            binding = getattr(kernel, f"{name}_binds")
        elif binding:
            binding = self.placed(kernel, name)
        if not binding:
            return None
        found: Access = getattr(kernel, f"{name}_access")
        return found

    def beats(self, kernel: UntimedKernel, name: str) -> int | None:
        if not self.placed(kernel, name):
            return None
        return beats(getattr(kernel, f"{name}_stream"))


def native_stream(
    name: str, endpoint: Endpoint, tensor: Any, sequence: Any, loop_ends: Any = None
) -> AxiStreamPort:
    """A channel stage's port on FinnLib's native pins, presenting ``sequence`` of
    ``tensor``: ``idat``, ``ivld``, ``irdy`` for a target, ``odat``, ``ovld``, ``ordy``
    for an initiator, with the loop-completion vector ``loop_ends`` where given. Its kernel
    runs on FinnLib's ``clk`` and ``rst`` (``NATIVE_CLOCKING``). Declare it in a kernel's
    body (``input = native_stream("input", Endpoint.TARGET, input_tensor, arriving)``)."""
    side = "i" if endpoint is Endpoint.TARGET else "o"
    return AxiStreamPort(
        name=name,
        endpoint=endpoint,
        tensor=tensor,
        sequence=sequence,
        signals=(f"{side}dat", f"{side}vld", f"{side}rdy"),
        loop_ends=loop_ends,
    )


# -- its views: functions of its declaration and its kernel's values -------------------------

Presenting = Stream | ScalarEncoding
"""What a port's views read of what it presents: its stream while placed; idle, the element
its pins carry (``<name>_element``)."""


def _given(port: AxiStreamPort | PhysicalPort, name: str) -> Any:
    """Argument ``name`` of ``port`` as a value: a view reads its declaration with its
    arguments read from its kernel (``AxiStreamPort.read``)."""
    value = getattr(port, name)
    if is_member(value):
        raise TypeError(
            f"{port.name}: its {name} is a member of its kernel; read it first (AxiStreamPort.read)"
        )
    return value


def _element_of(presenting: Presenting) -> ScalarEncoding:
    return presenting.element if isinstance(presenting, Stream) else presenting


def _markers(presenting: Presenting) -> tuple[LevelEnd, ...]:
    return presenting.sequence.markers if isinstance(presenting, Stream) else ()


def lane_count(port: AxiStreamPort | PhysicalPort, presenting: Presenting) -> int | Rejected:
    """The lanes a beat carries: what it presents; idle, one."""
    if isinstance(presenting, Stream):
        return presenting.sequence.form.lanes
    lanes = _given(port, "lanes")
    if lanes:
        return reject(
            "port-idle",
            f"{_given(port, 'name')}: an idle port carries one lane, not lanes of {lanes}",
        )
    return 1


def axis(port: AxiStreamPort | PhysicalPort, presenting: Presenting) -> AxisBeat | Rejected:
    """Its AXIS beat: its element's lanes, with a ``TLAST`` when it presents a marker."""
    label = _given(port, "name")
    markers = _markers(presenting)
    if len(markers) > 1:
        return reject("port-markers", f"{label} has one TLAST; the sequence needs more")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    try:
        return AxisBeat(
            label,
            _element_of(presenting).dtype,
            lanes,
            endpoint=_given(port, "endpoint"),
            last=bool(markers),
            spelling=_given(port, "spelling"),
        )
    except ValueError as error:
        return reject("port-lanes", f"{label}: {error}")


def transport(
    port: AxiStreamPort | PhysicalPort, presenting: Presenting, clocking: Clocking | None
) -> ReadyValidTransport | Rejected:
    """Its ready/valid pins, on its kernel's clock and reset: its AXIS beat's, or its
    ``signals`` with its loop-completion vector."""
    clock = None if clocking is None else clocking.clock
    reset = None if clocking is None else clocking.reset
    label, signals = _given(port, "name"), _given(port, "signals")
    if not signals:
        beat = axis(port, presenting)
        return beat if isinstance(beat, Rejected) else beat.native(clock=clock, reset=reset)
    if len(signals) != 3:
        return reject("port-signals", f"{label}: signals are data, valid and ready")
    data, valid, ready = signals
    markers: tuple[StreamMarker, ...] = ()
    ends: LoopEnds | None = _given(port, "loop_ends")
    presented = _markers(presenting)
    if ends is not None:
        unclosed = [rule.beats for rule in presented if rule not in ends.levels]
        if not ends.levels or unclosed:
            return reject(
                "port-markers",
                f"{label}: no bit of {ends.signal} closes every {unclosed} beats"
                if unclosed
                else f"{label}: {ends.signal} closes no level",
            )
        markers = (StreamMarker(ends.signal, MarkerKind.LOOP_END, len(ends.levels)),)
    elif presented:
        return reject("port-markers", f"{label}: ready/valid pins carry no marker")
    lanes = lane_count(port, presenting)
    if isinstance(lanes, Rejected):
        return lanes
    bits = lanes * _element_of(presenting).bits
    endpoint = _given(port, "endpoint")
    try:
        return ReadyValidTransport(label, bits, endpoint, data, valid, ready, clock, reset, markers)
    except ValueError as error:
        return reject("port-signals", f"{label}: {error}")


def data_width(port: AxiStreamPort | PhysicalPort, presenting: Presenting) -> int | Rejected:
    """The bits of its words: its lanes times its element's bits, padded to bytes for
    AXIS."""
    found = transport(port, presenting, None)
    return found if isinstance(found, Rejected) else found.data_width


def pins(
    port: AxiStreamPort | PhysicalPort, presenting: Presenting, clocking: Clocking
) -> tuple[Pin, ...] | Rejected:
    """Its pins of the module: its ready/valid signals, or its AXIS bus."""
    if _given(port, "signals"):
        found = transport(port, presenting, clocking)
        return found if isinstance(found, Rejected) else found.pins()
    beat = axis(port, presenting)
    if isinstance(beat, Rejected):
        return beat
    return (beat.bus(clock=clocking.clock, reset=clocking.reset),)


def held(
    port: AxiStreamPort | PhysicalPort, presenting: Presenting, clocking: Clocking
) -> Held | Rejected:
    """While idle (its element alone): its forward pins of a target and its ready of an
    initiator held low; placed, none."""
    if isinstance(presenting, Stream):
        return Held()
    found = transport(port, presenting, clocking)
    if isinstance(found, Rejected):
        return found
    inputs: list[tuple[str, int]] = []
    unused: list[str] = []
    for signal in found.pins():
        if signal.direction is Direction.IN:
            inputs.append((signal.name, 0))
        else:
            unused.append(signal.name)
    return Held(tuple(inputs), tuple(unused))


def beats(stream: Stream) -> int:
    """The beats it carries a frame."""
    return stream.sequence.form.beats


def pace(port: AxiStreamPort | PhysicalPort) -> Pace | None:
    """When it presents each beat: its schedule, with its ``reduces`` and ``holds``; None
    for a given sequence."""
    schedule: Schedule | None = _given(port, "schedule")
    if schedule is None:
        return None
    return Pace(schedule, _given(port, "reduces"), _given(port, "holds"))


def contract(
    port: AxiStreamPort | PhysicalPort, stream: Stream, clocking: Clocking | None
) -> StreamContract | Rejected:
    """What it states to a channel: its transport, its stream (element, form, repetition,
    markers, each by the transport pin that carries it) and its pace."""
    sequence = stream.sequence
    found = transport(port, stream, clocking)
    if isinstance(found, Rejected):
        return found
    ends: LoopEnds | None = _given(port, "loop_ends")
    markers: tuple[tuple[str, LevelEnd], ...]
    if ends is not None:
        # Each bit whose level it presents, by its bit (``olst[1]``).
        markers = tuple(
            (f"{ends.signal}[{bit}]", level)
            for bit, level in enumerate(ends.levels)
            if level in sequence.markers
        )
    elif found.markers:
        markers = ((found.markers[0].signal, sequence.markers[0]),)
    else:
        markers = ()
    return StreamContract(
        found, stream.element, sequence.form, sequence.repetition, markers, pace(port)
    )


# -- a port a kernel with children presents for a child -------------------------------------


@dataclass(frozen=True, eq=False)
class Served(KernelDeclaration):
    """A port a kernel with children presents as its child's: ``child`` (a child of the
    kernel, or a selector over children, ``Select``) and that child's port ``port``. It
    lowers to ``<name>_stream`` and ``<name>_data_width``, the selected child's stream and
    word width, and ``<name>_label``, ``<name>_side`` and ``<name>_contract``, the child's
    label, side and contract labelled below the kernel (``compute.<case>.x``; ``child.x``
    for a child; ``<labelled>.x`` where the kernel states the label its child keeps: a
    channel's source, ``source.memstream.set``)."""

    child: Any
    port: str
    labelled: str | None = None

    def lower(self, owner: type[UntimedKernel], name: str) -> Mapping[str, object]:
        child = next(
            (
                attribute
                for base in owner.__mro__
                for attribute, value in vars(base).items()
                if value is self.child
            ),
            None,
        )
        if child is None or not hasattr(self.child, f"{self.port}_stream"):
            raise DefinitionError.broken(
                "served-child-unknown",
                "Served",
                f"{owner.__qualname__}.{name}: serves {self.port} of no child of the kernel "
                "that declares it",
                declaration=f"{owner.__qualname__}.{name}",
            )
        given = getattr(self.child, f"{self.port}_stream")
        widths = getattr(self.child, f"{self.port}_data_width")
        labels = getattr(self.child, f"{self.port}_label")
        sides = getattr(self.child, f"{self.port}_side")
        offered = getattr(self.child, f"{self.port}_contract")
        members: dict[str, object] = {}

        def stream(stream: Stream) -> Stream:
            return stream

        def width(width: int) -> int:
            return width

        members[f"{name}_stream"] = derived(stream=given)(stream)
        members[f"{name}_data_width"] = derived(width=widths)(width)
        try:
            case = selected(self.child)  # a selector over children: the case it selects
        except DefinitionError:
            case = None
        # Its label, side and contract, each the child's below the same label: the one the
        # kernel states its child keeps, the selected case's, or the child's.
        if self.labelled is not None:
            above = self.labelled

            def stated_label(label: str) -> str:
                return f"{above}.{label}"

            def stated_side(side: Side) -> Side:
                return side.under(above)

            def stated(contract: PortContract) -> PortContract:
                return contract.under(above)

            members[f"{name}_label"] = derived(label=labels)(stated_label)
            members[f"{name}_side"] = derived(side=sides)(stated_side)
            members[f"{name}_contract"] = view(contract=offered)(stated)
        elif case is not None:

            def chosen_label(case: str, label: str) -> str:
                return f"{child}.{case}.{label}"

            def chosen_side(case: str, side: Side) -> Side:
                return side.under(f"{child}.{case}")

            def chosen(case: str, contract: PortContract) -> PortContract:
                return contract.under(f"{child}.{case}")

            members[f"{name}_label"] = derived(case=case, label=labels)(chosen_label)
            members[f"{name}_side"] = derived(case=case, side=sides)(chosen_side)
            members[f"{name}_contract"] = view(case=case, contract=offered)(chosen)
        else:

            def placed_label(label: str) -> str:
                return f"{child}.{label}"

            def placed_side(side: Side) -> Side:
                return side.under(child)

            def placed(contract: PortContract) -> PortContract:
                return contract.under(child)

            members[f"{name}_label"] = derived(label=labels)(placed_label)
            members[f"{name}_side"] = derived(side=sides)(placed_side)
            members[f"{name}_contract"] = view(contract=offered)(placed)
        return members


def served(child: Any, port: str, *, label: str | None = None) -> Served:
    """A port a kernel with children presents as ``child``'s ``port`` (``Served``), labelled
    below the child's node, or below the ``label`` the kernel states its child keeps.
    Declare it in the class body after the child (``x = served(compute, "x")``)."""
    return Served(child, port, label)


def _child_types(child: object) -> tuple[type[Space], ...]:
    """The classes a ``Served`` port's child may be: a child's, or each case's of a selector
    over children (a ``None`` case places none)."""
    record = unwrap(child)
    if isinstance(record, NodeDecision):
        return tuple(
            case.space_type
            for case in record.candidates.values()
            if case is not None and issubclass(case.space_type, Space)
        )
    if isinstance(record, NodeDecl) and issubclass(record.space_type, Space):
        return (record.space_type,)
    return ()


def port_endpoint(space_type: type[Space], port: str) -> Endpoint | None:
    """The endpoint port ``port`` of ``space_type`` declares, as the class states it: an
    ``AxiStreamPort``'s (``INITIATOR`` for an output, ``TARGET`` for an input), or a served
    port's, its child's (every case's, where they agree). None where the class states no
    stream port of that name, or where its endpoint is not known when the class is made (an
    endpoint read from a member, or cases that disagree)."""
    if not (isinstance(space_type, type) and issubclass(space_type, UntimedKernel)):
        return None
    # A real port child's static endpoint is its explicit binding or formal default.
    from finn.kernels.port_model import PortModel  # noqa: PLC0415 - breaks base/port cycle

    child = getattr(space_type, port, None)
    if isinstance(child, PortModel):
        record = unwrap(child)
        if not isinstance(record, NodeDecl):
            return None
        endpoint = record.bindings.get(
            "endpoint", space_formals(record.space_type)["endpoint"].default
        )
        return endpoint if isinstance(endpoint, Endpoint) else None
    declared = declarations(space_type, UntimedKernel).get(port)
    if isinstance(declared, AxiStreamPort):
        return None if is_member(declared.endpoint) else declared.endpoint
    if isinstance(declared, Served):
        found = {port_endpoint(child, declared.port) for child in _child_types(declared.child)}
        return found.pop() if len(found) == 1 else None
    return None


def port_surface(namespace: type[Space] | Space, port: str) -> PortSurface:
    """Actual references for a validated logical port key, at an authoring namespace.

    Child ports and still-unmigrated declarations are distinct authored forms.
    The latter's already-lowered members are captured here only for drivers and
    composition; this metadata never authorizes a receiver's reads. No aliases,
    exception fallback or guessed same-spelled references repair an unknown key.
    Parent served declarations intentionally expose no physical/element fields.
    """
    from finn.kernels.port_model import PortModel, PortSurface  # noqa: PLC0415

    space_type = namespace if isinstance(namespace, type) else type(namespace)
    if not issubclass(space_type, Space) or (
        not isinstance(namespace, type) and declared_path(namespace) is None
    ):
        raise TypeError("port_surface requires a Space class or node declaration")
    # Dotted keys from discovery traverse lexical Group frames only.
    frame: object = namespace
    segments = port.split(".")
    for segment in segments[:-1]:
        group = getattr(frame, segment, None)
        if not isinstance(group, Group):
            raise KeyError(port)
        frame = group
    child = getattr(frame, segments[-1], None)
    if isinstance(child, PortModel):
        return PortSurface.of(child)
    if len(segments) != 1 or not issubclass(space_type, UntimedKernel):
        raise KeyError(port)
    declaration = declarations(space_type, UntimedKernel).get(port)
    if not isinstance(declaration, (AxiStreamPort, Served)):
        raise KeyError(port)
    element: ScalarEncoding | None = (
        getattr(namespace, f"{port}_element") if isinstance(declaration, AxiStreamPort) else None
    )
    if isinstance(declaration, AxiStreamPort) and declaration.placement() is False:
        return PortSurface(element, port_endpoint(space_type, port), None, None)
    return PortSurface(
        element,
        port_endpoint(space_type, port),
        None,
        None,
        getattr(namespace, f"{port}_stream"),
        getattr(namespace, f"{port}_data_width"),
        getattr(namespace, f"{port}_side"),
        getattr(namespace, f"{port}_contract"),
    )


__all__ = [
    "AxiStreamPort",
    "LoopEnds",
    "Served",
    "axis",
    "beats",
    "Presenting",
    "PhysicalPort",
    "contract",
    "data_width",
    "held",
    "lane_count",
    "native_stream",
    "port_endpoint",
    "port_surface",
    "pace",
    "pins",
    "served",
    "transport",
]
