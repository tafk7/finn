# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's ports: real child Spaces, and the ports a kernel with children serves.

``StreamPort`` is the one stream-port construct: a child Space with zero Decisions and
intrinsic element admission at the port's real scope, presenting a given sequence or its
kernel's schedule's projection, or idle. Folding and composition remain on the parent. A
kernel with children presents a child's port as its own by serving it (``served``), which
its base lowers into the members a channel reads; nothing forwards a port by hand.

Public physical Views characterize the child's realization. Module assembly (the
base's, ``Kernel.netlist``) calls each port's ordinary methods, ``pin_values``,
``held_values`` and ``pins_presented``, over literal model inputs and the pure
physical functions; it must not read these Views, so that it collects directly
returned refusals. ``OtherPins`` is the one raw pin declaration: a module's pins
outside its ports and its control bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finn.core.space import (
    Group,
    Inapplicable,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    reject,
    view,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.schedule import Access, Affine, Index, Pace, Refused, Schedule, plain_index
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Clocking, Direction, Pin
from finn.kernels.artifacts.module import Held
from finn.kernels.base import PortPins
from finn.kernels.laws import Advance, TransportKind
from finn.kernels.port import (
    LoopEnds,
    PhysicalPort,
    Presenting,
    data_width,
    held,
    pins,
    stream_endpoint,
    stream_refusal,
)
from finn.kernels.port import stream_pins as pins_of
from finn.kernels.sides import Side
from finn.kernels.values.domains import Integer, admit_element


class StreamPort(Space):
    """A stream port: where one tensor leaves or enters its kernel as one stream, the one
    construct every port of a kernel with a module is declared by (CONTRACT.md § 7). It is
    declared as a module header states a port, its **header**, with exactly one
    **presentation** and its **relation**; it is read by its readouts.

    Its header: its ``direction`` (``finn.kernels.artifacts.abi.Direction``, ``IN`` or
    ``OUT``, stated, never ``INOUT``: ``port-direction``), from which its bus role follows
    (``finn.kernels.port.stream_endpoint``); its tensor, an idle port's element (``dtype``)
    and the elements an input admits (``admits``); its ``transport`` kind
    (``finn.kernels.laws``: ``AXIS``, ``NATIVE``, or the names its RTL states,
    ``Axis(bus=...)``, ``Native(prefix=...)``), never its pins: they, its idle holds and its
    packing follow from the library. ``label`` is the port's name in its kernel and
    ``clocking`` its kernel's clock domain (L8: the kernel's, never the port's): a kernel
    supplies both for a port declared in its own body that does not state them
    (``finn.kernels.base.Kernel``). Its ``transport`` has no default: a native port taken
    for AXIS would be a bus the RTL does not have.

    Its presentation, exactly one of: its kernel's ``schedule`` projected through its
    ``index`` (``lanes``, ``reduces``, ``holds``, ``closes``, ``reshaped``), or a given
    ``sequence`` (a stage's side); with either, a native port's ``loop_ends``. Its kernel's
    class refuses a port binding none, both, or another presentation's parameters
    (``port-presentation``, ``finn.kernels.base``). A port with no tensor is idle and
    presents nothing, whatever it declares; it states its element (``dtype``).

    Its relation, with a given sequence: an input selecting another port's beats
    (``governs``), an output of stored values (``stored``); its advance rule follows
    (``advance``). Its readouts: ``placed``, ``element``, ``stream``, ``advance``, ``pace``,
    ``width``, ``side``, ``binds`` and ``access`` (an input with an index reads its tensor
    to bind its kernel's extents), derived; ``stream_pins``, a view. A placed port's
    presentation is accepted and carried, or its design point is refused (``presented``,
    R3's level 1).
    """

    # -- its header: what a module header states of the port ----------------------------
    direction: Direction = Param()
    tensor: Tensor | None = Param(default=None)
    dtype: QONNXDataType | None = Param(default=None)
    admits: Integer | None = Param(default=None)
    transport: TransportKind = Param()
    label: str = Param()
    clocking: Clocking = Param()

    # -- its presentation: exactly one of (a) and (b), checked when its kernel's class is made
    # (a) a projection of its kernel's schedule
    schedule: Schedule | None = Param(default=None)
    index: tuple[Index | Affine, ...] = Param(default=())
    lanes: tuple[Index, ...] = Param(default=())
    reduces: tuple[Index, ...] = Param(default=())
    holds: tuple[Index, ...] = Param(default=())
    closes: tuple[Index, ...] = Param(default=())
    reshaped: bool = Param(default=False)
    # (b) a given sequence, a stage's side
    sequence: BeatSequence | None = Param(default=None)
    # with (a) or (b): a native port's loop-completion vector (L9)
    loop_ends: LoopEnds | None = Param(default=None)

    # -- its relation: what makes a port of a given sequence selected or sourced ---------
    governs: str | None = Param(default=None)
    stored: bool = Param(default=False)

    @derived
    def placed(self) -> bool:
        return self.tensor is not None

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        dtype = self.dtype
        if dtype is not None:
            if self.direction is Direction.OUT:
                return reject("port-element", f"{self.label}: an output's element is its tensor's")
            return admit_element(dtype, None)
        tensor = self.tensor
        if tensor is None:
            return reject("port-element", f"{self.label}: an idle port states its dtype")
        return tensor.element

    @constraint
    def element_taken(self) -> bool | Rejected:
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def advance(self) -> Advance:
        """Its advance rule, by how it is declared: scheduled, a projection of its kernel's
        schedule; selected, a selector port, which ``governs`` the port whose beats it goes
        with; sourced, a memory's output, ``stored``; else buffered, a stage's side."""
        if self.schedule is not None:
            return Advance.SCHEDULED
        if self.governs is not None:
            return Advance.SELECTED
        if self.stored:
            return Advance.SOURCED
        return Advance.BUFFERED

    @derived
    def pace(self) -> Pace | Inapplicable:
        """Its pace (CONTRACT.md § 4), a scheduled port's only: the iterations of its
        kernel's schedule at which it moves a beat (``Pace.times``, a reduced index's on its
        last value, a held one's on its first; an output's are when it issues them) and the
        iterations a frame takes (``Pace.span``). Derived, never declared. A port of
        another rule has none (its rule states its relation), nor has an idle port."""
        if not self.placed:
            return Inapplicable()
        schedule = self.schedule
        if schedule is None:
            return Inapplicable()
        return Pace(schedule, self.reduces, self.holds)

    @derived
    def stream(self) -> Stream | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        schedule, given = self.schedule, self.sequence
        if (schedule is None) == (given is None):
            # Its kernel's class admits a port binding exactly one (``port-presentation``):
            # here a bound presentation answers nothing at this design point.
            answered = "neither" if schedule is None else "both"
            return reject(
                "port-presented",
                f"{self.label}: its presentation answers {answered} of a schedule and a sequence",
            )
        if given is not None:
            sequence = given
        else:
            assert schedule is not None
            index = self.index
            try:
                shaped = None
                if self.reshaped:
                    plain = [each for each in map(plain_index, index) if each is not None]
                    if len(plain) != len(index):
                        raise Refused(
                            "reshape-not-plain",
                            "AxiStreamPort.presented",
                            "a reshaped port reads plain indices",
                        )
                    shaped = tuple(schedule.extent(axis) for axis in plain)
                tensor = self.tensor
                assert tensor is not None
                form = schedule.present(
                    tensor.shape,
                    index,
                    lanes=self.lanes,
                    reduces=self.reduces,
                    holds=self.holds,
                    reshape=shaped,
                )
                closes = self.closes
                markers = (schedule.closing(closes),) if closes else ()
                sequence = BeatSequence(form, markers=markers)
            except Refused as error:
                return reject("port-schedule", f"{self.label}: {error}")
        stream = Stream(self.element, sequence)
        unsupported = stream_refusal(stream)
        return unsupported if unsupported is not None else stream

    @derived
    def width(self) -> int | Rejected | Inapplicable:
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        return data_width(self.physical_spec(idle=False), stream)

    @constraint
    def presented(self) -> bool | Rejected:
        """R3's level 1 (``port-presented``): a stream port is an input or an output, and a
        placed port's presentation is applicable, accepted and carried: its ``stream`` (its
        schedule's projection, or its given sequence, on the physical schemas) and its
        ``width`` (its transport carries the stream) are answers, or its design point is
        refused with their refusal (a native port presenting markers its pins cannot
        carry, ``port-markers``). An idle port presents nothing."""
        if self.direction not in (Direction.IN, Direction.OUT):
            return reject("port-direction", f"{self.label}: a stream port is an input or an output")
        if not self.placed:
            return True
        _ = self.stream
        _ = self.width
        return True

    @derived
    def side(self) -> Side | Inapplicable:
        """The side it gives a channel: its stream, its label and, an output, its width."""
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        label = self.label
        if self.direction is Direction.OUT:
            return Side.kernel(stream, label, self.width)
        return Side.kernel(stream, label)

    @derived
    def binds(self) -> bool:
        """Whether it reads its tensor to bind its kernel's extents: a placed input with an
        index."""
        if not self.placed:
            return False
        return self.direction is Direction.IN and bool(self.index)

    @derived
    def access(self) -> Access | Inapplicable:
        """Only a placed input with indices reads its tensor to bind parent extents."""
        if not self.binds:
            return Inapplicable()
        tensor = self.tensor
        assert tensor is not None
        return Access(self.label, tensor.shape, self.index, self.reshaped)

    def physical_spec(self, *, idle: bool) -> PhysicalPort:
        """Resolved physical fields in existing read order; placed ports skip idle lanes.

        An undecorated native method: no receiver escapes as a model value and
        no generated names or manually declared reads participate.
        """
        return PhysicalPort(
            name=self.label,
            endpoint=stream_endpoint(self.direction),
            transport=self.transport,
            loop_ends=self.loop_ends,
            lanes=self.lanes if idle else (),
        )

    def presenting(self) -> Presenting:
        """Model input to pure pin assembly: its stream, placed; idle, its element."""
        return self.stream if self.placed else self.element

    # -- what its kernel's module takes of it (``Kernel.netlist``, ``Kernel.cycles``) -------

    def pin_values(self) -> tuple[Pin, ...] | Rejected:
        """Its pins of the module, from its transport (``finn.kernels.port.pins``): placed,
        its stream's; idle, one lane of its element."""
        presenting = self.presenting()
        return pins(
            self.physical_spec(idle=not isinstance(presenting, Stream)),
            presenting,
            self.clocking,
        )

    def held_values(self) -> Held | Rejected:
        """Placed, none; idle, its inputs held (L10)."""
        if self.placed:
            return Held()
        return held(self.physical_spec(idle=True), self.element, self.clocking)

    def pins_presented(self) -> bool:
        """Placed, its pins are presented (a channel connects them)."""
        placed: bool = self.placed
        return placed

    def frame_beats(self) -> int | None:
        """Placed, the beats a frame its stream carries; idle, None: what its kernel's
        cycles read of it where the kernel has no schedule (``Kernel.cycles``, the rate
        law)."""
        if not self.placed:
            return None
        beats: int = self.stream.sequence.form.beats
        return beats

    @view
    def physical_pins(self) -> tuple[Pin, ...] | Rejected:
        return self.pin_values()

    @view
    def idle_pins(self) -> Held | Rejected:
        return self.held_values()

    @view
    def stream_pins(self) -> PortPins | Rejected | Inapplicable:
        """What a channel wires to, realization: its stream on its transport's pins, under
        its label (``finn.kernels.port.stream_pins``). Its contract is its stream, its
        carrier (``transport``, ``width``), its label and, scheduled, its pace."""
        if not self.placed:
            return Inapplicable()
        stream = self.stream
        found = pins_of(self.physical_spec(idle=False), stream, self.clocking)
        return found if isinstance(found, Rejected) else PortPins(self.label, found)


class OtherPins(Group):
    """Pins of a module that are no stream port's and no control bus's (a mode pin), and
    what holds its inputs: the one raw pin declaration a kernel keeps (CONTRACT.md § 7).
    Its pins go before its kernel's control bus and ports, its holds after its ports'."""

    signals: tuple[Pin, ...] = Param()
    holding: Held = Param(default=Held())

    def pin_values(self) -> tuple[Pin, ...]:
        return self.signals

    def held_values(self) -> Held:
        return self.holding

    def pins_presented(self) -> bool:
        """Never presented: its inputs are held, or the module is refused (``kernel-pins``)."""
        return False


@dataclass(frozen=True)
class ServedPort:
    """A port a kernel with children presents as one of its child's stream ports (canon
    AUTHORING's ``served``): ``child``, a child of the kernel or a selector over children,
    and that child's port ``port``. Its stream, pins, pace, advance rule and intake are the
    child's port's; its label is the port's below the child's case (``compute.<case>.x`` for
    a selector's, MatMul's), or below the ``label`` the kernel states for its child
    (``source.memstream.set``, a channel's source's).

    Declared in a kernel's own body, after its child (``x = served(compute, "x")``); when
    the kernel's class is made (``finn.kernels.base.Kernel``), it is lowered into the
    members a channel and its sizing read of a port, ``<name>_side`` and
    ``<name>_stream_pins``, each the child's port's labelled below it, ``<name>_pace`` and
    ``<name>_advance``, the child's port's, and, for a selector, ``<child>_case``, its case.
    A reader finds them by the port's name (``finn.kernels.port.port_side``,
    ``port_stream_pins``, ``port_pace``, ``port_advance``), never by the members' names. Its
    intake is its child's, under its name (``finn.kernels.base.Kernel.intake``)."""

    child: object
    port: str
    label: str | None = None


def served(child: Any, port: str, *, label: str | None = None) -> ServedPort:
    """A port the kernel declaring it presents as ``child``'s stream port ``port``
    (``ServedPort``)."""
    return ServedPort(child, port, label)


def stream_ports(space_type: type[Space]) -> dict[str, StreamPort]:
    """The stream ports ``space_type`` declares, in inherited declaration order, through
    its Groups (by dotted path, ``group.port``), each bound where it is declared. Stops at
    child Spaces: a child's ports are not its parent's; a served port is not one."""

    def below(
        namespace: type[Space] | type[Group], placed: Space | Group | None, prefix: str
    ) -> dict[str, StreamPort]:
        effective: dict[str, object] = {}
        for base in reversed(namespace.__mro__):
            effective.update(vars(base))
        result: dict[str, StreamPort] = {}
        for name, declaration in effective.items():
            if isinstance(declaration, StreamPort):
                result[prefix + name] = declaration if placed is None else getattr(placed, name)
            elif isinstance(declaration, Group):
                group = declaration if placed is None else getattr(placed, name)
                result.update(below(type(declaration), group, prefix + name + "."))
        return result

    if not isinstance(space_type, type) or not issubclass(space_type, Space):
        raise TypeError("stream_ports requires a real Space class")
    return below(space_type, None, "")


__all__ = [
    "OtherPins",
    "ServedPort",
    "StreamPort",
    "served",
    "stream_ports",
]
