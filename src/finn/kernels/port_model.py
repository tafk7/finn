# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's ports: real child Spaces, and the ports a kernel with children exposes.

``StreamPort`` is the one stream-port construct, declared as one of its two kinds, an input
(``InPort``) or an output (``OutPort``): a child Space with zero Decisions and intrinsic
element admission at the port's real scope, presenting a given sequence or its kernel's
schedule's projection, or idle. Folding and composition remain on the parent. A
kernel with children exposes a member's port as its own by a node alias (``x = compute.x``,
``finn.core.space``): the port stays its member's, one node read by two names, and nothing
forwards it by hand. ``exposed_ports`` is a kernel's inventory of both.

Public physical Views characterize the child's realization. Module assembly (the
base's, ``Kernel.netlist``) calls each port's ordinary methods, ``pin_values``,
``held_values`` and ``pins_presented``, over literal model inputs and the pure
physical functions; it must not read these Views, so that it collects directly
returned refusals. ``OtherPins`` is the one raw pin declaration: a module's pins
outside its ports and its control bus.
"""

from __future__ import annotations

from typing import Any, cast

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
from finn.kernels.laws import RATE_INTERVAL, Advance, Interval, TimedPace, TransportKind
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
    construct every port of a kernel with a module is declared by (CONTRACT.md § 7), as one
    of its two kinds: an input (``InPort``) or an output (``OutPort``). It is declared as a
    module header states a port, its **header**, with exactly one **presentation** and its
    **relation**; it is read by its readouts.

    Its header: its kind, the class it is declared by, from which its ``direction``
    (``finn.kernels.artifacts.abi.Direction``, ``IN`` or ``OUT``, a read-only derived value)
    and its bus role follow (``finn.kernels.port.stream_endpoint``); a ``StreamPort`` itself
    is neither, ``INOUT``, and refused (``port-direction``). Its tensor and the elements it
    admits (``admits``); its ``transport`` kind (``finn.kernels.laws``: ``AXIS``, ``NATIVE``,
    or the names its RTL states, ``Axis(bus=...)``, ``Native(prefix=...)``), never its pins:
    they, its idle holds and its packing follow from the library. ``label`` is the port's
    name in its kernel, ``clocking`` its kernel's clock and reset (L8: the kernel's, never
    the port's) and ``interval`` its kernel's initiation interval (L6,
    ``finn.kernels.laws.Interval``), which it carries with its pace; its ``transport``, where
    it states none, is its kernel's interface protocol's
    (``finn.kernels.laws.InterfaceProtocol``): a kernel supplies all four for a port declared
    in its own body that does not state them (``finn.kernels.base.Kernel``). A port with its
    own pin names, or of another interface protocol than its kernel's (MemStream's AXIS-named
    output), states its transport.

    Its presentation, exactly one of: its kernel's ``schedule`` projected through its
    ``index`` (``lanes``, ``reduces``, ``holds``, ``closes``, ``reshaped``), or a given
    ``sequence`` (a stage's side); with either, a native port's ``loop_ends``. Its kernel's
    class refuses a port binding none, both, or another presentation's parameters
    (``port-presentation``, ``finn.kernels.base``). A port with no tensor is idle and
    presents nothing, whatever it declares.

    Its relation, with a given sequence, is its kind's: an input naming the sets another
    port's groups are made from (``InPort.governs``, a selector), a memory's output
    (``OutPort.sourced``); its advance rule follows (``advance``, its firing relation, L2-L5).
    Its readouts: ``placed``, ``element``, ``stream``, ``advance``, ``pace``, ``timed_pace``,
    ``group`` (a selector's), ``width``, ``side``, ``binds`` and ``access`` (an input with an
    index reads its tensor to bind its kernel's extents), derived;
    ``stream_pins``, a view. A placed port's presentation is accepted and carried, or its
    design point is refused (``presented``).
    """

    # -- its header: what a module header states of the port ----------------------------
    # (its kind, InPort or OutPort, is its class)
    tensor: Tensor | None = Param(default=None)
    admits: Integer | None = Param(default=None)
    transport: TransportKind = Param()
    label: str = Param()
    clocking: Clocking = Param()
    interval: Interval = Param(default=RATE_INTERVAL)

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

    # -- its relation: its kind's (InPort.governs, OutPort.sourced) ------------------------

    @derived
    def direction(self) -> Direction:
        """Whether it is an input or an output, by its kind, read-only: what its bus role
        and its ABI follow from. A ``StreamPort`` of neither kind is ``INOUT``, which no
        stream is (``port-direction``)."""
        return Direction.INOUT

    @derived
    def placed(self) -> bool:
        return self.tensor is not None

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        """Its tensor's element; an idle output has none (an input states its own,
        ``InPort.dtype``)."""
        tensor = self.tensor
        if tensor is None:
            return reject("port-element", f"{self.label}: an idle output presents no element")
        return tensor.element

    @constraint
    def element_taken(self) -> bool | Rejected:
        policy = self.admits
        return True if policy is None else policy.check(self.element.dtype)

    @derived
    def advance(self) -> Advance:
        """Its advance rule, by how it is declared: scheduled, a projection of its kernel's
        schedule; else stepped, a stage's side. An input that ``governs`` another port is
        selected (``InPort``), a ``sourced`` output sourced (``OutPort``)."""
        if self.schedule is not None:
            return Advance.SCHEDULED
        return Advance.STEPPED

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
    def timed_pace(self) -> TimedPace | Inapplicable:
        """Its pace in clock cycles: its ``pace`` at its kernel's ``interval``
        (``finn.kernels.laws.TimedPace``, by the one conversion, ``in_cycles``), a scheduled
        port's only. Read through a node alias (``x = compute.x``), it is its own kernel's,
        however deep: what FIFO sizing reads of the port beside a channel."""
        if not self.placed or self.schedule is None:
            return Inapplicable()
        return TimedPace(self.pace, self.interval)

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
        """A stream port is an input or an output, and a placed port's presentation is
        applicable, accepted and carried: its ``stream`` (its schedule's projection, or its
        given sequence, on the physical schemas) and its ``width`` (its transport carries the
        stream) are answers, or its design point is refused with their refusal (a native
        port presenting markers its pins cannot carry, ``port-markers``). An idle port
        presents nothing. Its kernel's class refuses what it sees of these in a declaration
        (a ``StreamPort`` of neither kind, ``port-direction``); this refuses one used outside
        a kernel's class, at the design point."""
        if self.direction is Direction.INOUT:
            return reject(
                "port-direction",
                f"{self.label}: a stream port is an input or an output (InPort, OutPort)",
            )
        if not self.placed:
            return True
        _ = self.stream
        _ = self.width
        if self.advance is Advance.SELECTED:
            _ = self.group
        return True

    @derived
    def group(self) -> int | Rejected | Inapplicable:
        """A selector's group (R2, L4): the governed port's beats its one beat names the set
        of (``InPort.group``); a port of another rule has none."""
        return Inapplicable()

    @derived
    def side(self) -> Side | Inapplicable:
        """The side it gives a channel: its stream and its label (an output's, its width
        too, ``OutPort``)."""
        if not self.placed:
            return Inapplicable()
        return Side.kernel(self.stream, self.label)

    @derived
    def binds(self) -> bool:
        """Whether it reads its tensor to bind its kernel's extents: a placed input with an
        index (``InPort``); an output never does."""
        return False

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

    # -- what its kernel's module takes of it (``Kernel.netlist``, ``Kernel.frame_cycles``) -

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
        cycles read of it where the kernel has no schedule (``Kernel.frame_cycles``, the rate
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


class InPort(StreamPort):
    """An input stream port (``StreamPort``): its stream enters its kernel, a target (AXIS
    slave, ``s_axis_*``). An input's own: the element it states where its tensor's is not
    its (``dtype``; an idle input's), and its relation with a given sequence, the port whose
    beats it selects (``governs``, a selector: its advance is selected, its ``group`` the
    governed port's beats each of its beats names the set of). It reads its tensor
    to bind its kernel's extents where it has an index (``binds``, ``access``)."""

    dtype: QONNXDataType | None = Param(default=None)
    governs: str | None = Param(default=None)
    # A selector's: the beats a frame of the port it governs, None where that is idle; its
    # kernel supplies it (``finn.kernels.base.Kernel``).
    governed_beats: int | None = Param(default=None)

    @derived
    def direction(self) -> Direction:
        return Direction.IN

    @derived
    def element(self) -> ScalarEncoding | Rejected:
        """Its ``dtype``'s element where it states one, else its tensor's; an idle input
        states its ``dtype``."""
        dtype = self.dtype
        if dtype is not None:
            return admit_element(dtype, None)
        tensor = self.tensor
        if tensor is None:
            return reject("port-element", f"{self.label}: an idle input states its dtype")
        return tensor.element

    @derived
    def advance(self) -> Advance:
        """Scheduled, a projection of its kernel's schedule; selected, a selector, which
        ``governs`` the port whose groups it names; else stepped, a stage's side."""
        if self.schedule is not None:
            return Advance.SCHEDULED
        if self.governs is not None:
            return Advance.SELECTED
        return Advance.STEPPED

    @derived
    def group(self) -> int | Rejected | Inapplicable:
        """A placed selector's group: the governed port's beats a frame over its own, the
        governed port's beats each of its beats names the set of (L4), refused where it does
        not divide them, or the governed port is idle (``port-group``); a port it governs
        none of has none."""
        governs = self.governs
        if not self.placed or governs is None:
            return Inapplicable()
        governed = self.governed_beats
        if governed is None:
            return reject("port-group", f"{self.label}: governs {governs}, which is idle")
        beats: int = self.stream.sequence.form.beats
        if beats == 0 or governed % beats:
            return reject(
                "port-group",
                f"{self.label}: {beats} beats a frame do not group {governs}'s {governed}",
            )
        return governed // beats

    @derived
    def binds(self) -> bool:
        """Whether it reads its tensor to bind its kernel's extents: placed, with an index."""
        if not self.placed:
            return False
        return bool(self.index)


class OutPort(StreamPort):
    """An output stream port (``StreamPort``): its stream leaves its kernel, an initiator
    (AXIS master, ``m_axis_*``). Its element is its tensor's. An output's own: its relation
    with a given sequence, ``sourced`` (a memory's output, each beat its stored value at its
    position of its consumer's order, issued whenever its consumer takes one: its advance is
    sourced, L5), and its width on the side it gives a channel."""

    sourced: bool = Param(default=False)

    @derived
    def direction(self) -> Direction:
        return Direction.OUT

    @derived
    def advance(self) -> Advance:
        """Scheduled, a projection of its kernel's schedule; sourced, a memory's output,
        ``sourced``; else stepped, a stage's side."""
        if self.schedule is not None:
            return Advance.SCHEDULED
        if self.sourced:
            return Advance.SOURCED
        return Advance.STEPPED

    @derived
    def side(self) -> Side | Inapplicable:
        """The side it gives a channel: its stream, its label and its width."""
        if not self.placed:
            return Inapplicable()
        return Side.kernel(self.stream, self.label, self.width)


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


def stream_ports(space_type: type[Space]) -> dict[str, StreamPort]:
    """The stream ports ``space_type`` declares, in inherited declaration order, through
    its Groups (by dotted path, ``group.port``), each bound where it is declared. Stops at
    child Spaces: a child's ports are not its parent's, but where it exposes one
    (``exposed_ports``)."""

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


def exposed_ports(space_type: type[Space]) -> dict[str, StreamPort]:
    """The stream ports kernel class ``space_type`` presents, by the name it presents each
    under: its own (``stream_ports``), then each node alias of a member's stream port
    (``x = compute.x``: every node it may name a stream port), as a class reference to read
    its readouts of (``exposed_ports(MatMulKernel)["x"].side``). The inventory a channel
    may connect; a member's port it does not expose is no port of its."""
    from finn.core.space._aliases import alias_table  # noqa: PLC0415 - the engine's own table
    from finn.core.space._nodes import space_members  # noqa: PLC0415

    found = stream_ports(space_type)
    for name, entry in alias_table(cast(Any, space_type)).items():
        if name not in space_members(space_type):
            continue
        if entry.ends and all(issubclass(end.space_type, StreamPort) for end in entry.ends):
            found[name] = getattr(space_type, name)
    return found


__all__ = [
    "InPort",
    "OtherPins",
    "OutPort",
    "StreamPort",
    "exposed_ports",
    "stream_ports",
]
