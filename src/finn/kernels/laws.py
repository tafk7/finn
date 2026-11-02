# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The library's interaction laws: how a kernel's pins behave around its streams.

They are the same across FINN's kernel library, so they are stated once, here, and
inherited; a kernel states only its exceptions, where its hardware differs. Each law
carries the ID the library's contract gives it (``L1``-``L10``), so a change to one is
one edit in that contract and one here. Until the conformance harness checks them, the
laws are stated, not checked.

Beside the laws, the defaults they fix, which the base, the ports and the transports
read here rather than spell themselves:

- each transport's pin names (``Axis``, ``Native``): an author declares a port's
  transport kind, not its pins, and overrides a name only where its RTL names a pin
  otherwise (``Axis(bus="s_axis_input")``, ``Native(prefix="a")``);
- the two interface protocols (``AXIS_PROTOCOL``, ``NATIVE_PROTOCOL``,
  ``InterfaceProtocol``), one a kernel, its module's own: its clock and reset pins
  (``AXIS_CLOCKING``, ``NATIVE_CLOCKING``) and its ports' transport where a port states
  none; and the doubled clock's name (``doubled``);
- the idle level (``IDLE_LEVEL``), AXIS's byte padding (``axis_carrier``) and the one
  marker an AXIS beat carries (``AXIS_MARKERS``).

And the terms a kernel states its intake in (``Intake``, with its ``joined`` inputs;
``PortIntake``, with how far an input is accepted ``Ahead`` of another; ``Standing``, and a
fitted intake's ``IntakeProfile``): intake is no law, every kernel states its own from its
RTL, but its shape, its unit and its standing are the library's, and so is its answer
where a kernel states none (``INTAKE_UNSTATED``). The pacing laws (L2-L5: a port that owns
its pacing, by its kernel's schedule, or one of the three ways a port follows another) say
when a beat is **consumed** or **issued**, relative to the kernel's other beats (``Pacing``,
``Follows``); how far input beats are **accepted** ahead of output beats **delivered**, and
whether one is accepted only with another port's, is its kernel's intake.

And the terms of the rate (L6), derived by default and stated where the model cannot give
it: a kernel's initiation interval (``Interval``: L6's 1, ``RATE_INTERVAL``, unless a kernel
states its own, with its standing and source), with the one conversion from iterations to
clock cycles (``in_cycles``) that a kernel's cycles a frame and FIFO sizing both read, and
the pace in cycles of a port that owns its pacing (``TimedPace``); and the rate a frame a
kernel states where its schedule and ports at its interval cannot express it
(``StatedRate``: a memory mover, whose memory words and calls take cycles its stream does
not show), with its reason, standing and source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from finn.dataflow.schedule import Pace
from finn.kernels.artifacts.abi import Clocking, Endpoint


@dataclass(frozen=True)
class Law:
    """One interaction law: its ``id``, its ``topic``, the library's ``default`` and the
    ``exception`` a kernel may state where its hardware differs."""

    id: str
    topic: str
    default: str
    exception: str


HANDSHAKE = Law(
    "L1",
    "handshake",
    "every stream port is ready/valid: a beat transfers on the rising clock edge where "
    "valid and ready, outside reset; valid, data and markers hold until accepted; a "
    "producer never waits for ready before raising valid",
    "a port with another transport (none known)",
)
PACING_SCHEDULE = Law(
    "L2",
    "pacing, schedule",
    "a port owns its pacing as a projection of its kernel's schedule, and the inputs it paces "
    "meet at an iteration: an iteration consumes the beats it reads together, and issues its "
    "results into the kernel's own storage; issue is not delivery. When each input beat is "
    "accepted, and whether only with another port's, is the kernel's intake",
    "none known: a kernel whose inputs fire at different iterations is outside the scope, "
    "and comes with a refinement of this contract",
)
PACING_PASS_THROUGH = Law(
    "L3",
    "follows, pass-through",
    "a stage's side follows its other side, one side's beats the other's, re-cut: an output "
    "beat is issued once the input beats it is made from are consumed; it accepts input "
    "while its storage has room (its intake)",
    "none known",
)
PACING_GROUPS = Law(
    "L4",
    "follows, groups",
    "a selector follows the port it governs, one beat a group: its beat names the set its "
    "governed port's next group of beats are made from. When it is accepted is its intake",
    "none known: its group is the governed port's beats a frame over its own (StreamPort.group)",
)
PACING_READER = Law(
    "L5",
    "follows, reader",
    "a memory's output follows its reader: it presents its next beat whenever its reader takes one",
    "none known",
)
RATE = Law(
    "L6",
    "rate",
    "by default, a kernel advances one iteration an interval (a stage, one beat a side an "
    "interval) when nothing blocks it, and the default interval is one cycle; so its cycles "
    "a frame at full rate are derived from its pace (its iterations a frame) or, for a "
    "stage, its larger beat count, at its interval, by one conversion",
    "a kernel whose iterations take more than one cycle, or whose one rests on what its RTL "
    "does not show (an HLS kernel's requested pipeline), states its initiation interval, with "
    "its standing and source; a kernel whose rate its schedule and ports cannot express (a "
    "memory mover) states its rate a frame, its port's beat times and its span, with its "
    "reason, standing and source",
)
RESET_LAW = Law(
    "L7",
    "reset",
    "one synchronous reset; the contract holds within a reset epoch; reset cancels all "
    "work in flight, including results completed but not yet delivered",
    "none known",
)
CLOCK_LAW = Law(
    "L8",
    "clock",
    "one clock, the kernel's interface protocol's clock and reset "
    "(AXIS_PROTOCOL or NATIVE_PROTOCOL); "
    "clock and reset pins are the kernel's, never a port's",
    "a doubled clock at twice the kernel's clock, where the kernel takes one, declared "
    "(doubled): named <clock>2x (ap_clk2x, clk2x), held low while the kernel does not double",
)
MARKERS = Law(
    "L9",
    "markers",
    "a marker closes a level of the stream's own traversal; on AXIS, TLAST carries one "
    "marker; a native port carries none",
    "a native loop-completion vector, one bit a level (<p>lst)",
)
IDLE = Law(
    "L10",
    "idle and packing",
    "an idle port holds its inputs low (a target's data and valid, an initiator's ready) "
    "and leaves its outputs unconnected, as does an unexposed control bus; lanes pack lane "
    "0 lowest; an AXIS beat is padded to a whole byte, a producer's padding bits left "
    "unconnected and a consumer's driven with zeros; a pure lane permutation, padding and "
    "reset polarity leave the stream unchanged: they are the connection's wires, not "
    "adaptation",
    "none known",
)

LAWS: tuple[Law, ...] = (
    HANDSHAKE,
    PACING_SCHEDULE,
    PACING_PASS_THROUGH,
    PACING_GROUPS,
    PACING_READER,
    RATE,
    RESET_LAW,
    CLOCK_LAW,
    MARKERS,
    IDLE,
)
"""The laws, in the order of their IDs."""


class Pacing(Enum):
    """What paces a port's beats, by ordinal: what each of its beats is consumed or issued
    with, or after, among its kernel's other beats (the pacing laws, L2-L5).

    - ``SCHEDULE``: the port owns its pacing, a projection of its kernel's schedule:
      iteration *t* consumes the input beats and issues the output beats its kernel's ports'
      projections place at *t*. Only such a port has a pace, in iterations (L2);
    - ``FOLLOWS``: it follows another port, by the relation its ``follows`` names
      (``Follows``), and has no pace of its own.

    When a beat is **accepted**, and whether only with another port's, is not its pacing but
    its kernel's intake (``Intake``). A port declares what it is and its pacing follows
    (``finn.kernels.port_model.StreamPort.pacing``, ``StreamPort.follows``): a projection of
    its kernel's schedule, an input that ``governs`` another port, an output that is
    ``sourced`` (a memory's output, that declaration), else a stage's side; it never names
    its pacing, and no port class states its own (``port-pacing``)."""

    SCHEDULE = "schedule"
    FOLLOWS = "follows"


class Follows(Enum):
    """How a port that does not own its pacing (``Pacing.FOLLOWS``) follows another port's
    beats, each relation with its event and its law:

    - ``PASS_THROUGH``: a stage's side (a FIFO, a width converter, ``input_gen``), one side's
      beats the other's, re-cut: output beat *j* is issued once the input beats it is made
      from are consumed (a FIFO's *j* from *j*, VPC's widths, ``input_gen``'s replay) (L3);
    - ``GROUPS``: a selector, one beat a group of the port it ``governs``: its beat *g* names
      the set the governed port's group *g* is made from, and is consumed no later than that
      group's first beat (L4);
    - ``READER``: a memory's output, one beat whenever its reader takes one: beat *j* is its
      stored value at position *j* of its reader's order (L5)."""

    PASS_THROUGH = "pass-through"
    GROUPS = "groups"
    READER = "reader"

    @property
    def law(self) -> Law:
        """The law it follows."""
        return _FOLLOWS_LAWS[self]


_FOLLOWS_LAWS = {
    Follows.PASS_THROUGH: PACING_PASS_THROUGH,
    Follows.GROUPS: PACING_GROUPS,
    Follows.READER: PACING_READER,
}


# -- intake (CONTRACT.md § 5) -------------------------------------------------------------


class Standing(Enum):
    """What a statement (an intake, an initiation interval) stands on: ``STRUCTURAL``, read
    from the RTL (exact or bounded, cited by file and line); ``MEASURED``, for an HLS
    kernel, from recorded traces, at their conditions (CONTRACT.md § 5); ``ASSUMED``, an
    admission assumption, named, with what it rests on and no measurement behind it (an HLS
    pragma's requested initiation interval; a form fitted to traces, between them, within
    its ``IntakeProfile``)."""

    STRUCTURAL = "structural"
    MEASURED = "measured"
    ASSUMED = "assumed"


@dataclass(frozen=True)
class IntakeProfile:
    """Where an intake fitted to traces holds: the ``part`` and ``tool`` version the traces
    were taken on, and its form's domain, ``ranges``, each a quantity's name and the least
    and most it was traced at, inclusive (``("K", 4, 9)``). Inside it the form's value
    stands **assumed** (it holds between traces); outside, the kernel states no intake
    (``INTAKE_UNSTATED``), naming the profile. A kernel checks its ranges (``outside``);
    the part and tool are stated and not checked, until a platform names its part."""

    part: str
    tool: str
    ranges: tuple[tuple[str, int | float, int | float], ...]

    def __post_init__(self) -> None:
        if not self.part or not self.tool:
            raise ValueError("a profile names the part and the tool its traces were taken on")
        names = [name for name, _, _ in self.ranges]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"a profile states each quantity's range once, not {names}")
        for name, least, most in self.ranges:
            bounds = (least, most)
            if not name or any(type(b) not in (int, float) for b in bounds) or least > most:
                raise ValueError(f"{name!r}: a range is two numbers, least first, not {bounds}")

    def outside(self, **values: int | float) -> tuple[str, ...]:
        """Each of ``values``, by its quantity's name, that lies outside its range."""
        ranges = {name: (least, most) for name, least, most in self.ranges}
        if set(values) != set(ranges):
            raise ValueError(f"a profile is checked at {sorted(ranges)}, not {sorted(values)}")
        return tuple(
            f"{name} {value}"
            for name, value in values.items()
            if not ranges[name][0] <= value <= ranges[name][1]
        )

    def __str__(self) -> str:
        domain = ", ".join(f"{name} {least}..{most}" for name, least, most in self.ranges)
        return f"{domain}; traced on {self.part} with {self.tool} (part and tool not checked)"


@dataclass(frozen=True)
class Ahead:
    """How far an input is accepted ahead of another input of its kernel, ``port`` (a label
    in its kernel's namespace): the most beats it accepts beyond the beats of ``port``
    accepted, ``beats``, a bound read from the RTL. Eltwise's operands: 1 each, its sidestep
    register."""

    port: str
    beats: int

    def __post_init__(self) -> None:
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("an input is ahead of a port it names")
        if type(self.beats) is not int or self.beats < 0:
            raise ValueError(f"ahead of {self.port}: a count of beats, not {self.beats!r}")


@dataclass(frozen=True)
class PortIntake:
    """How far a kernel takes input ahead of delivering output, at one input port (its
    ``port``, a label below the kernel stating it), in beats of that port.

    ``before_stop``: the beats it takes, from reset with its output stalled, before it
    **may** stop accepting, a lower bound (what a FIFO sizing must assume it takes);
    ``most``: the most beats of this port it holds at once, an upper bound (a kernel that
    drops input it never presents may take more before it stops than it holds). Where the
    RTL counts something else than the port's beats, ``before_stop`` rounds down and
    ``most`` up.
    ``rule``: when it stops and when it resumes; ``source``: where both are read, the RTL's
    file and lines at the library's pin, or the trace and its conditions; ``profile``: for a
    form fitted to traces, where it holds (``IntakeProfile``), its standing **assumed**.

    The counts are relative to output delivery, never to when beats are consumed or issued
    (the pacing laws, L2-L5). Beside them, its coupling to the kernel's other inputs:
    ``ahead``, for an input in no ``Intake.joined`` group, how far it is accepted beyond
    each other input it names (``Ahead``); none stated, none known."""

    port: str
    before_stop: int
    most: int
    rule: str
    standing: Standing
    source: str
    profile: IntakeProfile | None = None
    ahead: tuple[Ahead, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("an intake names its port")
        for name in ("before_stop", "most"):
            count = getattr(self, name)
            if type(count) is not int or count < 0:
                raise ValueError(f"{self.port}: {name} is a count of beats, not {count!r}")
        if not isinstance(self.standing, Standing):
            raise ValueError(f"{self.port}: an intake's standing is a Standing")
        if not self.rule or not self.source:
            raise ValueError(f"{self.port}: an intake states its rule and where it is read")
        if self.profile is not None and (
            not isinstance(self.profile, IntakeProfile) or self.standing is not Standing.ASSUMED
        ):
            raise ValueError(f"{self.port}: a fitted intake's profile is one, its standing assumed")
        object.__setattr__(self, "ahead", tuple(self.ahead))
        partners = [each.port for each in self.ahead if isinstance(each, Ahead)]
        if len(partners) != len(self.ahead) or len(set(partners)) != len(partners):
            raise ValueError(f"{self.port}: ahead states each other input once, as an Ahead")
        if self.port in partners:
            raise ValueError(f"{self.port}: an input is ahead of another input, not of itself")

    def renamed(self, name: Callable[[str], str]) -> PortIntake:
        """The intake with every port it names renamed by ``name``: its own and those it is
        ahead of."""
        ahead = tuple(replace(each, port=name(each.port)) for each in self.ahead)
        return replace(self, port=name(self.port), ahead=ahead)

    def under(self, member: str) -> PortIntake:
        """The intake as seen from where ``member`` is held."""
        return self.renamed(lambda port: f"{member}.{port}")


@dataclass(frozen=True)
class Intake:
    """A kernel's intake (contract item 5): one ``PortIntake`` an input port it places, by
    its label, and its inputs' coupling: ``joined``, the groups of its inputs whose beats
    transfer only together, in the same cycle, one beat of each (an atomic transfer group,
    read from its RTL's ready equations: dotp's ``x`` and ``w``); each port's ``ahead``
    beside it. Every name in it is a label in one namespace, its kernel's, of a placed input
    (``intake-port-unknown``, ``finn.kernels.base.Kernel.intake``). An intake states no
    coupling it does not know: none stated is none known, never assumed. A kernel with
    children derives its own from its members': a port it exposes by the name it exposes it
    under, any other member's below the member (``renamed``)."""

    ports: tuple[PortIntake, ...] = ()
    joined: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        names = [each.port for each in self.ports]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"an intake states each port once, not {twice}")
        joined = tuple(tuple(group) for group in self.joined)
        object.__setattr__(self, "joined", joined)
        grouped = [name for group in joined for name in group]
        if any(len(set(group)) != len(group) or len(group) < 2 for group in joined):
            raise ValueError(f"a joined group is two or more inputs, each once, not {joined}")
        if len(set(grouped)) != len(grouped):
            raise ValueError(f"an input is joined in one group at most, not {joined}")
        partners = [ahead.port for each in self.ports for ahead in each.ahead]
        unstated = sorted({name for name in (*grouped, *partners) if name not in names})
        if unstated:
            raise ValueError(f"an intake's coupling names the inputs it states, not {unstated}")
        both = sorted(each.port for each in self.ports if each.ahead and each.port in grouped)
        if both:
            raise ValueError(f"an input joined with another is ahead of none: {both}")

    def renamed(self, name: Callable[[str], str]) -> Intake:
        """The intake with every port it names renamed by ``name``: its ports, its groups'
        and its partners'."""
        return Intake(
            tuple(each.renamed(name) for each in self.ports),
            tuple(tuple(map(name, group)) for group in self.joined),
        )

    def under(self, member: str) -> Intake:
        """The intake as seen from where ``member`` is held."""
        return self.renamed(lambda port: f"{member}.{port}")

    def of(self, port: str) -> PortIntake:
        """Port ``port``'s; ``KeyError`` for a port it states none of."""
        for each in self.ports:
            if each.port == port:
                return each
        raise KeyError(port)

    @property
    def names(self) -> tuple[str, ...]:
        """The ports it states, in order."""
        return tuple(each.port for each in self.ports)


INTAKE_UNSTATED = "intake-unstated"
"""The answer of a kernel that states no intake (CONTRACT.md § 5): named, never zero, and
no refusal of its design point, since no rule reads intake."""


# -- the rate (L6): the initiation interval, the one conversion, a stated rate -------------


@dataclass(frozen=True)
class Interval:
    """A kernel's initiation interval (L6): the clock cycles an iteration takes when nothing
    blocks it (a stage's: a beat a side), at least 1. L6's is 1 (``RATE_INTERVAL``: no
    ``standing``, no ``source``, the law's); a kernel whose hardware differs, or that rests
    its 1 on what its RTL does not show (an HLS kernel's requested pipeline), states its
    own with its ``standing`` and its ``source``. A kernel with a module states it; a
    composite has none, having no schedule (``finn.kernels.base.Kernel.interval``)."""

    cycles: int = 1
    standing: Standing | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if type(self.cycles) is not int or self.cycles < 1:
            raise ValueError(f"an interval is a positive count of cycles, not {self.cycles!r}")
        if self.standing is None:
            if self.cycles != 1 or self.source:
                raise ValueError(
                    "an interval without a standing is the law's (L6: 1); a kernel stating "
                    "another states its standing and source"
                )
        elif not isinstance(self.standing, Standing) or not self.source:
            raise ValueError("a stated interval has a Standing and states where it is read")

    def __str__(self) -> str:
        if self.standing is None:
            return f"{self.cycles} (L6)"
        return f"{self.cycles}, {self.standing.value}: {self.source}"


RATE_INTERVAL = Interval()
"""L6's interval: an iteration a cycle, the law's."""


def in_cycles(iterations: int, interval: Interval) -> int:
    """The clock cycles ``iterations`` take, from the first's start, at ``interval`` (L6):
    an interval's cycles each. The one conversion from a kernel's iterations (a stage's
    beats) to cycles: a kernel's cycles a frame (``Kernel.frame_cycles``) and FIFO sizing's
    beat times (``TimedPace``) both read it."""
    return iterations * interval.cycles


@dataclass(frozen=True)
class TimedPace:
    """The pace of a port that owns its pacing (``finn.dataflow.schedule.Pace``, in
    iterations: THEORY says nothing about time) at its kernel's interval: its beat ``times``
    and its ``span`` in clock cycles (``in_cycles``). The port carries both
    (``StreamPort.pace``, ``StreamPort.interval``); FIFO sizing reads this."""

    pace: Pace
    interval: Interval

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(in_cycles(time, self.interval) for time in self.pace.times)

    @property
    def span(self) -> int:
        return in_cycles(self.pace.span, self.interval)


def check_beat_times(times: object, span: object) -> None:
    """Raise ``ValueError`` unless ``times`` are a port's beat times over a frame of ``span``
    cycles: a tuple of at least one cycle, strictly ascending from cycle 0 or later (a port
    moves one beat a cycle at most), each within a span of at least one cycle. A stated
    rate's (``StatedRate``) and the outside's (``finn.kernels.sides.BeatTimes``)."""
    if type(times) is not tuple or not times or not all(type(time) is int for time in times):
        raise ValueError("a pace's beat times are a tuple of at least one cycle")
    if type(span) is not int or span < 1:
        raise ValueError(f"a frame spans at least one cycle, not {span!r}")
    if times[0] < 0 or any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("a pace's beat times start at cycle 0 or later, one beat a cycle")
    if times[-1] >= span:
        raise ValueError(
            f"a pace's beat times lie within its span: a beat at {times[-1]} of {span}"
        )


@dataclass(frozen=True)
class StatedRate:
    """The rate a frame a kernel states where the derived rate cannot express it (L6): the
    cycles a frame ``span``s and each beat of its ``port`` from the frame's first (``times``,
    one a cycle at most, each within the span: ``check_beat_times``), with its ``reason``
    (why its schedule and ports at its interval cannot give it), its ``standing`` and its
    ``source``. A memory mover's: its memory side's words and each call's constant take
    cycles its stream does not show. The base reads it checked
    (``finn.kernels.base.Kernel.checked_rate``): its span is the kernel's cycles a frame, and
    its times and span the named port's pace in cycles (``StreamPort.timed_pace``), which
    FIFO sizing reads as it reads a ``TimedPace``'s (``finn.kernels.fifo_sizing.Paced``).
    Each is a marker for later work on the schedule and dataflow model, listed where the
    contract lists the rates stated."""

    port: str
    times: tuple[int, ...]
    span: int
    standing: Standing
    source: str
    reason: str

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("a stated rate names the port whose beats it times")
        if not isinstance(self.standing, Standing) or not self.source:
            raise ValueError("a stated rate has a Standing and states where it is read")
        if not self.reason:
            raise ValueError("a stated rate gives its reason: why the derived rate cannot give it")
        check_beat_times(self.times, self.span)

    def __str__(self) -> str:
        return f"{self.span} cycles a frame, {self.standing.value}: {self.source}"


# -- the transports' pin defaults ---------------------------------------------------------


class AxisSpelling(Enum):
    """How a module spells its AXIS member pins after the bus's name: FinnLib's RTL in
    lower case (``s_axis_tdata``), HLS's exported RTL in upper case (``s_axis_TDATA``)."""

    LOWER = "lower"
    UPPER = "upper"


AXIS_MEMBERS = ("tdata", "tvalid", "tready", "tlast")
"""An AXIS bus's members (L1, L9), in the order its pins are named."""

AXIS_MARKERS = 1
"""The markers an AXIS beat carries at most: one, on ``tlast`` (L9)."""


@dataclass(frozen=True)
class Axis:
    """AXI-Stream pins: ``<bus>_tdata``, ``_tvalid``, ``_tready`` and, where the stream has a
    marker, ``_tlast``. The bus is ``s_axis_<port>`` for a target, ``m_axis_<port>`` for an
    initiator, unless the kernel states its ``bus``; its members are spelled in lower case
    unless the kernel states its module's ``spelling`` (an HLS module's, upper)."""

    bus: str | None = None
    spelling: AxisSpelling = AxisSpelling.LOWER

    def __post_init__(self) -> None:
        if self.bus is not None and (not isinstance(self.bus, str) or not self.bus):
            raise ValueError("an AXIS bus a kernel states has a nonempty name")
        if not isinstance(self.spelling, AxisSpelling):
            raise ValueError("AXIS member pins are spelled by an AxisSpelling")

    def bus_name(self, port: str, endpoint: Endpoint) -> str:
        """The bus of the port named ``port`` at ``endpoint``: the stated one, else the
        default."""
        if self.bus is not None:
            return self.bus
        return f"{'s' if endpoint is Endpoint.TARGET else 'm'}_axis_{port}"


def axis_pin(bus: str, member: str, spelling: AxisSpelling) -> str:
    """The pin carrying ``member`` (one of ``AXIS_MEMBERS``) of ``bus``, spelled."""
    return f"{bus}_{member.upper() if spelling is AxisSpelling.UPPER else member}"


def axis_carrier(payload_bits: int) -> int:
    """The bits of an AXIS beat carrying ``payload_bits``: padded to a whole byte (L10)."""
    return (payload_bits + 7) // 8 * 8


@dataclass(frozen=True)
class Native:
    """FinnLib's native ready/valid pins: ``<p>dat``, ``<p>vld``, ``<p>rdy`` and, where the
    kernel states a loop-completion vector (L9), ``<p>lst``. ``<p>`` is ``i`` for a target
    and ``o`` for an initiator, unless the kernel states its ``prefix``, or ``names``, all
    three, where its RTL does not follow ``dat``/``vld``/``rdy`` (a port so named carries no
    loop-completion vector)."""

    prefix: str | None = None
    names: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if self.prefix is not None and self.names is not None:
            raise ValueError("a native port states its prefix or its names, not both")
        if self.prefix is not None and (not isinstance(self.prefix, str) or not self.prefix):
            raise ValueError("a native prefix a kernel states is a nonempty name")
        if self.names is not None:
            names = tuple(self.names)
            if (
                len(names) != 3
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != 3
            ):
                raise ValueError("a native port's names are its data, valid and ready pins")
            object.__setattr__(self, "names", names)

    def _prefix(self, endpoint: Endpoint) -> str:
        if self.prefix is not None:
            return self.prefix
        return "i" if endpoint is Endpoint.TARGET else "o"

    def pins(self, endpoint: Endpoint) -> tuple[str, str, str]:
        """Its data, valid and ready pins at ``endpoint``."""
        if self.names is not None:
            return self.names
        prefix = self._prefix(endpoint)
        return (f"{prefix}dat", f"{prefix}vld", f"{prefix}rdy")

    def loop_ends(self, endpoint: Endpoint) -> str | None:
        """Its loop-completion vector's pin at ``endpoint``; None for a port named in full."""
        return None if self.names is not None else f"{self._prefix(endpoint)}lst"


AXIS = Axis()
"""An AXIS port on the library's default names."""

NATIVE = Native()
"""A native port on the library's default names."""

TransportKind = Axis | Native
"""A port's transport kind, with the pin names a kernel states for it."""


# -- the interface protocols: clock and reset (L7, L8), and a port's default transport ------------

AXIS_CLOCKING = Clocking(clock="ap_clk", reset="ap_rst_n", active_low=True)
"""The AXIS protocol's clock and reset: ``ap_clk`` and the synchronous, active-low
``ap_rst_n`` (dotp, thresholding, pool)."""

NATIVE_CLOCKING = Clocking(clock="clk", reset="rst", active_low=False)
"""FinnLib's native protocol's clock and reset: ``clk`` and the synchronous, active-high
``rst`` (vpc, the FIFO, the input generator, MemStream, eltwise, pwpolyf, transpose)."""


@dataclass(frozen=True)
class InterfaceProtocol:
    """A module's interface protocol, one a kernel (``Kernel.interface_protocol``): its clock and
    reset pins and the reset's polarity (``clocking``, L7, L8), and the transport its ports
    take where a port states none (``transport``). A port with its own pin names, or of
    another interface protocol than its kernel's, states its transport."""

    clocking: Clocking
    transport: TransportKind

    def __post_init__(self) -> None:
        if not isinstance(self.clocking, Clocking) or self.clocking.doubled is not None:
            raise ValueError(
                "an interface protocol's clocking is a plain Clocking, with no doubled clock"
            )
        if not isinstance(self.transport, (Axis, Native)):
            raise ValueError("an interface protocol's transport is an Axis or a Native")


AXIS_PROTOCOL = InterfaceProtocol(AXIS_CLOCKING, AXIS)
"""The AXIS protocol: ``ap_clk``, ``ap_rst_n``, and AXIS ports on the library's default names
(dotp, thresholding, pool)."""

NATIVE_PROTOCOL = InterfaceProtocol(NATIVE_CLOCKING, NATIVE)
"""FinnLib's native protocol: ``clk``, ``rst``, and native ports on the library's default names
(vpc, the FIFO, the input generator, MemStream, eltwise, pwpolyf, transpose)."""


def doubled(domain: Clocking, doubling: bool) -> Clocking:
    """``domain`` with its doubled clock (L8's exception), ``<clock>2x``: a clock at twice
    ``domain``'s while ``doubling``, an unused input held low otherwise."""
    return replace(domain, doubled=f"{domain.clock}2x", doubling=doubling)


IDLE_LEVEL = 0
"""The level an idle port's inputs are held at (L10)."""


__all__ = [
    "AXIS",
    "Ahead",
    "AXIS_CLOCKING",
    "AXIS_PROTOCOL",
    "AXIS_MARKERS",
    "AXIS_MEMBERS",
    "Axis",
    "AxisSpelling",
    "CLOCK_LAW",
    "Follows",
    "HANDSHAKE",
    "IDLE",
    "IDLE_LEVEL",
    "INTAKE_UNSTATED",
    "Intake",
    "IntakeProfile",
    "Interval",
    "LAWS",
    "Law",
    "MARKERS",
    "NATIVE",
    "NATIVE_CLOCKING",
    "NATIVE_PROTOCOL",
    "Native",
    "InterfaceProtocol",
    "PACING_GROUPS",
    "PACING_PASS_THROUGH",
    "PACING_READER",
    "PACING_SCHEDULE",
    "Pacing",
    "PortIntake",
    "RATE",
    "RATE_INTERVAL",
    "RESET_LAW",
    "Standing",
    "StatedRate",
    "TimedPace",
    "TransportKind",
    "axis_carrier",
    "axis_pin",
    "check_beat_times",
    "doubled",
    "in_cycles",
]
