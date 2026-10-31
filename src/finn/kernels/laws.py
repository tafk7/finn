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
- the two pin families (``AXIS_FAMILY``, ``NATIVE_FAMILY``, ``PinFamily``), one a kernel,
  chosen by its module's family: its clock and reset pins (``AXIS_CLOCKING``,
  ``NATIVE_CLOCKING``) and its ports' transport where a port states none; and the doubled
  clock's name (``doubled``);
- the idle level (``IDLE_LEVEL``), AXIS's byte padding (``axis_carrier``) and the one
  marker an AXIS beat carries (``AXIS_MARKERS``).

And the terms a kernel states its intake in (``Intake``, ``PortIntake``, ``Standing``, and a
fitted intake's ``IntakeProfile``):
intake is no law, every kernel states its own from its RTL, but its shape, its unit and
its standing are the library's, and so is its answer where a kernel states none
(``INTAKE_UNSTATED``).

And the rate law's one exception, a kernel's initiation interval (``Interval``: L6's 1,
``RATE_INTERVAL``, unless a kernel states its own, with its standing and source), with the
one conversion from iterations to clock cycles (``in_cycles``) that a kernel's cycles a
frame and FIFO sizing both read, and a scheduled port's pace in cycles (``TimedPace``).
"""

from __future__ import annotations

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
ADVANCE_SCHEDULED = Law(
    "L2",
    "advance, scheduled",
    "inputs meet: the input beats an iteration reads are accepted together, never one "
    "alone; issue is not delivery: an iteration issues its results into the kernel's own "
    "output storage, it stops accepting by its intake rule, and results leave in order, "
    "held under backpressure",
    "a scheduled kernel whose inputs advance independently",
)
ADVANCE_BUFFERED = Law(
    "L3",
    "advance, buffered",
    "input and output move separately: input is accepted while the buffer has room, "
    "output presented once the beats it needs are in",
    "none known",
)
ADVANCE_SELECTED = Law(
    "L4",
    "advance, selected",
    "a selector beat is accepted together with the data beats it governs",
    "the group size, which the kernel always states",
)
ADVANCE_SOURCED = Law(
    "L5",
    "advance, sourced",
    "a memory presents its next beat whenever its consumer takes one",
    "none known",
)
RATE = Law(
    "L6",
    "rate",
    "a kernel advances one iteration a cycle (a stage one beat a side a cycle) when nothing "
    "blocks it; its cycles a frame at full rate follow from its pace, or a stage's from its "
    "larger beat count",
    "a kernel whose iterations take more than one cycle, or whose one rests on what its RTL "
    "does not show (an HLS kernel's requested pipeline), states its initiation interval, with "
    "its standing and source",
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
    "one clock, the kernel's pin family's clock and reset (AXIS_FAMILY or NATIVE_FAMILY); "
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
    ADVANCE_SCHEDULED,
    ADVANCE_BUFFERED,
    ADVANCE_SELECTED,
    ADVANCE_SOURCED,
    RATE,
    RESET_LAW,
    CLOCK_LAW,
    MARKERS,
    IDLE,
)
"""The laws, in the order of their IDs."""


class Advance(Enum):
    """A port's advance rule: when it moves its next beat, relative to its kernel's other
    ports (L2-L5). A port follows one by how it is declared and never names it
    (``finn.kernels.port_model.StreamPort.advance``)."""

    SCHEDULED = "scheduled"
    BUFFERED = "buffered"
    SELECTED = "selected"
    SOURCED = "sourced"

    @property
    def law(self) -> Law:
        """The law it follows."""
        return _ADVANCE_LAWS[self]


_ADVANCE_LAWS = {
    Advance.SCHEDULED: ADVANCE_SCHEDULED,
    Advance.BUFFERED: ADVANCE_BUFFERED,
    Advance.SELECTED: ADVANCE_SELECTED,
    Advance.SOURCED: ADVANCE_SOURCED,
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
    form fitted to traces, where it holds (``IntakeProfile``), its standing **assumed**."""

    port: str
    before_stop: int
    most: int
    rule: str
    standing: Standing
    source: str
    profile: IntakeProfile | None = None

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

    def under(self, member: str) -> PortIntake:
        """The intake as seen from where ``member`` is held."""
        return replace(self, port=f"{member}.{self.port}")


@dataclass(frozen=True)
class Intake:
    """A kernel's intake (contract item 5): one ``PortIntake`` an input port it places, by
    its label. A kernel with children derives its own from its members': a port it exposes
    by the name it exposes it under, any other member's below the member."""

    ports: tuple[PortIntake, ...] = ()

    def __post_init__(self) -> None:
        names = [each.port for each in self.ports]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"an intake states each port once, not {twice}")

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


# -- the initiation interval (L6) ---------------------------------------------------------


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
    """A scheduled port's pace (``finn.dataflow.schedule.Pace``, in iterations: THEORY says
    nothing about time) at its kernel's interval: its beat ``times`` and its ``span`` in
    clock cycles (``in_cycles``). The port carries both (``StreamPort.pace``,
    ``StreamPort.interval``); FIFO sizing reads this."""

    pace: Pace
    interval: Interval

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(in_cycles(time, self.interval) for time in self.pace.times)

    @property
    def span(self) -> int:
        return in_cycles(self.pace.span, self.interval)


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


# -- the pin families: clock and reset (L7, L8), and a port's default transport ------------

AXIS_CLOCKING = Clocking(clock="ap_clk", reset="ap_rst_n", active_low=True)
"""The AXIS family's clock and reset: ``ap_clk`` and the synchronous, active-low
``ap_rst_n`` (dotp, thresholding, pool)."""

NATIVE_CLOCKING = Clocking(clock="clk", reset="rst", active_low=False)
"""FinnLib's native family's clock and reset: ``clk`` and the synchronous, active-high
``rst`` (vpc, the FIFO, the input generator, MemStream, eltwise, pwpolyf, transpose)."""


@dataclass(frozen=True)
class PinFamily:
    """A module family's pin conventions, one a kernel (``Kernel.pin_family``): its clock and
    reset pins and the reset's polarity (``clocking``, L7, L8), and the transport its ports
    take where a port states none (``transport``). A port with its own pin names, or of
    another family than its kernel's, states its transport."""

    clocking: Clocking
    transport: TransportKind

    def __post_init__(self) -> None:
        if not isinstance(self.clocking, Clocking) or self.clocking.doubled is not None:
            raise ValueError("a pin family's clocking is a plain Clocking, with no doubled clock")
        if not isinstance(self.transport, (Axis, Native)):
            raise ValueError("a pin family's transport is an Axis or a Native")


AXIS_FAMILY = PinFamily(AXIS_CLOCKING, AXIS)
"""The AXIS family: ``ap_clk``, ``ap_rst_n``, and AXIS ports on the library's default names
(dotp, thresholding, pool)."""

NATIVE_FAMILY = PinFamily(NATIVE_CLOCKING, NATIVE)
"""FinnLib's native family: ``clk``, ``rst``, and native ports on the library's default names
(vpc, the FIFO, the input generator, MemStream, eltwise, pwpolyf, transpose)."""


def doubled(domain: Clocking, doubling: bool) -> Clocking:
    """``domain`` with its doubled clock (L8's exception), ``<clock>2x``: a clock at twice
    ``domain``'s while ``doubling``, an unused input held low otherwise."""
    return replace(domain, doubled=f"{domain.clock}2x", doubling=doubling)


IDLE_LEVEL = 0
"""The level an idle port's inputs are held at (L10)."""


__all__ = [
    "ADVANCE_BUFFERED",
    "ADVANCE_SCHEDULED",
    "ADVANCE_SELECTED",
    "ADVANCE_SOURCED",
    "AXIS",
    "Advance",
    "AXIS_CLOCKING",
    "AXIS_FAMILY",
    "AXIS_MARKERS",
    "AXIS_MEMBERS",
    "Axis",
    "AxisSpelling",
    "CLOCK_LAW",
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
    "NATIVE_FAMILY",
    "Native",
    "PinFamily",
    "PortIntake",
    "RATE",
    "RATE_INTERVAL",
    "RESET_LAW",
    "Standing",
    "TimedPace",
    "TransportKind",
    "axis_carrier",
    "axis_pin",
    "doubled",
    "in_cycles",
]
