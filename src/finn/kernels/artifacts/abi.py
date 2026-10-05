# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A module's pins: physical facts only.

Nothing here names a kernel, a channel or a port declaration path: two kernels
binding the same module the same way present equal pins.

**Direction is declared once and flipped.** A bus signature gives its member
directions for the *initiator*; a target reuses the same signature flipped.

**A name is build ABI, not meaning.** A module or pin name is a symbol a tool
resolves; it carries no semantics.

``Clock(Derived(of, ratio))`` says a doubled clock is not a free one: its rate
is the reference clock's, and a consumer must not pin a frequency for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Union


class AbiError(Exception):
    """An ABI is not well formed, so it describes no module."""


class Direction(Enum):
    IN = "input"
    OUT = "output"
    INOUT = "inout"


def flip(direction: Direction) -> Direction:
    """The same signature, seen from the other end of the wire."""

    if direction is Direction.IN:
        return Direction.OUT
    if direction is Direction.OUT:
        return Direction.IN
    return Direction.INOUT


class Endpoint(Enum):
    """Which end of a bus or stream this module is."""

    INITIATOR = "initiator"
    TARGET = "target"


# -- roles ---------------------------------------------------------------------


@dataclass(frozen=True)
class Data:
    """Payload."""


@dataclass(frozen=True)
class Free:
    """A clock whose rate the enclosing design chooses."""


@dataclass(frozen=True)
class Derived:
    """A clock whose rate is a fixed multiple of another of this module's clocks."""

    of: str
    ratio: int

    def __post_init__(self) -> None:
        if not self.of:
            raise AbiError("a derived clock names the clock it is derived from")
        if self.ratio < 1:
            raise AbiError(f"a clock ratio is at least 1, got {self.ratio}")


Rate = Union[Free, Derived]


@dataclass(frozen=True)
class Clock:
    rate: Rate = Free()


@dataclass(frozen=True)
class Reset:
    active_low: bool = True
    synchronous: bool = False
    synchronous_to: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.synchronous_to is None:
            return
        domains = tuple(self.synchronous_to)
        if len(domains) != len(set(domains)):
            raise AbiError("a qualified reset names one synchronous clock twice")
        if any(not domain for domain in domains):
            raise AbiError("a qualified reset names non-empty synchronous clocks")
        domains = tuple(sorted(domains))
        if self.synchronous and not domains:
            raise AbiError("a qualified synchronous reset names at least one clock")
        if not self.synchronous and domains:
            raise AbiError("a qualified asynchronous reset has no synchronous clocks")
        object.__setattr__(self, "synchronous_to", domains)


Role = Union[Data, Clock, Reset]


# -- protocols and their signatures --------------------------------------------


class StandardProtocol(Enum):
    AXIS = "amba.axis"
    AXILITE = "amba.axilite"


#: Member directions **for the initiator**.  The target's are these flipped.
SIGNATURES: Mapping[StandardProtocol, Mapping[str, Direction]] = {
    StandardProtocol.AXIS: {
        "tdata": Direction.OUT,
        "tvalid": Direction.OUT,
        "tready": Direction.IN,
        "tlast": Direction.OUT,
    },
    StandardProtocol.AXILITE: {
        "awaddr": Direction.OUT,
        "awprot": Direction.OUT,
        "awvalid": Direction.OUT,
        "awready": Direction.IN,
        "wdata": Direction.OUT,
        "wstrb": Direction.OUT,
        "wvalid": Direction.OUT,
        "wready": Direction.IN,
        "bresp": Direction.IN,
        "bvalid": Direction.IN,
        "bready": Direction.OUT,
        "araddr": Direction.OUT,
        "arprot": Direction.OUT,
        "arvalid": Direction.OUT,
        "arready": Direction.IN,
        "rdata": Direction.IN,
        "rresp": Direction.IN,
        "rvalid": Direction.IN,
        "rready": Direction.OUT,
    },
}


# -- pins ----------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One loose pin."""

    name: str
    direction: Direction
    width: int
    role: Role = Data()

    def __post_init__(self) -> None:
        if not self.name:
            raise AbiError("a signal needs a name")
        if self.width < 1:
            raise AbiError(f"{self.name} has width {self.width}; a pin is at least one bit")


@dataclass(frozen=True, order=True)
class Member:
    """One logical bus member, the pin that carries it, and how wide it is.

    One bit by default, which is right for every handshake member.
    """

    logical: str
    physical: str
    width: int = 1

    def __post_init__(self) -> None:
        if not self.logical or not self.physical:
            raise AbiError("a bus member needs a logical name and a physical pin")
        if self.width < 1:
            raise AbiError(f"{self.physical} has width {self.width}; a pin is at least one bit")


@dataclass(frozen=True, init=False)
class Bus:
    """A group of pins that a consumer connects as one interface.

    ``signals`` maps each logical member to the physical pin that carries it,
    so the ABI can describe RTL whose naming it does not control.  It is stored
    sorted: the map is a lookup and its order is not a fact.
    """

    name: str
    protocol: StandardProtocol
    signals: tuple[Member, ...]
    endpoint: Endpoint = Endpoint.TARGET
    role: Role = Data()
    associated_clock: str | None = None
    associated_reset: str | None = None

    def __init__(
        self,
        name: str,
        protocol: StandardProtocol,
        signals: Iterable[Member],
        endpoint: Endpoint = Endpoint.TARGET,
        role: Role = Data(),
        associated_clock: str | None = None,
        associated_reset: str | None = None,
    ) -> None:
        members = tuple(sorted(signals))
        if not members:
            raise AbiError(f"bus {name!r} groups no signals")
        logical = [member.logical for member in members]
        if len(logical) != len(set(logical)):
            raise AbiError(f"bus {name!r} maps one logical member twice")
        unknown = [member for member in logical if member not in SIGNATURES[protocol]]
        if unknown:
            raise AbiError(
                f"bus {name!r} declares {unknown} which {protocol.value} has no member for"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "signals", members)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "associated_clock", associated_clock)
        object.__setattr__(self, "associated_reset", associated_reset)

    def member_directions(self) -> tuple[tuple[str, Direction], ...]:
        """Each physical pin's direction, from the signature and the endpoint."""

        signature = SIGNATURES[self.protocol]
        return tuple(
            (
                member.physical,
                signature[member.logical]
                if self.endpoint is Endpoint.INITIATOR
                else flip(signature[member.logical]),
            )
            for member in self.signals
        )

    def widths(self) -> tuple[tuple[str, int], ...]:
        """Each physical pin and the width the ABI declares for it."""

        return tuple((member.physical, member.width) for member in self.signals)


#: One entry of a module's ABI: a loose pin, or a bus of them.
Pin = Union[Signal, Bus]


@dataclass(frozen=True, order=True)
class ClockAlignment:
    """The bounded aligned-2x environment contract for two input clocks.

    Frequency remains a property of :class:`Derived`.  This value additionally
    requires coincident rising edges at every reference-clock rising edge and
    the other aligned-clock rising edge during the reference low half-cycle.
    """

    reference_clock: str
    aligned_clock: str

    def __post_init__(self) -> None:
        if not self.reference_clock or not self.aligned_clock:
            raise AbiError("a clock alignment names both clocks")
        if self.reference_clock == self.aligned_clock:
            raise AbiError("a clock cannot be aligned-2x to itself")


def physical_names(pins: Sequence[Pin]) -> tuple[str, ...]:
    """Every pin the module actually has, in declared order."""

    names: list[str] = []
    for pin in pins:
        if isinstance(pin, Signal):
            names.append(pin.name)
        else:
            names.extend(member.physical for member in pin.signals)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class PinInfo:
    """One physical pin: its direction, width and role, and the bus member it carries."""

    direction: Direction
    width: int
    role: Role
    bus: str | None = None
    member: str | None = None


def abi_pins(pins: Sequence[Pin]) -> Mapping[str, PinInfo]:
    """Every physical pin of ``pins`` by name, in declared order; a bus member takes
    its bus's role."""

    physical: dict[str, PinInfo] = {}
    for pin in pins:
        if isinstance(pin, Signal):
            physical[pin.name] = PinInfo(pin.direction, pin.width, pin.role)
            continue
        directions = dict(pin.member_directions())
        for member in pin.signals:
            physical[member.physical] = PinInfo(
                directions[member.physical], member.width, pin.role, pin.name, member.logical
            )
    return MappingProxyType(physical)


def validate_pins(pins: Sequence[Pin], clock_alignments: Sequence[ClockAlignment]) -> None:
    """Refuse pins that name one pin twice or whose clock relations do not hold."""

    names = [pin.name for pin in pins]
    if len(names) != len(set(names)):
        raise AbiError("an ABI names one pin twice")
    physical = physical_names(pins)
    if len(physical) != len(set(physical)):
        raise AbiError("an ABI carries one physical pin in two places")

    by_name = {pin.name: pin for pin in pins}
    clocks = {
        pin.name: pin for pin in pins if isinstance(pin, Signal) and isinstance(pin.role, Clock)
    }
    for pin in pins:
        if isinstance(pin, Signal) and isinstance(pin.role, Reset):
            missing = tuple(d for d in pin.role.synchronous_to or () if d not in clocks)
            if missing:
                raise AbiError(
                    f"reset {pin.name!r} is synchronous to clocks this ABI does not have: "
                    f"{missing!r}"
                )
        if not isinstance(pin, Bus):
            continue
        if pin.associated_clock is not None and pin.associated_clock not in clocks:
            raise AbiError(
                f"bus {pin.name!r} associated clock {pin.associated_clock!r} "
                "does not name a clock signal"
            )
        if pin.associated_reset is not None:
            reset = by_name.get(pin.associated_reset)
            if not isinstance(reset, Signal) or not isinstance(reset.role, Reset):
                raise AbiError(
                    f"bus {pin.name!r} associated reset {pin.associated_reset!r} "
                    "does not name a reset signal"
                )
            domains = reset.role.synchronous_to
            if (
                reset.role.synchronous
                and domains is not None
                and pin.associated_clock not in domains
            ):
                raise AbiError(
                    f"bus {pin.name!r} uses clock {pin.associated_clock!r}, but reset "
                    f"{pin.associated_reset!r} is synchronous to {domains!r}"
                )

    if len(set(clock_alignments)) != len(clock_alignments):
        raise AbiError("an ABI declares one clock alignment twice")
    for alignment in clock_alignments:
        reference = clocks.get(alignment.reference_clock)
        aligned = clocks.get(alignment.aligned_clock)
        if reference is None or aligned is None:
            missing = tuple(
                name
                for name, clock in (
                    (alignment.reference_clock, reference),
                    (alignment.aligned_clock, aligned),
                )
                if clock is None
            )
            raise AbiError(f"clock alignment names clocks this ABI does not have: {missing!r}")
        if reference.direction is not Direction.IN or aligned.direction is not Direction.IN:
            raise AbiError("aligned clocks are input signals supplied by the environment")
        if aligned.role != Clock(Derived(alignment.reference_clock, 2)):
            raise AbiError(
                f"aligned clock {alignment.aligned_clock!r} must be "
                f"Derived({alignment.reference_clock!r}, 2)"
            )


# -- the RTL check -------------------------------------------------------------


@dataclass(frozen=True)
class ObservedPort:
    """A port as the RTL actually declares it, supplied by whoever parsed the source."""

    name: str
    direction: Direction
    width: int


def check_against_rtl(pins: Sequence[Pin], observed: Sequence[ObservedPort]) -> tuple[str, ...]:
    """Every way the source contradicts the declared pins; empty when they agree.

    Refusal only: the declaration stays authoritative and nothing is read back
    into it.  SystemVerilog identifiers are case sensitive, so a name the
    source spells in another case is reported as such.
    """

    actual = {port.name: port for port in observed}
    folded: dict[str, list[str]] = {}
    for name in actual:
        folded.setdefault(name.casefold(), []).append(name)

    issues: list[str] = []
    declared_names = physical_names(pins)
    for declared in declared_names:
        if declared in actual:
            continue
        near = folded.get(declared.casefold())
        if near:
            issues.append(
                f"the ABI declares {declared!r} but the source spells it {near[0]!r}; "
                "SystemVerilog identifiers are case sensitive"
            )
        else:
            issues.append(f"the ABI declares {declared!r}, which the source does not have")
    for name in actual:
        if name not in declared_names:
            issues.append(f"the source has {name!r}, which the ABI does not declare")

    for pin in pins:
        if isinstance(pin, Signal):
            expected: tuple[tuple[str, Direction], ...] = ((pin.name, pin.direction),)
            widths: tuple[tuple[str, int], ...] = ((pin.name, pin.width),)
        else:
            expected, widths = pin.member_directions(), pin.widths()
        for name, direction in expected:
            found = actual.get(name)
            if found is not None and found.direction is not direction:
                issues.append(
                    f"the ABI declares {name!r} as {direction.value} and the source "
                    f"declares it {found.direction.value}"
                )
        for name, width in widths:
            found = actual.get(name)
            if found is not None and found.width != width:
                issues.append(
                    f"the ABI declares {name!r} as {width} bits and the source "
                    f"resolves it to {found.width}"
                )

    return tuple(issues)


__all__ = [
    "SIGNATURES",
    "AbiError",
    "Bus",
    "Clock",
    "ClockAlignment",
    "Data",
    "Derived",
    "Direction",
    "Endpoint",
    "Free",
    "Member",
    "ObservedPort",
    "Pin",
    "PinInfo",
    "Rate",
    "Reset",
    "Role",
    "Signal",
    "StandardProtocol",
    "abi_pins",
    "check_against_rtl",
    "flip",
    "physical_names",
    "validate_pins",
]
