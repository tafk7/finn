# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``ComponentABI``: the portability boundary, and physical facts only.

Packaging is married to one vendor today, and the reason is not the code -- it
is that the *interface knowledge* lives only in the Tcl: ``infer_bus_interface``,
``associate_bus_interfaces``, the abstraction VLNVs, the ``FREQ_HZ``
workaround.  Declare it structurally and every format becomes a projection.

Three rules hold the type down.

**Physical facts only.**  Nothing here names a Region, a port declaration path,
an Operation, or a source scope.  Two Operations binding the same memstream
Kernel through different Region declaration paths must produce an ABI that
compares *equal*, or cross-Operation reuse is lost -- and adding an exclusion
list to ABI equality to recover it is the error this whole design is against.
Semantic associations live outside, in the association index.

**Direction is declared once and flipped.**  A bus signature gives its member
directions for the *initiator*; a target reuses the same signature flipped.
A direction table written on both sides of a bus is one fact stated twice, and
therefore a fact that will eventually disagree.

**A name in an ABI is build ABI, not meaning.**  A module name is a symbol a
linker resolves.  It carries no semantics and must never be read as any.

``Clock(Derived(of, ratio))`` earns its place against an observed failure: a
doubled clock is not a free clock, its rate is not the component's to declare,
and a packager that has to guess pins a ``FREQ_HZ`` the component was never
told -- so any enclosing design at another rate fails validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Union


class AbiError(Exception):
    """An ABI is not well formed, so it describes no component."""


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
    """Which end of a bus this component is.

    Not in the design's §4.1 field list, and it has to be: "declared for the
    initiator and flipped for the target" is not a rule a value can follow
    without saying which one it is.
    """

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
    """A clock whose rate is a fixed multiple of another of this component's.

    ``ap_clk2x`` is this, and the reason the type exists: a packager that
    cannot express the relation pins a frequency the component never declared.
    """

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


@dataclass(frozen=True)
class Config:
    """Static configuration driven from outside."""


@dataclass(frozen=True)
class Status:
    """Observable state driven outward."""


@dataclass(frozen=True)
class Interrupt:
    """An event line."""


Role = Union[Data, Clock, Reset, Config, Status, Interrupt]


# -- protocols and their signatures --------------------------------------------


class StandardProtocol(Enum):
    AXIS = "amba.axis"
    AXILITE = "amba.axilite"
    AXI = "amba.axi"
    BRAM = "bram"


@dataclass(frozen=True)
class CustomProtocol:
    """A protocol we do not model, named so a packager can refuse it."""

    protocol_id: str
    spec_ref: str = ""

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise AbiError("a custom protocol needs an id")


Protocol = Union[StandardProtocol, CustomProtocol]

#: Member directions **for the initiator**.  The target's are these flipped.
SIGNATURES: Mapping[StandardProtocol, Mapping[str, Direction]] = {
    StandardProtocol.AXIS: {
        "tdata": Direction.OUT,
        "tvalid": Direction.OUT,
        "tready": Direction.IN,
        "tlast": Direction.OUT,
        "tkeep": Direction.OUT,
        "tstrb": Direction.OUT,
        "tuser": Direction.OUT,
        "tid": Direction.OUT,
        "tdest": Direction.OUT,
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
    StandardProtocol.BRAM: {
        "addr": Direction.OUT,
        "din": Direction.OUT,
        "dout": Direction.IN,
        "en": Direction.OUT,
        "we": Direction.OUT,
        "clk": Direction.OUT,
        "rst": Direction.OUT,
    },
}

#: The members that make a group of loose signals recognisably one AXI-Stream.
#: A packager infers exactly this, so declaring a different grouping publishes
#: interfaces the unit does not have -- and the failure surfaces in a block
#: design rather than at authoring.
AXIS_REQUIRED = frozenset({"tdata", "tvalid", "tready"})


# -- ports ---------------------------------------------------------------------


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

    The width is here rather than only on ``Signal`` because a declared
    ``tdata`` width that disagrees with the RTL is the most likely real
    mismatch, and the one a packager must get right.  A bus whose members
    carried no width would be the one part of an ABI nothing could check.

    One bit by default, which is right for every handshake member; only the
    payload members are ever wider.
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
    protocol: Protocol
    signals: tuple[Member, ...]
    endpoint: Endpoint = Endpoint.TARGET
    role: Role = Data()
    associated_clock: str | None = None
    associated_reset: str | None = None

    def __init__(
        self,
        name: str,
        protocol: Protocol,
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
        if isinstance(protocol, StandardProtocol):
            if protocol not in SIGNATURES:
                # Refused here rather than at member_directions().  A bus that
                # constructs and then cannot say which way its pins point is a
                # value that is only half a value, and the failure surfaces at
                # whichever consumer happens to ask first.
                raise AbiError(
                    f"bus {name!r} speaks {protocol.value}, which this package has no "
                    "declared signature for; declare one, or say CustomProtocol and be "
                    "refused by the formats that publish interfaces"
                )
            known = SIGNATURES[protocol]
            unknown = [member for member in logical if member not in known]
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
        """Each physical pin's direction, from the signature and the endpoint.

        The one place bus directions come from.  Nobody writes them twice.
        """

        if not isinstance(self.protocol, StandardProtocol) or self.protocol not in SIGNATURES:
            raise AbiError(
                f"bus {self.name!r} speaks {self.protocol}, which has no declared signature; "
                "a packager must refuse it rather than guess its directions"
            )
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


Port = Union[Signal, Bus]


@dataclass(frozen=True)
class ComponentABI:
    """Everything a packaging format may read besides a target and its options.

    ``ports`` is ordered, because a port list is a declaration and reordering
    it is a change to what was declared.  ``parameters`` is sorted: it is a
    table.
    """

    entry_point: str
    ports: tuple[Port, ...]
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.entry_point:
            raise AbiError("an ABI needs an entry point")
        names = [port.name for port in self.ports]
        if len(names) != len(set(names)):
            raise AbiError("an ABI names one port twice")
        physical = list(self.physical_names())
        if len(physical) != len(set(physical)):
            raise AbiError("an ABI carries one physical pin in two places")
        parameters = tuple(sorted(self.parameters, key=lambda item: item[0]))
        if len({name for name, _ in parameters}) != len(parameters):
            raise AbiError("an ABI names one parameter twice")
        object.__setattr__(self, "parameters", parameters)

    def physical_names(self) -> tuple[str, ...]:
        """Every pin the module actually has, in declared order."""

        names: list[str] = []
        for port in self.ports:
            if isinstance(port, Signal):
                names.append(port.name)
            else:
                names.extend(member.physical for member in port.signals)
        return tuple(names)

    def clocks(self) -> tuple[Signal, ...]:
        return tuple(
            port for port in self.ports if isinstance(port, Signal) and isinstance(port.role, Clock)
        )


# -- inference and the checks it makes possible --------------------------------


def infer_buses(names: Sequence[str]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Group ``<prefix>_tdata``/``_tvalid``/``_tready`` into one interface.

    This is the same inference a packager performs.  Running it over the
    *declared* ABI is what catches a declared grouping the tool will not agree
    with -- earlier than a block design would.
    """

    groups: dict[str, list[tuple[str, str]]] = {}
    for name in names:
        prefix, _, suffix = name.rpartition("_")
        if prefix and suffix in SIGNATURES[StandardProtocol.AXIS]:
            groups.setdefault(prefix, []).append((suffix, name))
    return tuple(
        (prefix, tuple(sorted(members)))
        for prefix, members in sorted(groups.items())
        if AXIS_REQUIRED <= {member for member, _ in members}
    )


@dataclass(frozen=True)
class ObservedPort:
    """A port as the RTL actually declares it.

    Supplied by whoever parsed the source.  This module does no parsing: the
    checker is refusal-only and never supplies a value.
    """

    name: str
    direction: Direction
    width: int


def _conventional_prefix(physical: str) -> str | None:
    """The prefix suffix inference would file this pin under, if it would.

    ``None`` means the name does not end in an AXI-Stream member suffix at all,
    so inference never sees it.  That is not a disagreement -- it is the
    ``Bus.signals`` map doing its job on RTL whose naming the ABI does not
    control.
    """

    prefix, _, suffix = physical.rpartition("_")
    if prefix and suffix in SIGNATURES[StandardProtocol.AXIS]:
        return prefix
    return None


def check_declared_grouping(abi: ComponentABI) -> tuple[str, ...]:
    """Where the declared bus grouping and the inferred one disagree.

    A packager infers interfaces from suffixes.  If the declaration groups
    differently, the component publishes interfaces the unit does not have.

    A bus whose pins do not follow the suffix convention **at all** is not
    reported.  Mapping a logical member onto an unconventional pin is the
    feature ``Bus.signals`` exists for, and complaining about it would make the
    feature unusable without an accompanying complaint.  What is reported is a
    bus that is *partly* conventional: those pins do reach inference, and where
    it puts them is then a fact that can disagree.
    """

    inferred = {
        frozenset(physical for _, physical in members)
        for _, members in infer_buses(abi.physical_names())
    }
    declared: set[frozenset[str]] = set()
    issues: list[str] = []
    for port in abi.ports:
        if not isinstance(port, Bus) or port.protocol is not StandardProtocol.AXIS:
            continue
        pins = frozenset(member.physical for member in port.signals)
        declared.add(pins)
        prefixes = {physical: _conventional_prefix(physical) for physical in pins}
        conventional = {name for name, prefix in prefixes.items() if prefix is not None}
        if not conventional:
            continue
        if conventional != pins:
            issues.append(
                f"declared AXI-Stream {port.name!r} mixes {sorted(conventional)}, which suffix "
                f"inference groups, with {sorted(pins - conventional)}, which it does not; a "
                "packager would publish part of this interface and leave the rest loose"
            )
            continue
        if pins not in inferred:
            issues.append(
                f"declared AXI-Stream {sorted(pins)} is not what suffix inference would "
                "group; a packager would publish a different interface"
            )
    for group in sorted(inferred - declared, key=sorted):
        issues.append(
            f"{sorted(group)} infers as an AXI-Stream but is not declared as one; a "
            "stitcher would see loose pins"
        )
    return tuple(issues)


def check_against_rtl(abi: ComponentABI, observed: Sequence[ObservedPort]) -> tuple[str, ...]:
    """Refuse a declaration the source contradicts.

    Refusal-only, by contract.  The declaration stays authoritative -- generated
    RTL is generated *from* the ABI, so parsing it back would be circular, and
    the ABI is the packaging contract and must not move whenever the RTL does.
    What a checker adds is that a declaration nothing checks is a second
    authority waiting to disagree.

    The live case: the physical model says ``in0_V_TDATA`` and the generated
    wrapper says ``in0_V_tdata``.  SystemVerilog identifiers are case
    sensitive, so a consumer taking the reported name into a ``connect_bd_net``
    names a pin that does not exist.  Nothing compared the two authorities,
    which is why it survived.
    """

    actual = {port.name: port for port in observed}
    folded: dict[str, list[str]] = {}
    for name in actual:
        folded.setdefault(name.casefold(), []).append(name)

    issues: list[str] = []
    for declared in abi.physical_names():
        if declared in actual:
            continue
        near = folded.get(declared.casefold())
        if near:
            issues.append(
                f"the ABI declares {declared!r} but the source spells it {near[0]!r}; "
                "SystemVerilog identifiers are case sensitive, so a consumer connecting "
                "the declared name would name a pin that does not exist"
            )
        else:
            issues.append(f"the ABI declares {declared!r}, which the source does not have")

    declared_names = set(abi.physical_names())
    for name in actual:
        if name not in declared_names:
            issues.append(f"the source has {name!r}, which the ABI does not declare")

    for port in abi.ports:
        if isinstance(port, Signal):
            expected: tuple[tuple[str, Direction], ...] = ((port.name, port.direction),)
        else:
            expected = port.member_directions()
        for name, direction in expected:
            found = actual.get(name)
            if found is not None and found.direction is not direction:
                issues.append(
                    f"the ABI declares {name!r} as {direction.value} and the source "
                    f"declares it {found.direction.value}"
                )

    # Bus members are checked here too, and that is the point of them carrying
    # a width: a declared ``tdata`` that disagrees with the RTL is the most
    # likely real mismatch, and it used to pass because only loose signals were
    # compared.
    for port in abi.ports:
        if isinstance(port, Signal):
            widths: tuple[tuple[str, int], ...] = ((port.name, port.width),)
        else:
            widths = port.widths()
        for name, width in widths:
            found = actual.get(name)
            if found is not None and found.width != width:
                issues.append(
                    f"the ABI declares {name!r} as {width} bits and the source "
                    f"resolves it to {found.width}"
                )

    return tuple(issues)


__all__ = [
    "AXIS_REQUIRED",
    "SIGNATURES",
    "AbiError",
    "Bus",
    "Clock",
    "ComponentABI",
    "Config",
    "CustomProtocol",
    "Data",
    "Derived",
    "Direction",
    "Endpoint",
    "Free",
    "Interrupt",
    "Member",
    "ObservedPort",
    "Port",
    "Protocol",
    "Rate",
    "Reset",
    "Role",
    "Signal",
    "StandardProtocol",
    "Status",
    "check_against_rtl",
    "check_declared_grouping",
    "flip",
    "infer_buses",
]
