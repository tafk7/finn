# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Kernel protocol: one module's choices, and the plumbing every kernel shares.

A kernel is a Space whose choices configure one module. It either binds one
FinnLib module (a leaf) or places kernel children and the channels between
them; never both. On the protocol a leaf declares only what is its own:

- ``rtl_module``: the RTL module it instantiates, and ``sources()``, the
  source files that provide it and any data they read;
- one ``Port`` node per interface (``finn.kernels.port``): each exports its
  pins, its clock, and what it holds while idle;
- ``parameters()``: the module's parameters, from its choices;
- ``clocking``: its clock and reset pins, when not the plain ``ap_clk`` and
  ``ap_rst_n``;
- ``admission``: the constraint group by which it refuses a configuration it
  cannot build;
- ``other_pins()``, ``held()`` and ``controlled()``: pins that are no port's (an
  AXI-Lite bus), what it holds of them, and the buses among them it presents
  through a ``ControlBus`` (``finn.kernels.control``).

The base derives the rest. ``codegen`` is the leaf (``Leaf``): its pins
(clocking, other pins, then every port's), parameters, sources and data, and
what it holds (the doubled clock while unused, idle ports, and its own
``held()``). Two constraints account for its pins at creation:
``pins_accounted`` (every input among its other pins is held or presented) and
``clocked`` (every port runs on its clock).

A kernel with children merges its members' netlists (``NETLIST``: each
child's and each channel's ``Fragment``, under its node) and the control buses
its ``ControlBus`` nodes present. Its interface channels are reference inputs
its parent supplies; the channels between its children are its own. Placed
alone, as the root of what is emitted, it is one ``Composed`` module whose
pins are ``ap_clk``, ``ap_clk2x`` when an instance takes a doubled clock,
``ap_rst_n``, the AXIS bus of each boundary channel it declares (inputs, then
outputs) and each presented bus.

Every kernel exports its netlist (``NETLIST``) and its ``module`` (``MODULE``):
the ``Leaf``, or the ``Composed`` module, accepted under ``admission``,
``pins_accounted`` and ``clocked``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, cast

from finn.core.space import (
    ConstraintGroup,
    DefinitionError,
    Domain,
    Members,
    Rejected,
    Space,
    ValueSemantics,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    divisors_of,
    domain,
    reject,
)
from finn.dataflow.schedule import Access, Index, Refused, Schedule, bind_extents
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Data,
    Derived,
    Direction,
    Endpoint,
    Free,
    Pin,
    Reset,
    Signal,
    abi_pins,
)
from finn.kernels.artifacts.contributions import Contribution, CopiedSource, GeneratedData
from finn.kernels.artifacts.module import (
    BuildError,
    BusExport,
    Composed,
    Fragment,
    Held,
    Leaf,
    Module,
    Pins,
    ProducerIdentity,
    merge,
)
from finn.kernels.control import EXPORTED, top_bus
from finn.kernels.transport import STREAM_CONTRACT


def _frozen(name: str, *kinds: type) -> ValueSemantics[object]:
    """A frozen build value: compared by value, shared without a copy."""
    return ValueSemantics(
        kinds[0],
        name,
        lambda value: isinstance(value, kinds),
        lambda left, right: left == right,
        lambda value: value,
    )


MODULE_SEMANTICS = cast("ValueSemantics[Module]", _frozen("module", Leaf, Composed))
MODULE = ViewKey("module", MODULE_SEMANTICS)
"""A kernel's module: its ``Leaf``, or its children's netlist as one ``Composed`` module."""

NETLIST_SEMANTICS = cast("ValueSemantics[Fragment]", _frozen("netlist", Fragment))
NETLIST = ViewKey("netlist", NETLIST_SEMANTICS)
"""A kernel's or a channel's netlist (``finn.kernels.artifacts.module.Fragment``), labelled
relative to itself; its parent merges its members' under their nodes."""

BOUNDARY = ViewKey("boundary", default_semantics(tuple))
"""A channel's AXIS bus on the root's boundary: one, or none when both of its ends are
kernels."""

PORT = ViewKey("port", STREAM_CONTRACT)
"""A kernel's port on one channel, exported per reference input."""

PINS = ViewKey("pins", default_semantics(tuple))
"""A port's pins (signals, or one bus), collected by its kernel into the module's pins."""

CLOCKED = ViewKey("clocked", default_semantics(str))
"""The clock pin a port runs on, collected by its kernel (``clocked``)."""

ACCESS = ViewKey("access", default_semantics(Access))
"""A placed scheduled port's read of its channel's tensor, collected by its kernel to bind
the extents of its indices (``finn.dataflow.schedule.bind_extents``)."""

HELD_SEMANTICS = default_semantics(Held)
HELD = ViewKey("held", HELD_SEMANTICS)
"""What an idle port holds, collected by its kernel's leaf."""

# The composed module's clocking pins: its interface convention, not a routing rule.
CLOCK, CLOCK2X, RESET = "ap_clk", "ap_clk2x", "ap_rst_n"


@dataclass(frozen=True)
class Clocking:
    """A module's clock and reset pins.

    ``clock`` is free-running and ``reset`` synchronous, active low unless
    ``active_low`` is False (FinnLib's native ``clk`` and ``rst``). A
    ``doubled`` pin, when the module has one, is a clock at twice ``clock``
    while ``doubling``, and otherwise an unused input held low.
    """

    clock: str = "ap_clk"
    reset: str = "ap_rst_n"
    doubled: str | None = None
    doubling: bool = False
    active_low: bool = True

    def signals(self) -> tuple[Signal, ...]:
        doubled = () if self.doubled is None else (self.doubled,)
        clocks = (self.clock, *doubled) if self.doubling else (self.clock,)
        return (
            Signal(self.clock, Direction.IN, 1, Clock(Free())),
            *(
                Signal(
                    name,
                    Direction.IN,
                    1,
                    Clock(Derived(self.clock, 2)) if self.doubling else Data(),
                )
                for name in doubled
            ),
            Signal(
                self.reset,
                Direction.IN,
                1,
                Reset(active_low=self.active_low, synchronous=True, synchronous_to=clocks),
            ),
        )

    def alignments(self) -> tuple[ClockAlignment, ...]:
        if self.doubled is None or not self.doubling:
            return ()
        return (ClockAlignment(self.clock, self.doubled),)

    def held(self) -> tuple[tuple[str, int], ...]:
        """The doubled clock input, held low while unused."""
        if self.doubled is None or self.doubling:
            return ()
        return ((self.doubled, 0),)


NATIVE_CLOCKING = Clocking(clock="clk", reset="rst", active_low=False)
"""FinnLib's native ``clk`` and synchronous active-high ``rst``."""


def _inputs(pin: Pin) -> tuple[str, ...]:
    if isinstance(pin, Signal):
        return (pin.name,) if pin.direction is Direction.IN else ()
    return tuple(name for name, direction in pin.member_directions() if direction is Direction.IN)


class Kernel(Space):
    """A named Space class configuring one module; see the module docstring for the protocol."""

    id: ClassVar[str] = ""
    version: ClassVar[int] = 1
    # The RTL module it instantiates; empty for a kernel with children.
    rtl_module: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if type(cls.id) is not str or not cls.id:
            raise DefinitionError(f"{cls.__qualname__} must declare a nonempty string id")
        if type(cls.version) is not int or cls.version < 1:
            raise DefinitionError(f"{cls.__qualname__} must declare a positive int version")

    # -- the protocol: what a kernel declares ----------------------------------------------

    port_pins = Members(PINS)
    port_holds = Members(HELD)
    port_clocks = Members(CLOCKED)
    port_accesses = Members(ACCESS)
    admission = ConstraintGroup()

    def parameters(self) -> Mapping[str, int | str] | Rejected:
        """The module's parameters, read from the kernel's choices; refused when it has none."""
        return {}

    def sources(self) -> tuple[Contribution, ...]:
        """The source files that provide ``rtl_module``, and any data they read."""
        return ()

    def other_pins(self) -> tuple[Pin, ...]:
        """Pins that are no port's, such as an AXI-Lite configuration bus."""
        return ()

    def held(self) -> Held | Rejected:
        """What the kernel itself holds idle, beyond its idle ports and doubled clock."""
        return Held()

    def controlled(self) -> tuple[Bus, ...]:
        """The buses among its other pins it presents through a ``ControlBus``."""
        return ()

    def stem(self) -> str:
        """A kernel with children: its composed module's name stem."""
        return "finn_" + type(self).__name__.lower()

    def producer_identity(self) -> ProducerIdentity:
        """A kernel with children: what derives its composed module."""
        space_type = type(self)
        return ProducerIdentity(space_type.id, str(space_type.version))

    @derived
    def clocking(self) -> Clocking:
        return Clocking()

    # -- extents and the schedule --------------------------------------------------------

    def _bound(self, extents: Mapping[Index, int] | None = None) -> dict[Index, int] | Rejected:
        accesses = [replace(item.value, name=item.node or "") for item in self.port_accesses]
        try:
            return bind_extents(accesses, extents)
        except Refused as error:
            return reject("kernel-extents", str(error))

    @derived
    def extents(self) -> dict[Index, int] | Rejected:
        """Each index's extent, bound from the tensors its placed ports read."""
        return self._bound()

    def bound_schedule(
        self,
        order: tuple[Index, ...],
        factors: Mapping[Index, int] | None = None,
        extents: Mapping[Index, int] | None = None,
    ) -> Schedule | Rejected:
        """The schedule in beat ``order`` (outer to inner), each index's extent bound from the
        ports' tensors (and any ``extents`` the kernel gives), with folding ``factors``."""
        bound = self.extents if extents is None else self._bound(extents)
        if isinstance(bound, Rejected):
            return bound
        missing = [index for index in order if index not in bound]
        if missing:
            return reject("kernel-extents", f"{missing} are bound by no placed port")
        try:
            return Schedule({index: bound[index] for index in order}, factors, order)
        except ValueError as error:
            return reject("kernel-schedule", str(error))

    # -- a leaf: its module, and its pins accounted for -------------------------------------

    @derived
    def holds(self) -> Held | Rejected:
        """What the module holds: the doubled clock while unused, idle ports, its own."""
        held = self.held()
        if isinstance(held, Rejected):
            return held
        parts = (Held(self.clocking.held()), *(item.value for item in self.port_holds), held)
        return Held(
            tuple(pin for part in parts for pin in part.inputs),
            tuple(pin for part in parts for pin in part.unused),
        )

    @derived
    def codegen(self) -> Leaf | Rejected:
        """The module: clocking, its other pins, then every port's pins; its parameters,
        sources and data; and what it holds."""
        space_type = type(self)
        if not space_type.rtl_module:
            return reject("kernel-module", f"{space_type.__qualname__} declares no module")
        clocking = self.clocking
        chosen = self.parameters()
        if isinstance(chosen, Rejected):
            return chosen
        parameters = tuple(sorted(chosen.items()))
        pins = tuple(pin for item in self.port_pins for pin in item.value)
        contributions = self.sources()
        return Leaf(
            space_type.id,
            str(space_type.version),
            space_type.rtl_module,
            parameters,
            Pins(
                (*clocking.signals(), *self.other_pins(), *pins),
                tuple((name, str(value)) for name, value in parameters),
                clocking.alignments(),
            ),
            tuple(item for item in contributions if isinstance(item, CopiedSource)),
            tuple(item for item in contributions if isinstance(item, GeneratedData)),
            self.holds,
        )

    @constraint
    def pins_accounted(self) -> bool | Rejected:
        """Every input among its other pins is held, or presented as a control bus."""
        held = self.held()
        if isinstance(held, Rejected):
            return held
        accounted = {pin for pin, _ in held.inputs} | {
            member.physical for bus in self.controlled() for member in bus.signals
        }
        loose = [pin for item in self.other_pins() for pin in _inputs(item) if pin not in accounted]
        if loose:
            return reject(
                "kernel-pins",
                f"{type(self).__qualname__}: the inputs {loose} are neither held nor presented",
            )
        return True

    @constraint
    def clocked(self) -> bool | Rejected:
        """Every port runs on the module's clock."""
        clock = self.clocking.clock
        other = [f"{item.node} on {item.value}" for item in self.port_clocks if item.value != clock]
        if other:
            return reject("kernel-clock", f"ports run off the clock {clock}: {', '.join(other)}")
        return True

    # -- a kernel with children: their netlists, merged -------------------------------------

    netlists = Members(NETLIST)
    presented = Members(EXPORTED)
    boundary_buses = Members(BOUNDARY)

    @derived
    def fragment(self) -> Fragment | Rejected:
        """A leaf: its accepted module, the empty label (its parent's ``under(node)`` names
        it ``node``). A kernel with children: each member's netlist under its node, and the
        buses its ``ControlBus`` nodes present; its own module is complete only as the
        root, whose channels it declares, so a parent reads its netlist, not its module."""
        space_type = type(self)
        if space_type.rtl_module:
            if self.netlists:
                return reject("kernel-children", "a kernel binds one module or has children")
            leaf = self.module
            assert isinstance(leaf, Leaf)
            return Fragment((("", leaf),))
        if not self.netlists:
            return reject(
                "kernel-module",
                f"{space_type.__qualname__} declares no module and places no kernel",
            )
        exports = tuple(
            BusExport(item.node, item.child, item.port)
            for located in self.presented
            for item in located.value
        )
        try:
            return merge(
                *(item.value.under(str(item.node)) for item in self.netlists),
                Fragment(exports=exports),
            )
        except BuildError as error:
            return reject("kernel-netlist", str(error))

    @derived
    def composed_pins(self) -> Pins:
        """Its clocks and reset, each boundary channel's AXIS bus (inputs, then outputs), then
        each presented bus."""
        fragment = self.fragment
        doubled = any(
            isinstance(info.role, Clock) and isinstance(info.role.rate, Derived)
            for _, leaf in fragment.instances
            for info in abi_pins(leaf.pins.ports).values()
        )
        clocks = (CLOCK, CLOCK2X) if doubled else (CLOCK,)
        buses = [bus for item in self.boundary_buses for bus in item.value]
        return Pins(
            (
                Signal(CLOCK, Direction.IN, 1, Clock(Free())),
                *((Signal(CLOCK2X, Direction.IN, 1, Clock(Derived(CLOCK, 2))),) if doubled else ()),
                Signal(RESET, Direction.IN, 1, Reset(True, True, clocks)),
                *(bus for bus in buses if bus.endpoint is Endpoint.TARGET),
                *(bus for bus in buses if bus.endpoint is not Endpoint.TARGET),
                *(top_bus(item.bus, item.port, CLOCK, RESET) for item in fragment.exports),
            ),
            (),
            (ClockAlignment(CLOCK, CLOCK2X),) if doubled else (),
        )

    @derived(semantics=MODULE_SEMANTICS)
    def built(self) -> Leaf | Composed | Rejected:
        if type(self).rtl_module:
            return self.codegen
        producer = self.producer_identity()
        try:
            return Composed(
                producer.producer_id,
                producer.contract_version,
                self.stem(),
                self.composed_pins,
                self.fragment,
            )
        except BuildError as error:
            return reject("kernel-netlist", str(error))

    module = View(
        built, requires=(admission, pins_accounted, clocked, netlists, presented, boundary_buses)
    )
    netlist = View(
        fragment, requires=(admission, pins_accounted, clocked, netlists, presented, boundary_buses)
    )

    # A kernel adding exports of its own extends these: ``{**Kernel.exports, KEY: ...}``.
    exports = {NETLIST: netlist, MODULE: module}


def extent_of(index: Index) -> int:
    """A derived member: ``index``'s extent, bound from the kernel's placed ports.

    Name it in the class body (``channels = extent_of(c)``) and read that name,
    in a folding factor's ``divisors_of`` domain for instance; used inline inside another
    declaration it is not a member of the class, and linking refuses it.
    """

    def extent(self: Kernel) -> int | Rejected:
        extents = self.extents
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return extents[index]

    return derived(extent)


def factor_domain(index: Index, bound: int = 1 << 32) -> Domain[int]:
    """A folding factor Decision's domain: the divisors of ``index``'s bound extent.

    While no placed port binds ``index`` (a flat build of a module whose
    parameters need no extents), the factor is any the RTL takes, ``1 <= factor
    < bound``, and is committed as a choice; there is nothing to enumerate.
    """
    candidates = divisors_of(1).candidates
    assert candidates is not None

    def accepts(*, candidate: int, extents: Mapping[Index, int]) -> bool:
        if type(candidate) is not int or candidate < 1:
            return False
        return extents[index] % candidate == 0 if index in extents else candidate < bound

    def enumerate_(*, extents: Mapping[Index, int]) -> tuple[int, ...] | Rejected:
        if index not in extents:
            return reject("kernel-extents", f"{index!r} is bound by no placed port")
        return tuple(cast("tuple[int, ...]", candidates(extent=extents[index])))

    return domain(
        accepts=accepts,
        candidates=enumerate_,
        semantics=default_semantics(int),
        extents=Kernel.extents,
    )


__all__ = [
    "ACCESS",
    "BOUNDARY",
    "CLOCK",
    "CLOCK2X",
    "CLOCKED",
    "Clocking",
    "HELD",
    "HELD_SEMANTICS",
    "Kernel",
    "MODULE",
    "MODULE_SEMANTICS",
    "NATIVE_CLOCKING",
    "NETLIST",
    "NETLIST_SEMANTICS",
    "PINS",
    "PORT",
    "RESET",
    "extent_of",
    "factor_domain",
]
