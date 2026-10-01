# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Kernel protocol: one generated module's choices, and the plumbing every kernel shares.

A kernel is a Space whose choices configure one module. On the protocol it
declares only what is its own:

- ``module``: the RTL module it instantiates, and ``sources()``, the source
  files that provide it;
- one ``Port`` node per interface (``finn.kernels.port``): each exports its
  pins, and what it holds while idle;
- ``parameters()``: the module's parameters, from its choices;
- ``clocking``: its clock and reset pins, when not the plain ``ap_clk`` and
  ``ap_rst_n``;
- ``admission``: the constraint group by which it refuses a configuration it
  cannot build;
- ``other_pins()`` and ``held()``: pins that are no port's (an AXI-Lite bus)
  and what it holds of them.

The base derives the rest: the module's ABI (clocking, other pins, then every
port's pins), ``build_requirements`` (accepted under ``admission``), the
``tieoffs`` (the doubled clock while unused, idle ports, and what it holds
itself) and the exports ``MODULE`` and ``TIEOFFS``. A composite kernel wires
its children's modules through streams instead (``finn.kernels.composite``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, cast

from finn.core.space import (
    ConstraintGroup,
    Domain,
    Members,
    Rejected,
    Space,
    View,
    ViewKey,
    default_semantics,
    derived,
    divisors_of,
    domain,
    reject,
    view,
)

from finn.core.space.errors import DefinitionError
from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    ClockAlignment,
    Data,
    Derived,
    Direction,
    Free,
    Reset,
    Signal,
)
from finn.dataflow.schedule import Access, Index, Refused, Schedule, bind_extents
from finn.kernels.physical.contract import STREAM_CONTRACT
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
    RequirementContribution,
)

MODULE_REQUIREMENTS = default_semantics(ModuleBuildRequirements)
MODULE = ViewKey("module", MODULE_REQUIREMENTS)
"""A kernel's generated module, collected by its composite."""

PORT = ViewKey("port", STREAM_CONTRACT)
"""A kernel's port on one stream, exported per reference input."""

PINS = ViewKey("pins", default_semantics(tuple))
"""A port's pins (signals, or one bus), collected by its kernel into the module's ABI."""

ACCESS = ViewKey("access", default_semantics(Access))
"""A placed scheduled port's read of its stream's tensor, collected by its kernel to bind
the extents of its indices (``finn.dataflow.schedule.bind_extents``)."""


@dataclass(frozen=True)
class Tieoffs:
    """Pins a kernel leaves out of the composition in this configuration.

    ``inputs`` are held constant, as (pin, value); ``unused`` outputs are left
    unconnected.
    """

    inputs: tuple[tuple[str, int], ...] = ()
    unused: tuple[str, ...] = ()


TIEOFFS_SEMANTICS = default_semantics(Tieoffs)
TIEOFFS = ViewKey("tieoffs", TIEOFFS_SEMANTICS)
HELD = ViewKey("held", TIEOFFS_SEMANTICS)
"""What an idle port holds, collected by its kernel's tie-offs."""


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


class Kernel(Space):
    """A named family configuring one module; see the module docstring for the protocol.

    A kernel may describe plain pins, a stream boundary, source requirements,
    or other values. Identity does not imply any particular interface or view.
    """

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    # The RTL module it instantiates; empty for a kernel off the protocol.
    module: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("id", "version"):
            value = getattr(cls, name)
            if type(value) is not str or not value:
                raise DefinitionError(f"{cls.__qualname__} must declare a nonempty string {name}")

    # -- the protocol: what a kernel declares ----------------------------------------------

    port_pins = Members(PINS)
    port_holds = Members(HELD)
    port_accesses = Members(ACCESS)
    admission = ConstraintGroup()

    def parameters(self) -> Mapping[str, int | str] | Rejected:
        """The module's parameters, read from the kernel's choices; refused when it has none."""
        return {}

    def sources(self) -> tuple[RequirementContribution, ...]:
        """The source files that provide ``module``, and any data they read."""
        return ()

    def other_pins(self) -> tuple[Signal | Bus, ...]:
        """Pins that are no port's, such as an AXI-Lite configuration bus."""
        return ()

    def held(self) -> Tieoffs | Rejected:
        """What the kernel itself holds idle, beyond its idle ports and doubled clock."""
        return Tieoffs()

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

    # -- derived plumbing ------------------------------------------------------------------

    @derived
    def codegen(self) -> ModuleBuildRequirements | Rejected:
        """The module: clocking, its other pins, then every port's pins; and parameters."""
        family = type(self)
        if not family.module:
            return reject("kernel-module", f"{family.__qualname__} declares no module")
        clocking = self.clocking
        chosen = self.parameters()
        if isinstance(chosen, Rejected):
            return chosen
        parameters = tuple(sorted(chosen.items()))
        pins = tuple(pin for item in self.port_pins for pin in item.value)
        abi = ModuleABIRequirements(
            FixedModuleName(family.module),
            (*clocking.signals(), *self.other_pins(), *pins),
            tuple((name, str(value)) for name, value in parameters),
            clocking.alignments(),
        )
        return ModuleBuildRequirements(family.id, family.version, parameters, abi, self.sources())

    build_requirements = View(codegen, requires=(admission,))

    @view
    def tieoffs(self) -> Tieoffs | Rejected:
        held = self.held()
        if isinstance(held, Rejected):
            return held
        parts = (Tieoffs(self.clocking.held()), *(item.value for item in self.port_holds), held)
        return Tieoffs(
            tuple(pin for part in parts for pin in part.inputs),
            tuple(pin for part in parts for pin in part.unused),
        )

    # A kernel adding exports of its own extends these: ``{**Kernel.exports, KEY: ...}``.
    exports = {MODULE: build_requirements, TIEOFFS: tieoffs}


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
    "Clocking",
    "HELD",
    "Kernel",
    "MODULE",
    "MODULE_REQUIREMENTS",
    "NATIVE_CLOCKING",
    "PINS",
    "PORT",
    "TIEOFFS",
    "TIEOFFS_SEMANTICS",
    "Tieoffs",
    "extent_of",
    "factor_domain",
]
