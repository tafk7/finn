# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Kernel protocol: one generated module's choices, and the plumbing every kernel shares.

A kernel is a Space whose choices configure one module. On the protocol it
declares only what is its own:

- ``module``: the RTL module it instantiates, and ``sources()``, the source
  files that provide it;
- one ``Port`` node per stream interface (``finn.kernels.port``): each
  admits its element, presents its beat sequence and exports its bus;
- ``parameters()``: the module's parameters, from its choices;
- ``clocking``: its clock and reset pins, when not the plain ``ap_clk`` and
  ``ap_rst_n``;
- ``admission``: the constraint group by which it refuses a configuration it
  cannot build.

The base derives the rest: the module's ABI (clocking, then every port's bus),
``build_requirements`` (accepted under ``admission``), the ``tieoffs`` of
pins the configuration leaves unused, and the exports ``MODULE`` and
``TIEOFFS``. A composite kernel wires its children's modules through streams
instead (``finn.kernels.streams.netlist``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from finn.core.space import (
    ConstraintGroup,
    Members,
    Rejected,
    Space,
    View,
    ViewKey,
    default_semantics,
    derived,
    reject,
    view,
)
from finn.core.space.errors import DefinitionError
from finn.core.space.inspection import NodeInfo, members
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
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.artifacts.requirements import (
    FixedModuleName,
    ModuleABIRequirements,
    ModuleBuildRequirements,
)

MODULE_REQUIREMENTS = default_semantics(ModuleBuildRequirements)
MODULE = ViewKey("module", MODULE_REQUIREMENTS)
"""A kernel's generated module, collected by its composite."""

BUS = ViewKey("bus", default_semantics(Bus))
"""A port's pins, collected by its kernel into the module's ABI."""


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


@dataclass(frozen=True)
class Clocking:
    """A module's clock and reset pins.

    ``clock`` is free-running and ``reset`` active low and synchronous. A
    ``doubled`` pin, when the module has one, is a clock at twice ``clock``
    while ``doubling``, and otherwise an unused input held low.
    """

    clock: str = "ap_clk"
    reset: str = "ap_rst_n"
    doubled: str | None = None
    doubling: bool = False

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
                Reset(active_low=True, synchronous=True, synchronous_to=clocks),
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


CLOCKING = default_semantics(Clocking)


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

    def capabilities(self) -> tuple[NodeInfo, ...]:
        """Inspect authored views in this scope and its children without evaluating."""

        return tuple(member for member in members(self) if member.kind == "view")

    # -- the protocol: what a kernel declares ----------------------------------------------

    buses = Members(BUS)
    admission = ConstraintGroup()

    def parameters(self) -> Mapping[str, int | str]:
        """The module's parameters, read from the kernel's choices."""
        return {}

    def sources(self) -> tuple[CopiedSource, ...]:
        """The source files that provide ``module``."""
        return ()

    @derived(semantics=CLOCKING)
    def clocking(self) -> Clocking:
        return Clocking()

    # -- derived plumbing ------------------------------------------------------------------

    @derived(semantics=MODULE_REQUIREMENTS)
    def codegen(self) -> ModuleBuildRequirements | Rejected:
        """The module: clocking, then every port's bus in declaration order, and parameters."""
        family = type(self)
        if not family.module:
            return reject("kernel-module", f"{family.__qualname__} declares no module")
        clocking = self.clocking
        parameters = tuple(sorted(self.parameters().items()))
        abi = ModuleABIRequirements(
            FixedModuleName(family.module),
            (*clocking.signals(), *(item.value for item in self.buses)),
            tuple((name, str(value)) for name, value in parameters),
            clocking.alignments(),
        )
        return ModuleBuildRequirements(family.id, family.version, parameters, abi, self.sources())

    build_requirements = View(codegen, requires=(admission,))

    @view(semantics=TIEOFFS_SEMANTICS)
    def tieoffs(self) -> Tieoffs:
        return Tieoffs(self.clocking.held())

    exports = {MODULE: build_requirements, TIEOFFS: tieoffs}


__all__ = [
    "BUS",
    "CLOCKING",
    "Clocking",
    "Kernel",
    "MODULE",
    "MODULE_REQUIREMENTS",
    "TIEOFFS",
    "TIEOFFS_SEMANTICS",
    "Tieoffs",
]
