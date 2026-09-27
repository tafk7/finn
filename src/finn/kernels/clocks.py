# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Clock domains as ordinary Spaces that kernels reference.

A ``ClockDomain`` is a node declared in the composite beside the kernels it
clocks. A kernel has one reference input per domain it runs in
(``clock: ClockDomain = Param()``) and exports, under ``CLOCKING``, which of
its pins that domain drives: ``exports = {CLOCKING: {clock: clock_pins}}``.
The domain sees its kernels through ``Users(CLOCKING)`` and exports a
``Domain`` under ``DOMAIN``; ``netlist`` drives every child clock and reset pin
from the domains, never from pin names.

A domain some kernel runs in becomes top-level pins of the composite, named by
its ``clock`` and ``reset`` inputs: ABI names are design data of the domain, as
a stream's ``port`` is. A domain no kernel runs in is absent from the top. A
``DerivedClock`` runs at twice its ``base`` domain's rate, phase aligned, and
has no reset of its own: its kernels take the base domain's reset, which is
then synchronous to both clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.core.space import (
    Located,
    Param,
    Rejected,
    Space,
    Users,
    ViewKey,
    default_semantics,
    reject,
    view,
)
from finn.kernels.artifacts.abi import Clock, ClockAlignment, Derived, Direction, Free, Signal


@dataclass(frozen=True)
class Clocking:
    """A kernel's pins one domain drives: its clock, and its reset if the domain has one.

    ``Clocking()`` drives nothing: the kernel does not run in that domain in
    this configuration, and the domain does not count it.
    """

    clock: str | None = None
    reset: str | None = None

    @property
    def used(self) -> bool:
        return self.clock is not None or self.reset is not None


CLOCKING_SEMANTICS = default_semantics(Clocking)
CLOCKING = ViewKey("clocking", CLOCKING_SEMANTICS)


@dataclass(frozen=True)
class Domain:
    """One clock domain of a composite: its top pins and the kernel pins they drive.

    ``clock`` is None when no kernel runs in the domain. ``reset`` names the top
    reset pin; a derived domain has none and names its ``base`` clock instead.
    ``driven`` lists each kernel by node name with the pins it named.
    """

    clock: Signal | None
    reset: str | None
    base: str | None
    alignment: ClockAlignment | None
    driven: tuple[tuple[str, Clocking], ...]


DOMAIN = ViewKey("domain", default_semantics(Domain))


def _driven(clocked: tuple[Located[Clocking], ...]) -> tuple[tuple[str, Clocking], ...]:
    return tuple((str(user.node), user.value) for user in clocked if user.value.used)


class ClockDomain(Space):
    """A free-running clock and its synchronous active-low reset, by top pin name."""

    clock: str = Param()
    reset: str = Param()
    clocked = Users(CLOCKING)

    @view(semantics=default_semantics(Domain), requires=(clocked,))
    def domain(self) -> Domain:
        driven = _driven(self.clocked)
        if not driven:
            return Domain(None, None, None, None, ())
        return Domain(
            Signal(self.clock, Direction.IN, 1, Clock(Free())), self.reset, None, None, driven
        )

    exports = {DOMAIN: domain}


class DerivedClock(Space):
    """A clock at twice ``base``'s rate with coincident rising edges; no reset of its own."""

    clock: str = Param()
    base: ClockDomain = Param()
    clocked = Users(CLOCKING)

    @view(semantics=default_semantics(Domain), requires=(clocked,))
    def domain(self) -> Domain | Rejected:
        driven = _driven(self.clocked)
        if not driven:
            return Domain(None, None, None, None, ())
        resets = [f"{node}.{pins.reset}" for node, pins in driven if pins.reset is not None]
        if resets:
            return reject(
                "clock-reset", f"{', '.join(resets)}: a derived clock has no reset of its own"
            )
        base = self.base.clock
        return Domain(
            Signal(self.clock, Direction.IN, 1, Clock(Derived(base, 2))),
            None,
            base,
            ClockAlignment(base, self.clock),
            driven,
        )

    exports = {DOMAIN: domain}


__all__ = [
    "CLOCKING",
    "CLOCKING_SEMANTICS",
    "ClockDomain",
    "Clocking",
    "DOMAIN",
    "DerivedClock",
    "Domain",
]
