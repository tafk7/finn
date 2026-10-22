# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A chain of units whose widths follow their neighbours' choices (``Pipeline``), and the
cores and links it is built from.

Each ``Unit`` selects between two cores (``Packed``, ``Stub``) and its output width follows
the selected core's choice; a packed core's ``pe`` divides the width it is given, a
constraint (its width reads the choices upstream, and a domain reads facts only). A
``Link`` between units reads whether the upstream core is packed, through constraints on its
``burst`` and its ``beat``; a ``Probe`` and a ``Meter`` close the chain with a Decision read
through ``present`` and ``supplied``. Its rules read choices that move when a neighbour's
choice moves: the Space the bounded change's walks (``core.space.test_bounded_change``), the
change of facts' tests (``core.space.test_with_facts``) and the oracle's toys share, imported
as ``support.space_pipeline``.

Engine-only: this module imports ``finn.core.space`` and nothing of FINN.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    reject,
    supplied,
    view,
)


class Packed(Space):
    """Refuses more than 64 lanes; its ``pe`` divides its width."""

    width: int = Param()
    pe: int = Decision(values=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
        return True

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.pe:
            return reject("pe-divides", "the PE divides the width")
        return True


class Stub(Space):
    """Refuses an odd width."""

    width: int = Param()
    rows: int = Decision(values=(1, 2, 4))

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True


CORES: dict[str, type[Space] | Space] = {"packed": Packed, "stub": Stub}


class Unit(Space):
    """A selector over two cores; its output width follows the selected core's choice."""

    width: int = Param()
    compute: Packed | Stub = Select(**CORES, width=width)

    @derived
    def lanes(self) -> int:
        core = self.compute
        return core.pe if isinstance(core, Packed) else core.rows

    @derived
    def out(self) -> int:
        return self.width + self.lanes

    @derived
    def packed(self) -> bool:
        return isinstance(self.compute, Packed)


class Direct(Space):
    width: int = Param()
    upstream_packed: bool = Param()


class Gearbox(Space):
    """Takes only what a packed core feeds."""

    width: int = Param()
    upstream_packed: bool = Param()

    @constraint
    def fed(self) -> bool | Rejected:
        if not self.upstream_packed:
            return reject("gearbox-source", "the gearbox takes a packed core's stream")
        return True


class Link(Space):
    """Between two units: its verdicts read the upstream unit's selected core."""

    width: int = Param()
    upstream_packed: bool = Param()
    adapter: Direct | Gearbox = Select(
        direct=Direct, gearbox=Gearbox, width=width, upstream_packed=upstream_packed
    )
    burst: int = Decision(values=(1, 2, 4))
    beat: int = Decision(values=(1, 2))

    @constraint
    def bursts(self) -> bool | Rejected:
        if self.burst in (2, 4) and not self.upstream_packed:
            return reject("burst-source", "bursts need a packed core")
        return True

    @constraint
    def beats(self) -> bool | Rejected:
        """One beat, or two from a packed core."""
        if self.beat == 2 and not self.upstream_packed:
            return reject("beat-source", "two beats need a packed core")
        return True

    @derived
    def out(self) -> int:
        return self.width


class Probe(Space):
    enabled: bool = Decision(values=(False, True))

    @view(when=enabled)
    def level(self) -> int:
        return 3


class Meter(Space):
    """Reads through ``present`` and ``supplied``, guarded by the probe's Decision."""

    reading: int = Param(required=False)
    valued = supplied(reading)
    scale: int = Decision(values=(1, 2), when=valued)

    @derived
    def seen(self) -> int:
        return self.reading if self.present(Meter.reading) else 0


class Pipeline(Space):
    width: int = Param()
    u0 = Unit(width=width)
    l0 = Link(width=u0.out, upstream_packed=u0.packed)
    u1 = Unit(width=l0.out)
    l1 = Link(width=u1.out, upstream_packed=u1.packed)
    u2 = Unit(width=l1.out)
    probe = Probe()
    meter = Meter(reading=probe.level)
