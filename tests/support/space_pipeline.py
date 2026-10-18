# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A chain of units whose widths follow their neighbours' choices (``Pipeline``), and the
cores and links it is built from.

Each ``Unit`` selects between two cores (``Packed``, ``Stub``) and its output width follows
the selected core's choice; a ``Link`` between units reads whether the upstream core is packed,
through a constraint on its ``burst`` (which forcing tries each case against) and through the
domain of its ``beat`` (whose own check waits while the upstream core is open);
a ``Probe`` and a ``Meter`` close the chain with a Decision read through ``present`` and
``supplied``. Its verdicts and evaluations read **forced** Decisions, which move when a
neighbour's choice moves: the Space the bounded change's walks
(``core.space.test_bounded_change``) and the cache's source check
(``core.space.test_cache_writes``) share, imported as ``support.space_pipeline``.

Engine-only: this module imports ``finn.core.space`` and nothing of FINN.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Domain,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    divisors_of,
    domain,
    reject,
    supplied,
    view,
)


class Packed(Space):
    """Refuses more than 64 lanes."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed core takes at most 64 lanes")
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
    compute: Packed | Stub = Decision(CORES, width=width)

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


def beats(packed: object) -> Domain[int]:
    """One beat, or two from a packed core: a domain over whether the upstream core is
    packed, so a beat's own check waits while that core is open."""

    def accepts(*, candidate: int, packed: bool) -> bool:
        return candidate == 1 or (packed and candidate == 2)

    def candidates(*, packed: bool) -> tuple[int, ...]:
        return (1, 2) if packed else (1,)

    return domain(accepts=accepts, candidates=candidates, packed=packed)


class Link(Space):
    """Between two units: its verdicts read the upstream unit's selected core."""

    width: int = Param()
    upstream_packed: bool = Param()
    adapter: Direct | Gearbox = Decision(
        {"direct": Direct, "gearbox": Gearbox}, width=width, upstream_packed=upstream_packed
    )
    burst: int = Decision(values=(1, 2, 4))
    beat: int = Decision(domain=beats(upstream_packed))

    @constraint
    def bursts(self) -> bool | Rejected:
        if self.burst in (2, 4) and not self.upstream_packed:
            return reject("burst-source", "bursts need a packed core")
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
