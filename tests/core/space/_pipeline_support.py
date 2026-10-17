# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The toy Pipeline: a chain of units whose widths follow their neighbours' choices.

Copied from the bounded-change walks' test (``test_bounded_change.py`` on the lane
``fix/bounded-change``, at its commit ``34446ceee``) for the Space trace prototype
(``finn.core.space._trace``), which reads where each check reads another choice. Nothing
here is about hardware.

- ``Unit``: a selector over two cores (``Packed``, ``Stub``); its output width follows
  the selected core's folding choice.
- ``Link``: between two units; its ``adapter`` (``Direct`` or ``Gearbox``) and its
  ``burst`` read whether the upstream unit's core is packed (``upstream_packed``).
- ``Probe`` and ``Meter``: a Decision that guards a view, and a Decision guarded by
  whether that view is supplied.
"""

from __future__ import annotations

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    divisors_of,
    reject,
    requires,
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

    admission = ConstraintGroup(fits)


class Stub(Space):
    """Refuses an odd width."""

    width: int = Param()
    rows: int = Decision(values=(1, 2, 4))

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub core takes an even width")
        return True

    admission = ConstraintGroup(even)


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

    admission = ConstraintGroup(fed)


class Link(Space):
    """Between two units: its verdicts read the upstream unit's selected core."""

    width: int = Param()
    upstream_packed: bool = Param()
    adapter: Direct | Gearbox = Decision(
        {"direct": Direct, "gearbox": Gearbox}, width=width, upstream_packed=upstream_packed
    )
    burst: int = Decision(
        values=(1, 2, 4),
        requires=(
            requires(upstream_packed, "burst-source: bursts need a packed core", cases=(2, 4)),
        ),
    )

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


#: The chain from nothing, Pipeline seed 6, step 23, reduced (the bounded-change walks'
#: ``test_a_retained_choice_waits_on_a_selector_the_change_opened``): each step's
#: commitments, then the keys it clears.
SEED_6_STEP_23: tuple[tuple[dict[str, object], tuple[str, ...]], ...] = (
    ({"u0.compute": "packed"}, ()),
    ({"u0.compute.packed.pe": 5}, ()),
    ({}, ("u1.compute",)),
    ({"l1.burst": 1}, ()),
    ({"u0.compute.packed.pe": 10}, ()),
)
