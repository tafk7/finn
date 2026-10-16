# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A successor answers as a recomputation: walks of random changes over engine-only Spaces.

``support.space_walk`` applies seeded changes (a value, a clear, a pair, one the configuration
refuses) and compares every successor with the configuration of the same assignments
built from nothing: its assignments, forced values, viable cases and every evaluation
it cached, and the evidence each cached evaluation carries. The Spaces here are where
that matters: verdicts and evaluations that read **forced** Decisions, which move when
a neighbour's choice moves.

A walk fails on any difference but one, which today's engine has and the walks tally
apart: a change may leave a retained choice waiting on a selector it opened. A strict
xfail states it, so it flips when the engine stops having it. A choice a change
publishes records its guard's reads, as a recomputation does, and answers as one.
"""

from __future__ import annotations

import pytest
from support.space_walk import check, fresh, walk

from core.space._choices_support import replay
from core.space._toys_support import House
from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    occurrence,
    reject,
    requires,
    supplied,
    view,
)
from finn.core.space import _runtime as runtime

# -- a chain of units whose widths follow their neighbours' choices -------------------------


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


def test_a_chain_whose_forced_selectors_move_with_their_neighbours() -> None:
    check(walk(design_space(Pipeline(width=10)), seed=1, steps=250))


def test_a_chain_walked_from_a_configured_start() -> None:
    start = replay(
        design_space(Pipeline(width=10)), {"u0.compute": "packed", "probe.enabled": True}
    )
    check(walk(start, seed=4, steps=250))


def test_the_house() -> None:
    check(walk(design_space(House(budget=400)), seed=1, steps=250), forced=False)


def test_a_house_whose_budget_refuses() -> None:
    check(walk(design_space(House(budget=60)), seed=2, steps=250), forced=False)


# -- a published choice's evidence --------------------------------------------------------


# The way today's engine answers differently from a recomputation, which the walks tally
# apart (``Report.known``). The strict xfail below states it, and passes once the engine
# does not have it: remove its mark then.
WAITING = "an assignment waits on a choice the change opened"


class Option(Space):
    size: int = Decision(values=(1, 2, 3))


class Host(Space):
    """A Decision under a guard that reads a Decision, read from outside the guard."""

    want: bool = Decision(values=(False, True))
    extra = Option(when=want)

    @derived
    def extra_size(self) -> int:
        return self.extra.size

    @derived
    def doubled(self) -> int:
        return 2 * self.extra_size


def decisions_read(point: Space, reference: object) -> set[str]:
    """The keys of the Decisions ``reference``'s evaluation records."""
    current = occurrence.state(point)
    entry = runtime.evaluate(current, current.model.resolve(point._scope, reference))
    return {current.linked.nodes[i].key for i in entry.decisions}


def test_a_walk_over_a_guarded_decision_answers_as_a_recomputation() -> None:
    # Every answer, forced value, verdict and recorded read agrees.
    for seed in (1, 2, 3):
        check(walk(design_space(Host()), seed=seed, steps=150), forced=False, known=True)


def test_a_published_choice_records_its_guards_decisions() -> None:
    point = design_space(Host()).with_choices({Host.want: True})
    point = point.with_choices({Host.extra.size: 2})
    assert point.extra_size == 2
    assert decisions_read(point, Host.extra_size) >= {"extra.size", "want"}


def test_a_retained_choice_under_a_selector_the_change_opened_answers_as_a_recomputation() -> None:
    # The chain from nothing, seed 2, step 195 (reduced to the sequence below).
    report = walk(design_space(Pipeline(width=10)), seed=2, steps=250)
    check(report)


def test_a_retained_choice_whose_selector_the_neighbour_stops_forcing_is_unresolved() -> None:
    # The second unit's pe is committed under its committed core. Clearing the core
    # leaves it forced (the first unit's odd width refuses the stub), so the pe is still
    # answered; a neighbour's change that makes the stub viable again un-forces the core,
    # and the pe is unresolved, as a configuration of the same assignments built from
    # nothing answers it. The pe's entry, published under the forced core, recorded no
    # read of it, so the change did not reach it.
    point = design_space(Pipeline(width=10))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    steps: list[tuple[dict[str, object], tuple[str, ...]]] = [
        ({"u0.compute": "packed"}, ()),
        ({"u1.compute": "packed"}, ()),
        ({"u0.compute.packed.pe": 10}, ()),
        ({"u1.compute.packed.pe": 1}, ()),
        ({"u0.compute.packed.pe": 5}, ()),
        ({}, ("u1.compute",)),
        ({"u0.compute.packed.pe": 10}, ()),
    ]
    for values, clears in steps:
        changes = [point.field(owned[key]).clear() for key in clears]
        result = point.try_with_choices({owned[k]: v for k, v in values.items()}, *changes)
        assert result.accepted, (values, clears)
        point = result.instance
    pe = owned["u1.compute.packed.pe"]
    answer = inspection.explain(point, pe).result
    assert isinstance(answer, Unresolved), answer
    assert answer == inspection.explain(fresh(point), pe).result


@pytest.mark.xfail(strict=True, reason=WAITING)
def test_a_retained_choice_is_admitted_on_the_successor_of_a_change_that_opens_a_selector() -> None:
    # The trial reads its base's forced value of a selector the change un-forces, so a
    # retained choice that reads it is admitted there, and waits on the successor.
    report = walk(design_space(Pipeline(width=10)), seed=6, steps=100)
    check(report)
    assert WAITING not in report.known, report.known[WAITING]


# -- presence reads the Decisions of the guards it reads ------------------------------------


def test_presence_records_the_guards_decisions() -> None:
    point = design_space(Pipeline(width=10)).with_choices({Pipeline.probe.enabled: True})
    assert point.meter.seen == 3
    assert "probe.enabled" in decisions_read(point.meter, Meter.seen)
    off = point.with_choices({Pipeline.probe.enabled: False})
    assert off.meter.seen == 0
    assert "probe.enabled" in decisions_read(off.meter, Meter.seen)


def test_supplied_records_the_guards_decisions() -> None:
    class Sees(Space):
        reading: int = Param(required=False)
        valued = supplied(reading)

        @derived
        def has_reading(self) -> bool:
            return self.valued

    class Plant(Space):
        probe = Probe()
        sees = Sees(reading=probe.level)

    on = design_space(Plant()).with_choices({Plant.probe.enabled: True})
    assert on.sees.has_reading is True
    assert "probe.enabled" in decisions_read(on.sees, Sees.has_reading)
    off = on.with_choices({Plant.probe.enabled: False})
    assert off.sees.has_reading is False
    assert "probe.enabled" in decisions_read(off.sees, Sees.has_reading)
