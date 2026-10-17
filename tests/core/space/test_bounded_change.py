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
apart: a change may leave a retained choice waiting on a selector it opened. That is the
engine's rule (DESIGN section 7), and a plain test below states it. A choice a change
publishes records its guard's reads, as a recomputation does, and answers as one.
"""

from __future__ import annotations

import pytest
from support.space_cores import Exclusive, Gated, Mutual
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
    _forcing,
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

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")

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


def test_a_walk_over_decisions_that_exclude_each_other() -> None:
    # Two forced Decisions that refuse together: the successor's seeded rounds refuse one
    # and the plain rounds run instead, and every answer agrees with a recomputation.
    for space in (Exclusive, Mutual, Gated):
        for seed in (1, 2):
            report = walk(design_space(space()), seed=seed, steps=100)
            assert not report.differences, "\n".join(report.differences[:3])
            assert report.tallies.accepted and report.tallies.forced_moved, report.tallies


def test_the_house() -> None:
    check(walk(design_space(House(budget=400)), seed=1, steps=250), forced=False)


def test_a_house_whose_budget_refuses() -> None:
    check(walk(design_space(House(budget=60)), seed=2, steps=250), forced=False)


# -- a published choice's evidence --------------------------------------------------------


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
    queried = point.query(pe)
    assert isinstance(queried, Unresolved), queried
    assert queried == fresh(point).query(pe)
    answer = inspection.explain(point, pe).result
    assert isinstance(answer, Unresolved), answer
    assert answer == inspection.explain(fresh(point), pe).result


def test_an_admission_read_under_a_forced_value_the_successor_does_not_give_is_not_reused() -> None:
    # The chain from nothing, seed 1, step 128 (reduced to the sequence below). Each
    # successor answers every node, so the next trial starts from a warm cache. The
    # last step admits the third unit's pe under the base's forced packed core, which
    # the change leaves open (the successor does not force it): the admission assumed
    # that core. The gearbox change must then admit it again, and refuse as a
    # recomputation does.
    point = design_space(Pipeline(width=10))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    steps: list[tuple[dict[str, object], tuple[str, ...]]] = [
        ({"probe.enabled": True}, ()),
        ({"u1.compute": "packed"}, ()),
        ({"u0.compute": "packed"}, ()),
        ({"u0.compute.packed.pe": 10}, ()),
        ({"u1.compute.packed.pe": 5}, ()),
        ({}, ("l1.burst",)),
        ({"u2.compute.packed.pe": 1}, ()),
        ({"u1.compute.packed.pe": 2}, ()),
    ]
    for values, clears in steps:
        changes = [point.field(owned[key]).clear() for key in clears]
        result = point.try_with_choices({owned[k]: v for k, v in values.items()}, *changes)
        assert result.accepted, (values, clears)
        point = result.instance
        current = occurrence.state(point)
        _forcing.forced(current)
        for index in range(len(current.linked.nodes)):
            runtime.evaluate(current, index)
    got = point.try_with_choices({owned["l0.adapter"]: "gearbox"})
    want = fresh(point).try_with_choices({owned["l0.adapter"]: "gearbox"})
    assert not want.accepted
    assert not got.accepted
    refusals = [
        {item.owner: item.result for item in each.outcomes if item.status == "refused"}
        for each in (got, want)
    ]
    assert refusals[0] == refusals[1]


def test_a_retained_choice_waits_on_a_selector_the_change_opened() -> None:
    """DESIGN section 7's rule as it stands: a trial reads an open Decision it does not
    change as its base forces it, so a retained choice that reads it is admitted there.

    The chain from nothing, Pipeline seed 6, step 23 (reduced to the sequence below).
    The second link's burst is committed while the second unit's core is forced (the
    first unit's pe leaves its width odd, which refuses the stub), and clearing that
    core is accepted. The last change moves the first unit's pe to an even width, which
    un-forces the core: the burst stays published, and its admission now waits on the
    core, ``Unresolved`` (decision unassigned), not refused. A configuration of the same
    assignments built from nothing answers the same.

    The inconsistency is that a new commit whose admission waits is refused (the last
    lines), while a retained one is published. The gate for the human in
    ``records/space-composition-2026-10-15/E2-BRIEF.md`` decides between the rules, and
    a decision for one changes this assertion.
    """
    point = design_space(Pipeline(width=10))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    steps: list[tuple[dict[str, object], tuple[str, ...]]] = [
        ({"u0.compute": "packed"}, ()),
        ({"u0.compute.packed.pe": 5}, ()),
        ({}, ("u1.compute",)),
        ({"l1.burst": 1}, ()),
        ({"u0.compute.packed.pe": 10}, ()),
    ]
    for values, clears in steps:
        changes = [point.field(owned[key]).clear() for key in clears]
        result = point.try_with_choices({owned[k]: v for k, v in values.items()}, *changes)
        assert result.accepted, (values, clears)
        point = result.instance
    burst = owned["l1.burst"]
    current = occurrence.state(point)
    index = next(i for i, node in enumerate(current.linked.nodes) if node.key == "l1.burst")
    assert current.assignments[index] == 1
    answer = runtime.membership(current, index, 1).result
    assert isinstance(answer, Unresolved), answer
    assert not isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"decision-unassigned"}
    assert answer == runtime.membership(occurrence.state(fresh(point)), index, 1).result
    # A new commit of the same value, with the choice cleared first, is refused as waiting.
    cleared = point.try_with_choices({}, point.field(burst).clear())
    assert cleared.accepted
    again = cleared.instance.try_with_choices({burst: 1})
    assert not again.accepted
    assert [item.result for item in again.outcomes if item.status == "refused"] == [answer]


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
