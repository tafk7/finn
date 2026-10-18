# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A successor answers as a recomputation: walks of random changes over engine-only Spaces.

``support.space_walk`` applies seeded changes (a value, a clear, a pair, one the configuration
refuses) and compares every successor with the configuration of the same assignments
built from nothing: its assignments, forced values, viable cases and every evaluation
it cached, and the evidence each cached evaluation carries. The Spaces here are where
that matters: verdicts and evaluations that read **forced** Decisions, which move when
a neighbour's choice moves.

A walk fails on any difference. A choice whose own check cannot conclude is kept unchecked
(stored, read unresolved, listed by ``inspection.unchecked``): a successor holds the same
unchecked set as a recomputation, and every unchecked choice's blockers are open or
unsupplied there. A choice a change publishes records its guard's reads, as a recomputation
does, and answers as one.
"""

from __future__ import annotations

import pytest
from support.space_cores import Exclusive, Gated, Mutual, forced
from support.space_pipeline import Meter, Pipeline, Probe
from support.space_walk import check, fresh, walk

from core.space._choices_support import replay
from core.space._toys_support import House
from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    _forcing,
    constraint,
    derived,
    design_space,
    domain,
    inspection,
    occurrence,
    reject,
    supplied,
)
from finn.core.space import _runtime as runtime
from finn.core.space.inspection import candidate

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")


def test_a_chain_whose_forced_selectors_move_with_their_neighbours() -> None:
    check(walk(design_space(Pipeline(width=10)), seed=1, steps=250))


@pytest.mark.parametrize("seed", range(1, 9))
def test_a_chain_that_keeps_choices_unchecked_and_checks_them_again(seed: int) -> None:
    # The walk draws values for choices whose check waits (their selector still open): a
    # change keeps one unchecked, a later one makes it checked or refuses it, and each
    # successor holds the unchecked set, the forced values and the answers a recomputation does.
    report = walk(design_space(Pipeline(width=10)), seed=seed, steps=250)
    check(report, forced=False, unchecked=True)


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
        check(walk(design_space(Host()), seed=seed, steps=150), forced=False)


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


def test_a_gearbox_change_after_a_warm_chain_answers_as_a_recomputation() -> None:
    # The chain from nothing, seed 1, step 128 (reduced to the sequence below). Each
    # successor answers every node, so the next trial starts from a warm cache. The
    # last step commits the gearbox. A trial reads the forced values of the
    # configuration it would publish, so no admission is taken under a forced value that
    # configuration does not give: the change is answered as a recomputation answers it.
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
    assert got.accepted and want.accepted
    assert [(o.owner, o.status, o.unchecked) for o in got.outcomes] == [
        (o.owner, o.status, o.unchecked) for o in want.outcomes
    ]
    assert inspection.unchecked(got.instance) == inspection.unchecked(want.instance)


def test_a_retained_choice_waits_on_a_selector_the_change_opened() -> None:
    """A choice whose own check cannot conclude is kept unchecked: stored, read unresolved,
    and reported with what it waits on, as a new commit of it is.

    The chain from nothing (reduced to the sequence below). The second link's beat is
    committed while the second unit's core is forced (the first unit's pe leaves its width
    odd, which refuses the stub), and clearing that core is accepted. The beat's domain
    reads whether that core is packed. The last change moves the first unit's pe to an even
    width, which un-forces the core: the beat stays, and its own check now waits on the core
    (``Unresolved``, decision unassigned), not refused. It is kept unchecked: reads of
    it answer unresolved and its stored value is visible through ``inspection.unchecked``
    with the core as its blocker, the change's report says so, and a configuration of the
    same assignments built from nothing holds it unchecked too. A new commit of the same
    value is kept unchecked as well, and once the core is committed the choice is checked.
    """
    point = design_space(Pipeline(width=10))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    steps: list[tuple[dict[str, object], tuple[str, ...]]] = [
        ({"u0.compute": "packed"}, ()),
        ({"u0.compute.packed.pe": 5}, ()),
        ({}, ("u1.compute",)),
        ({"l1.beat": 1}, ()),
    ]
    for values, clears in steps:
        changes = [point.field(owned[key]).clear() for key in clears]
        result = point.try_with_choices({owned[k]: v for k, v in values.items()}, *changes)
        assert result.accepted, (values, clears)
        point = result.instance
    assert inspection.unchecked(point) == ()
    last = point.try_with_choices({owned["u0.compute.packed.pe"]: 10})
    assert last.accepted
    # The report: the choice the change named is as it was, and the one it left unchecked is said.
    assert [(o.owner, o.status, o.requested, o.unchecked) for o in last.outcomes] == [
        ("u0.compute.packed.pe", "changed", True, False),
        ("l1.beat", "unchanged", False, True),
    ]
    point = last.instance
    beat = owned["l1.beat"]
    current = occurrence.state(point)
    index = next(i for i, node in enumerate(current.linked.nodes) if node.key == "l1.beat")
    assert current.assignments[index] == 1 and current.unchecked == {index}
    answer = runtime.membership(current, index, 1).result
    assert isinstance(answer, Unresolved), answer
    assert not isinstance(answer, Rejected)
    assert {finding.code for finding in answer.findings} == {"decision-unassigned"}
    assert answer == runtime.membership(occurrence.state(fresh(point)), index, 1).result
    # Reads answer unresolved; the stored value and what it waits on are inspection's.
    assert isinstance(point.query(beat), Unresolved)
    assert isinstance(point.field(beat).state, Unresolved)
    assert inspection.unchecked(point) == (inspection.Unchecked("l1.beat", 1, ("u1.compute",)),)
    assert inspection.unchecked(fresh(point)) == inspection.unchecked(point)
    # A new commit of the same value, with the choice cleared first, is kept unchecked too.
    cleared = point.try_with_choices({}, point.field(beat).clear())
    assert cleared.accepted and inspection.unchecked(cleared.instance) == ()
    again = cleared.instance.try_with_choices({beat: 1})
    assert again.accepted
    assert [(o.owner, o.unchecked, o.result) for o in again.outcomes] == [("l1.beat", True, answer)]
    assert inspection.unchecked(again.instance) == inspection.unchecked(point)
    # Once the core it waits on is committed, its check succeeds and it reads as its value.
    settled = again.instance.try_with_choices({owned["u1.compute"]: "packed"})
    assert settled.accepted and inspection.unchecked(settled.instance) == ()
    assert settled.instance.query(beat) == Available(1)


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


# -- an unchecked choice that becomes checked while its stored candidate stays equal ----------


def within(*, candidate: int, limit: int) -> bool:
    return candidate <= limit


class Wide(Space):
    """A core that takes widths up to 4."""

    width: int = Param()

    @constraint
    def fits(self) -> bool | Rejected:
        return True if self.width <= 4 else reject("too-wide", "the core takes widths up to 4")


class Never(Space):
    """A core that is always refused."""

    width: int = Param()

    @constraint
    def refused(self) -> bool | Rejected:
        return reject("never", "this core is always refused")


class Gauge(Space):
    """``size`` is checked against ``limit``: it is unchecked until ``limit`` is known. The core
    is forced to the wide one while ``size`` reads unresolved (the other is refused, and waiting
    is not a refusal), and refused outright once ``size`` reads 8."""

    limit: int = Decision(values=(3, 9))
    size: int = Decision(domain=domain(accepts=within, limit=limit))
    compute: Wide | Never = Decision({"wide": Wide, "never": Never}, width=size)


def test_a_forced_value_is_not_kept_when_a_stored_choice_it_read_unresolved_becomes_checked() -> (
    None
):
    # The stored candidate of ``size`` is 8 before and after: only its check concludes. A
    # forced value found while it read unresolved must be found again once it reads 8, though
    # the candidate is equal (a verdict observes what a read answers, not the stored value).
    point = design_space(Gauge())
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    assert forced(point) == {"compute": "wide"}
    held = point.try_with_choices({owned["size"]: 8})
    assert held.accepted
    assert [(o.owner, o.unchecked) for o in held.outcomes] == [("size", True)]
    point = held.instance
    assert forced(point) == {"compute": "wide"}  # the unchecked choice is not read as 8
    assert inspection.unchecked(point) == (inspection.Unchecked("size", 8, ("limit",)),)
    verdict = _forcing.forced(occurrence.state(point)).verdicts
    keys = occurrence.state(point).linked.keys
    assert verdict[keys["compute"]].reads[keys["size"]] is _forcing.UNCHECKED
    settled = point.try_with_choices({owned["limit"]: 9})
    assert settled.accepted and inspection.unchecked(settled.instance) == ()
    point = settled.instance
    assert occurrence.state(point).assignments[keys["size"]] == 8
    # Read as 8, the wide core is refused too: no case is viable, as a recomputation says.
    want = fresh(point)
    assert forced(point) == forced(want) == {}
    assert point.query(Gauge.compute) == want.query(Gauge.compute)
    assert inspection.viable(point) == inspection.viable(want)


def test_a_forced_value_is_not_kept_when_a_checked_choice_becomes_unchecked() -> None:
    # The reverse of the case above: ``size`` stays 8 while its check goes from true (the limit
    # 9 is committed) to unknown (the limit is cleared). Read as 8 the wide core is refused, and
    # read as unresolved it is not, so the core's verdict must not be reused on the candidate.
    point = design_space(Gauge())
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    first = point.try_with_choices({owned["limit"]: 9, owned["size"]: 8})
    assert first.accepted and not isinstance(first.instance.query(Gauge.compute), Available)
    assert forced(first.instance) == {}
    cleared = first.instance.try_with_choices({}, first.instance.field(owned["limit"]).clear())
    assert cleared.accepted
    point = cleared.instance
    assert inspection.unchecked(point) == (inspection.Unchecked("size", 8, ("limit",)),)
    want = fresh(point)
    assert forced(point) == forced(want) == {"compute": "wide"}
    assert point.query(Gauge.compute) == want.query(Gauge.compute)
    assert inspection.viable(point) == inspection.viable(want)


class Sized(Space):
    """A core whose ``size`` is checked against its ``limit``."""

    limit: int = Param()
    size: int = Decision(domain=domain(accepts=within, limit=limit))


class Plain(Space):
    """A core with no choice of its own."""

    limit: int = Param()


class Selected(Space):
    """``mode`` selects the core; the cores read the root's ``limit``."""

    limit: int = Decision(values=(3, 9))
    mode: Sized | Plain = Decision({"sized": Sized, "plain": Plain}, limit=limit)


def test_an_unchecked_choice_whose_selector_settles_to_another_case_is_refused() -> None:
    """A choice kept unchecked under an open selector is not silently removed when the
    selector is committed to another case: the change is refused, in one Space, and the
    refusal names the choice (it is inactive: ``Inapplicable``), though the change did not
    request it. Settling the selector to the case that holds the choice keeps it
    unchecked; clearing the choice in the same change as the selector is accepted."""
    point = design_space(Selected())
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    core = candidate(point, Selected.mode, "sized")
    assert core is not None
    held = point.try_with_choices(core.field(Sized.size).change(8))
    assert held.accepted
    point = held.instance
    size = owned["mode.sized.size"]
    # Kept unchecked: it waits on the limit and on the selector, both still open.
    assert inspection.unchecked(point) == (
        inspection.Unchecked("mode.sized.size", 8, ("limit", "mode")),
    )
    refused = point.try_with_choices({owned["mode"]: "plain"})
    assert not refused.accepted and refused.instance is point
    assert [(o.owner, o.status, o.requested) for o in refused.outcomes] == [
        ("mode", "admissible", True),
        ("mode.sized.size", "refused", False),
    ]
    assert isinstance(refused.outcomes[-1].result, Inapplicable)
    assert inspection.unchecked(point) == (
        inspection.Unchecked("mode.sized.size", 8, ("limit", "mode")),
    )
    # The same case as the choice's keeps it, now waiting on the limit alone.
    same = point.try_with_choices({owned["mode"]: "sized"})
    assert same.accepted
    assert inspection.unchecked(same.instance) == (
        inspection.Unchecked("mode.sized.size", 8, ("limit",)),
    )
    # Clearing it in the same change removes it, and the selector settles.
    cleared = point.try_with_choices({owned["mode"]: "plain"}, point.field(size).clear())
    assert cleared.accepted
    assert inspection.unchecked(cleared.instance) == ()
    assert cleared.instance.query(owned["mode"]) == Available("plain")
