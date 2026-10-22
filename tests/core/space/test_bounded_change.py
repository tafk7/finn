# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A successor answers as a recomputation: walks of random changes over engine-only Spaces.

``support.space_walk`` applies seeded changes (a value, a clear, a pair) and compares every
successor with the configuration of the same facts and stored choices built from nothing:
its stored choices, every node's answer and what each read, and its validity. The Spaces
here are where that matters: a successor's world takes from its base's what the change did
not reach, and choices under selectors still open wait, go dormant and come back into play
as the selectors move.

A walk fails on any difference. A choice an entry reads records its guard's reads, as a
recomputation does, and answers as one.
"""

from __future__ import annotations

import pytest
from support.space_cores import chain
from support.space_pipeline import Meter, Pipeline, Probe
from support.space_walk import check, fresh, walk

from core.space._choices_support import replay
from core.space._toys_support import House
from finn.core.space import (
    Available,
    Decision,
    DecisionState,
    Inapplicable,
    Param,
    Select,
    Space,
    Unresolved,
    derived,
    design_space,
    domain,
    inspection,
    occurrence,
    supplied,
)
from finn.core.space.inspection import candidate


def test_a_chain_walked_from_nothing() -> None:
    check(walk(design_space(Pipeline(width=10)), seed=1, steps=250))


def test_a_chain_walked_from_a_configured_start() -> None:
    start = replay(
        design_space(Pipeline(width=10)), {"u0.compute": "packed", "probe.enabled": True}
    )
    check(walk(start, seed=4, steps=250))


@pytest.mark.parametrize("stages_first", [True, False])
def test_a_chain_whose_changes_reach_far(stages_first: bool) -> None:
    # A change to one stage reaches the hops next to it: the successor keeps what read the
    # rest. Each successor answers as a recomputation, in either order of declaration
    # (stages or hops first).
    for seed in (1, 2):
        report = walk(design_space(chain(8, stages_first=stages_first)()), seed=seed, steps=150)
        assert not report.differences, "\n".join(report.differences[:3])
        tallies = report.tallies
        assert tallies.accepted and tallies.clears and tallies.pairs, tallies


def test_the_house() -> None:
    check(walk(design_space(House(budget=400)), seed=1, steps=250))


def test_a_house_whose_budget_refuses() -> None:
    check(walk(design_space(House(budget=60)), seed=2, steps=250))


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
    """The keys of the choices ``reference``'s entry read, directly or through another."""
    current = occurrence.state(point)
    entry = current.world.entry(current.model.resolve(point._scope, reference))
    nodes = current.linked.nodes
    read = current.evaluator.bits.nodes(entry.reads)
    return {nodes[i].key for i in read if nodes[i].kind == "decision"}


def test_a_walk_over_a_guarded_decision_answers_as_a_recomputation() -> None:
    # Every answer, recorded read and validity agrees. The host has no rule: nothing
    # refuses a step.
    for seed in (1, 2, 3):
        check(walk(design_space(Host()), seed=seed, steps=150), refused=False)


def test_a_published_choice_records_its_guards_decisions() -> None:
    point = design_space(Host()).with_choices({Host.want: True})
    point = point.with_choices({Host.extra.size: 2})
    assert point.extra_size == 2
    assert decisions_read(point, Host.extra_size) >= {"extra.size", "want"}


def test_a_retained_choice_under_a_selector_the_change_opened_answers_as_a_recomputation() -> None:
    report = walk(design_space(Pipeline(width=10)), seed=2, steps=250)
    check(report)


def test_a_gearbox_change_after_a_warm_chain_answers_as_a_recomputation() -> None:
    # Each successor answers every node, so the next change starts from a warm world. The
    # last step commits the gearbox: the change is answered as a recomputation answers it.
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
        for node in current.linked.nodes:
            current.world.entry(node.index)
    got = point.try_with_choices({owned["l0.adapter"]: "gearbox"})
    want = fresh(point).try_with_choices({owned["l0.adapter"]: "gearbox"})
    assert got.accepted and want.accepted
    assert [(o.owner, o.status, o.result) for o in got.outcomes] == [
        (o.owner, o.status, o.result) for o in want.outcomes
    ]
    assert inspection.validity(got.instance) == inspection.validity(want.instance)


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


# -- a choice under a selector: waiting, dormant, back in play ------------------------------


def within(*, candidate: int, limit: int) -> bool:
    return candidate <= limit


class Sized(Space):
    """A core whose ``size`` is checked against its ``limit``."""

    limit: int = Param()
    size: int = Decision(domain=domain(accepts=within, limit=limit))


class Plain(Space):
    """A core with no choice of its own."""

    limit: int = Param()


class Selected(Space):
    """``mode`` selects the core; the cores read the root's ``limit``, a fact."""

    limit: int = Param()
    mode: Sized | Plain = Select(sized=Sized, plain=Plain, limit=limit)


def test_a_choice_whose_selector_settles_to_another_case_goes_dormant_and_comes_back() -> None:
    """A choice stored under an open selector waits on it. Settling the selector on
    another case keeps it, dormant; settling it back on its case brings it into play as
    stored. Clearing it in the same change as the selector removes it."""
    point = design_space(Selected(limit=9))
    owned = {item.key: item.reference for item in inspection.decisions(point)}
    core = candidate(point, Selected.mode, "sized")
    assert core is not None
    held = point.try_with_choices(core.field(Sized.size).change(8))
    assert held.accepted and held.outcomes[0].result == Available(True)
    point = held.instance
    size = owned["mode.sized.size"]
    waiting = point.query(size)
    assert isinstance(waiting, Unresolved)
    assert {(f.code, f.owner) for f in waiting.findings} == {("decision-unassigned", "mode")}
    # Outside its domain at its case, it is refused though its case is not selected.
    assert not point.try_with_choices({size: 10}).accepted
    plain = point.try_with_choices({owned["mode"]: "plain"})
    assert plain.accepted
    assert isinstance(plain.instance.query(size), Inapplicable)
    assert plain.instance.field(size).state == Available(
        DecisionState("mode.sized.size", "committed", 8)
    )
    back = plain.instance.with_choices({owned["mode"]: "sized"})
    assert back.query(size) == Available(8)
    cleared = point.try_with_choices({owned["mode"]: "plain"}, point.field(size).clear())
    assert cleared.accepted
    assert cleared.instance.query(owned["mode"]) == Available("plain")
    assert isinstance(cleared.instance.field(size).state, Inapplicable)  # open and absent
