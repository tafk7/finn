# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A configuration with other facts: the facts an evaluation read, and ``with_facts``.

An entry records the root formals it read (facts) beside the choices, through everything
it read. ``with_facts`` is the same model with other values for some formals: its world
takes what read none of the ones that changed, every stored value is checked again
against its domain over the new facts (dormant ones and those waiting on a selector
included, each where its selectors read its case), and it refuses as ``with_choices``
does. Each answer is what a binding built from nothing with the same facts and choices
gives: ``support.space_walk`` walks random changes of formals and choices and compares
every successor with that binding.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.space_cores import CORES
from support.space_cores import Packed as CorePacked
from support.space_cores import Stub as CoreStub
from support.space_pipeline import Pipeline, Stub, Unit
from support.space_walk import check, fresh, walk

from core.space._choices_support import committed, replay
from core.space._toys_support import House
from finn.core.space import (
    Available,
    ConfigurationError,
    Decision,
    DecisionState,
    Inapplicable,
    Param,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ValueHandle,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    occurrence,
    reject,
    supplied,
)

WIDTHS = (4, 6, 8, 9, 10, 12, 14, 70)


# -- walked against a binding built from nothing ----------------------------------------------


def test_a_chain_walked_through_other_widths() -> None:
    for seed in (1, 2, 3):
        report = walk(
            design_space(Pipeline(width=10)), seed=seed, steps=120, facts={"width": WIDTHS}
        )
        check(report, facts=True)


def test_a_chain_walked_through_other_widths_from_a_configured_start() -> None:
    start = replay(
        design_space(Pipeline(width=10)), {"u0.compute": "packed", "probe.enabled": True}
    )
    check(walk(start, seed=4, steps=120, facts={"width": WIDTHS}), facts=True)


def test_the_house_walked_through_other_budgets() -> None:
    for seed in (1, 2):
        report = walk(
            design_space(House(budget=400)),
            seed=seed,
            steps=120,
            facts={"budget": (60, 100, 150, 300, 400, 500)},
        )
        check(report, facts=True)


# -- an optional formal: its presence is a fact an evaluation reads ---------------------------


class Gauge(Space):
    """Reads an optional formal through ``present`` and through a ``supplied`` guard."""

    reading: int = Param(required=False)
    valued = supplied(reading)
    scale: int = Decision(values=(1, 2), when=valued)

    @derived
    def seen(self) -> int:
        return self.reading if self.present(Gauge.reading) else 0


class Office(Space):
    """A root with an optional formal, bound to a child and read by ``present`` here too."""

    width: int = Param()
    rate: int = Param(required=False)
    gauge = Gauge(reading=rate)
    mode: int = Decision(values=(1, 2, 4))

    @derived
    def charged(self) -> int:
        return self.rate * self.mode if self.present(Office.rate) else self.width


def test_an_optional_formal_walked_through_being_supplied_and_changed() -> None:
    # Every walk starts with `rate` omitted. A fact can supply it and change it; none can
    # omit it again (``with_facts`` keeps the formals it is not given).
    for seed in (1, 2, 3):
        report = walk(
            design_space(Office(width=4)),
            seed=seed,
            steps=100,
            facts={"width": (4, 6, 8), "rate": (1, 2, 3)},
        )
        assert not report.differences, "\n".join(report.differences[:5])
        assert report.tallies.facts_changed > 10


def test_supplying_an_omitted_formal_replaces_what_read_it_omitted() -> None:
    point = design_space(Office(width=4))
    assert (point.present(Office.rate), point.charged, point.gauge.seen) == (False, 4, 0)
    assert point.gauge.present(Gauge.reading) is False
    assert isinstance(point.gauge.query(Gauge.scale), Inapplicable)
    moved = point.with_facts({Office.rate: 3}, {Office.mode: 2})
    assert (moved.present(Office.rate), moved.charged, moved.gauge.seen) == (True, 6, 3)
    assert moved.gauge.present(Gauge.reading) is True
    assert isinstance(moved.gauge.query(Gauge.scale), Unresolved)
    assert read(moved.with_facts({Office.rate: 5}), Office.charged) == 10
    assert (point.present(Office.rate), point.charged) == (False, 4)  # the base is as it was


def test_a_formal_cannot_be_set_back_to_omitted() -> None:
    point = design_space(Office(width=4, rate=2))
    with pytest.raises(RequestError, match="expected value of nominal type int"):
        point.with_facts({Office.rate: None})
    assert point.present(Office.rate) is True


# -- the facts an evaluation reads ---------------------------------------------------------


def test_an_evaluation_records_the_formals_it_read_beside_its_decisions() -> None:
    class Plate(Space):
        wide: int = Param()
        tall: int = Param()
        lanes: int = Decision(values=(1, 2))

        @derived
        def across(self) -> int:
            return self.wide * 2

        @derived
        def both(self) -> int:
            return self.across + self.tall

        @derived
        def per_lane(self) -> int:
            return self.wide * self.lanes

    point = design_space(Plate(wide=3, tall=4)).with_choices(lanes=2)
    current = occurrence.state(point)
    keys = current.linked.keys

    def read(key: str) -> set[str]:
        entry = current.world.entry(keys[key])
        return {current.linked.nodes[i].key for i in current.evaluator.bits.nodes(entry.reads)}

    assert read("across") == {"wide"}
    assert read("both") == {"wide", "tall"}  # through the evaluation it read
    assert read("per_lane") == {"wide", "lanes"}


class Pair(Space):
    """Two units whose widths are formals of their own."""

    left: int = Param()
    right: int = Param()
    a = Unit(width=left)
    b = Unit(width=right)


def test_other_facts_keep_what_read_none_of_the_changed_formals() -> None:
    point = replay(design_space(Pair(left=10, right=8)), {"a.compute": "packed"})
    before = occurrence.state(point)
    keys = before.linked.keys
    held = {formal: before.world.entry(keys[formal]) for formal in ("left", "right")}

    moved = point.with_facts({Pair.right: 12})
    after = occurrence.state(moved)
    assert dict(after.facts) == {keys["left"]: 10, keys["right"]: 12}
    # What read only the left formal is the base's own, and what read the right is found again.
    assert after.world.entry(keys["left"]) is held["left"]
    assert after.world.entry(keys["right"]) is not held["right"]
    assert (moved.a.width, moved.b.width) == (10, 12)


def read(point: Space, reference: Any) -> object:
    answer = point.query(reference)
    assert isinstance(answer, Available)
    return answer.value


def test_other_facts_evaluate_what_read_a_changed_formal_and_only_that() -> None:
    calls: list[str] = []

    class Plate(Space):
        wide: int = Param()
        tall: int = Param()
        lanes: int = Decision(values=(1, 2))

        @derived
        def across(self) -> int:
            calls.append("across")
            return self.wide * 2

        @derived
        def down(self) -> int:
            calls.append("down")
            return self.tall * 2

        @derived
        def total(self) -> int:
            calls.append("total")
            return self.across + self.down

        @derived
        def per_lane(self) -> int:
            calls.append("per_lane")
            return self.down * self.lanes

    point = design_space(Plate(wide=3, tall=4)).with_choices(lanes=2)
    assert [read(point, Plate.across), read(point, Plate.per_lane)] == [6, 16]
    assert read(point, Plate.total) == 14
    assert sorted(calls) == ["across", "down", "per_lane", "total"]
    calls.clear()

    moved = point.with_facts({Plate.wide: 5})
    assert read(moved, Plate.across) == 10
    assert read(moved, Plate.total) == 18
    assert read(moved, Plate.per_lane) == 16
    assert sorted(calls) == ["across", "total"]  # `down` and `per_lane` read only `tall`
    calls.clear()

    # The other formal: the first two are kept, and the base is as it was.
    assert read(point.with_facts({Plate.tall: 1}), Plate.across) == 6
    assert calls == []
    assert read(point, Plate.total) == 14
    assert calls == []


def test_a_value_equal_by_its_semantics_changes_nothing() -> None:
    point = design_space(Pair(left=10, right=8))
    report = point.try_with_facts({Pair.left: 10})
    assert report.accepted and report.instance is point
    assert [(o.owner, o.status) for o in report.outcomes] == [("left", "unchanged")]
    report = point.try_with_facts({Pair.left: 12})
    assert report.accepted and report.instance is not point
    assert [(o.owner, o.status) for o in report.outcomes] == [("left", "changed")]
    assert point.left == 10


def test_facts_and_choices_change_together_and_the_formals_must_be_the_roots() -> None:
    point = design_space(Pair(left=10, right=8))
    moved = point.with_facts({Pair.left: 12}, {Pair.a.compute: "stub"})
    assert moved.left == 12 and isinstance(moved.a.compute, Stub)
    with pytest.raises(RequestError, match="a.width is not an exposed parameter"):
        point.with_facts({Pair.a.width: 12})
    linked = inspection.model(point).linked
    twice = ValueHandle[object](linked, linked.keys["left"])
    with pytest.raises(RequestError, match="duplicate fact for left"):
        point.with_facts({Pair.left: 12, twice: 14})


# -- a refusal is as ``with_choices`` makes it, and names the choice --------------------------


def reference_of(point: Space, key: str) -> Any:
    """The reference of the Decision ``key`` of ``point``."""
    return {item.key: item.reference for item in inspection.decisions(point)}[key]


def state_of(point: Space, key: str) -> object:
    """The state of the Decision ``key`` of ``point``."""
    return point.field(reference_of(point, key)).state


def read_of(point: Space, key: str) -> object:
    """What a read of the Decision ``key`` of ``point`` answers."""
    return point.query(reference_of(point, key))


def refused(point: Space, facts: dict[object, object]) -> dict[str, list[str]]:
    """The owners the change to ``facts`` refuses, with the codes of their findings."""
    report = point.try_with_facts(facts)
    assert not report.accepted
    return {
        outcome.owner: [item.code for item in outcome.result.findings]  # type: ignore[union-attr]
        for outcome in report.outcomes
        if outcome.status == "refused"
    }


class Head(Space):
    """A unit over the support's two cores, given its width."""

    width: int = Param()
    compute: CorePacked | CoreStub = Decision(CORES, width=width)


class Front(Space):
    """One unit whose width is the root's fact: its packed core's ``pe`` lists the
    divisors of a fact, a domain over facts (the chain's units read their neighbours'
    choices, so theirs is a constraint)."""

    width: int = Param()
    u0 = Head(width=width)


def test_other_facts_refuse_a_retained_choice_their_domain_excludes_and_name_its_domain() -> None:
    point = replay(design_space(Front(width=10)), {"u0.compute": "packed"})
    point = replay(point, {"u0.compute.packed.pe": 2})
    # 2 divides 10 and 8, not 9 or 7.
    assert committed(point.with_facts({Front.width: 8}))["u0.compute.packed.pe"] == 2
    for width in (9, 7):
        assert refused(point, {Front.width: width}) == {
            "u0.compute.packed.pe": ["domain-membership"]
        }
    with pytest.raises(ConfigurationError):
        point.with_facts({Front.width: 9})
    assert committed(point)["u0.compute.packed.pe"] == 2  # the base is as it was


def test_a_stored_choice_under_an_open_selector_is_checked_again_by_new_facts() -> None:
    # The selector is open at 9 and at 12: ``pe`` waits on it, checked where its case is
    # selected. 3 divides both widths: the facts change keeps it, waiting.
    point = design_space(Front(width=9))
    assert isinstance(point.query(Front.u0.compute), Unresolved)
    point = replay(point, {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Front.width: 12})
    assert report.accepted
    pe = reference_of(report.instance, "u0.compute.packed.pe")
    waiting = report.instance.query(pe)
    assert isinstance(waiting, Unresolved)
    assert {(f.code, f.owner) for f in waiting.findings} == {("decision-unassigned", "u0.compute")}
    assert waiting == fresh(report.instance).query(pe)
    # The same choice once the selector is set with it: in play at width 12.
    both = point.try_with_facts({Front.width: 12}, {Front.u0.compute: "packed"})
    assert both.accepted and both.instance.width == 12
    assert committed(both.instance)["u0.compute.packed.pe"] == 3
    assert both.instance.query(pe) == Available(3)


def test_a_refusal_names_the_choices_domain_alone() -> None:
    # `pe = 3` is stored at 9, waiting on the open selector. At 71, 3 does not divide 71: the
    # facts change is refused by its type check, which names its domain alone.
    point = replay(design_space(Front(width=9)), {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Front.width: 71})
    assert not report.accepted
    (outcome,) = (o for o in report.outcomes if o.status == "refused")
    assert outcome.owner == "u0.compute.packed.pe"
    assert {(item.code, item.owner) for item in outcome.result.findings} == {  # type: ignore[union-attr]
        ("domain-membership", "u0.compute.packed.pe"),
    }


# -- a selector's case refused by the choice it holds -----------------------------------------


class Lanes(Space):
    """A core whose constraint reads its own choice: it needs the lanes to divide the width."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.pe:
            return reject("lanes-divide", "the lanes do not divide the width")
        return True


class Tight(Space):
    """A core whose constraint reads its own choice against a quarter of the width."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def quarter(self) -> bool | Rejected:
        if self.pe * 4 > self.width:
            return reject("lanes-quarter", "the core takes at most a quarter of the width in lanes")
        return True


class Holder(Space):
    """A selector with one case, whose validity reads the choice committed under it."""

    width: int = Param()
    compute: Lanes = Decision({"core": Lanes}, width=width)


class Narrow(Space):
    width: int = Param()
    compute: Tight = Decision({"core": Tight}, width=width)


def test_a_choice_failing_its_domain_under_new_facts_is_refused_by_its_domain() -> None:
    # 3 divides 12. At width 7 it fails its domain: the facts change is refused, naming it.
    point = replay(design_space(Holder(width=12)), {"compute.core.pe": 3})
    assert committed(point) == {"compute.core.pe": 3}
    report = point.try_with_facts({Holder.width: 7})
    assert not report.accepted
    (outcome,) = (o for o in report.outcomes if o.status == "refused")
    assert outcome.owner == "compute.core.pe"
    assert {(item.code, item.owner) for item in outcome.result.findings} == {  # type: ignore[union-attr]
        ("domain-membership", "compute.core.pe"),
    }
    assert committed(point) == {"compute.core.pe": 3}  # the base is as it was


def test_a_choice_set_under_an_open_selector_waits_on_it() -> None:
    # At 12 a quarter of the width is 3, so `pe = 4` is in its domain (4 divides 12) while
    # the core's constraint refuses it. The selector is open: the choice is stored, and
    # waits on it.
    point = design_space(Narrow(width=12))
    assert isinstance(point.query(Narrow.compute), Unresolved)
    accepted = point.try_with_choices({reference_of(point, "compute.core.pe"): 4})
    assert accepted.accepted
    state = state_of(accepted.instance, "compute.core.pe")
    assert state == Available(DecisionState("compute.core.pe", "committed", 4))
    read = read_of(accepted.instance, "compute.core.pe")
    assert isinstance(read, Unresolved)
    assert {(item.code, item.owner) for item in read.findings} == {
        ("decision-unassigned", "compute")
    }


def test_the_published_reading_accepts_a_commit_the_choice_reading_accepts() -> None:
    point = design_space(Narrow(width=12))
    change = {reference_of(point, "compute.core.pe"): 4}
    by_choices = point.try_with_choices(change)
    by_facts = point.try_with_facts({}, change)
    assert by_choices.accepted and by_facts.accepted
    # A change checks the choice's own values: both leave the choice committed and waiting
    # on its selector, the same as each other.
    states = [state_of(report.instance, "compute.core.pe") for report in (by_choices, by_facts)]
    assert states[0] == states[1] == Available(DecisionState("compute.core.pe", "committed", 4))
    reads = [read_of(report.instance, "compute.core.pe") for report in (by_choices, by_facts)]
    assert reads[0] == reads[1] and isinstance(reads[0], Unresolved)


def refusals(report: Any) -> dict[str, set[tuple[str, str]]]:
    """The owners a report refuses, each with its findings' codes and owners."""
    return {
        outcome.owner: {(item.code, item.owner) for item in outcome.result.findings}
        for outcome in report.outcomes
        if outcome.status == "refused"
    }


# -- a choice under a selector whose case the facts refuse -------------------------------------


class Closed(Space):
    """A core that refuses every width past 8, whatever its choice is."""

    width: int = Param()
    n: int = Decision(values=(1, 2, 3))

    @derived
    def doubled(self) -> int:
        return self.n * 2

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 8:
            return reject("too-wide", "the core takes widths up to 8")
        return True


class Gate(Space):
    """A selector with one case, whose constraint is false at width 12 without reading the
    core's choice."""

    width: int = Param()
    compute: Closed = Decision({"closed": Closed}, width=width)
    other: int = Decision(values=(1, 2))


def test_a_choice_under_a_selector_whose_case_the_facts_refuse_answers_as_from_nothing() -> None:
    # The selector is open: the choice is stored, waiting on it, and every answer of the
    # successor is that of a configuration built from nothing.
    point = design_space(Gate(width=12))
    assert isinstance(point.query(Gate.compute), Unresolved)
    n = reference_of(point, "compute.closed.n")
    moved = point.try_with_facts({}, {n: 3}, {Gate.other: 2})
    assert moved.accepted
    got, want = occurrence.state(moved.instance), occurrence.state(fresh(moved.instance))
    wrong = [
        node.key
        for node in got.linked.nodes
        if got.world.result(node.index) != want.world.result(node.index)
    ]
    assert not wrong


# -- a choice under a selector still open ------------------------------------------------------


class Spread(Space):
    """A core whose choice's domain reads a value of its own, and with no constraint reading
    the choice: a value outside the domain leaves the core viable."""

    width: int = Param()

    @derived
    def lanes(self) -> int:
        return self.width

    pe: int = Decision(domain=divisors_of(lanes))


SPREAD_OR_CLOSED: dict[str, type[Space] | Space] = {"core": Spread, "closed": Closed}


class Either(Space):
    """A selector whose two cases are both viable up to width 8, so it stays open."""

    width: int = Param()
    compute: Spread | Closed = Decision(SPREAD_OR_CLOSED, width=width)


def test_a_choice_under_an_open_selector_is_refused_by_its_own_check_or_waits() -> None:
    # The selector is open, and `pe` waits on it. Its value is checked against its domain
    # where its case is selected: 3 divides neither 4 nor 8, and the change is refused by
    # `pe`'s own finding. 2 divides both: stored, waiting on the selector. Both forms (8 by
    # other facts).
    point = design_space(Either(width=4))
    assert isinstance(point.query(Either.compute), Unresolved)
    pe = reference_of(point, "compute.core.pe")

    def readings(value: int) -> tuple[Any, Any]:
        change = {pe: value}
        return point.try_with_choices(change), point.try_with_facts({Either.width: 8}, change)

    for three in readings(3):
        assert not three.accepted
        assert refusals(three) == {"compute.core.pe": {("domain-membership", "compute.core.pe")}}
    for two in readings(2):
        assert two.accepted
        waiting = two.instance.query(pe)
        assert isinstance(waiting, Unresolved)
        assert {(f.code, f.owner) for f in waiting.findings} == {("decision-unassigned", "compute")}
