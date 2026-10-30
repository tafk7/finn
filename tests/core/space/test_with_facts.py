# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A configuration with other facts: the facts an evaluation read, ``with_facts``, and the
bind.

An entry records the root formals it read (facts) beside the choices, through everything
it read. New facts instantiate a new design space, keeping the choices that still fall
within their domains: ``with_facts`` takes what read none of the formals that changed,
checks every stored value again against its domain under the new facts (dormant ones and
those waiting on a selector included, each where its selectors read its case), drops each
now outside it, and is never refused for that or for a constraint the new facts make
false: the design space reads refused, and the report names the rule with the stored
choices standing in its way. Choices given beside the facts are a change of choices on
the new design space, refused as ``with_choices`` refuses, judged against it. ``bind`` is
the same instantiation of saved choices on a configuration's facts. Each answer is what a
binding built from nothing with the same facts and choices gives: ``support.space_walk``
walks random changes of formals and choices and compares every successor with that
binding.
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
    Select,
    Space,
    Unresolved,
    ValueHandle,
    bind,
    constraint,
    derived,
    design_space,
    divisors_of,
    inspection,
    occurrence,
    reject,
)
from finn.core.space._running.evaluate import Ans, Session, evaluation
from finn.core.space._scopes import Address
from finn.core.space.edits import ConfigurationResult
from finn.core.space.inspection import Absent, Dormant, Open

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


# -- an optional formal: None where it is left out, a fact an evaluation reads ---------------


class Gauge(Space):
    """Reads an optional formal, and a guard over it."""

    reading: int | None = Param(default=None)

    @derived
    def valued(self) -> bool:
        return self.reading is not None

    scale: int = Decision(domain=(1, 2), when=valued)

    @derived
    def seen(self) -> int:
        return 0 if self.reading is None else self.reading


class Office(Space):
    """A root with an optional formal, bound to a child and read here too."""

    width: int = Param()
    rate: int | None = Param(default=None)
    gauge = Gauge(reading=rate)
    mode: int = Decision(domain=(1, 2, 4))

    @derived
    def charged(self) -> int:
        return self.width if self.rate is None else self.rate * self.mode


def test_an_optional_formal_walked_through_being_given_changed_and_left_out() -> None:
    # Every walk starts with `rate` left out (None). A fact can give it, change it and set
    # it back to None, the fact left out.
    for seed in (1, 2, 3):
        report = walk(
            design_space(Office(width=4)),
            seed=seed,
            steps=100,
            facts={"width": (4, 6, 8), "rate": (None, 1, 2, 3)},
        )
        assert not report.differences, "\n".join(report.differences[:5])
        assert report.tallies.facts_changed > 10


def test_giving_an_optional_formal_replaces_what_read_it_none() -> None:
    point = design_space(Office(width=4))
    assert (point.rate, point.charged, point.gauge.seen) == (None, 4, 0)
    assert isinstance(point.gauge.query(Gauge.scale), Inapplicable)
    moved = point.with_facts({Office.rate: 3}, {Office.mode: 2})
    assert (moved.rate, moved.charged, moved.gauge.seen) == (3, 6, 3)
    assert isinstance(moved.gauge.query(Gauge.scale), Unresolved)
    assert read(moved.with_facts({Office.rate: 5}), Office.charged) == 10
    assert (point.rate, point.charged) == (None, 4)  # the base is as it was


def test_an_optional_formal_set_back_to_none_is_the_one_left_out() -> None:
    point = design_space(Office(width=4, rate=2))
    back = point.with_facts({Office.rate: None})
    assert (back.rate, back.charged) == (None, 4)
    left_out = design_space(Office(width=4))
    assert occurrence.held(back).ds == occurrence.held(left_out).ds
    assert occurrence.held(back).ds.layer is occurrence.held(left_out).ds.layer


# -- the facts an evaluation reads ---------------------------------------------------------


def test_an_evaluation_records_the_formals_it_read_beside_its_decisions() -> None:
    class Plate(Space):
        wide: int = Param()
        tall: int = Param()
        lanes: int = Decision(domain=(1, 2))

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
    ds = occurrence.held(point).ds

    def read(key: str) -> set[str]:
        return {ds.key(i) for i in answer(point, ds.node(key)).reads}

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
    keys = inspection.model(point).scopes.keys
    held = {formal: answer(point, keys[formal]) for formal in ("left", "right")}

    moved = point.with_facts({Pair.right: 12})
    after = occurrence.held(moved).ds
    assert dict(after.facts) == {keys["left"]: 10, keys["right"]: 12}
    # What read only the left formal is the base's own (a hit in the model's cache), and
    # what read the right is found again.
    assert answer(moved, keys["left"]) is held["left"]
    assert answer(moved, keys["right"]) is not held["right"]
    assert (moved.a.width, moved.b.width) == (10, 12)


def read(point: Space, reference: Any) -> object:
    answer = point.query(reference)
    assert isinstance(answer, Available)
    return answer.value


@pytest.mark.usefixtures("purity_unchecked")
def test_other_facts_evaluate_what_read_a_changed_formal_and_only_that() -> None:
    """Other facts evaluate again what read a changed formal, and only that. The test
    counts work, so it runs with the purity check off: the check runs every cache hit's
    function again."""
    calls: list[str] = []

    class Plate(Space):
        wide: int = Param()
        tall: int = Param()
        lanes: int = Decision(domain=(1, 2))

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
    model = inspection.model(point)
    twice = ValueHandle[object](model, model.scopes.find("left"))
    with pytest.raises(RequestError, match="duplicate fact for left"):
        point.with_facts({Pair.left: 12, twice: 14})


# -- a value outside its new domain is dropped, and names its domain ---------------------------


def reference_of(point: Space, key: str) -> Any:
    """The reference of the Decision ``key`` of ``point``."""
    return {item.key: item.reference for item in inspection.decisions(point)}[key]


def state_of(point: Space, key: str) -> object:
    """The state of the Decision ``key`` of ``point``."""
    return point.field(reference_of(point, key)).state


def read_of(point: Space, key: str) -> object:
    """What a read of the Decision ``key`` of ``point`` answers."""
    return point.query(reference_of(point, key))


def dropped(report: ConfigurationResult[Any]) -> dict[str, set[tuple[str, str]]]:
    """The choices a change dropped, each with its check's findings' codes and owners."""
    return {
        outcome.owner: {(item.code, item.owner) for item in outcome.result.findings}  # type: ignore[union-attr]
        for outcome in report.outcomes
        if outcome.status == "dropped"
    }


def states(point: Space) -> dict[str, object]:
    """Every choice's state, by key."""
    return dict(inspection.states(point))


def broken(report: ConfigurationResult[Any]) -> list[tuple[str, tuple[str, ...]]]:
    """A report's broken rules, each with the choices standing in its way."""
    return [(rule.rule, rule.standing) for rule in report.broken]


class Head(Space):
    """A unit over the support's two cores, given its width."""

    width: int = Param()
    compute: CorePacked | CoreStub = Select(**CORES, width=width)


class Front(Space):
    """One unit whose width is the root's fact: its packed core's ``pe`` lists the
    divisors of a fact, a domain over facts (the chain's units read their neighbours'
    choices, so theirs is a constraint)."""

    width: int = Param()
    u0 = Head(width=width)


def test_a_walk_over_a_domain_of_facts_drops_what_falls_outside() -> None:
    for seed in (1, 2):
        report = walk(design_space(Front(width=10)), seed=seed, steps=80, facts={"width": WIDTHS})
        check(report, refused=False, facts=True)
        assert report.tallies.dropped, report.tallies


def test_other_facts_drop_a_stored_choice_their_domain_excludes_and_name_its_domain() -> None:
    point = replay(design_space(Front(width=10)), {"u0.compute": "packed"})
    point = replay(point, {"u0.compute.packed.pe": 2})
    # 2 divides 10 and 8, not 9 or 7: dropped there, and the decision is open.
    assert committed(point.with_facts({Front.width: 8}))["u0.compute.packed.pe"] == 2
    for width in (9, 7):
        report = point.try_with_facts({Front.width: width})
        assert report.accepted and report.broken == ()
        assert dropped(report) == {
            "u0.compute.packed.pe": {("domain-membership", "u0.compute.packed.pe")}
        }
        (outcome,) = (o for o in report.outcomes if o.status == "dropped")
        assert outcome.requested is False
        assert states(report.instance)["u0.compute.packed.pe"] == Open()
        assert "u0.compute.packed.pe" not in committed(point.with_facts({Front.width: width}))
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


def test_a_value_waiting_on_an_open_selector_is_dropped_naming_its_domain_alone() -> None:
    # `pe = 3` is stored at 9, waiting on the open selector. At 71, 3 does not divide 71:
    # the facts change drops it, naming its domain alone, and the choice is open again,
    # waiting on the selector.
    point = replay(design_space(Front(width=9)), {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Front.width: 71})
    assert report.accepted
    assert dropped(report) == {
        "u0.compute.packed.pe": {("domain-membership", "u0.compute.packed.pe")}
    }
    assert states(report.instance)["u0.compute.packed.pe"] == Open(("u0.compute",))


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
    compute: Lanes = Select(core=Lanes, width=width)


class Narrow(Space):
    width: int = Param()
    compute: Tight = Select(core=Tight, width=width)


def test_a_choice_failing_its_domain_under_new_facts_is_dropped_by_its_domain() -> None:
    # 3 divides 12. At width 7 it fails its domain: the facts change drops it, naming it.
    point = replay(design_space(Holder(width=12)), {"compute.core.pe": 3})
    assert committed(point) == {"compute.core.pe": 3}
    report = point.try_with_facts({Holder.width: 7})
    assert report.accepted
    assert dropped(report) == {"compute.core.pe": {("domain-membership", "compute.core.pe")}}
    assert committed(report.instance) == {}
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
    n: int = Decision(domain=(1, 2, 3))

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
    compute: Closed = Select(closed=Closed, width=width)
    other: int = Decision(domain=(1, 2))


def test_a_choice_under_a_selector_whose_case_the_facts_refuse_answers_as_from_nothing() -> None:
    # The selector is open: the choice is stored, waiting on it, and every answer of the
    # successor is that of a configuration built from nothing.
    point = design_space(Gate(width=12))
    assert isinstance(point.query(Gate.compute), Unresolved)
    n = reference_of(point, "compute.closed.n")
    moved = point.try_with_facts({}, {n: 3}, {Gate.other: 2})
    assert moved.accepted
    assert answers(moved.instance) == answers(fresh(moved.instance), cached=False)


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
    compute: Spread | Closed = Select(**SPREAD_OR_CLOSED, width=width)


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


# -- the clauses of a change of facts ----------------------------------------------------------


class Rated(Space):
    """A rule reading a choice against a fact: new facts can make it false while the choice
    stays inside its domain. ``lanes`` lists the divisors of another fact."""

    cap: int = Param()
    extent: int = Param()
    c: int = Decision(domain=(1, 2, 3, 4))
    lanes: int = Decision(domain=divisors_of(extent))
    other: int = Decision(domain=(1, 2))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.c > self.cap:
            return reject("over-cap", "more than the cap")
        return True


class Gated(Space):
    extent: int = Param()
    enabled: bool = Decision(domain=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=enabled)


def answer(point: Space, index: Address) -> Ans:
    """``index``'s answer in the design space ``point`` holds, in its model's session."""
    ds = occurrence.held(point).ds
    return occurrence.ask(ds.model, lambda session: evaluation(ds, session).get(index))


def answers(point: Space, *, cached: bool = True) -> list[tuple[str, object]]:
    """Every node's answer in the design space ``point`` holds, by key: in its model's
    session, or (``cached`` false) in a session of its own with no cache."""
    ds = occurrence.held(point).ds

    def every(session: Session) -> list[tuple[str, object]]:
        ev = evaluation(ds, session)
        return [(i.key, ev.result(i)) for i in ds.model.scopes.all_addresses()]

    if cached:
        return occurrence.ask(ds.model, every)
    with ds.model.lock:
        return every(Session())


def codes(result: object) -> set[str]:
    assert isinstance(result, (Rejected, Unresolved)), result
    return {finding.code for finding in result.findings}


def same_design_space(left: Space, right: Space) -> None:
    """Two design spaces of one model, equal (the same facts and stored choices), with the
    same answers."""
    got, want = occurrence.held(left).ds, occurrence.held(right).ds
    assert got == want
    assert dict(got.facts) == dict(want.facts)
    assert dict(got.stored) == dict(want.stored)
    assert answers(left) == answers(right)
    assert inspection.validity(left) == inspection.validity(right)


def test_a_dormant_value_outside_its_new_domain_is_dropped_and_reported() -> None:
    point = design_space(Gated(extent=12)).with_choices(enabled=False, lanes=4)
    assert states(point)["lanes"] == Dormant(4)
    report = point.try_with_facts({Gated.extent: 10})
    assert report.accepted and report.broken == ()
    assert [(o.owner, o.status, o.requested) for o in report.outcomes] == [
        ("extent", "changed", True),
        ("lanes", "dropped", False),
    ]
    assert dropped(report) == {"lanes": {("domain-membership", "lanes")}}
    assert states(report.instance)["lanes"] == Absent()
    # Back in play, it is open: nothing is stored for it.
    assert states(report.instance.with_choices(enabled=True))["lanes"] == Open()


def test_a_batch_whose_choice_breaks_a_rule_with_the_new_facts_is_refused() -> None:
    point = design_space(Rated(cap=4, extent=12))
    report = point.try_with_facts({Rated.cap: 2}, {Rated.c: 3})
    assert not report.accepted and report.instance is point
    assert [(o.owner, o.status) for o in report.outcomes] == [("cap", "checked"), ("c", "checked")]
    assert broken(report) == [("fits", ())]  # the choice the batch names does not stand
    with pytest.raises(ConfigurationError):
        point.with_facts({Rated.cap: 2}, {Rated.c: 3})
    # A value the batch names outside its domain under the new facts refuses it too.
    outside = point.try_with_facts({Rated.extent: 10}, {Rated.lanes: 4})
    assert not outside.accepted
    assert [(o.owner, o.status) for o in outside.outcomes] == [
        ("extent", "checked"),
        ("lanes", "refused"),
    ]


def test_a_batch_whose_facts_break_a_rule_its_choices_do_not_touch_is_accepted() -> None:
    point = design_space(Rated(cap=4, extent=12)).with_choices(c=3)
    report = point.try_with_facts({Rated.cap: 2}, {Rated.other: 2})
    assert report.accepted and report.instance.other == 2
    assert broken(report) == [("fits", ("c",))]
    assert codes(report.broken[0].result) == {"over-cap"}
    assert inspection.validity(report.instance).status == "refused"
    # The receiver's pairs are the baseline: a rule already false is not reported again.
    again = report.instance.try_with_facts({Rated.cap: 1})
    assert again.accepted and again.broken == ()
    # A choice set beside the facts so that the rule holds: nothing false is published,
    # nothing reported.
    fixed = point.try_with_facts({Rated.cap: 2}, {Rated.c: 1})
    assert fixed.accepted and fixed.broken == ()
    assert inspection.validity(fixed.instance).status == "valid"


def test_a_batch_is_its_facts_then_its_choices() -> None:
    """The new facts make ``fits`` false with ``c = 3``; the batch sets ``c = 2``, which
    leaves it false through the same origin: accepted, as the facts and then the choices
    are, with the same design space and answers. ``lanes = 4`` falls outside at 10 and
    the batch sets it: left out by the facts, not reported dropped, then set."""
    point = design_space(Rated(cap=4, extent=12)).with_choices(c=3, lanes=4)
    facts = {Rated.cap: 1, Rated.extent: 10}
    batch = point.try_with_facts(facts, {Rated.c: 2, Rated.lanes: 5})
    assert batch.accepted
    assert [(o.owner, o.status) for o in batch.outcomes] == [
        ("cap", "changed"),
        ("extent", "changed"),
        ("c", "changed"),
        ("lanes", "changed"),
    ]
    # The standing choices are read in the published design space, named ones included.
    assert broken(batch) == [("fits", ("c",))]
    sequence = point.with_facts(facts).with_choices(c=2, lanes=5)
    same_design_space(batch.instance, sequence)
    # Alone, the facts drop ``lanes`` and report it.
    assert dropped(point.try_with_facts(facts)) == {"lanes": {("domain-membership", "lanes")}}


def test_bind_keeps_saved_choices_that_break_a_rule_and_drops_one_outside_its_domain() -> None:
    point = design_space(Rated(cap=2, extent=10))
    report = bind(point, {Rated.c: 3, Rated.lanes: 4, Rated.other: 2})
    assert report.accepted
    assert [(o.owner, o.status, o.requested) for o in report.outcomes] == [
        ("c", "changed", True),
        ("lanes", "dropped", True),
        ("other", "changed", True),
    ]
    assert codes(report.outcomes[1].result) == {"domain-membership"}
    assert broken(report) == [("fits", ("c",))]
    assert inspection.validity(report.instance).status == "refused"
    assert committed(report.instance) == {"c": 3, "other": 2}
    # The same saved choices again: the receiver itself, the report read from nothing
    # stored (each kept choice ``"changed"``, the rule broken).
    held = report.instance
    again = bind(held, {Rated.c: 3, Rated.other: 2})
    assert again.instance is held
    assert [o.status for o in again.outcomes] == ["changed", "changed"]
    assert broken(again) == [("fits", ("c",))]
    # Exactly the saved choices: what the receiver held and they do not name is gone.
    fewer = bind(held, {Rated.other: 1})
    assert committed(fewer.instance) == {"other": 1} and fewer.broken == ()
    with pytest.raises(RequestError, match="clears none"):
        bind(held, held.field(Rated.c).clear())
    with pytest.raises(RequestError, match="not an owned Decision"):
        bind(held, {Rated.cap: 1})


def test_a_facts_change_is_the_bind_of_its_stored_choices_under_the_new_facts() -> None:
    point = design_space(Rated(cap=4, extent=12)).with_choices(c=3, lanes=4, other=2)
    by_facts = point.try_with_facts({Rated.cap: 2, Rated.extent: 10})
    nothing = design_space(Rated(cap=2, extent=10))
    saved = {reference_of(nothing, key): value for key, value in committed(point).items()}
    by_bind = bind(nothing, saved)
    assert by_facts.accepted and by_bind.accepted
    same_design_space(by_facts.instance, by_bind.instance)
    assert set(dropped(by_facts)) == set(dropped(by_bind)) == {"lanes"}
    assert broken(by_facts) == broken(by_bind) == [("fits", ("c",))]


def test_a_bind_moves_from_its_facts_with_nothing_stored_whatever_the_receiver_stores() -> None:
    # ``lanes`` is saved under ``enabled = False``: put dormant, from nothing stored (where
    # ``enabled`` is open) and from a receiver where ``enabled`` is already false alike.
    saved = {Gated.enabled: False, Gated.lanes: 4}
    nothing = bind(design_space(Gated(extent=12)), saved)
    held = bind(design_space(Gated(extent=12)).with_choices(enabled=False), saved)
    assert nothing.dormant == held.dormant == ("lanes",)
    assert nothing.in_play == held.in_play == () and nothing.opened == held.opened == ()
    same_design_space(nothing.instance, held.instance)


def whole_report(report: ConfigurationResult[Any]) -> tuple[object, ...]:
    """Everything a bind reports: each outcome (owner, status, requested, result), the rules
    broken and what moved."""
    return (
        report.accepted,
        tuple((o.owner, o.status, o.requested, o.result) for o in report.outcomes),
        report.broken,
        report.dormant,
        report.in_play,
        report.opened,
    )


@pytest.mark.parametrize(
    ("nothing", "saved", "other", "expected"),
    [
        pytest.param(
            Rated(cap=2, extent=10),
            {Rated.c: 3, Rated.lanes: 4, Rated.other: 2},
            {"c": 1, "lanes": 5, "other": 1},
            [("c", "changed"), ("lanes", "dropped"), ("other", "changed")],
            id="drops-and-breaks",
        ),
        pytest.param(
            Gated(extent=12),
            {Gated.enabled: False, Gated.lanes: 4},
            {"enabled": True, "lanes": 6},
            [("enabled", "changed"), ("lanes", "changed")],
            id="moves",
        ),
    ],
)
def test_a_bind_reports_the_same_from_any_design_space_with_its_facts(
    nothing: Space, saved: dict[Any, object], other: dict[str, object], expected: list[Any]
) -> None:
    """The same saved choices bound from a design space with nothing stored, from one
    holding them and from one holding other values: one report, read from nothing stored.
    Every saved choice kept is ``"changed"`` and every one dropped ``"dropped"``, whatever
    the receiver holds; the receiver holding them is the design space published."""
    empty = design_space(nothing)
    from_nothing = bind(empty, saved)
    holding = from_nothing.instance
    from_holding = bind(holding, saved)
    from_other = bind(empty.with_choices(**other), saved)
    assert from_holding.instance is holding
    assert [(o.owner, o.status) for o in from_nothing.outcomes] == expected
    assert all(o.requested for o in from_nothing.outcomes)
    assert whole_report(from_holding) == whole_report(from_nothing)
    assert whole_report(from_other) == whole_report(from_nothing)
    same_design_space(from_holding.instance, from_nothing.instance)
    same_design_space(from_other.instance, from_nothing.instance)
