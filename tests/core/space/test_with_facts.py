# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A configuration with other facts: the facts an evaluation read, ``with_facts``, and a
trial that reads the configuration it would publish.

An evaluation, an admission and a verdict record the root formals they read (``facts``)
beside the Decisions. ``with_facts`` is the same model with other values for some
formals: it keeps what read none of the ones that changed, admits the retained choices as
a trial admits them, and refuses as ``with_choices`` does. Its trial reads a touched
Decision's forced value from the configuration it would publish, since the base's forced
values are of other facts. Each answer is what a binding built from nothing with the same
facts and choices gives: ``support.space_walk`` walks random changes of formals and
choices and compares every successor with that binding.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.space_pipeline import Packed, Pipeline, Stub, Unit
from support.space_walk import check, fresh, walk

from core.space._choices_support import committed, replay
from core.space._toys_support import House
from finn.core.space import (
    Available,
    ConfigurationError,
    Decision,
    DecisionState,
    Domain,
    Inapplicable,
    Param,
    Rejected,
    RequestError,
    Space,
    Unresolved,
    ValueHandle,
    _forcing,
    _runtime,
    constraint,
    derived,
    design_space,
    divisors_of,
    domain,
    inspection,
    occurrence,
    reject,
    supplied,
)

# Each test runs with the readers index used for every cache, then for none: the same answers.
pytestmark = pytest.mark.usefixtures("index_path")

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
        check(report, forced=False, facts=True)


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
    state = occurrence.state(point)
    assert isinstance(
        _runtime.evaluate(state, state.linked.keys["gauge.scale"]).result, Inapplicable
    )
    moved = point.with_facts({Office.rate: 3}, {Office.mode: 2})
    assert (moved.present(Office.rate), moved.charged, moved.gauge.seen) == (True, 6, 3)
    assert moved.gauge.present(Gauge.reading) is True
    assert isinstance(
        _runtime.evaluate(occurrence.state(moved), state.linked.keys["gauge.scale"]).result,
        Unresolved,
    )
    assert read(moved.with_facts({Office.rate: 5}), Office.charged) == 10
    assert (point.present(Office.rate), point.charged) == (False, 4)  # the base is as it was


def test_a_formal_cannot_be_set_back_to_omitted() -> None:
    point = design_space(Office(width=4, rate=2))
    with pytest.raises(RequestError, match="expected value of nominal type int"):
        point.with_facts({Office.rate: None})
    assert point.present(Office.rate) is True


# -- the facts an evaluation, an admission and a verdict read ------------------------------


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
    snapshot = occurrence.state(point)
    keys = snapshot.linked.keys
    for reference in (Plate.across, Plate.both, Plate.per_lane):
        _runtime.evaluate(snapshot, snapshot.model.resolve(0, reference))
    cache = snapshot.cache
    wide, tall, lanes = keys["wide"], keys["tall"], keys["lanes"]
    assert cache[keys["across"]].facts == {wide}
    assert cache[keys["both"]].facts == {wide, tall}  # through the evaluation it read
    assert cache[keys["per_lane"]].facts == {wide}
    assert cache[keys["per_lane"]].decisions == {lanes}
    assert cache[keys["both"]].decisions == frozenset()


class Pair(Space):
    """Two units whose widths are formals of their own."""

    left: int = Param()
    right: int = Param()
    a = Unit(width=left)
    b = Unit(width=right)


def test_a_verdict_and_an_admission_record_the_formals_they_read() -> None:
    point = replay(design_space(Pair(left=10, right=8)), {"a.compute": "packed"})
    keys = occurrence.state(point).linked.keys
    found = _forcing.forced(occurrence.state(point))
    assert found.verdicts[keys["b.compute"]].facts == {keys["right"]}
    assert found.verdicts[keys["a.compute.packed.pe"]].facts == {keys["left"]}
    # The admission of a retained choice reads the formal its domain reads.
    point = replay(point, {"a.compute.packed.pe": 2})
    assert occurrence.state(point).admission_facts[keys["a.compute.packed.pe"]] == {keys["left"]}


def test_other_facts_keep_what_read_none_of_the_changed_formals() -> None:
    point = replay(design_space(Pair(left=10, right=8)), {"a.compute": "packed"})
    before = occurrence.state(point)
    keys = before.linked.keys
    base_verdicts = dict(_forcing.forced(before).verdicts)
    for formal in ("left", "right"):
        _runtime.evaluate(before, keys[formal])

    moved = point.with_facts({Pair.right: 12})
    after = occurrence.state(moved)
    assert dict(after.parameters) == {keys["left"]: 10, keys["right"]: 12}
    # What read only the left formal is the base's own, and what read the right is found again.
    assert after.cache[keys["left"]] is before.cache[keys["left"]]
    assert keys["right"] not in after.cache
    pe = keys["a.compute.packed.pe"]
    assert after.verdicts[pe] is base_verdicts[pe]
    assert keys["b.compute"] in base_verdicts and keys["b.compute"] not in after.verdicts
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
    assert occurrence.state(point).parameters[occurrence.state(point).linked.keys["left"]] == 10


def test_facts_and_choices_change_together_and_the_formals_must_be_the_roots() -> None:
    point = design_space(Pair(left=10, right=8))
    moved = point.with_facts({Pair.left: 12}, {Pair.a.compute: "stub"})
    assert moved.left == 12 and isinstance(moved.a.compute, Stub)
    with pytest.raises(RequestError, match="a.width is not an exposed parameter"):
        point.with_facts({Pair.a.width: 12})
    twice = ValueHandle[object](
        occurrence.state(point).linked, occurrence.state(point).linked.keys["left"]
    )
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


def test_other_facts_refuse_a_retained_choice_their_domain_excludes_and_name_its_domain() -> None:
    point = replay(design_space(Pipeline(width=10)), {"u0.compute": "packed"})
    point = replay(point, {"u0.compute.packed.pe": 2})
    # 2 divides 10 and 8, not 9 or 7.
    assert committed(point.with_facts({Pipeline.width: 8}))["u0.compute.packed.pe"] == 2
    for width in (9, 7):
        assert refused(point, {Pipeline.width: width}) == {
            "u0.compute.packed.pe": ["domain-membership"]
        }
    with pytest.raises(ConfigurationError):
        point.with_facts({Pipeline.width: 9})
    assert committed(point)["u0.compute.packed.pe"] == 2  # the base is as it was


def test_a_retained_choice_whose_selector_the_new_facts_open_is_kept_unchecked() -> None:
    # At 9 the stub core refuses (an odd width), so the selector is forced to the packed
    # core and `pe` is admitted. At 12 both cores are viable: the selector is open, and the
    # choice's own check cannot conclude. It is kept unchecked with the selector as its
    # blocker, not published as an ordinary choice and not refused.
    point = design_space(Pipeline(width=9))
    assert isinstance(point.u0.compute, Packed)
    point = replay(point, {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Pipeline.width: 12})
    assert report.accepted
    kept = [(o.owner, o.status, o.unchecked) for o in report.outcomes if o.unchecked]
    assert kept == [("u0.compute.packed.pe", "unchanged", True)]
    assert inspection.unchecked(report.instance) == (
        inspection.Unchecked("u0.compute.packed.pe", 3, ("u0.compute",)),
    )
    assert inspection.unchecked(fresh(report.instance)) == inspection.unchecked(report.instance)
    # What a binding at width 12 says of the same choice: it is kept unchecked there too.
    # The same choice once the selector is committed with it: checked at width 12.
    both = point.try_with_facts({Pipeline.width: 12}, {Pipeline.u0.compute: "packed"})
    assert both.accepted and both.instance.width == 12
    assert committed(both.instance)["u0.compute.packed.pe"] == 3
    assert inspection.unchecked(both.instance) == ()


def test_a_refusal_names_the_choice_domain_beside_a_selector_left_no_case() -> None:
    # At 9 only the packed core is viable (the stub refuses an odd width), so `pe = 3` is
    # admitted. At 71 (odd, and past the packed core's 64 lanes) neither core is viable, and
    # 3 does not divide 71: the choice's own check is false as well. A choice's own check is
    # its domain and the requirements naming its value, on the stored value; a selector with
    # no case is the configuration's refusal, reported beside it and never in its place.
    point = replay(design_space(Pipeline(width=9)), {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Pipeline.width: 71})
    assert not report.accepted
    (outcome,) = (o for o in report.outcomes if o.status == "refused")
    assert outcome.owner == "u0.compute.packed.pe"
    assert {(item.code, item.owner) for item in outcome.result.findings} == {  # type: ignore[union-attr]
        ("domain-membership", "u0.compute.packed.pe"),
        ("decision-no-viable-case", "u0.compute"),
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


def test_a_choice_failing_its_domain_is_reported_where_that_leaves_its_selector_no_case() -> None:
    # 3 divides 12. At width 7 it fails its domain, and that is why the selector's only case
    # (whose constraint needs the lanes to divide the width) is refused: the circular shape.
    point = replay(design_space(Holder(width=12)), {"compute.core.pe": 3})
    assert committed(point) == {"compute.core.pe": 3}
    report = point.try_with_facts({Holder.width: 7})
    assert not report.accepted
    (outcome,) = (o for o in report.outcomes if o.status == "refused")
    assert outcome.owner == "compute.core.pe"
    assert {(item.code, item.owner) for item in outcome.result.findings} == {  # type: ignore[union-attr]
        ("domain-membership", "compute.core.pe"),
        ("decision-no-viable-case", "compute"),
    }
    assert committed(point) == {"compute.core.pe": 3}  # the base is as it was


def test_a_commit_the_choice_reading_accepts_leaves_a_configuration_no_case_reads_viable() -> None:
    # At 12 a quarter of the width is 3, so `pe = 4` is the core's own check true (4 divides
    # 12) while the core's constraint refuses it. The choice reading, which reads the selector
    # as the base forces it, accepts the commit; the configuration it leaves has no viable case.
    # The choice is committed, and read refused through its selector's guard.
    point = design_space(Narrow(width=12))
    assert isinstance(point.compute, Tight)
    accepted = point.try_with_choices({reference_of(point, "compute.core.pe"): 4})
    assert accepted.accepted
    state = state_of(accepted.instance, "compute.core.pe")
    assert state == Available(DecisionState("compute.core.pe", "committed", 4))
    read = read_of(accepted.instance, "compute.core.pe")
    assert isinstance(read, Rejected)
    assert [item.code for item in read.findings] == ["decision-no-viable-case"]


def test_the_published_reading_accepts_a_commit_the_choice_reading_accepts() -> None:
    point = design_space(Narrow(width=12))
    change = {reference_of(point, "compute.core.pe"): 4}
    by_choices = point.try_with_choices(change)
    by_facts = point.try_with_facts({}, change)
    assert by_choices.accepted and by_facts.accepted
    # A change checks only the choice's own values: both leave the choice committed, the
    # configuration reading it refused through its selector's guard, and the same as each other.
    states = [state_of(report.instance, "compute.core.pe") for report in (by_choices, by_facts)]
    assert states[0] == states[1] == Available(DecisionState("compute.core.pe", "committed", 4))
    reads = [read_of(report.instance, "compute.core.pe") for report in (by_choices, by_facts)]
    assert reads[0] == reads[1] and isinstance(reads[0], Rejected)


# -- a choice whose own check reads its core's own values (a kernel's folding) --------------------


def pumpings(split: object) -> Domain[bool]:
    """Unpumped, or pumped where the lanes split: a domain over another choice's value."""

    def accepts(*, candidate: bool, split: bool) -> bool | Rejected:
        if candidate and not split:
            return reject("pumped-split", "pumping needs two lanes")
        return True

    def candidates(*, split: bool) -> tuple[bool, ...]:
        return (False, True) if split else (False,)

    return domain(accepts=accepts, candidates=candidates, split=split)


class Folded(Space):
    """A core whose choices' domains read a value of its own, as a kernel's folding reads its
    extents, so every read of them is under the case's guard. Its constraints need the lanes
    to divide the width, and up to 8 of it. ``pumped``'s domain reads ``simd`` (through
    ``split``): its own check reads another choice of the core."""

    width: int = Param()

    @derived
    def lanes(self) -> int:
        return self.width

    pe: int = Decision(domain=divisors_of(lanes))
    simd: int = Decision(domain=divisors_of(lanes))

    @derived
    def split(self) -> bool:
        return self.simd >= 2

    pumped: bool = Decision(domain=pumpings(split))

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.pe:
            return reject("lanes-divide", "the lanes do not divide the width")
        return True

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 8:
            return reject("too-wide", "the core takes widths up to 8")
        return True


class Folder(Space):
    width: int = Param()
    compute: Folded = Decision({"core": Folded}, width=width)


def folder_change(point: Space, **values: object) -> dict[Any, object]:
    return {reference_of(point, f"compute.core.{name}"): value for name, value in values.items()}


def refusals(report: Any) -> dict[str, set[tuple[str, str]]]:
    """The owners a report refuses, each with its findings' codes and owners."""
    return {
        outcome.owner: {(item.code, item.owner) for item in outcome.result.findings}
        for outcome in report.outcomes
        if outcome.status == "refused"
    }


def test_a_choice_whose_domain_reads_its_core_is_refused_by_its_own_finding() -> None:
    # MatMul's shape: the selector is forced to the one core while `pe` is open. `pe = 3`
    # fails its domain (3 does not divide 4), and that is also why the core's constraint
    # refuses, so the configuration the change would publish has no case. The domains read
    # the core's `lanes`, under the case's guard: each choice's own check is read with its
    # guard's case assumed. `pe` is refused with its own finding beside the selector's; `simd`
    # and `pumped`, whose own checks hold, are not refused through the selector. Both readings.
    point = design_space(Folder(width=4))
    assert isinstance(point.compute, Folded)
    change = folder_change(point, pe=3, simd=2, pumped=False)
    for report in (point.try_with_choices(change), point.try_with_facts({}, change)):
        assert not report.accepted
        assert refusals(report) == {
            "compute.core.pe": {
                ("domain-membership", "compute.core.pe"),
                ("decision-no-viable-case", "compute"),
            }
        }
    # A domain reading another choice of the core reads it through that choice's own
    # check, in the configuration with the case assumed: one lane cannot be pumped.
    change = folder_change(point, pe=3, simd=1, pumped=True)
    assert refusals(point.try_with_choices(change)) == {
        "compute.core.pe": {
            ("domain-membership", "compute.core.pe"),
            ("decision-no-viable-case", "compute"),
        },
        "compute.core.pumped": {
            ("pumped-split", "compute.core.pumped"),
            ("decision-no-viable-case", "compute"),
        },
    }


def test_an_own_check_under_a_refused_selector_is_checked_again_on_a_change() -> None:
    # At 12 the core refuses whatever its choices are (past 8), so the selector is refused,
    # and its verdict reads no choice of the core. `pumped`'s own check holds with two lanes:
    # admitted, the configuration reading it refused through its guard. Its check read `simd`
    # with the case assumed, and such a check is never reused, so on the change to one lane
    # (the selector, all the guard read, is stable) it is checked again and refused, as a
    # configuration built from nothing refuses it.
    point = design_space(Folder(width=12))
    two = point.try_with_choices(folder_change(point, simd=2, pumped=True))
    assert two.accepted
    assert state_of(two.instance, "compute.core.pumped") == Available(
        DecisionState("compute.core.pumped", "committed", True)
    )
    assert isinstance(read_of(two.instance, "compute.core.pumped"), Rejected)
    one = two.instance.try_with_choices(folder_change(two.instance, simd=1))
    built = design_space(Folder(width=12))
    cold = built.try_with_choices(folder_change(built, simd=1, pumped=True))
    assert not one.accepted and not cold.accepted
    pumped = {("pumped-split", "compute.core.pumped"), ("decision-no-viable-case", "compute")}
    assert refusals(one) == refusals(cold) == {"compute.core.pumped": pumped}


# -- a choice admitted under a selector that stays refused ----------------------------------------


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
    """A selector with one case, which is refused at width 12 without reading the core's choice."""

    width: int = Param()
    compute: Closed = Decision({"closed": Closed}, width=width)
    other: int = Decision(values=(1, 2))


def test_a_choice_under_a_selector_that_stays_refused_is_read_refused_in_the_successor() -> None:
    # The selector is refused in the base, and its verdict does not read the choice, so the
    # change reaches neither. The choice's own check holds (3 is in its values): it is admitted,
    # and the configuration reads it refused through its guard, as a binding built from nothing.
    point = design_space(Gate(width=12))
    assert isinstance(point.query(Gate.compute), Rejected)
    n = reference_of(point, "compute.closed.n")
    moved = point.try_with_facts({}, {n: 3}, {Gate.other: 2})
    assert moved.accepted
    got, want = occurrence.state(moved.instance), occurrence.state(fresh(moved.instance))
    for index in range(len(got.linked.nodes)):
        _runtime.evaluate(got, index)
    assert got.cache
    wrong = [
        got.linked.nodes[index].key
        for index, entry in got.cache.items()
        if _runtime.evaluate(want, index).result != entry.result
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
    # Both cores are viable at 4 and at 8, so the selector is open and `pe` waits on it. A
    # commit is a check of the value against its domain where the choice exists, so `pe`'s own
    # check is read with the case assumed all the same: 3 divides neither 4 nor 8, which no
    # later commitment mends, and the commit is refused by `pe`'s own finding. 2 divides both,
    # and is kept unchecked, waiting on the selector. Both readings (8 by other facts).
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
        assert inspection.unchecked(two.instance) == (
            inspection.Unchecked("compute.core.pe", 2, ("compute",)),
        )


# -- a forced value read with the case assumed, under a selector that stays refused ----------------


class Forced(Space):
    """A core whose choice's domain reads a Decision of its own forced by a formal (``y``: 1
    where ``x`` is 1, else 2, by its constraints), and whose ``fits`` refuses widths past 8
    reading neither."""

    width: int = Param()
    x: int = Param()

    @derived
    def small(self) -> bool:
        return self.x == 1

    @derived
    def big(self) -> bool:
        return self.x != 1

    y: int = Decision(values=(1, 2))

    @constraint
    def y_small(self) -> bool | Rejected:
        if self.y == 1 and not self.small:
            return reject("y-small", "1 needs x = 1")
        return True

    @constraint
    def y_big(self) -> bool | Rejected:
        if self.y == 2 and not self.big:
            return reject("y-big", "2 needs x > 1")
        return True

    @derived
    def ys(self) -> int:
        return self.y

    pe: int = Decision(domain=divisors_of(ys))

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 8:
            return reject("too-wide", "the core takes widths up to 8")
        return True


class Reach(Space):
    """The core under a selector with one case, given ``x`` as a fact."""

    width: int = Param()
    x: int = Param()
    compute: Forced = Decision({"core": Forced}, width=width, x=x)


class ReachByChoice(Space):
    """The same, given ``x`` as a choice."""

    width: int = Param()
    x: int = Decision(values=(1, 2))
    compute: Forced = Decision({"core": Forced}, width=width, x=x)


def test_an_own_check_read_with_the_case_assumed_is_checked_again_when_a_fact_changes() -> None:
    # At 12 the core refuses whatever its choices are, so the selector is refused, and `pe`'s
    # own check is read with the case assumed, where `x` forces `y`: at x = 2 to 2, which 2
    # divides, so `pe = 2` is admitted. At x = 1 `y` is forced to 1 there, which 2 does not
    # divide. In the configuration `y` is under the refused guard and its reading does not
    # follow `x`, so no reach connects `x` to `pe`'s admission: an own check read with the case
    # assumed is never reused, and is checked again on every change. The change and a
    # configuration built from nothing refuse it alike. At 4 the guard holds, for comparison.
    for width in (4, 12):
        point = design_space(Reach(width=width, x=2))
        step = point.try_with_choices({reference_of(point, "compute.core.pe"): 2})
        assert step.accepted
        bounded = step.instance.try_with_facts({Reach.x: 1})
        built = design_space(Reach(width=width, x=1))
        cold = built.try_with_choices({reference_of(built, "compute.core.pe"): 2})
        assert not bounded.accepted and not cold.accepted
        assert refusals(bounded) == refusals(cold)
        assert ("domain-membership", "compute.core.pe") in refusals(bounded)["compute.core.pe"]


def test_an_own_check_read_with_the_case_assumed_is_checked_again_when_a_choice_changes() -> None:
    # The same with `x` a choice of the root: the change to `x` touches the choice it changes,
    # and `pe`'s admission is checked again as above.
    for width in (4, 12):
        point = design_space(ReachByChoice(width=width))
        change = {reference_of(point, "x"): 2, reference_of(point, "compute.core.pe"): 2}
        step = point.try_with_choices(change)
        assert step.accepted
        bounded = step.instance.try_with_choices({reference_of(step.instance, "x"): 1})
        built = design_space(ReachByChoice(width=width))
        change = {reference_of(built, "x"): 1, reference_of(built, "compute.core.pe"): 2}
        cold = built.try_with_choices(change)
        assert not bounded.accepted and not cold.accepted
        assert refusals(bounded) == refusals(cold)
        assert ("domain-membership", "compute.core.pe") in refusals(bounded)["compute.core.pe"]


def test_a_forced_value_read_with_the_case_assumed_walked_through_other_facts() -> None:
    for seed in (1, 2, 3):
        report = walk(
            design_space(Reach(width=12, x=2)),
            seed=seed,
            steps=60,
            facts={"width": (4, 12), "x": (1, 2)},
        )
        assert not report.differences, "\n".join(report.differences[:5])
        report = walk(design_space(ReachByChoice(width=12)), seed=seed, steps=60)
        assert not report.differences, "\n".join(report.differences[:5])
