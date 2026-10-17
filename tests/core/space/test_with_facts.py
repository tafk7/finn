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
from support.space_walk import check, walk

from core.space._choices_support import committed, replay
from core.space._toys_support import House
from finn.core.space import (
    Available,
    ConfigurationError,
    ConstraintGroup,
    Decision,
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
        check(report, facts=True, known=True)


def test_a_chain_walked_through_other_widths_from_a_configured_start() -> None:
    start = replay(
        design_space(Pipeline(width=10)), {"u0.compute": "packed", "probe.enabled": True}
    )
    check(walk(start, seed=4, steps=120, facts={"width": WIDTHS}), facts=True, known=True)


def test_the_house_walked_through_other_budgets() -> None:
    for seed in (1, 2):
        report = walk(
            design_space(House(budget=400)),
            seed=seed,
            steps=120,
            facts={"budget": (60, 100, 150, 300, 400, 500)},
        )
        check(report, forced=False, facts=True, known=True)


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


def test_a_retained_choice_whose_selector_the_new_facts_open_is_refused_not_published() -> None:
    # At 9 the stub core refuses (an odd width), so the selector is forced to the packed
    # core and `pe` is admitted. At 12 both cores are viable: the selector is open, and a
    # trial that read the base's forced value would publish a choice that waits on it.
    point = design_space(Pipeline(width=9))
    assert isinstance(point.u0.compute, Packed)
    point = replay(point, {"u0.compute.packed.pe": 3})
    report = point.try_with_facts({Pipeline.width: 12})
    assert not report.accepted
    waiting = {o.owner: o.result for o in report.outcomes if o.status == "refused"}
    assert isinstance(waiting["u0.compute.packed.pe"], Unresolved)
    # What a binding at width 12 says of the same choice.
    with pytest.raises(ConfigurationError):
        replay(design_space(Pipeline(width=12)), {"u0.compute.packed.pe": 3})
    # The same choice once the selector is committed with it: admitted at width 12.
    both = point.try_with_facts({Pipeline.width: 12}, {Pipeline.u0.compute: "packed"})
    assert both.accepted and both.instance.width == 12
    assert committed(both.instance)["u0.compute.packed.pe"] == 3


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
    """A core whose admission reads its own choice: it needs the lanes to divide the width."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def divides(self) -> bool | Rejected:
        if self.width % self.pe:
            return reject("lanes-divide", "the lanes do not divide the width")
        return True

    admission = ConstraintGroup(divides)


class Tight(Space):
    """A core whose admission reads its own choice against a quarter of the width."""

    width: int = Param()
    pe: int = Decision(domain=divisors_of(width))

    @constraint
    def quarter(self) -> bool | Rejected:
        if self.pe * 4 > self.width:
            return reject("lanes-quarter", "the core takes at most a quarter of the width in lanes")
        return True

    admission = ConstraintGroup(quarter)


class Holder(Space):
    """A selector with one case, whose admission reads the choice committed under it."""

    width: int = Param()
    compute: Lanes = Decision({"core": Lanes}, width=width)


class Narrow(Space):
    width: int = Param()
    compute: Tight = Decision({"core": Tight}, width=width)


def test_a_choice_failing_its_domain_is_reported_where_that_leaves_its_selector_no_case() -> None:
    # 3 divides 12. At width 7 it fails its domain, and that is why the selector's only case
    # (whose admission needs the lanes to divide the width) is refused: the circular shape.
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
    # 12) while the core's admission refuses it. The choice reading, which reads the selector
    # as the base forces it, accepts the commit; the configuration it leaves has no viable case.
    point = design_space(Narrow(width=12))
    assert isinstance(point.compute, Tight)
    accepted = point.try_with_choices({reference_of(point, "compute.core.pe"): 4})
    assert accepted.accepted
    state = state_of(accepted.instance, "compute.core.pe")
    assert isinstance(state, Rejected)
    assert [item.code for item in state.findings] == ["decision-no-viable-case"]


@pytest.mark.xfail(
    strict=True,
    reason="the published reading refuses a commit the choice reading accepts: the selector's "
    "only case reads the new value, and the trial reads the selector as the successor forces "
    "it (a stop for the switch of `try_with_choices` onto the published reading)",
)
def test_the_published_reading_accepts_a_commit_the_choice_reading_accepts() -> None:
    point = design_space(Narrow(width=12))
    change = {reference_of(point, "compute.core.pe"): 4}
    assert point.try_with_choices(change).accepted
    assert point.try_with_facts({}, change).accepted
