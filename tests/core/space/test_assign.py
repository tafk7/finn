# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change as assignment, on a configuration's state (``finn.core.space._assign``).

Each value set is type-checked against its domain, which reads facts only: outside it
refuses the change; an input of the domain refusing, or not applying, refuses it,
naming the input; a missing fact leaves the choice stored and waiting. A choice's
membership is read as if it existed: under a case not selected or a false guard it is
checked and kept dormant, under a selector still open it is checked and waits. A change
of choices that passes its type checks and makes a rule newly false is refused, naming
the rule and the stored choices standing in its way. A facts change is never refused: it
checks every stored value again, dormant ones included, and drops each now outside its
domain.
"""

from __future__ import annotations

import random
from typing import cast

from support.space_cores import chain

from core.space._state_support import change, codes, key, start
from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Select,
    Space,
    constraint,
    derived,
    divisors_of,
    domain,
    reject,
)
from finn.core.space._state import State, decision_state
from finn.core.space._world import BASE_SHARE, FLATTEN, World
from finn.core.space.ir import NodeId
from finn.core.space.results import DecisionState, QueryResult


def read(current: State, name: str) -> QueryResult[object]:
    return current.world.result(key(current, name))


class Core(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))
    width: int = Decision(values=(8, 16))

    @constraint
    def small(self) -> bool:
        return self.lanes * self.width <= 32


def test_a_value_in_its_domain_is_published_and_one_outside_refuses_the_change() -> None:
    base = start(Core(extent=12))
    report = change(base, {"lanes": 4})
    assert report.accepted
    assert [(o.owner, o.status, o.result) for o in report.outcomes] == [
        ("lanes", "changed", Available(True))
    ]
    point = cast(State, report.state)
    assert read(point, "lanes") == Available(4)
    assert point.stored == {key(base, "lanes"): 4}
    assert read(base, "lanes") == read(start(Core(extent=12)), "lanes")  # the receiver as it was

    refused = change(point, {"lanes": 5, "width": 16})
    assert not refused.accepted and refused.state is None
    assert [(o.owner, o.status) for o in refused.outcomes] == [
        ("lanes", "refused"),
        ("width", "checked"),
    ]
    assert codes(refused.outcomes[0].result) == [("domain-membership", "lanes")]


def test_a_change_breaking_a_rule_is_refused_naming_it_after_the_type_checks() -> None:
    """``small``: 4 lanes of 16 bits is 64 > 32, false on the values the change names, so
    nothing stored stands in its way. Each value named passes its type check."""
    report = change(start(Core(extent=12)), {"lanes": 4, "width": 16})
    assert not report.accepted and report.state is None
    assert [(o.owner, o.status, o.result) for o in report.outcomes] == [
        ("lanes", "checked", Available(True)),
        ("width", "checked", Available(True)),
    ]
    (rule,) = report.broken
    assert (rule.rule, rule.standing) == ("small", ())
    assert codes(rule.result) == [("constraint-false", "small")]
    # Set one at a time, the second change is refused with the first standing in its way.
    point = cast(State, change(start(Core(extent=12)), {"lanes": 4}).state)
    (rule,) = change(point, {"width": 16}).broken
    assert (rule.rule, rule.standing) == ("small", ("lanes",))
    # A value outside its domain reports its type check alone.
    alone = change(start(Core(extent=12)), {"lanes": 5, "width": 16})
    assert [o.status for o in alone.outcomes] == ["refused", "checked"] and alone.broken == ()


def test_the_same_values_keep_the_configuration() -> None:
    point = cast(State, change(start(Core(extent=12)), {"lanes": 4}).state)
    again = change(point, {"lanes": 4}, {"extent": 12})
    assert again.state is point
    assert [o.status for o in again.outcomes] == ["unchanged", "unchanged"]
    cleared = change(point, clear=("width",))  # clearing an open choice changes nothing
    assert cleared.state is point


def test_clearing_opens_the_choice_and_never_refuses() -> None:
    point = cast(State, change(start(Core(extent=12)), {"lanes": 4, "width": 8}).state)
    report = change(point, clear=("lanes",))
    assert report.accepted and [o.status for o in report.outcomes] == ["removed"]
    assert codes(read(cast(State, report.state), "lanes")) == [("decision-unassigned", "lanes")]


class Mode(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))


class Plain(Space):
    pass


class Selected(Space):
    extent: int = Param()
    mode: Mode | Plain = Select(mode=Mode(extent=extent), plain=Plain)


def test_a_choice_under_a_case_not_selected_is_checked_there_and_kept_dormant() -> None:
    plain = cast(State, change(start(Selected(extent=12)), {"mode": "plain"}).state)
    report = change(plain, {"mode.mode.lanes": 3})
    assert report.accepted
    (outcome,) = report.outcomes
    # Checked as if it existed: 3 divides 12.
    assert outcome.status == "changed" and outcome.result == Available(True)
    point = cast(State, report.state)
    assert isinstance(read(point, "mode.mode.lanes"), Inapplicable)  # dormant
    assert decision_state(point, key(point, "mode.mode.lanes")) == Available(
        DecisionState("mode.mode.lanes", "committed", 3)
    )
    # A value outside its domain there refuses the change, dormant or not.
    refused = change(plain, {"mode.mode.lanes": 5})
    assert not refused.accepted
    assert codes(refused.outcomes[0].result) == [("domain-membership", "mode.mode.lanes")]
    # Switching to its case brings it into play, as stored.
    back = cast(State, change(point, {"mode": "mode"}).state)
    assert read(back, "mode.mode.lanes") == Available(3)


def test_an_open_selectors_case_adds_a_piece_only_where_its_list_would_be_empty() -> None:
    """Presence unknown, a choice's list is made as if it existed: a list with a value
    adds nothing, so a case with no rule leaves the configuration valid; an empty one is
    an unknown piece, waiting on the selector."""
    base = start(Selected(extent=12))
    root = cast(NodeId, base.linked.scopes[0].validity)
    assert base.world.result(root) == Available(True)
    empty = start(Selected(extent=0))
    assert codes(empty.world.result(root)) == [("decision-unassigned", "mode")]


def test_a_choice_under_an_open_selector_is_checked_at_its_case_and_waits() -> None:
    base = start(Selected(extent=12))
    report = change(base, {"mode.mode.lanes": 4})
    assert report.accepted and report.outcomes[0].result == Available(True)
    point = cast(State, report.state)
    assert codes(read(point, "mode.mode.lanes")) == [("decision-unassigned", "mode")]
    assert not change(base, {"mode.mode.lanes": 5}).accepted


class Gated(Space):
    extent: int = Param()
    enabled: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=enabled)


def test_a_choice_under_a_when_over_a_choice_is_checked_as_if_it_existed() -> None:
    """``lanes`` exists while ``enabled`` holds: its membership is read as if it did,
    whether ``enabled`` is open or false, so a value outside its domain is refused at
    once and one inside it is kept, waiting or dormant, and comes into play unchecked."""
    base = start(Gated(extent=12))
    for choices in ({"lanes": 5}, {"enabled": False, "lanes": 5}):
        refused = change(base, choices)
        assert not refused.accepted
        assert codes(refused.outcomes[-1].result) == [("domain-membership", "lanes")]
    waiting = change(base, {"lanes": 4})
    assert waiting.accepted and waiting.outcomes[0].result == Available(True)
    off = cast(State, change(base, {"enabled": False, "lanes": 4}).state)
    assert isinstance(read(off, "lanes"), Inapplicable)  # dormant
    on = change(off, {"enabled": True})
    assert on.accepted and [o.owner for o in on.outcomes] == ["enabled"]
    assert read(cast(State, on.state), "lanes") == Available(4)


class Windowed(Space):
    channels: int = Param()
    lanes: int = Decision(domain=divisors_of(channels))


class Fenced(Space):
    wide: bool = Param()
    extent: int = Param()
    window = Windowed(channels=extent, when=wide)


def test_a_choice_under_a_when_over_a_fact_is_checked_where_its_inputs_exist_there() -> None:
    """``window`` is absent while ``wide`` is false; its ``lanes`` reads the window's own
    ``channels``, which exists under the window's guard: taken as holding, its value
    decides."""
    narrow = start(Fenced(wide=False, extent=12))
    refused = change(narrow, {"window.lanes": 5})
    assert codes(refused.outcomes[0].result) == [("domain-membership", "window.lanes")]
    kept = change(narrow, {"window.lanes": 3})
    assert kept.accepted and kept.outcomes[0].result == Available(True)
    assert isinstance(read(cast(State, kept.state), "window.lanes"), Inapplicable)


def test_a_facts_change_checks_a_dormant_value_again_as_if_it_existed_and_drops_it() -> None:
    off = cast(State, change(start(Gated(extent=12)), {"enabled": False, "lanes": 4}).state)
    dropped = change(off, facts={"extent": 10})
    assert dropped.accepted and dropped.broken == ()
    assert [(o.owner, o.status, o.requested) for o in dropped.outcomes] == [
        ("extent", "changed", True),
        ("lanes", "dropped", False),
    ]
    assert codes(dropped.outcomes[1].result) == [("domain-membership", "lanes")]
    assert key(off, "lanes") not in cast(State, dropped.state).stored
    assert key(off, "lanes") in off.stored  # the receiver as it was
    assert change(off, {"lanes": 2}, facts={"extent": 10}).accepted
    narrow = cast(State, change(start(Fenced(wide=False, extent=12)), {"window.lanes": 4}).state)
    under = change(narrow, facts={"extent": 10})
    assert [(o.owner, o.status) for o in under.outcomes] == [
        ("extent", "changed"),
        ("window.lanes", "dropped"),
    ]


class Waits(Space):
    extent: int = Param(required=False)
    lanes: int = Decision(domain=divisors_of(extent))


def test_a_missing_fact_leaves_the_choice_waiting_until_the_facts_change_checks_it() -> None:
    base = start(Waits())
    report = change(base, {"lanes": 5})
    assert report.accepted
    assert codes(report.outcomes[0].result) == [("input-missing", "extent")]
    point = cast(State, report.state)
    assert codes(read(point, "lanes")) == [("input-missing", "extent")]
    # The fact arrives and 5 does not divide 12: the waiting value is dropped.
    dropped = change(point, facts={"extent": 12})
    assert dropped.accepted
    assert [(o.owner, o.status, o.requested) for o in dropped.outcomes] == [
        ("extent", "changed", True),
        ("lanes", "dropped", False),
    ]
    assert codes(read(cast(State, dropped.state), "lanes")) == [("decision-unassigned", "lanes")]
    supplied = change(point, facts={"extent": 10})
    assert supplied.accepted and read(cast(State, supplied.state), "lanes") == Available(5)


class Lanes(Space):
    w: int = Param()

    @derived
    def width(self) -> int | Rejected:
        return reject("bad-width", "an odd width") if self.w % 2 else self.w

    lanes: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, width: width % candidate == 0,
            candidates=lambda *, width: tuple(d for d in (1, 2, 4, 8) if width % d == 0),
            width=width,
        )
    )


def test_an_input_of_the_domain_refusing_on_facts_refuses_the_assignment_naming_it() -> None:
    report = change(start(Lanes(w=7)), {"lanes": 1})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert outcome.status == "refused" and codes(outcome.result) == [("bad-width", "width")]
    # A facts change that makes it refuse while a value is stored drops the value: the
    # open choice's list cannot be made, false through ``width``, on the facts alone.
    point = cast(State, change(start(Lanes(w=8)), {"lanes": 4}).state)
    dropped = change(point, facts={"w": 7})
    assert dropped.accepted
    assert [(o.owner, o.status, o.requested) for o in dropped.outcomes] == [
        ("w", "changed", True),
        ("lanes", "dropped", False),
    ]
    assert codes(dropped.outcomes[1].result) == [("bad-width", "width")]
    assert [(rule.rule, rule.standing) for rule in dropped.broken] == [("width", ())]
    # Cleared in the same change: the same design space, the value not reported.
    cleared = change(point, facts={"w": 7}, clear=("lanes",))
    assert cleared.accepted and [o.status for o in cleared.outcomes] == ["changed", "removed"]
    assert [(rule.rule, rule.standing) for rule in cleared.broken] == [("width", ())]


class Banks(Space):
    """``width`` applies only while the fact ``wide`` holds; ``lanes`` lists its
    divisors."""

    wide: bool = Param()
    width: int = Param(default=8)

    @derived(when=wide)
    def span(self) -> int:
        return self.width

    lanes: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, span: span % candidate == 0,
            candidates=lambda *, span: tuple(d for d in (1, 2, 4, 8) if span % d == 0),
            span=span,
        )
    )


def test_a_domain_input_that_does_not_apply_refuses_the_assignment_and_the_open_choice() -> None:
    """The input sits under a guard over facts that is false: it does not exist for these
    facts, and nothing the configuration could set would make the domain."""
    narrow = start(Banks(wide=False))
    report = change(narrow, {"lanes": 2})
    assert not report.accepted
    (outcome,) = report.outcomes
    assert outcome.status == "refused"
    assert isinstance(outcome.result, Rejected)
    assert [(f.code, f.owner, f.details) for f in outcome.result.findings] == [
        ("domain-input-absent", "lanes", (("input", "span"),))
    ]
    # The open choice's domain piece is false through the input.
    validity = narrow.world.result(cast(NodeId, narrow.linked.scopes[0].validity))
    assert codes(validity) == [("domain-input-absent", "lanes")]
    # The facts that make the input apply make the domain.
    assert change(start(Banks(wide=True)), {"lanes": 2}).accepted


def test_an_omitted_optional_fact_leaves_the_domain_unresolved() -> None:
    base = start(Waits())
    validity = base.world.result(cast(NodeId, base.linked.scopes[0].validity))
    assert codes(validity) == [("input-missing", "extent")]


def test_a_long_chain_of_changes_stays_short_and_answers_as_one_built_at_once() -> None:
    point = start(Core(extent=64))
    values = (1, 2, 4, 8, 16, 32, 64)
    for step in range(3 * FLATTEN):
        point = cast(State, change(point, {"lanes": values[step % len(values)]}).state)
        assert point.world.depth <= FLATTEN
    fresh = cast(
        State, change(start(Core(extent=64)), {"lanes": point.stored[key(point, "lanes")]}).state
    )
    for node in point.linked.nodes:
        assert point.world.result(node.index) == fresh.world.result(node.index), node.key


def test_a_walk_of_hundreds_of_changes_shares_its_base_and_answers_as_one_built_at_once() -> None:
    """A world published past ``FLATTEN`` sits on the ancestor holding the most entries,
    which stands alone: a lookup walks two worlds. A flattened world shares its base
    (nothing of it copied) until its own entries reach ``BASE_SHARE`` of the base's, when
    it stands alone itself. Every fifty steps the walk's configuration answers every node
    as the same facts and stored choices built at once."""
    point = start(chain(40)())
    ones = {f"s{i}.size": 1 for i in range(40)} | {f"h{i}.depth": 1 for i in range(39)}
    point = cast(State, change(point, ones).state)
    rng = random.Random(7)
    flattened = shared = 0
    for step in range(301):
        if step % 50 == 0:
            world = point.world
            built = World.start(world.evaluator, dict(world.facts), dict(world.stored))
            for node in point.linked.nodes:
                assert world.result(node.index) == built.result(node.index), node.key
        # A stage's size and the depth of the hop after it, moved together.
        stage = rng.randrange(40)
        size = 3 - cast(int, point.stored[key(point, f"s{stage}.size")])
        moved = {f"s{stage}.size": size} | ({f"h{stage}.depth": size} if stage < 39 else {})
        before = point.world.depth
        report = change(point, moved)
        if report.state is None or report.state is point:
            continue
        point = report.state
        world = point.world
        assert world.depth <= FLATTEN
        if world.depth <= before:  # flattened on publication
            flattened += 1
            assert world.depth <= 1
            if world.parent is not None:
                shared += 1
                assert world.parent.parent is None
                assert len(world.cache) < BASE_SHARE * len(world.parent.cache)
    assert flattened >= 10 and shared >= 5, (flattened, shared)
