# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A change as assignment, on a configuration's state (``finn.core.space._assign``).

Each value set is type-checked against its domain, which reads facts only: outside it
refuses the change; an input of the domain refusing, or not applying, refuses it,
naming the input; a missing fact leaves the choice stored and waiting. A choice under a
case not selected is accepted (dormant); under a selector still open it is checked
where the selector reads its case. A facts change checks every stored value again. No
rule refuses a change here: constraints only read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Param,
    Rejected,
    Space,
    Unresolved,
    constraint,
    derived,
    divisors_of,
    domain,
    reject,
)
from finn.core.space._assign import Assigned, assign
from finn.core.space._state import State, bind, decision_state, prepare
from finn.core.space._world import FLATTEN
from finn.core.space.compiler import compile_model, root_parameters, root_record
from finn.core.space.edits import Change
from finn.core.space.ir import NodeId
from finn.core.space.results import DecisionState, QueryResult


def start(declaration: Space) -> State:
    """The design space of a root declaration, as a state: its facts bound, every choice
    open."""
    record = root_record(declaration)
    model = compile_model(record.space_type)
    facts = root_parameters(model, record)
    return bind(model, prepare(model.linked, {i: (v, "parameter") for i, v in facts.items()}))


def key(current: State, name: str) -> NodeId:
    return current.linked.keys[name]


def change(
    current: State,
    choices: Mapping[str, object] | None = None,
    facts: Mapping[str, object] | None = None,
    clear: tuple[str, ...] = (),
) -> Assigned:
    """``current`` with ``choices`` set and ``clear`` cleared (by key), and ``facts``."""
    nodes = current.linked.nodes
    requests: list[Change[object]] = [
        Change(current, nodes[key(current, name)].scope, key(current, name), value)
        for name, value in (choices or {}).items()
    ]
    requests += [
        Change(current, nodes[key(current, name)].scope, key(current, name), remove=True)
        for name in clear
    ]
    given = {key(current, name): value for name, value in (facts or {}).items()}
    return assign(current, given, requests)


def read(current: State, name: str) -> QueryResult[object]:
    return current.world.result(key(current, name))


def codes(result: object) -> list[tuple[str, str]]:
    assert isinstance(result, (Inapplicable, Rejected, Unresolved))
    return [(finding.code, finding.owner) for finding in result.findings]


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


def test_no_rule_refuses_a_change_here() -> None:
    """Until rules gate a change, one breaking a constraint is accepted and the
    configuration reads refused (``small``: 4 lanes of 16 bits is 64 > 32)."""
    report = change(start(Core(extent=12)), {"lanes": 4, "width": 16})
    assert report.accepted
    point = cast(State, report.state)
    assert isinstance(read(point, "small"), Rejected)


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
    mode: Mode | Plain = Decision({"mode": Mode(extent=extent), "plain": Plain})


def test_a_choice_under_a_case_not_selected_is_checked_there_and_kept_dormant() -> None:
    plain = cast(State, change(start(Selected(extent=12)), {"mode": "plain"}).state)
    report = change(plain, {"mode.mode.lanes": 3})
    assert report.accepted
    (outcome,) = report.outcomes
    # Checked where its selector reads its case: 3 divides 12.
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


def test_a_choice_whose_presence_another_guard_decides_is_checked_when_it_comes_into_play() -> None:
    """``lanes`` exists while ``enabled`` holds, which no selector case decides: set while
    ``enabled`` is open or false, its value is accepted unchecked (the guard's answer is
    the outcome's result), and checked when a change brings it into play."""
    base = start(Gated(extent=12))
    waiting = change(base, {"lanes": 5})
    assert waiting.accepted
    assert codes(waiting.outcomes[0].result) == [("decision-unassigned", "enabled")]
    off = change(base, {"enabled": False, "lanes": 5})
    assert off.accepted and isinstance(off.outcomes[1].result, Inapplicable)
    point = cast(State, off.state)
    refused = change(point, {"enabled": True})
    assert not refused.accepted
    assert [(o.owner, o.status, o.requested) for o in refused.outcomes] == [
        ("enabled", "checked", True),
        ("lanes", "refused", False),
    ]
    assert codes(refused.outcomes[1].result) == [("domain-membership", "lanes")]
    assert change(point, {"enabled": True, "lanes": 4}).accepted


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
    refused = change(point, facts={"extent": 12})
    assert not refused.accepted
    assert [(o.owner, o.status, o.requested) for o in refused.outcomes] == [
        ("extent", "checked", True),
        ("lanes", "refused", False),
    ]
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
    # A facts change that makes it refuse while a value is stored is refused alike, unless
    # the change clears it.
    point = cast(State, change(start(Lanes(w=8)), {"lanes": 4}).state)
    refused = change(point, facts={"w": 7})
    assert not refused.accepted
    assert [(o.owner, o.status, o.requested) for o in refused.outcomes] == [
        ("w", "checked", True),
        ("lanes", "refused", False),
    ]
    assert change(point, facts={"w": 7}, clear=("lanes",)).accepted


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
