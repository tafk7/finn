# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle (``support.space_oracle``) on FINN's own toys: seeded walks of
every registered implementation against the public API, the planted faults each reported
by key, and the forms' rendering.

The walks compare the public API with itself and with the public API with no cache: what
they show is that the oracle runs over the toys' changes and compares what it says it
compares, and that the cache changes no answer. The
planted faults show that it would report a difference: each alters one thing, and
``compare`` and ``forms.differences`` (what ``diff`` runs) name it by key.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest
from support.space_oracle import IMPLEMENTATIONS, Answers, compare, forms, toys
from support.space_oracle.public import FAULTS, PUBLIC, Fault, Historied, Planted, PublicAnswers
from support.space_walk import fresh

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    Decision,
    Group,
    Param,
    QueryResult,
    RequestError,
    Select,
    Space,
    Unresolved,
    design_space,
    inspection,
)

from ._running_type import Running
from ._toys_support import House

TOYS = toys.TOYS
"""FINN's toys, opened, with the values a walk may give their root facts (the dump tool's
``toys`` target opens the same)."""

STEPS = 20


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", sorted(TOYS))
@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_implementation_walks_the_toys_as_the_public_api(
    name: str, toy: str, seed: int
) -> None:
    found = compare.walk(
        TOYS[toy].make, TOYS[toy].facts, seed, STEPS, PUBLIC, IMPLEMENTATIONS[name]
    )
    assert found.lines() == []
    # It compared what it says: every design space reached, every change's report.
    assert found.tally["changes"] > 0
    asked = found.tally["changes"] + found.tally["binds"]
    assert found.tally["reports"] + found.tally["requests raising alike"] == asked
    assert found.tally["design spaces"] >= 1 + found.tally["reports"] // 2
    assert found.tally["answers"] > 0 and found.tally["viable values"] > 0


@pytest.mark.parametrize("seed", [0, 1])
def test_the_boundary_walks_open_and_close_its_guard(seed: int) -> None:
    """The walks of the boundary set ``enabled`` both ways: ``outside.width`` is asked with
    the scope it reads absent and present."""
    boundary = TOYS["boundary"]
    found = compare.walk(boundary.make, boundary.facts, seed, STEPS, PUBLIC, PUBLIC)
    assert found.lines() == []
    asked = " / ".join(found.requests)
    assert "enabled=False" in asked and "enabled=True" in asked, found.requests


BOUNDARY: list[tuple[compare.Request, forms.Form]] = [
    (compare.Request({"enabled": False}), {"kind": "inapplicable", "findings": []}),
    (compare.Request({"enabled": True}), {"kind": "value", "value": 3}),
    (
        compare.Request(clears=("enabled",)),
        {
            "kind": "unresolved",
            "findings": [
                {
                    "kind": "blocker",
                    "code": "decision-unassigned",
                    "owner": "enabled",
                    "message": "no choice has set this decision",
                    "details": [],
                    "causes": [],
                }
            ],
        },
    ),
]
"""The public API's answer for ``outside.width`` after each change, in turn: inapplicable
while the scope it reads (``guarded``) is absent, the source's value while it is present, waiting
on ``enabled`` once cleared."""


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_formal_read_from_outside_an_absent_scope_answers_as_the_public_api(name: str) -> None:
    """The boundary of S1 (``toys.Boundary``): ``outside.width`` reads ``guarded.leaf.width``
    from outside ``guarded``. With ``enabled`` false, the public API's answer is
    inapplicable (``core.space.test_collapse``: a formal bound by reference answers within
    its scope), pinned here through the oracle's forms, and every registered implementation
    answers as the public API does."""
    configuration = TOYS["boundary"].make()
    base, other = PUBLIC.open(configuration), IMPLEMENTATIONS[name].open(configuration)
    for request, expected in BOUNDARY:
        found, base_after, other_after = compare.change(base, other, request)
        assert found.lines() == [], request
        assert base_after is not None and other_after is not None, request
        assert forms.answer(base_after.query("outside.width")) == expected, request
        base, other = base_after, other_after


def _house(**stored: object) -> Space:
    point = design_space(House(budget=100))
    if stored:
        point = point.with_choices(
            {getattr(House, key).finish: value for key, value in stored.items()}
        )
    return point


# What each fault moves, by key, at the house with the kitchen's finish at 1: the budget is
# the first member answering a value; the root's validity is open (the dining room and the
# hall wait); the garage's want is the first choice with a possible value; the dining
# room's finish 2 is refused by ``matched`` with the kitchen's finish standing.
EXPECTED: dict[Fault, str] = {
    "value": "members/budget/answer/value",
    "validity": f"validity/{forms.ROOT}/status",
    "possible": "viable/want_garage/values/0/",
    "standing": "viable/dining.finish/values/1/refused/0/standing",
}


@pytest.mark.parametrize("fault", FAULTS)
def test_compare_reports_each_planted_fault_by_key(fault: Fault) -> None:
    point = _house(kitchen=1)
    found = compare.everything(PUBLIC.open(point), Planted(PUBLIC, fault).open(point))
    paths = [difference.path for difference in found.disagreements]
    assert paths and all(path.startswith(EXPECTED[fault]) for path in paths), paths


@pytest.mark.parametrize("fault", FAULTS)
def test_diff_reports_each_planted_fault_by_key(fault: Fault) -> None:
    """``diff`` compares two dumps' forms with ``forms.differences``."""
    point = _house(kitchen=1)
    base = forms.space(PUBLIC.open(point))
    other = forms.space(Planted(PUBLIC, fault).open(point))
    found = forms.differences({"house": {"space": base}}, {"house": {"space": other}})
    assert found and all(
        difference.path.startswith(f"house/space/{EXPECTED[fault]}") for difference in found
    )
    assert forms.differences(base, forms.space(PUBLIC.open(point))) == []


def test_a_planted_standing_choice_is_reported_in_a_change() -> None:
    """A change refused by the kitchen's finish standing in the way: the planted report
    drops it."""
    point = _house(kitchen=1)
    request = compare.Request({"dining.finish": 2})
    found, after, _ = compare.change(
        PUBLIC.open(point), Planted(PUBLIC, "standing").open(point), request
    )
    assert after is None  # refused, on both
    assert [str(difference) for difference in found.disagreements] == [
        'report/broken/0/standing: ["kitchen.finish"] vs []'
    ]


@pytest.mark.parametrize("fault", FAULTS)
def test_a_walk_reports_each_planted_fault(fault: Fault) -> None:
    house = TOYS["house"]
    found = compare.walk(house.make, house.facts, 0, STEPS, PUBLIC, Planted(PUBLIC, fault))
    assert found.disagreements
    kinds = {"value": "members", "validity": "validity", "possible": "viable"}
    expected = {kinds[fault]} if fault in kinds else {"viable", "report"}
    assert set(found.by_kind) <= expected, found.by_kind


# -- the cold build (``Implementation.make``) ------------------------------------------------


def _reached(toy: toys.Toy, seed: int, steps: int = 5) -> list[Answers]:
    """The design spaces a short seeded walk of the public API reaches on ``toy``: the start,
    the dump's change, then a listed value set or a stored choice cleared at each step."""
    rng = random.Random(seed)
    answers = PUBLIC.open(toy.make())
    found = [answers]
    changed = answers.change(toy.change, ()).after
    if changed is not None:
        answers = changed
        found.append(answers)
    for _ in range(steps):
        key = rng.choice([key for key, _ in answers.states()])
        stored = answers.stored()
        if key in stored and rng.random() < 0.3:
            after = answers.change({}, (key,)).after
        else:
            (listed,) = answers.viable([key])
            values = [item.value for item in listed.values or ()]
            if not values:
                continue
            after = answers.change({key: rng.choice(values)}, ()).after
        if after is not None:
            answers = after
            found.append(answers)
    return found


@pytest.mark.parametrize("toy", sorted(TOYS))
def test_the_cold_build_is_the_recomputation_on_the_toys(toy: str) -> None:
    """``PUBLIC.make`` of a design space's Space class, facts and stored choices (answered
    with no cache) answers as ``support.space_walk.fresh`` of it (its recomputation, bound
    from nothing, asked through the public API), at every design space a short walk reaches
    (``compare.walk`` holds the walked one to it)."""
    configuration = TOYS[toy].make()
    space_class, facts = type(configuration), compare.facts_of(configuration)
    reached = [answers for seed in (0, 1) for answers in _reached(TOYS[toy], seed)]
    assert any(answers.stored() for answers in reached)
    for answers in reached:
        assert isinstance(answers, PublicAnswers)
        made = forms.space(PUBLIC.make(space_class, facts, answers.stored()))
        assert forms.differences(forms.space(PUBLIC.open(fresh(answers.point))), made) == []


def test_a_cold_build_holds_exactly_the_facts_and_choices_given() -> None:
    """``make`` checks no rule and drops nothing: the house's ``matched`` broken by the
    kitchen's and the dining room's finishes, both kept; a value its type check refuses,
    or a key that is no choice, is refused."""
    made = PUBLIC.make(House, {"budget": 100}, {"kitchen.finish": 1, "dining.finish": 2})
    assert made.stored() == {"kitchen.finish": 1, "dining.finish": 2}
    assert forms.validity(made.validity())["status"] == "refused"
    with pytest.raises(ValueError, match="kitchen.finish"):
        PUBLIC.make(House, {"budget": 100}, {"kitchen.finish": 9})
    with pytest.raises(KeyError, match="budget"):
        PUBLIC.make(House, {"budget": 100}, {"budget": 1})


@pytest.mark.parametrize("historied", ["base", "other", "both"])
def test_a_walk_reports_history_by_the_cold_build(historied: str) -> None:
    """``Historied``: what a change reaches answers one member otherwise than the same design
    space made from nothing. Against the public API, it is reported on its own side by the
    cold build, beside the members it answers otherwise than the public API does; walked
    against itself, the two sides agree everywhere, and the cold build alone reports it."""
    house = TOYS["house"]
    base = Historied(PUBLIC) if historied in ("base", "both") else PUBLIC
    other = Historied(PUBLIC) if historied in ("other", "both") else PUBLIC
    found = compare.walk(house.make, house.facts, 0, STEPS, base, other)
    sides = {path.split("/cold/")[1].split("/")[0] for path in _paths(found) if "/cold/" in path}
    assert found.by_kind["cold"] > 0
    if historied == "both":
        # one implementation: one side held to its cold build
        assert set(found.by_kind) == {"cold"} and sides == {"base"}
        assert found.tally["cold builds"] == found.tally["design spaces"]
    else:
        assert set(found.by_kind) == {"cold", "members"} and sides == {historied}
        assert found.tally["cold builds"] == 2 * found.tally["design spaces"]


def _paths(found: compare.Compared) -> list[str]:
    return [difference.path for difference in found.disagreements]


class _Reordered:
    """The public API's answers, each answer's findings in reverse order."""

    def __init__(self, inner: Answers) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def query(self, key: str) -> QueryResult[object]:
        found = self.inner.query(key)
        if isinstance(found, Unresolved) and len(found.findings) > 1:
            return _Unordered(tuple(reversed(found.findings)))
        return found


@dataclass(frozen=True)
class _Unordered(Unresolved):
    """An unresolved answer keeping its findings in the order given."""

    def __post_init__(self) -> None:
        pass


def test_findings_in_another_order_are_no_disagreement() -> None:
    point = design_space(House(budget=100))
    other: Any = _Reordered(PUBLIC.open(point))
    found = compare.everything(PUBLIC.open(point), other)
    assert found.disagreements == []
    assert found.tally["order-only"] > 0


# -- the forms ----------------------------------------------------------------------------


class Colour(enum.Enum):
    RED = 1


@dataclass(frozen=True)
class Pair:
    left: object
    right: object


class Grid:
    """Array-like: a dtype, a shape and contents (the engine's tests import no numpy)."""

    dtype = "int8"
    shape = (2, 1)

    def tolist(self) -> list[list[int]]:
        return [[1], [-2]]


def test_values_render_canonically() -> None:
    assert forms.value(frozenset({"b", "a", "c"})) == {"set": ["a", "b", "c"]}
    assert forms.value({"b": 1, "a": 2}) == {"mapping": [["a", 2], ["b", 1]]}
    assert forms.value((1, "x", None, True)) == [1, "x", None, True]
    assert forms.value(-0.0) == {"float": "-0.0"} != forms.value(0.0)
    assert forms.value(Colour.RED) == {"enum": "RED"}
    assert forms.value(Pair(frozenset({2, 1}), 0.5)) == {
        "type": "Pair",
        "fields": {"left": {"set": [1, 2]}, "right": {"float": "0.5"}},
    }
    assert forms.value(Grid()) == {
        "array": {"dtype": "int8", "shape": [2, 1], "contents": [[1], [-2]]}
    }
    assert forms.value(b"ab") == {"repr": "b'ab'"}


def test_a_repr_holding_an_address_is_refused() -> None:
    with pytest.raises(forms.AddressInRepr, match="object"):
        forms.value(object())
    with pytest.raises(forms.AddressInRepr):
        forms.value([1, {"x": object()}])


def test_an_answer_is_its_kind_value_and_sorted_findings() -> None:
    first = Finding(FindingKind.BLOCKER, "b-code", "b", "waits on b")
    second = Finding(FindingKind.BLOCKER, "a-code", "a", "waits on a")
    expected = {
        "kind": "unresolved",
        "findings": [forms.finding(second), forms.finding(first)],
    }
    assert forms.answer(Unresolved((first, second))) == expected
    assert forms.answer(Unresolved((second, first))) == expected
    assert forms.finding(second) == {
        "kind": "blocker",
        "code": "a-code",
        "owner": "a",
        "message": "waits on a",
        "details": [],
        "causes": [],
    }


def test_a_finding_is_rendered_whole() -> None:
    """Its kind, its details and its causes (each a finding, recursively) are compared: a
    blocker apart from a rejection of the same code, findings apart by their subject or
    their causes."""
    cause = Finding(FindingKind.REJECTION, "inner", "x", "x refused", (("value", 3),))
    found = Finding(
        FindingKind.BLOCKER,
        "flow-loop",
        "a",
        "a loop",
        (("choices", ("a.p", "b.q")), ("node", "a")),
        (cause,),
    )
    assert forms.finding(found) == {
        "kind": "blocker",
        "code": "flow-loop",
        "owner": "a",
        "message": "a loop",
        "details": [["choices", ["a.p", "b.q"]], ["node", "a"]],
        "causes": [
            {
                "kind": "rejection",
                "code": "inner",
                "owner": "x",
                "message": "x refused",
                "details": [["value", 3]],
                "causes": [],
            }
        ],
    }
    others = [
        replace(found, kind=FindingKind.REJECTION),
        replace(found, details=(("choices", ("a.p",)), ("node", "a"))),
        replace(found, causes=(replace(cause, details=(("value", 4),)),)),
    ]
    paths = [
        [
            difference.path
            for difference in forms.differences(
                forms.answer(Unresolved((found,))), forms.answer(Unresolved((other,)))
            )
        ]
        for other in others
    ]
    assert paths == [
        ["findings/0/kind"],
        ["findings/0/details/0/1"],
        ["findings/0/causes/0/details/0/1"],
    ]


def test_differences_name_their_keys() -> None:
    base = {"members": {"a.x": {"kind": "derived", "answer": {"kind": "value", "value": 1}}}}
    other = {"members": {"a.x": {"kind": "derived", "answer": {"kind": "value", "value": 2}}}}
    (found,) = forms.differences(base, other)
    assert (found.path, found.base, found.other, found.kind) == (
        "members/a.x/answer/value",
        1,
        2,
        "members",
    )
    (missing,) = forms.differences(base, {"members": {}})
    assert missing.path == "members/a.x" and missing.other == forms.MISSING


# -- every scope reached -------------------------------------------------------------------


class Leaf(Space):
    x: int = Decision(domain=(1, 2))


class Inner(Space):
    on: bool = Decision(domain=(False, True))
    leaf = Leaf(when=on)
    pick: Leaf = Select(a=Leaf(), b=Leaf())


class Outer(Space):
    slot: Inner | Leaf = Select(inner=Inner(), leaf=Leaf())
    held = Inner()


def test_every_scope_is_reached_under_open_selectors_and_guards() -> None:
    """Candidates of an open selector, nested in a candidate, and a child whose presence
    waits on an open choice: each scope's validity is asked through the public API."""
    answers = PUBLIC.open(design_space(Outer()))
    assert answers.scopes() == (
        "",
        "held",
        "held.leaf",
        "held.pick.a",
        "held.pick.b",
        "slot.inner",
        "slot.inner.leaf",
        "slot.inner.pick.a",
        "slot.inner.pick.b",
        "slot.leaf",
    )
    found = forms.space(answers)
    assert all("status" in validity for validity in found["validity"].values()), found


def test_group_members_are_compared_without_independent_validity_scopes() -> None:
    class Included(Group):
        number: int = Param(default=7)

    class Host(Space):
        group = Included()

    point = design_space(Host())
    public = PUBLIC.open(point)
    uncached = Running().open(point)
    assert public.scopes() == uncached.scopes() == ("",)
    assert public.members() == uncached.members() == (("group.number", "const"),)
    assert public.query("group.number") == uncached.query("group.number") == Available(7)
    assert forms.space(public) == forms.space(uncached)
    with pytest.raises(RequestError, match="a Group has no independent validity"):
        inspection.validity(cast(Any, point.group))  # intentionally invalid public request
    with pytest.raises(RequestError, match="a Group has no independent validity"):
        uncached.validity("group")
