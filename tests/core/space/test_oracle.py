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
from typing import Any

import pytest
from support.space_oracle import IMPLEMENTATIONS, Answers, compare, forms, toys
from support.space_oracle.public import FAULTS, PUBLIC, Fault, Historied, Planted, PublicAnswers
from support.space_walk import fresh

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Decision,
    QueryResult,
    Select,
    Space,
    Unresolved,
    design_space,
)

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
                    "message": "decision requires a commitment",
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


# -- the missing-fact shapes (``toys.WaitingMembership``, ``AcceptedWaiting``, ``ThreeDeep``) -----
#
# The reference model reads an optional fact left out as ``None``, so these shapes left it
# (T4c's row). The public API reads it unresolved until T4c: they are pinned here, through
# the oracle's forms, for every registered implementation. Each opens with its optional
# fact left out and gives it in a facts change.

MISSING_FACTS = ("waiting-membership", "accepted-waiting", "three-deep")


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", MISSING_FACTS)
def test_the_missing_fact_toys_walk_from_the_fact_left_out_to_the_fact_given(
    toy: str, seed: int
) -> None:
    (fact,) = TOYS[toy].facts
    # left out at the start: the public API reads it unresolved
    assert forms.answer(PUBLIC.open(TOYS[toy].make()).query(fact)) == _missing(fact)
    found = compare.walk(TOYS[toy].make, TOYS[toy].facts, seed, STEPS, PUBLIC, PUBLIC)
    assert found.lines() == []
    assert found.tally["steps giving facts"], found.requests
    assert any(request.startswith(f"{fact}=") for request in found.requests), found.requests


def _unassigned(owner: str) -> forms.Form:
    return {
        "kind": "unresolved",
        "findings": [
            {
                "kind": "blocker",
                "code": "decision-unassigned",
                "owner": owner,
                "message": "decision requires a commitment",
                "details": [],
                "causes": [],
            }
        ],
    }


def _missing(fact: str) -> forms.Form:
    return {
        "kind": "unresolved",
        "findings": [
            {
                "kind": "limitation",
                "code": "input-missing",
                "owner": fact,
                "message": "optional input was omitted at start",
                "details": [],
                "causes": [],
            }
        ],
    }


def _asked(answers: Answers, keys: tuple[str, ...]) -> dict[str, forms.Form]:
    """The root's validity (its status and the owners of its pieces), and each of ``keys``'
    answer, and state where it is a choice, through the oracle's forms."""
    validity = forms.validity(answers.validity())
    found: dict[str, forms.Form] = {
        "validity": (validity["status"], [owner for owner, _ in validity["pieces"]])
    }
    states = dict(answers.states())
    for key in keys:
        found[f"{key} answer"] = forms.answer(answers.query(key))
        if key in states:
            found[f"{key} state"] = forms.state(states[key])
    return found


def _reported(report: forms.Form) -> dict[str, forms.Form]:
    """A change's report: accepted, each outcome's owner, status and result's kind, and
    each broken rule with its standing choices."""
    return {
        "accepted": report["accepted"],
        "outcomes": [(o["owner"], o["status"], o["result"]["kind"]) for o in report["outcomes"]],
        "broken": [(b["rule"], b["standing"]) for b in report["broken"]],
    }


def _accepted(*outcomes: tuple[str, str, str], broken: tuple[Any, ...] = ()) -> forms.Form:
    return {"accepted": True, "outcomes": list(outcomes), "broken": list(broken)}


OPEN: forms.Form = {"state": "open", "waits": []}
LEAF = "outer.middle.leaf"

#: per path: the toy, the keys asked, the public API's answers at the start, and each change
#: with its report and the public API's answers after it
MissingFactPath = tuple[
    str,
    tuple[str, ...],
    dict[str, forms.Form],
    list[tuple[compare.Request, forms.Form, dict[str, forms.Form]]],
]

#: membership waiting on a fact: ``size`` lists nothing and is no piece of validity (only
#: ``fits`` is); set, it is accepted and waits on ``limit``, and ``fits`` reading it waits
#: on ``limit`` too; ``limit`` given checks it: dropped at 2, set at 4
_SIZE_SET = {
    "validity": ("open", ["fits"]),
    "size answer": _missing("limit"),
    "size state": {"state": "waiting", "value": 3, "waits": ["limit"]},
    "fits answer": _missing("limit"),
}
_SIZE_START = {
    "validity": ("open", ["fits"]),
    "size answer": _unassigned("size"),
    "size state": OPEN,
    "fits answer": _unassigned("size"),
}
#: a choice accepted while its membership waits: ``pe``'s list waits on ``width`` (an
#: unknown piece); ``pe = 3`` is accepted waiting; ``width`` given drops it (8) or keeps it (6)
_PE_START = {
    "validity": ("open", ["pe"]),
    "pe answer": _unassigned("pe"),
    "pe state": OPEN,
}
_PE_SET = {
    "validity": ("open", ["pe"]),
    "pe answer": _missing("width"),
    "pe state": {"state": "waiting", "value": 3, "waits": ["width"]},
}
#: a rule three deep waiting on a fact: ``fits`` waits on ``pe``, then, ``pe`` set, on
#: ``width``, and the root's validity with it; ``width`` given: true (8), or broken with
#: ``pe`` standing (2)
_DEEP_START = {
    "validity": ("open", [f"{LEAF}.fits"]),
    f"{LEAF}.pe answer": _unassigned(f"{LEAF}.pe"),
    f"{LEAF}.pe state": OPEN,
    f"{LEAF}.fits answer": _unassigned(f"{LEAF}.pe"),
}
_DEEP_SET = {
    "validity": ("open", [f"{LEAF}.fits"]),
    f"{LEAF}.pe answer": {"kind": "value", "value": 4},
    f"{LEAF}.pe state": {"state": "set", "value": 4},
    f"{LEAF}.fits answer": _missing("width"),
}
_SET_SIZE = (compare.Request({"size": 3}), _accepted(("size", "changed", "unresolved")), _SIZE_SET)
_SET_PE = (compare.Request({"pe": 3}), _accepted(("pe", "changed", "unresolved")), _PE_SET)
_SET_DEEP = (
    compare.Request({f"{LEAF}.pe": 4}),
    _accepted((f"{LEAF}.pe", "changed", "value")),
    _DEEP_SET,
)
_GIVEN = ("changed", "value")

MISSING_FACT_PATHS: list[MissingFactPath] = [
    (
        "waiting-membership",
        ("size", "fits"),
        _SIZE_START,
        [
            _SET_SIZE,
            (
                compare.Request(facts={"limit": 2}),
                _accepted(("limit", *_GIVEN), ("size", "dropped", "refused")),
                _SIZE_START,
            ),
        ],
    ),
    (
        "waiting-membership",
        ("size", "fits"),
        _SIZE_START,
        [
            _SET_SIZE,
            (
                compare.Request(facts={"limit": 4}),
                _accepted(("limit", *_GIVEN)),
                {
                    "validity": ("open", ["fits"]),
                    "size answer": {"kind": "value", "value": 3},
                    "size state": {"state": "set", "value": 3},
                    "fits answer": _unassigned("lanes"),
                },
            ),
        ],
    ),
    (
        "accepted-waiting",
        ("pe",),
        _PE_START,
        [
            _SET_PE,
            (
                compare.Request(facts={"width": 8}),
                _accepted(("width", *_GIVEN), ("pe", "dropped", "refused")),
                {"validity": ("valid", []), "pe answer": _unassigned("pe"), "pe state": OPEN},
            ),
        ],
    ),
    (
        "accepted-waiting",
        ("pe",),
        _PE_START,
        [
            _SET_PE,
            (
                compare.Request(facts={"width": 6}),
                _accepted(("width", *_GIVEN)),
                {
                    "validity": ("valid", []),
                    "pe answer": {"kind": "value", "value": 3},
                    "pe state": {"state": "set", "value": 3},
                },
            ),
        ],
    ),
    (
        "three-deep",
        (f"{LEAF}.pe", f"{LEAF}.fits"),
        _DEEP_START,
        [
            _SET_DEEP,
            (
                compare.Request(facts={"width": 2}),
                _accepted(("width", *_GIVEN), broken=((f"{LEAF}.fits", [f"{LEAF}.pe"]),)),
                {
                    **_DEEP_SET,
                    "validity": ("refused", [f"{LEAF}.fits"]),
                    f"{LEAF}.fits answer": {
                        "kind": "refused",
                        "findings": [
                            {
                                "kind": "rejection",
                                "code": "constraint-false",
                                "owner": f"{LEAF}.fits",
                                "message": "constraint returned False",
                                "details": [],
                                "causes": [],
                            }
                        ],
                    },
                },
            ),
        ],
    ),
    (
        "three-deep",
        (f"{LEAF}.pe", f"{LEAF}.fits"),
        _DEEP_START,
        [
            _SET_DEEP,
            (
                compare.Request(facts={"width": 8}),
                _accepted(("width", *_GIVEN)),
                {
                    **_DEEP_SET,
                    "validity": ("valid", []),
                    f"{LEAF}.fits answer": {"kind": "value", "value": True},
                },
            ),
        ],
    ),
]
"""The public API's answers on the missing-fact shapes, along each path from the fact left
out."""


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize(
    "path",
    MISSING_FACT_PATHS,
    ids=[f"{toy}-{steps[-1][0]}" for toy, _, _, steps in MISSING_FACT_PATHS],
)
def test_a_missing_fact_answers_as_the_public_api(name: str, path: MissingFactPath) -> None:
    """The public API's answers pinned through the oracle's forms, and every registered
    implementation answering as the public API does (its reports, and everything at each
    design space)."""
    toy, keys, start, steps = path
    configuration = TOYS[toy].make()
    base, other = PUBLIC.open(configuration), IMPLEMENTATIONS[name].open(configuration)
    assert compare.everything(base, other).lines() == []
    assert _asked(base, keys) == start
    for request, report, expected in steps:
        made = base.change(request.commits, request.clears, request.facts)
        assert _reported(forms.report(made.report)) == report, request
        found, base_after, other_after = compare.change(base, other, request)
        assert found.lines() == [], request
        assert base_after is not None and other_after is not None, request
        assert _asked(base_after, keys) == expected, request
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
    x: int = Decision(values=(1, 2))


class Inner(Space):
    on: bool = Decision(values=(False, True))
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
