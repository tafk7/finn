# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The class-based oracle (``support.space_oracle``) on FINN's own toys: seeded walks of
every registered implementation against today's, the planted faults each reported by key,
and the forms' rendering.

With today's alone registered the walks compare today's with itself: what they show is
that the oracle runs over the toys' changes and compares what it says it compares. The
planted faults show that it would report a difference: each alters one thing, and
``compare`` and ``forms.differences`` (what ``diff`` runs) name it by key.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from support.space_cores import chain as stages
from support.space_oracle import IMPLEMENTATIONS, Answers, compare, forms
from support.space_oracle.today import FAULTS, TODAY, Fault, Planted
from support.space_pipeline import Pipeline

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Decision,
    QueryResult,
    Space,
    Unresolved,
    design_space,
)

from ._toys_support import House

Toy = tuple[Callable[[], Space], Mapping[str, Sequence[object]]]

TOYS: dict[str, Toy] = {
    "house": (lambda: design_space(House(budget=100)), {"budget": (60, 100, 400)}),
    "pipeline": (lambda: design_space(Pipeline(width=10)), {"width": (6, 10, 12)}),
    "stages": (lambda: design_space(stages(6)()), {}),
}
"""FINN's toys, opened, with the values a walk may give their root facts (the dump tool's
``toys`` target opens the same three)."""

STEPS = 20


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("toy", sorted(TOYS))
@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_implementation_walks_the_toys_as_today(name: str, toy: str, seed: int) -> None:
    make, facts = TOYS[toy]
    found = compare.walk(make, facts, seed, STEPS, TODAY, IMPLEMENTATIONS[name])
    assert found.lines() == []
    # It compared what it says: every design space reached, every change's report.
    assert found.tally["changes"] > 0
    asked = found.tally["changes"] + found.tally["binds"]
    assert found.tally["reports"] + found.tally["requests raising alike"] == asked
    assert found.tally["design spaces"] >= 1 + found.tally["reports"] // 2
    assert found.tally["answers"] > 0 and found.tally["viable values"] > 0


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
    found = compare.everything(TODAY.open(point), Planted(TODAY, fault).open(point))
    paths = [difference.path for difference in found.disagreements]
    assert paths and all(path.startswith(EXPECTED[fault]) for path in paths), paths


@pytest.mark.parametrize("fault", FAULTS)
def test_diff_reports_each_planted_fault_by_key(fault: Fault) -> None:
    """``diff`` compares two dumps' forms with ``forms.differences``."""
    point = _house(kitchen=1)
    base = forms.space(TODAY.open(point))
    other = forms.space(Planted(TODAY, fault).open(point))
    found = forms.differences({"house": {"space": base}}, {"house": {"space": other}})
    assert found and all(
        difference.path.startswith(f"house/space/{EXPECTED[fault]}") for difference in found
    )
    assert forms.differences(base, forms.space(TODAY.open(point))) == []


def test_a_planted_standing_choice_is_reported_in_a_change() -> None:
    """A change refused by the kitchen's finish standing in the way: the planted report
    drops it."""
    point = _house(kitchen=1)
    request = compare.Request({"dining.finish": 2})
    found, after, _ = compare.change(
        TODAY.open(point), Planted(TODAY, "standing").open(point), request
    )
    assert after is None  # refused, on both
    assert [str(difference) for difference in found.disagreements] == [
        'report/broken/0/standing: ["kitchen.finish"] vs []'
    ]


@pytest.mark.parametrize("fault", FAULTS)
def test_a_walk_reports_each_planted_fault(fault: Fault) -> None:
    make, facts = TOYS["house"]
    found = compare.walk(make, facts, 0, STEPS, TODAY, Planted(TODAY, fault))
    assert found.disagreements
    kinds = {"value": "members", "validity": "validity", "possible": "viable"}
    expected = {kinds[fault]} if fault in kinds else {"viable", "report"}
    assert set(found.by_kind) <= expected, found.by_kind


class _Reordered:
    """Today's answers, each answer's findings in reverse order."""

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
    other: Any = _Reordered(TODAY.open(point))
    found = compare.everything(TODAY.open(point), other)
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
        "findings": [["a-code", "a", "waits on a"], ["b-code", "b", "waits on b"]],
    }
    assert forms.answer(Unresolved((first, second))) == expected
    assert forms.answer(Unresolved((second, first))) == expected


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
    pick: Leaf = Decision({"a": Leaf(), "b": Leaf()})


class Outer(Space):
    slot: Inner | Leaf = Decision({"inner": Inner(), "leaf": Leaf()})
    held = Inner()


def test_every_scope_is_reached_under_open_selectors_and_guards() -> None:
    """Candidates of an open selector, nested in a candidate, and a child whose presence
    waits on an open choice: each scope's validity is asked through the public API."""
    answers = TODAY.open(design_space(Outer()))
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
