# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The evaluation core's worlds: how each kind of node reads, validity's pieces, what a
world takes from its parent, and early cutoff by value semantics.

The core (``finn.core.space._world``) is not wired to ``Space`` yet: each test opens a
world of a root declaration's facts and stored choices (``core.space._world_support``).
A choice reads its stored value through its type check over facts, or unassigned, or
(under a scope not in play) inapplicable; nothing is forced. Validity's pieces are the
constraints, present children, a present choice whose listed domain lists no value,
and a guard refused through a refused input.
"""

from __future__ import annotations

from typing import cast

import pytest

from core.space._world_support import answer, node, open_world, successor, today
from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Members,
    Param,
    Rejected,
    Space,
    Unresolved,
    ValueSemantics,
    ViewKey,
    constraint,
    derived,
    divisors_of,
    reject,
    view,
)
from finn.core.space.graph import Located
from finn.core.space.results import ConstraintAssessment


def _codes(result: object) -> list[tuple[str, str]]:
    assert isinstance(result, (Inapplicable, Rejected, Unresolved))
    return [(finding.code, finding.owner) for finding in result.findings]


class Choices(Space):
    extent: int = Param()
    enabled: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=enabled)
    width: int = Decision(values=(8, 16))


def test_a_choice_reads_its_stored_value_unassigned_waiting_or_dormant() -> None:
    world = open_world(Choices(extent=4))
    assert _codes(answer(world, "enabled")) == [("decision-unassigned", "enabled")]
    # Its presence waits on the open selector of its scope: unknown, never forced.
    assert _codes(answer(world, "lanes")) == [("decision-unassigned", "enabled")]

    dormant = open_world(Choices(extent=4), {"enabled": False, "lanes": 2})
    assert isinstance(answer(dormant, "lanes"), Inapplicable)
    assert dormant.stored[node(dormant, "lanes")] == 2  # kept, out of play

    present = open_world(Choices(extent=4), {"enabled": True})
    assert _codes(answer(present, "lanes")) == [("decision-unassigned", "lanes")]

    stored = open_world(Choices(extent=4), {"enabled": True, "lanes": 2})
    lanes = answer(stored, "lanes")
    assert lanes == Available(2)
    assert lanes is stored.values[node(stored, "lanes")]  # one answer object per value


class Facts(Space):
    extent: int = Param(required=False)

    @derived
    def checked(*, extent: int) -> int | Rejected:
        if extent < 2:
            return reject("extent-small", "the extent is below two")
        return extent

    lanes: int = Decision(domain=divisors_of(extent))
    banks: int = Decision(domain=divisors_of(checked))


def test_a_stored_value_reads_through_its_type_check_over_facts() -> None:
    outside = open_world(Facts(extent=4), {"lanes": 3})
    assert _codes(answer(outside, "lanes")) == [("domain-membership", "lanes")]

    missing = open_world(Facts(), {"lanes": 2})
    assert _codes(answer(missing, "lanes")) == [("input-missing", "extent")]

    refused = open_world(Facts(extent=1), {"banks": 1})
    assert _codes(answer(refused, "banks")) == [("extent-small", "checked")]


class Holder(Space):
    extent: int = Param()
    on: bool = Decision(values=(False, True))
    lanes: int = Decision(domain=divisors_of(extent), when=on)
    other: int = Decision(values=(1, 2))


@pytest.mark.parametrize(
    ("extent", "choices", "expected"),
    (
        (0, {"on": True}, [("domain-empty", "lanes")]),
        # Presence unknown: an empty list is an unknown piece, waiting on the selector.
        (0, {}, [("decision-unassigned", "on")]),
        (0, {"on": False}, None),
        # A non-empty list adds nothing, present or not; an open choice is no piece.
        (4, {}, None),
        (4, {"on": True}, None),
        # A stored value's type check is no piece.
        (4, {"on": True, "lanes": 3}, None),
    ),
)
def test_validity_counts_a_present_domain_that_lists_no_value(
    extent: int, choices: dict[str, object], expected: list[tuple[str, str]] | None
) -> None:
    world = open_world(Holder(extent=extent), choices)
    validity = world.entry(world.evaluator.linked.scopes[0].validity or 0)
    if expected is None:
        assert validity.result == Available(True)
    else:
        assert _codes(validity.result) == expected
    assert isinstance(validity.assessment, ConstraintAssessment)
    assert "other" not in validity.assessment.results


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))


class Guarded(Space):
    size: int = Param()

    @derived
    def big(*, size: int) -> bool | Rejected:
        if size < 0:
            return reject("size-negative", "a size is never negative")
        return size > 4

    @constraint(when=big)
    def limit(*, size: int) -> bool:
        return size < 100

    banks: int = Decision(values=(1, 2), when=big)
    leaf = Leaf(width=size, when=big)


def test_a_guard_refused_through_its_input_is_one_piece_per_guard() -> None:
    world = open_world(Guarded(size=-1))
    validity = world.entry(world.evaluator.linked.scopes[0].validity or 0)
    assessment = validity.assessment
    assert isinstance(assessment, ConstraintAssessment)
    assert {key: sorted(_codes(result)) for key, result in assessment.results.items()} == {
        "banks": [("guard-refused", "banks"), ("size-negative", "big")],
        "leaf.$validity": [("guard-refused", "leaf"), ("size-negative", "big")],
        "limit": [("guard-refused", "limit"), ("size-negative", "big")],
    }
    # The guarded members read refused through the guard; the guard is no origin.
    assert _codes(answer(world, "limit")) == [("size-negative", "big")]
    assert not world.entry(node(world, "limit")).own_refusal
    assert world.entry(node(world, "big")).own_refusal

    held = open_world(Guarded(size=8))
    assert held.entry(held.evaluator.linked.scopes[0].validity or 0).result == Available(True)


class Parity(Space):
    count: int = Decision(values=(1, 2, 3, 4))
    other: int = Decision(values=(1, 2))

    @derived
    def odd(*, count: int) -> bool:
        return count % 2 == 1

    @derived
    def label(*, odd: bool) -> str:
        return "odd" if odd else "even"

    @derived
    def doubled(*, other: int) -> int:
        return 2 * other


def test_early_cutoff_keeps_the_answer_object_where_what_it_read_answers_the_same() -> None:
    world = open_world(Parity(), {"count": 1, "other": 1})
    label = world.entry(node(world, "label"))
    assert label.result == Available("odd")
    counters = world.evaluator.counters
    counters.reset()

    after = successor(world, {"count": 3})
    found = after.entry(node(after, "label"))
    assert found.result is label.result
    # odd is called again (and count's type check); label is not.
    assert counters.cut == 1 and counters.calls == {"derived": 1, "domain membership": 1}
    assert counters.semantic >= 1  # odd's new answer, True again, compared by its semantics
    assert found.reads == label.reads


def test_entries_that_read_nothing_changed_are_taken_from_the_parent() -> None:
    world = open_world(Parity(), {"count": 1, "other": 1})
    held = {key: world.entry(node(world, key)) for key in ("odd", "label", "doubled")}
    world.evaluator.counters.reset()
    after = successor(world, {"other": 2})
    assert after.entry(node(after, "label")) is held["label"]
    assert after.entry(node(after, "odd")) is held["odd"]
    assert after.entry(node(after, "doubled")).result == Available(4)
    assert world.evaluator.counters.inherited == 2


def test_a_successor_with_equal_values_reads_nothing_otherwise() -> None:
    world = open_world(Parity(), {"count": 1, "other": 1})
    after = successor(world, {"count": 1})
    assert after.delta == 0
    assert after.values[node(after, "count")] is world.values[node(world, "count")]
    cleared = world.successor(stored={})
    bits = world.evaluator.bits
    assert set(bits.nodes(cleared.delta)) == {node(world, "count"), node(world, "other")}
    assert _codes(answer(cleared, "count")) == [("decision-unassigned", "count")]


def test_same_compares_by_value_semantics_never_bare_equality() -> None:
    world = open_world(Parity(), {"count": 1, "other": 1})
    odd = node(world, "odd")
    # ``True == 1`` in Python; the bool semantics does not recognize 1.
    assert not world.same(odd, Available(True), Available(1))
    assert world.same(odd, Available(True), Available(True))
    refused = reject("x", "y", owner="z")
    assert world.same(odd, refused, reject("x", "y", owner="z"))
    assert not world.same(odd, refused, Unresolved(refused.findings))


class Grid:
    """A value like an array: ``==`` answers cell by cell, with no truth value."""

    def __init__(self, cells: tuple[int, ...]) -> None:
        self.cells = cells

    def __eq__(self, other: object) -> Cells:  # type: ignore[override]
        assert isinstance(other, Grid)
        return Cells(tuple(a == b for a, b in zip(self.cells, other.cells)))

    __hash__ = None  # type: ignore[assignment]


class Cells:
    def __init__(self, cells: tuple[bool, ...]) -> None:
        self.cells = cells

    def __bool__(self) -> bool:
        raise ValueError("the truth value of a grid comparison is ambiguous")


GRID = ValueSemantics(
    Grid,
    "grid",
    lambda value: type(value) is Grid,
    lambda left, right: left.cells == right.cells,
    lambda value: Grid(value.cells),
)


class Table(Space):
    table: Grid = Param(semantics=GRID)

    @derived
    def total(*, table: Grid) -> int:
        return sum(table.cells)

    @derived
    def large(*, total: int) -> bool:
        return total > 10


def test_cutoff_and_deltas_compare_by_semantics_values_without_a_truth_value() -> None:
    world = open_world(Table(table=Grid((1, 2, 3))))
    large = world.entry(node(world, "large"))
    facts = dict(world.facts)
    table = node(world, "table")
    facts[table] = Grid((1, 2, 3))
    assert world.successor(facts=facts).delta == 0
    facts[table] = Grid((3, 2, 1))
    after = world.successor(facts=facts)
    assert after.delta != 0
    assert after.entry(node(after, "large")).result is large.result
    assert world.same(table, Available(Grid((1, 2))), Available(Grid((1, 2))))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(Grid((1,)) == Grid((1,)))


def test_a_world_only_adds_to_its_cache() -> None:
    world = open_world(Guarded(size=8))
    first = {key: world.entry(index) for key, index in world.evaluator.linked.keys.items()}
    held = dict(world.cache)
    for index in world.evaluator.linked.keys.values():
        world.entry(index)
    assert all(world.cache[index] is entry for index, entry in held.items())
    assert all(
        world.cache[world.evaluator.linked.keys[key]] is entry for key, entry in first.items()
    )


COST = ViewKey("cost", int)


class Part(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @view
    def cost(*, area: int, finish: int) -> int:
        return area * finish

    exports = {COST: cost}


class Assembly(Space):
    budget: int = Param()
    extra: bool = Decision(values=(False, True))
    first = Part(area=4)
    second = Part(area=first.area)
    optional = Part(area=6, when=extra)
    kind: Part = Decision({"small": Part(area=2), "large": Part(area=3)})
    picked = Part(area=kind.area)
    costs = Members(COST)

    @constraint
    def within(*, costs: tuple[Located[int], ...], budget: int) -> bool:
        return sum(member.value for member in costs) <= budget

    @view
    def total(*, costs: tuple[Located[int], ...]) -> int:
        return sum(member.value for member in costs)


@pytest.mark.parametrize("budget", (40, 10))
def test_every_node_answers_as_the_current_evaluator_where_nothing_is_forced(budget: int) -> None:
    choices = {
        "extra": True,
        "first.finish": 1,
        "second.finish": 2,
        "optional.finish": 3,
        "kind": "large",
        "kind.large.finish": 2,
        "picked.finish": 1,
    }
    expected = today(Assembly(budget=budget), choices)
    world = open_world(Assembly(budget=budget), choices)
    found = {key: world.result(index) for key, index in world.evaluator.linked.keys.items()}
    assert found.keys() <= expected.keys()
    differ = {key: (found[key], expected[key]) for key in found if found[key] != expected[key]}
    assert differ == {}
    if budget == 40:
        assert cast(Available[int], found["total"]).value == 4 + 8 + 18 + 6 + 3
    else:
        assert _codes(found["total"]) == [("constraint-false", "within")]
