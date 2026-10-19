# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The bit table and the memo every world of one model shares.

A bit is allocated for a choice or a fact the first time it is read or changed. An
authored function's outcome is kept across worlds by what it read, one rule for
every function: its trace (an explicit function's arguments' answers, a method's
reads), the answers held, found again where each read answers the same (the same
object, else equal by value semantics); a few traces a function, and a budget of them
a model, the least recently used dropped first.
"""

from __future__ import annotations

from itertools import count

import pytest

from core.space._world_support import node, open_world, successor
from finn.core.space import Available, Decision, Param, Space, derived
from finn.core.space._failure import NativeEvaluationError
from finn.core.space._memo import TRACES_KEPT, BitTable, Counters, Memo
from finn.core.space._world import Called, World, check_purity
from finn.core.space.results import QueryResult


def test_bits_are_allocated_on_first_use_and_decoded_in_order() -> None:
    bits = BitTable()
    assert len(bits) == 0
    seven, three = bits.bit(7), bits.bit(3)
    assert (seven, three) == (1, 2) and bits.bit(7) == 1 and len(bits) == 2
    assert bits.bits((3, 9)) == 2 | 4
    assert bits.nodes(seven | three | bits.bit(9)) == (7, 3, 9)
    assert bits.nodes(0) == ()


def test_a_model_allocates_bits_only_for_what_it_reads() -> None:
    world = open_world(Pair(width=4), {"left": 1, "right": 2})
    assert len(world.evaluator.bits) == 0
    world.entry(node(world, "scaled"))
    assert set(world.evaluator.bits.nodes(world.entry(node(world, "scaled")).reads)) == {
        node(world, "width"),
        node(world, "left"),
    }
    assert len(world.evaluator.bits) == 2


def test_a_memo_keeps_a_few_traces_a_function_newest_first_holding_their_answers() -> None:
    memo: Memo[Called] = Memo()
    first: QueryResult[object] = Available(1)
    memo.keep((5, "call"), ((1, first),), Called(Available(3)))
    (kept,) = memo.kept((5, "call"))
    assert kept.outcome.result == Available(3) and kept.trace[0][1] is first  # held
    assert memo.kept((5, "membership")) == ()
    for number in range(TRACES_KEPT + 2):
        answer: QueryResult[object] = Available(number)
        memo.keep((4, "call"), ((1, answer),), Called(answer))
    traces = memo.kept((4, "call"))
    assert len(traces) == TRACES_KEPT and len(memo) == TRACES_KEPT + 1
    assert [each.outcome.result for each in traces][:2] == [
        Available(TRACES_KEPT + 1),
        Available(TRACES_KEPT),
    ]


def test_a_memo_keeps_its_budget_dropping_the_least_recently_used() -> None:
    memo: Memo[Called] = Memo(budget=3)
    for index in range(3):
        memo.keep((index, "call"), (), Called(Available(index)))
    (oldest,) = memo.kept((0, "call"))
    memo.used(oldest)  # found again: the most recently used
    memo.keep((3, "call"), (), Called(Available(3)))
    assert len(memo) == 3 and memo.kept((1, "call")) == ()
    assert memo.kept((0, "call")) == (oldest,)
    memo.budget = 1  # lowered: the least recently used go at once
    assert len(memo) == 1 and memo.kept((3, "call")) != ()
    memo.used(oldest)  # dropped meanwhile: nothing to move


calls: list[str] = []


class Pair(Space):
    width: int = Param()
    left: int = Decision(values=(1, 2, 3))
    right: int = Decision(values=(1, 2, 3))

    @derived
    def scaled(*, width: int, left: int) -> int:
        calls.append("scaled")
        return width * left

    @derived
    def total(*, scaled: int, right: int) -> int:
        calls.append("total")
        return scaled + right


@pytest.mark.usefixtures("purity_unchecked")
def test_sibling_worlds_share_what_either_called() -> None:
    calls.clear()
    base = open_world(Pair(width=4), {"left": 1, "right": 1})
    first = successor(base, {"right": 2})
    second = successor(base, {"right": 3})
    assert first.result(node(first, "total")) == Available(6)
    assert second.result(node(second, "total")) == Available(7)
    assert calls == ["scaled", "total", "total"]  # scaled once: both read the same answers
    again = successor(first, {"left": 1})  # nothing changes: the same answer objects
    assert again.delta == 0 and again.result(node(again, "total")) == Available(6)
    third = successor(second, {"left": 1, "right": 3})
    assert third.result(node(third, "total")) == Available(7)
    assert calls == ["scaled", "total", "total"]
    counters = base.evaluator.counters
    assert counters.calls["derived"] == 3 and counters.hits["derived"] == 1  # second's scaled


def test_counters_count_calls_and_hits_by_kind_and_node_and_reset() -> None:
    counters = Counters()
    counters.call("derived", 4)
    counters.call("derived", 5)
    counters.hit("constraint", 4)
    counters.evaluated += 3
    assert counters.calls == {"derived": 2} and counters.hits == {"constraint": 1}
    assert counters.calls_at == {4: 1, 5: 1} and counters.hits_at == {4: 1}
    counters.reset()
    assert counters == Counters()


@pytest.mark.usefixtures("purity_unchecked")
def test_worlds_started_apart_share_what_they_compute_alike() -> None:
    """A world started on its own makes its own answer objects; the memo finds what
    another world called where each argument is equal by its value semantics."""
    calls.clear()
    world = open_world(Pair(width=4), {"left": 2, "right": 1})
    other = World.start(world.evaluator, world.facts, world.stored)
    assert world.result(node(world, "total")) == other.result(node(other, "total"))
    assert calls == ["scaled", "total"]
    counters = world.evaluator.counters
    assert counters.hits["derived"] == 2 and counters.semantic >= 2
    # The other world's answer is the kept outcome, the same object.
    assert world.result(node(world, "total")) is other.result(node(other, "total"))


STAMPS = count()


class Stamped(Space):
    """Impure: each answer reads a counter outside the Space."""

    width: int = Param()

    @derived
    def stamped(self) -> int:
        return self.width + next(STAMPS) % 2

    @derived
    def numbered(*, width: int) -> int:
        return width + next(STAMPS) % 2


@pytest.mark.parametrize("key", ["stamped", "numbered"])
def test_the_purity_check_runs_a_memo_hit_again_and_names_an_impure_function(key: str) -> None:
    world = open_world(Stamped(width=4))
    first = world.result(node(world, key))
    other = World.start(world.evaluator, world.facts, world.stored)  # equal reads: a hit
    with pytest.raises(NativeEvaluationError) as caught:
        other.result(node(other, key))
    assert caught.value.primary.owner == key
    assert "answered differently when run again" in str(caught.value.primary)
    check_purity(False)  # unchecked, the hit answers what was kept, silently
    again = World.start(world.evaluator, world.facts, world.stored)
    assert again.result(node(again, key)) is first
