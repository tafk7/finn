# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The bit table and the memo every world of one model shares.

A bit is allocated for a choice or a fact the first time it is read or changed. An
authored function's outcome is kept across worlds by what it read: an explicit
function by its arguments' answers (their identities, the answers held), a method by
its traces.
"""

from __future__ import annotations

from core.space._world_support import node, open_world, successor
from finn.core.space import Available, Decision, Param, Space, derived
from finn.core.space._memo import TRACES_KEPT, BitTable, Counters, Memo
from finn.core.space._world import Called, World


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


def test_explicit_memo_matches_identities_and_holds_the_answers() -> None:
    memo: Memo[Called] = Memo()
    first, second = Available(1), Available(2)
    memo.keep_explicit(5, "call", (first, second), Called(Available(3)))
    held = memo.explicit(5, "call", (first, second))
    assert held is not None and held.outcome.result == Available(3)
    assert memo.explicit(5, "call", (Available(1), second)) is None  # equal, not the same
    assert memo.explicit(5, "membership", (first, second)) is None
    assert held.answers[0] is first  # held, so its identity is never reused


def test_method_traces_are_kept_newest_first_and_bounded() -> None:
    memo: Memo[Called] = Memo()
    for number in range(TRACES_KEPT + 2):
        memo.keep_method(4, ((1, Available(number)),), Called(Available(number)))
    traces = memo.traces(4)
    assert len(traces) == TRACES_KEPT
    assert [called.result for _, called in traces][:2] == [
        Available(TRACES_KEPT + 1),
        Available(TRACES_KEPT),
    ]
    assert memo.traces(5) == ()


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


def test_counters_count_calls_and_hits_by_kind_and_reset() -> None:
    counters = Counters()
    counters.call("derived")
    counters.call("derived")
    counters.hit("constraint")
    counters.evaluated += 3
    assert counters.calls == {"derived": 2} and counters.hits == {"constraint": 1}
    counters.reset()
    assert counters == Counters()


def test_worlds_started_apart_share_no_answer_objects() -> None:
    """The memo's keys are answer identities: a world started on its own makes its own
    answer objects, so it finds nothing another world started apart called."""
    calls.clear()
    world = open_world(Pair(width=4), {"left": 2, "right": 1})
    other = World.start(world.evaluator, world.facts, world.stored)
    assert world.result(node(world, "total")) == other.result(node(other, "total"))
    assert calls == ["scaled", "total", "scaled", "total"]
