# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The depth guard: dependency depth is not bounded by Python's stack.

The evaluation core evaluates on the Python stack to ``_world.DEPTH`` evaluations;
past it the evaluations in flight are abandoned and the run evaluates the deep node
first, then again what it abandoned. Authored functions are pure, so a function run
again answers the same: each completes once (one abandoned may have started before;
the current evaluator's "one start per callback" becomes "each callback completes
once"). The recursion limit is never raised.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import cast

import pytest

from core.space._world_support import node, open_world
from finn.core.space import (
    Available,
    Const,
    Derived,
    EvaluationError,
    Param,
    Space,
    _failure,
    _world,
    derived,
)
from finn.core.space._failure import NativeEvaluationError


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> object:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


def _chain(depth: int, completed: list[int], *, broken: int | None = None) -> type[Space]:
    """``value_<n>`` reads ``value_<n-1>`` through its receiver; ``completed`` gets ``n``
    once ``value_<n>`` returns (its read answered)."""
    members: dict[str, object] = {"__annotations__": {"value_0": int}, "value_0": Param()}

    def step(number: int, previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            if number == broken:
                raise LookupError("the bottom")
            found = cast(int, getattr(self, previous))
            completed.append(number)
            return found + 1

        return compute

    for number in range(1, depth + 1):
        members[f"value_{number}"] = derived(step(number, f"value_{number - 1}"))
    return cast(type[Space], type("DeepSelf", (Space,), members))


def test_a_chain_of_twenty_thousand_methods_completes_each_once() -> None:
    """Moved from ``test_native_admission``'s
    ``test_native_chain_20000_has_one_start_per_callback``: each callback completes once;
    one the depth guard abandoned started more than once."""
    limit = sys.getrecursionlimit()
    completed: list[int] = []
    world = open_world(_chain(20_000, completed)(value_0=0))  # type: ignore[call-arg]
    assert world.result(node(world, "value_20000")) == Available(20_000)
    assert sorted(completed) == list(range(1, 20_001))
    starts = world.evaluator.counters.calls["derived"]
    assert 20_000 < starts <= 2 * 20_000
    assert world.result(node(world, "value_20000")) == Available(20_000)
    assert world.evaluator.counters.calls["derived"] == starts  # held: nothing runs again
    assert sys.getrecursionlimit() == limit


def test_a_chain_of_twenty_thousand_explicit_functions() -> None:
    """``test_compiler``'s twenty-thousand explicit chain, on the core."""

    def step(*, previous: int) -> int:
        return previous + 1

    members: dict[str, object] = {"value0": Const(0)}
    for number in range(1, 20_001):
        members[f"value{number}"] = Derived(
            step, aliases={"previous": members[f"value{number - 1}"]}
        )
    world = open_world(cast(type[Space], type("Deep", (Space,), members))())
    assert world.result(node(world, "value20000")) == Available(20_000)
    assert world.evaluator.counters.calls["derived"] == 20_000  # explicit: never abandoned


def test_a_failure_at_the_bottom_of_a_deep_chain_is_the_primary() -> None:
    completed: list[int] = []
    world = open_world(_chain(500, completed, broken=1)(value_0=0))  # type: ignore[call-arg]
    with pytest.raises(NativeEvaluationError) as caught:
        world.entry(node(world, "value_500"))
    assert caught.value.primary.owner == "value_1"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == () and completed == []


def test_a_cycle_longer_than_the_guard_is_found() -> None:
    length = 5 * _world.DEPTH
    members: dict[str, object] = {}

    def step(previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1

        return compute

    for number in range(length):
        members[f"value_{number}"] = derived(step(f"value_{(number - 1) % length}"))
    world = open_world(cast(type[Space], type("Loop", (Space,), members))())
    with pytest.raises(EvaluationError, match="dependency cycle"):
        world.entry(node(world, "value_0"))


def test_wide_fan_in_and_a_deep_dependency_behind_a_cached_prefix() -> None:
    """Mirrors ``test_native_admission``'s
    ``test_native_wide_fan_in_and_cached_prefix_deep_dependency``: one method reading 400
    held values and a chain 800 deeper; one reading a thousand formals."""
    completed: list[int] = []
    chain = _chain(1_200, completed)

    def output(self: Space) -> int:
        prefix = sum(cast(int, getattr(self, f"value_{index}")) for index in range(1, 401))
        return prefix + cast(int, getattr(self, "value_1200"))

    output.__reads__ = (*(f"value_{index}" for index in range(1, 401)), "value_1200")  # type: ignore[attr-defined]
    mixed = cast(type[Space], type("Mixed", (chain,), {"output": derived(output)}))
    world = open_world(mixed(value_0=0))  # type: ignore[call-arg]
    assert world.result(node(world, "value_400")) == Available(400)
    completed.clear()
    entry = world.entry(node(world, "output"))
    assert entry.result == Available(sum(range(1, 401)) + 1_200)
    assert sorted(completed) == list(range(401, 1_201))  # each of the 800 completes once
    assert len({index for index, _ in entry.trace}) == 401

    leaves: dict[str, object] = {"__annotations__": {}}
    for index in range(1_000):
        leaves["__annotations__"][f"leaf_{index}"] = int  # type: ignore[index]
        leaves[f"leaf_{index}"] = Param()

    def total(self: Space) -> int:
        return sum(cast(int, getattr(self, f"leaf_{index}")) for index in range(1_000))

    leaves["total"] = derived(total)
    wide = cast(type[Space], type("WideSelf", (Space,), leaves))
    build: Callable[..., Space] = wide
    fan = open_world(build(**{f"leaf_{index}": index for index in range(1_000)}))
    assert fan.result(node(fan, "total")) == Available(sum(range(1_000)))
    assert fan.evaluator.counters.calls["derived"] == 1
