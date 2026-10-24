# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The depth guard: dependency depth is not bounded by Python's stack.

The running type evaluates on the Python stack to ``depth.DEPTH`` evaluations;
past it the evaluations in flight are abandoned and the run evaluates the deep node
first, then again what it abandoned. Authored functions are pure, so a function run
again answers the same: each completes once (one abandoned may have started before). The
recursion limit is never raised.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import cast

import pytest

from core.space._evaluation_support import entry, node, open_evaluation
from finn.core.space import (
    Available,
    EvaluationError,
    Param,
    Space,
    _failure,
    derived,
)
from finn.core.space._failure import NativeEvaluationError
from finn.core.space._running import depth


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> object:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


def _chain(
    depth: int,
    completed: list[int],
    *,
    broken: int | None = None,
    impure: int | None = None,
) -> type[Space]:
    """``value_<n>`` reads ``value_<n-1>`` through its receiver; ``completed`` gets ``n``
    once ``value_<n>`` returns (its read answered). ``value_<impure>`` adds how often it
    completed before, odd or even: run again, it answers otherwise."""
    members: dict[str, object] = {"__annotations__": {"value_0": int}, "value_0": Param()}

    def step(number: int, previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            if number == broken:
                raise LookupError("the bottom")
            found = cast(int, getattr(self, previous))
            completed.append(number)
            if number == impure:
                return found + 1 + completed.count(number) % 2
            return found + 1

        return compute

    for number in range(1, depth + 1):
        members[f"value_{number}"] = derived(step(number, f"value_{number - 1}"))
    return cast(type[Space], type("DeepSelf", (Space,), members))


def test_the_purity_check_runs_again_a_method_the_depth_guard_abandoned() -> None:
    """Past ``DEPTH`` the methods in flight are abandoned and run again; under the purity
    check one completed after it was abandoned runs once more, and an impure one is
    named. Pure, every abandoned method completes twice."""
    completed: list[int] = []
    world = open_evaluation(_chain(100, completed)(value_0=0))  # type: ignore[call-arg]
    assert world.result(node(world, "value_100")) == Available(100)
    assert 0 < sum(completed.count(number) == 2 for number in range(1, 101)) < 100
    completed.clear()
    impure = open_evaluation(_chain(100, completed, impure=50)(value_0=0))  # type: ignore[call-arg]
    with pytest.raises(NativeEvaluationError) as caught:
        entry(impure, "value_100")
    assert caught.value.primary.owner == "value_50"


@pytest.mark.usefixtures("purity_unchecked")
def test_a_chain_of_twenty_thousand_methods_completes_each_once() -> None:
    """Moved from ``test_native_admission``'s
    ``test_native_chain_20000_has_one_start_per_callback``: each callback completes once;
    one the depth guard abandoned started more than once."""
    limit = sys.getrecursionlimit()
    completed: list[int] = []
    world = open_evaluation(_chain(20_000, completed)(value_0=0))  # type: ignore[call-arg]
    assert world.result(node(world, "value_20000")) == Available(20_000)
    assert sorted(completed) == list(range(1, 20_001))
    starts = world.counters.calls["derived"]
    assert 20_000 < starts <= 2 * 20_000
    assert world.result(node(world, "value_20000")) == Available(20_000)
    assert world.counters.calls["derived"] == starts  # held: nothing runs again
    assert sys.getrecursionlimit() == limit


def _zero() -> int:
    return 0


@pytest.mark.usefixtures("purity_unchecked")
def test_a_chain_of_twenty_thousand_explicit_functions() -> None:
    """``test_compiler``'s twenty-thousand explicit chain, on the running type: each
    function is called once (an explicit function is called once its inputs are read, so
    the depth guard abandons none mid-call). Unchecked: the purity check evaluates every
    node the depth guard abandoned again."""

    def step(*, previous: int) -> int:
        return previous + 1

    members: dict[str, object] = {"value0": derived(_zero)}
    for number in range(1, 20_001):
        members[f"value{number}"] = derived(previous=members[f"value{number - 1}"])(step)
    world = open_evaluation(cast(type[Space], type("Deep", (Space,), members))())
    assert world.result(node(world, "value20000")) == Available(20_000)
    assert world.counters.calls["derived"] == 20_001  # explicit: never abandoned


def test_a_failure_at_the_bottom_of_a_deep_chain_is_the_primary() -> None:
    completed: list[int] = []
    world = open_evaluation(_chain(500, completed, broken=1)(value_0=0))  # type: ignore[call-arg]
    with pytest.raises(NativeEvaluationError) as caught:
        entry(world, "value_500")
    assert caught.value.primary.owner == "value_1"
    assert isinstance(caught.value.__cause__, LookupError)
    assert caught.value.cleanup_failures == () and completed == []


def test_a_cycle_longer_than_the_guard_is_found() -> None:
    length = 5 * depth.DEPTH
    members: dict[str, object] = {}

    def step(previous: str) -> Callable[[Space], int]:
        def compute(self: Space) -> int:
            return cast(int, getattr(self, previous)) + 1

        return compute

    for number in range(length):
        members[f"value_{number}"] = derived(step(f"value_{(number - 1) % length}"))
    world = open_evaluation(cast(type[Space], type("Loop", (Space,), members))())
    with pytest.raises(EvaluationError, match="dependency cycle"):
        entry(world, "value_0")


@pytest.mark.usefixtures("purity_unchecked")
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
    world = open_evaluation(mixed(value_0=0))  # type: ignore[call-arg]
    assert world.result(node(world, "value_400")) == Available(400)
    completed.clear()
    output_entry = entry(world, "output")
    assert output_entry.result == Available(sum(range(1, 401)) + 1_200)
    assert sorted(completed) == list(range(401, 1_201))  # each of the 800 completes once
    assert len({index for index, _ in output_entry.trace}) == 401

    leaves: dict[str, object] = {"__annotations__": {}}
    for index in range(1_000):
        leaves["__annotations__"][f"leaf_{index}"] = int  # type: ignore[index]
        leaves[f"leaf_{index}"] = Param()

    def total(self: Space) -> int:
        return sum(cast(int, getattr(self, f"leaf_{index}")) for index in range(1_000))

    leaves["total"] = derived(total)
    wide = cast(type[Space], type("WideSelf", (Space,), leaves))
    build: Callable[..., Space] = wide
    fan = open_evaluation(build(**{f"leaf_{index}": index for index in range(1_000)}))
    assert fan.result(node(fan, "total")) == Available(sum(range(1_000)))
    assert fan.counters.calls["derived"] == 1
