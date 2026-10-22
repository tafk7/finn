# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``self`` methods on the evaluation core: a plain call whose receiver reads through
the world it runs in (``_world.MethodReader``, reached through ``occurrence``).

Each read is recorded with its answer (the entry's trace), checked against the reads
inferred from the method's source (always: an undeclared read is the method's
failure), and a read of a forwarding alias goes to its source where the alias applies
whenever the method does. A method's outcome is memoized by its traces. One lock per
model: two models evaluate at once, one model's worlds one at a time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import cast

import pytest

from core.space._toys_support import House
from core.space._world_support import node, open_world, successor, today
from finn.core.space import (
    Available,
    Decision,
    EvaluationError,
    Param,
    Select,
    Space,
    Unresolved,
    _failure,
    derived,
)
from finn.core.space._failure import NativeEvaluationError
from finn.core.space._world import Evaluator, World


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> object:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


def value(world: World, key: str) -> object:
    result = world.result(node(world, key))
    assert isinstance(result, Available), result
    return result.value


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @derived
    def bits(self) -> int:
        return self.width * self.lanes


class Wide(Space):
    width: int = Param()
    span: int = Decision(values=(2, 4))


class Outer(Space):
    width: int = Param()
    enabled: bool = Decision(values=(False, True))
    leaf = Leaf(width=width)
    guarded = Leaf(width=width, when=enabled)
    mode: Leaf | Wide = Select(leaf=Leaf(width=width), wide=Wide(width=width))

    @derived
    def through_child(self) -> int:
        return self.leaf.bits + self.leaf.width

    @derived
    def through_choice(self) -> int:
        return self.mode.width

    @derived
    def presence(self) -> bool:
        return self.present(Outer.guarded)

    @derived
    def hidden(self) -> int:
        # A loop's name is no path from the receiver: what is read through it is not
        # inferred (a local bound once to a path is).
        for child in (self.leaf,):
            try:
                return child.width
            except EvaluationError:
                return 0
        return 0


def test_house_answers_as_a_published_configuration() -> None:
    choices = {
        "want_garage": True,
        "heating": "heat_pump",
        "heating.heat_pump.cop": 3,
        "hall.finish": 1,
        "kitchen.finish": 1,
        "dining.finish": 1,
        "garage.finish": 2,
    }
    for budget in (200, 100):
        expected = today(House(budget=budget), choices)
        world = open_world(House(budget=budget), choices)
        found = {key: world.result(index) for key, index in world.evaluator.linked.keys.items()}
        assert {key: answer for key, answer in found.items() if answer != expected[key]} == {}


def test_a_method_records_each_read_with_its_answer() -> None:
    world = open_world(Outer(width=4), {"leaf.lanes": 2})
    entry = world.entry(node(world, "through_child"))
    assert entry.result == Available(12)
    keys = [world.evaluator.nodes[index].key for index, _ in entry.trace]
    # ``self.leaf.width`` is an alias of the root's width, read at its source.
    assert keys == ["leaf.bits", "width"]
    assert entry.via == ((node(world, "leaf.width"), node(world, "width")),)
    assert set(world.evaluator.bits.nodes(entry.reads)) == {
        node(world, "width"),
        node(world, "leaf.lanes"),
    }


def test_a_candidates_member_reads_its_selector_first() -> None:
    world = open_world(Outer(width=4))
    assert isinstance(world.result(node(world, "through_choice")), Unresolved)
    chosen = successor(world, {"mode": "wide"})
    entry = chosen.entry(node(chosen, "through_choice"))
    assert entry.result == Available(4)
    assert [chosen.evaluator.nodes[index].key for index, _ in entry.trace][0] == "mode"


@pytest.mark.parametrize(("enabled", "expected"), ((None, None), (False, False), (True, True)))
def test_presence_reads_the_guard_and_halts_while_it_waits(
    enabled: bool | None, expected: bool | None
) -> None:
    world = open_world(Outer(width=4), {} if enabled is None else {"enabled": enabled})
    result = world.result(node(world, "presence"))
    if expected is None:
        assert isinstance(result, Unresolved)
    else:
        assert result == Available(expected)


def test_an_undeclared_read_is_the_methods_failure_even_when_caught() -> None:
    world = open_world(Outer(width=4))
    with pytest.raises(NativeEvaluationError) as caught:
        world.entry(node(world, "hidden"))
    assert caught.value.primary.owner == "hidden"
    assert caught.value.primary.role == "read-undeclared"
    assert "leaf.width" in caught.value.primary.detail


def test_dynamic_and_declared_reads() -> None:
    def computed(self: Space) -> int:
        return sum(cast(int, getattr(self, name)) for name in ("a", "b"))

    def declared(self: Space) -> int:
        return sum(cast(int, getattr(self, name)) for name in ("a", "b"))

    declared.__reads__ = ("a", "b")  # type: ignore[attr-defined]

    def short(self: Space) -> int:
        return sum(cast(int, getattr(self, name)) for name in ("a", "b"))

    short.__reads__ = ("a",)  # type: ignore[attr-defined]

    members: dict[str, object] = {
        "__annotations__": {"a": int, "b": int},
        "a": Param(),
        "b": Param(),
        "computed": derived(computed),
        "declared": derived(declared),
        "short": derived(short),
    }
    space_type = cast(type[Space], type("Reads", (Space,), members))
    world = open_world(space_type(a=1, b=2))  # type: ignore[call-arg]
    assert value(world, "computed") == 3  # dynamic: not checked
    assert value(world, "declared") == 3
    with pytest.raises(NativeEvaluationError, match="read-undeclared"):
        world.entry(node(world, "short"))


calls: list[str] = []


class Counted(Space):
    width: int = Param()
    left: int = Decision(values=(1, 2, 3))
    right: int = Decision(values=(1, 2, 3))

    @derived
    def scaled(self) -> int:
        calls.append("scaled")
        return self.width * self.left

    @derived
    def parity(self) -> bool:
        calls.append("parity")
        return self.scaled % 2 == 0

    @derived
    def label(self) -> str:
        calls.append("label")
        return "even" if self.parity else "odd"


@pytest.mark.usefixtures("purity_unchecked")
def test_a_method_is_memoized_by_its_traces() -> None:
    calls.clear()
    base = open_world(Counted(width=4), {"left": 1, "right": 1})
    first = successor(base, {"right": 2})
    second = successor(base, {"right": 3})
    assert value(first, "label") == "even" and value(second, "label") == "even"
    # The second world reads the same answers: every method is a memo hit there.
    assert calls == ["label", "parity", "scaled"]
    assert base.evaluator.counters.hits["derived"] == 3
    changed = successor(first, {"left": 2})
    assert value(changed, "label") == "even"
    # scaled again (its read of left differs), parity again (scaled's answer differs);
    # label is cut early (parity answers the same).
    assert calls == ["label", "parity", "scaled", "scaled", "parity"]


@pytest.mark.usefixtures("purity_unchecked")
def test_a_halted_method_is_memoized_with_its_blocked_read() -> None:
    """A trace is replayed by the reads' answers, compared by value semantics, so a world
    started apart (other answer objects, equal answers) finds it too."""
    calls.clear()
    world = open_world(Counted(width=4))
    assert isinstance(world.result(node(world, "label")), Unresolved)
    again = World.start(world.evaluator, world.facts, world.stored)
    assert isinstance(again.result(node(again, "label")), Unresolved)
    assert calls == ["label", "parity", "scaled"]
    assert world.evaluator.counters.hits["derived"] == 3


@pytest.mark.usefixtures("purity_unchecked")
def test_two_models_evaluate_at_once_one_model_one_at_a_time() -> None:
    """Moved from ``test_native_admission``'s
    ``test_concurrent_independent_and_cold_shared_snapshots``: the lock is the model's,
    so the barrier is met by two models' worlds, and one model's worlds share its
    entries and memo."""
    barrier = Barrier(2)
    seen: list[int] = []

    class First(Space):
        fact: int = Param()

        @derived
        def output(self) -> int:
            value = self.fact
            barrier.wait(timeout=5)
            seen.append(value)
            return value

    class Second(First):
        pass

    worlds = (open_world(First(fact=1)), open_world(Second(fact=2)))
    assert worlds[0].evaluator.model is not worlds[1].evaluator.model
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda world: value(world, "output"), worlds)) == [1, 2]
    assert sorted(seen) == [1, 2]

    class Shared(Space):
        fact: int = Param()

        @derived
        def output(self) -> int:
            seen.append(self.fact)
            return self.fact

    shared = open_world(Shared(fact=3))
    others = [World.start(shared.evaluator, shared.facts, shared.stored) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda world: value(world, "output"), [shared, *others] * 16))
    assert found == [3] * 64
    assert seen.count(3) == 1  # the model's memo: each later world replays the trace
    assert _failure.current() is None


def test_an_evaluator_is_per_model() -> None:
    world = open_world(Counted(width=4))
    assert isinstance(world.evaluator, Evaluator)
    assert world.evaluator.model.space_type is Counted
    assert world.evaluator.run is None
