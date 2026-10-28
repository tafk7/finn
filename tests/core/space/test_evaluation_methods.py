# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""``self`` methods on the running type: a plain call whose receiver reads through the
evaluation it runs in (``_running.evaluate.Reader``, on ``_running.receiver.MethodReader``).

Each read is recorded with its answer (the answer's trace), and checked against the reads
inferred from the method's source (always: an undeclared read is the method's failure).
A candidate's member reads its selector first; ``present()`` reads the guard, and halts
while it waits.
"""

from __future__ import annotations

from typing import cast

import pytest

from core.space._evaluation_support import entry, node, open_evaluation
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
from finn.core.space._running.cache import NodeCache
from finn.core.space._running.evaluate import Evaluation, Session, evaluation


@pytest.fixture(autouse=True)
def no_call_is_left_active() -> object:
    assert _failure.current() is None
    yield None
    assert _failure.current() is None


def value(world: Evaluation, key: str) -> object:
    result = world.result(node(world, key))
    assert isinstance(result, Available), result
    return result.value


class Leaf(Space):
    width: int = Param()
    lanes: int = Decision(domain=(1, 2))

    @derived
    def bits(self) -> int:
        return self.width * self.lanes


class Wide(Space):
    width: int = Param()
    span: int = Decision(domain=(2, 4))


class Outer(Space):
    width: int = Param()
    enabled: bool = Decision(domain=(False, True))
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


def test_a_method_records_each_read_with_its_answer() -> None:
    """Each read in order, with its answer. ``self.leaf.width``, a formal bound to the
    root's width, is read from the root as a member of its own scope (the root is not
    below the formal's scope), recorded as itself. Where a node cache is on, the answer
    carries the choices and facts it read, through anything."""
    world = open_evaluation(Outer(width=4), {"leaf.lanes": 2})
    found = entry(world, "through_child")
    assert found.result == Available(12)
    assert [index.key for index, _ in found.trace] == ["leaf.bits", "leaf.width"]
    cached = evaluation(world.ds, Session(cache=NodeCache()))
    assert cached.get(node(world, "through_child")).reads == {
        node(world, "width"),
        node(world, "leaf.lanes"),
    }


def test_a_candidates_member_reads_its_selector_first() -> None:
    world = open_evaluation(Outer(width=4))
    assert isinstance(world.result(node(world, "through_choice")), Unresolved)
    chosen = open_evaluation(Outer(width=4), {"mode": "wide"})
    found = entry(chosen, "through_choice")
    assert found.result == Available(4)
    assert [index.key for index, _ in found.trace][0] == "mode"


@pytest.mark.parametrize(("enabled", "expected"), ((None, None), (False, False), (True, True)))
def test_presence_reads_the_guard_and_halts_while_it_waits(
    enabled: bool | None, expected: bool | None
) -> None:
    world = open_evaluation(Outer(width=4), {} if enabled is None else {"enabled": enabled})
    result = world.result(node(world, "presence"))
    if expected is None:
        assert isinstance(result, Unresolved)
    else:
        assert result == Available(expected)


def test_an_undeclared_read_is_the_methods_failure_even_when_caught() -> None:
    world = open_evaluation(Outer(width=4))
    with pytest.raises(NativeEvaluationError) as caught:
        entry(world, "hidden")
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
    world = open_evaluation(space_type(a=1, b=2))  # type: ignore[call-arg]
    assert value(world, "computed") == 3  # dynamic: not checked
    assert value(world, "declared") == 3
    with pytest.raises(NativeEvaluationError, match="read-undeclared"):
        entry(world, "short")
