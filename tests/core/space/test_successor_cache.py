# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a successor keeps of its base's evaluations.

A successor is answered in its model's session, whose node cache keeps every answer with
the choices and facts it read: an answer that read nothing the change touched (the choices
it sets, replaces or clears, the facts it changes) is found there, and the rest is
evaluated again. Each answers as a design space bound from nothing with the same choices
does, evidence included. What lies outside the change's reach is not evaluated again, and
what lies inside it is: the tests that count calls assert both.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.space_cores import CORES, Packed, Stub, chain, commit
from support.space_walk import fresh as recomputed

from finn.core.space import (
    Decision,
    Param,
    Select,
    Space,
    Unresolved,
    derived,
    design_space,
    domain,
    inspection,
    occurrence,
)
from finn.core.space._running.evaluate import Ans, Session, evaluation


@pytest.mark.usefixtures("purity_unchecked")
def test_a_successor_keeps_what_its_change_does_not_reach_and_derives_the_rest() -> None:
    """A successor derives again only what its change reaches. The test counts work, so it
    runs with the purity check off: the check runs every cache hit's function again."""
    calls: list[str] = []

    class Room(Space):
        width: int = Param()
        lanes: int = Decision(values=(1, 2, 4))
        finish: str = Decision(values=("matte", "gloss"))

        @derived
        def area(self) -> int:
            calls.append("area")
            return self.width * self.width

        @derived
        def cycles(self) -> int:
            calls.append("cycles")
            return self.area // self.lanes

        @derived
        def cost(self) -> int:
            calls.append("cost")
            return self.cycles * (2 if self.finish == "gloss" else 1)

    base = commit(design_space(Room(width=8)), {"lanes": 2, "finish": "matte"})
    assert base.cost == 32 and calls == ["cost", "cycles", "area"]
    calls.clear()
    glossy = commit(base, {"finish": "gloss"})
    assert glossy.cost == 64 and calls == ["cost"]
    calls.clear()
    wider = commit(base, {"lanes": 4})
    # ``cycles`` read the lanes: found again first, its answer differs, and ``cost`` is too.
    assert wider.cost == 16 and sorted(calls) == ["cost", "cycles"]
    fresh = commit(design_space(Room(width=8)), {"lanes": 4, "finish": "matte"})
    assert inspection.explain(wider, Room.cost) == inspection.explain(fresh, Room.cost)


def test_a_successor_answers_as_a_configuration_built_from_nothing() -> None:
    class Two(Space):
        width: int = Param()
        left: Packed | Stub = Select(**CORES, width=width)
        right: Packed | Stub = Select(**CORES, width=width)

    def answers(point: Any) -> dict[str, object]:
        return {
            "validity": inspection.validity(point),
            "left": inspection.explain(point, Two.left),
            "pe": inspection.explain(point, Two.left["packed"].pe),  # type: ignore[index]
        }

    base = commit(design_space(Two(width=8)), {"left": "packed"})
    answers(base)
    point = commit(base, {"left.packed.pe": 4, "right": "stub"})
    fresh = commit(design_space(Two(width=8)), {"left": "packed", "right": "stub"})
    fresh = commit(fresh, {"left.packed.pe": 4})
    assert answers(point) == answers(fresh)
    assert isinstance(point.left, Packed) and point.left.pe == 4


def _answer(point: Space, index: int) -> Ans:
    """``index``'s answer in the design space ``point`` holds, in its model's session."""
    ds = occurrence.held(point).ds
    return occurrence.ask(ds.model, lambda session: evaluation(ds, session).get(index))


@pytest.mark.parametrize("stages_first", [True, False])
def test_a_change_to_one_stage_keeps_what_read_the_far_stages(stages_first: bool) -> None:
    # A hop far down the chain reads only its own stages: a change to the first stage
    # leaves its answer to the successor as it was (found in the model's cache), and it
    # answers as one bound from nothing.
    point = design_space(chain(8, stages_first=stages_first)())
    ds = occurrence.held(point).ds
    narrow = ds.node("h5.narrow")
    far = _answer(point, narrow)  # waits on an open stage
    assert isinstance(far.result, Unresolved)
    assert {ds.key(i) for i in far.reads} == {"s5.size"}
    wide = commit(point, {"s0.size": 2})
    assert _answer(wide, narrow) is far
    built = occurrence.held(recomputed(wide)).ds
    with ds.model.lock:
        assert evaluation(built, Session()).result(narrow) == far.result


@pytest.mark.usefixtures("purity_unchecked")
def test_a_change_does_not_evaluate_the_guard_of_a_choice_it_does_not_reach() -> None:
    guards: list[str] = []

    def within(*, candidate: int, limit: int) -> bool:
        return candidate <= limit

    class Guarded(Space):
        width: int = Param()
        finish: str = Decision(values=("matte", "gloss"))
        compute: Packed | Stub = Select(**CORES, width=width)

        @derived
        def packed(self) -> bool:
            guards.append("packed")
            return isinstance(self.compute, Packed)

        @derived
        def rows(self) -> int:
            return 2

        depth: int = Decision(domain=domain(accepts=within, limit=rows), when=packed)

    base = commit(design_space(Guarded(width=3)), {"compute": "packed", "depth": 1})
    guards.clear()
    # The change reaches neither ``depth`` nor its guard: the successor finds the guard's
    # answer in the model's node cache, and asks nothing again.
    glossy = commit(base, {"finish": "gloss"})
    assert guards == [] and glossy.depth == 1
    built = recomputed(glossy)
    # Built from nothing, in a session of its own with no cache (a recomputation shares
    # nothing with the model's cache), its bind calls the guard's method again: once to
    # check ``depth``'s stored value (its guard read first), once for what the bind moved
    # (``depth``'s presence).
    assert guards == ["packed", "packed"]
    assert inspection.explain(glossy, Guarded.depth) == inspection.explain(built, Guarded.depth)
    # Asked through the public API, the design space built is the successor's: found in
    # the model's node cache.
    assert guards == ["packed", "packed"]
