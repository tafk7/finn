# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a successor keeps of its base's evaluations.

A successor's world is made from its base's: it takes each entry that read nothing the
change touched (the choices it sets, replaces or clears, the facts it changes), and
finds the rest again, keeping an answer where everything it read answers the same
(early cutoff). Each answers as a configuration built from nothing with the same
choices does, evidence included. What lies outside the change's reach is not evaluated
again, and what lies inside it is: the tests that count calls assert both.
"""

from __future__ import annotations

from typing import Any

import pytest
from support.space_cores import CORES, Packed, Stub, chain, commit
from support.space_walk import fresh as recomputed

from finn.core.space import (
    Decision,
    Param,
    Space,
    Unresolved,
    derived,
    design_space,
    domain,
    inspection,
    occurrence,
)


def test_a_successor_keeps_what_its_change_does_not_reach_and_derives_the_rest() -> None:
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
        left: Packed | Stub = Decision(CORES, width=width)
        right: Packed | Stub = Decision(CORES, width=width)

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


@pytest.mark.parametrize("stages_first", [True, False])
def test_a_change_to_one_stage_keeps_what_read_the_far_stages(stages_first: bool) -> None:
    # A hop far down the chain reads only its own stages: a change to the first stage
    # leaves its entry to the successor as it was, and it answers as one built from nothing.
    point = design_space(chain(8, stages_first=stages_first)())
    base = occurrence.state(point)
    keys, nodes = base.linked.keys, base.linked.nodes
    far = base.world.entry(keys["h5.narrow"])  # waits on an open stage
    assert isinstance(far.result, Unresolved)
    assert {nodes[i].key for i in base.evaluator.bits.nodes(far.reads)} == {"s5.size"}
    wide = commit(point, {"s0.size": 2})
    assert occurrence.state(wide).world.entry(keys["h5.narrow"]) is far
    built = occurrence.state(recomputed(wide))
    assert built.world.result(keys["h5.narrow"]) == far.result


def test_a_change_does_not_evaluate_the_guard_of_a_choice_it_does_not_reach() -> None:
    guards: list[str] = []

    def within(*, candidate: int, limit: int) -> bool:
        return candidate <= limit

    class Guarded(Space):
        width: int = Param()
        finish: str = Decision(values=("matte", "gloss"))
        compute: Packed | Stub = Decision(CORES, width=width)

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
    # The change reaches neither ``depth`` nor its guard: the successor takes the guard's
    # entry from its base, and asks nothing again.
    glossy = commit(base, {"finish": "gloss"})
    assert guards == [] and glossy.depth == 1
    assert inspection.explain(glossy, Guarded.depth) == inspection.explain(
        recomputed(glossy), Guarded.depth
    )
    # Built from nothing, the guard's method is answered by the model's memo: it read the
    # same core.
    assert guards == []
