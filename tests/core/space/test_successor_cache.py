# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""What a successor and a trial keep of their base's evaluations, and what forcing
keeps of its own.

A change touches the Decisions it adds, removes or replaces, and each open Decision
whose forced value's verdict read one of those. A published successor starts from
its base's evaluations that read no touched Decision, and from its trial's that read
no Decision open in it but one the change does not touch; forcing's copies hand back
what reads no refused Decision. A trial takes a retained choice's admission as its
base recorded it while that admission read no touched Decision, and starts from what
reads only such choices and the open Decisions the change does not touch. Each
answers as a configuration built from nothing with the same choices does, evidence
included. What lies outside the change's reach is not evaluated again, and what lies
inside it is: the tests that count calls assert both.
"""

from __future__ import annotations

from typing import Any

from support.space_cores import CORES, Packed, Stub, commit, forced
from support.space_walk import fresh as recomputed

from finn.core.space import (
    Decision,
    Param,
    Rejected,
    Space,
    derived,
    design_space,
    domain,
    inspection,
    requires,
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
    assert wider.cost == 16 and calls == ["cost", "cycles"]
    fresh = commit(design_space(Room(width=8)), {"lanes": 4, "finish": "matte"})
    assert inspection.explain(wider, Room.cost) == inspection.explain(fresh, Room.cost)


def test_what_read_a_forced_decision_is_derived_again_where_the_forced_value_moves() -> None:
    class Odd(Space):
        lanes: int = Decision(values=(3, 128))
        # Three lanes refuse the stub core (odd), 128 the packed core (over 64).
        compute: Packed | Stub = Decision(CORES, width=lanes)

        @derived
        def kind(self) -> str:
            return type(self.compute).__name__  # reads the forced selector, not the lanes

    three = commit(design_space(Odd()), {"lanes": 3})
    assert three.kind == "Packed" and forced(three) == {"compute": "packed"}
    wide = commit(three, {"lanes": 128})
    assert forced(wide) == {"compute": "stub"} and wide.kind == "Stub"


def test_forcing_keeps_its_evaluations_on_the_configuration_except_a_refused_reads() -> None:
    derivations: list[int] = []

    class Shared(Space):
        lanes: int = Param()

        @derived
        def width(self) -> int:
            derivations.append(self.lanes)
            return self.lanes

        # Its requirement reads ``width``: forcing's copy derives it, before the cases.
        wide: bool = Decision(
            values=(False, True),
            requires=(requires(width, "width-absent: no lanes", cases=(True,)),),
        )
        compute: Packed | Stub = Decision(CORES, width=width)

        @derived
        def kind(self) -> str:
            return type(self.compute).__name__

        # Its requirement reads ``kind``: forcing's copy evaluates it, ``compute`` open.
        typed: bool = Decision(
            values=(False, True),
            requires=(requires(kind, "kind-absent: no core", cases=(True,)),),
        )

    point = design_space(Shared(lanes=128))
    assert forced(point) == {"compute": "stub"}
    assert point.width == 128 and point.kind == "Stub" and derivations == [128]
    # No case of ``compute`` is viable at 129 lanes: ``kind`` is refused here, though
    # forcing's copy, where ``compute`` reads as unassigned, found it waiting.
    refused = design_space(Shared(lanes=129))
    assert forced(refused) == {}
    answer = refused.query(Shared.kind)
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["decision-no-viable-case"]


def test_a_retained_choice_is_admitted_again_whatever_its_base_evaluated() -> None:
    """A retained choice whose admission read what the change touched is admitted
    again, and so is one whose admission read it, even through an evaluation of a
    configuration: neither, nor what read them, is inherited."""

    def within(*, candidate: int, limit: int) -> bool:
        return candidate <= limit

    class Bounded(Space):
        limit: int = Decision(values=(1, 2, 4))
        lanes: int = Decision(domain=domain(accepts=within, limit=limit))

        @derived
        def doubled(self) -> int:
            return 2 * self.lanes

        # Its admission reads ``doubled``, which reads the retained ``lanes``.
        depth: int = Decision(domain=domain(accepts=within, limit=doubled))

    first = commit(design_space(Bounded()), {"limit": 4, "lanes": 2})
    assert first.doubled == 4  # evaluated on a configuration, reading ``lanes`` only
    base = commit(first, {"depth": 3})  # which keeps it
    report = base.try_with_choices({Bounded.limit: 1})
    assert not report.accepted
    outcomes = {(item.owner, item.status, item.requested) for item in report.outcomes}
    assert outcomes == {
        ("limit", "admissible", True),
        ("lanes", "refused", False),
        ("depth", "refused", False),  # ``doubled`` is not inherited: ``lanes`` refuses it
    }
    kept = base.try_with_choices({Bounded.limit: 2})
    assert kept.accepted and kept.instance.doubled == 4


def test_a_retained_admission_the_change_does_not_reach_is_reused_with_what_read_it() -> None:
    calls: list[str] = []

    def within(owner: str) -> Any:
        def accepts(*, candidate: int, limit: int) -> bool:
            calls.append(owner)
            return candidate <= limit

        return accepts

    class Fold(Space):
        width: int = Param()
        lanes: int = Decision(domain=domain(accepts=within("lanes"), limit=width))

        @derived
        def cycles(self) -> int:
            calls.append("cycles")
            return self.width // self.lanes

        # Its admission reads ``cycles``, which reads the retained ``lanes``.
        depth: int = Decision(domain=domain(accepts=within("depth"), limit=cycles))
        finish: str = Decision(values=("matte", "gloss"))

    base = commit(design_space(Fold(width=8)), {"lanes": 2, "depth": 4})
    calls.clear()
    glossy = commit(base, {"finish": "gloss"})
    assert calls == [] and glossy.depth == 4
    deeper = commit(glossy, {"depth": 3})
    assert calls == ["depth"]  # ``cycles`` is inherited: it read ``lanes`` only
    calls.clear()
    wider = commit(deeper, {"lanes": 4, "depth": 2})
    assert sorted(calls) == ["cycles", "depth", "lanes"]
    # A refusal reads the same whether the trial reused ``lanes`` or admitted it again.
    reused = deeper.try_with_choices({Fold.depth: 5})
    admitted = wider.try_with_choices({Fold.lanes: 2, Fold.depth: 5})
    refusals = [
        {item.owner: item.result for item in report.outcomes if item.status == "refused"}
        for report in (reused, admitted)
    ]
    assert refusals[0] == refusals[1] and list(refusals[0]) == ["depth"]
    fresh = commit(design_space(Fold(width=8)), {"lanes": 2, "depth": 3, "finish": "gloss"})
    assert inspection.explain(deeper, Fold.depth) == inspection.explain(fresh, Fold.depth)


def test_a_successor_answers_as_a_configuration_built_from_nothing() -> None:
    class Two(Space):
        width: int = Param()
        left: Packed | Stub = Decision(CORES, width=width)
        right: Packed | Stub = Decision(CORES, width=width)

    def answers(point: Any) -> dict[str, object]:
        return {
            "forced": forced(point),
            "viable": {item.key: item.cases for item in inspection.viable(point)},
            "left": inspection.explain(point, Two.left),
        }

    base = commit(design_space(Two(width=8)), {"left": "packed"})
    answers(base)
    point = commit(base, {"left.packed.pe": 4, "right": "stub"})
    fresh = commit(design_space(Two(width=8)), {"left": "packed", "right": "stub"})
    fresh = commit(fresh, {"left.packed.pe": 4})
    assert answers(point) == answers(fresh)
    assert isinstance(point.left, Packed) and point.left.pe == 4


def test_what_read_a_forced_decision_is_kept_unless_the_change_reaches_its_verdict() -> None:
    calls: list[str] = []

    class Odd(Space):
        lanes: int = Decision(values=(3, 128))
        finish: str = Decision(values=("matte", "gloss"))
        # Three lanes refuse the stub core (odd), 128 the packed core (over 64): its
        # verdict reads ``lanes`` and nothing of ``finish``.
        compute: Packed | Stub = Decision(CORES, width=lanes)

        @derived
        def kind(self) -> str:
            calls.append("kind")
            return type(self.compute).__name__  # reads the forced selector

    base = commit(design_space(Odd()), {"lanes": 128})
    assert forced(base) == {"compute": "stub"} and base.kind == "Stub"
    calls.clear()
    glossy = commit(base, {"finish": "gloss"})
    assert glossy.kind == "Stub" and calls == []  # outside the reach: kept
    narrow = commit(base, {"lanes": 3})
    assert narrow.kind == "Packed" and calls == ["kind"]  # inside it: derived again
    fresh = commit(design_space(Odd()), {"lanes": 3})
    assert forced(narrow) == forced(fresh) == {"compute": "packed"}
    assert inspection.explain(narrow, Odd.kind) == inspection.explain(fresh, Odd.kind)


def test_an_admission_that_read_a_forced_decision_is_reused_unless_the_change_reaches_it() -> None:
    calls: list[str] = []

    def within(owner: str) -> Any:
        def accepts(*, candidate: int, limit: int) -> bool:
            calls.append(owner)
            return candidate <= limit

        return accepts

    class Fit(Space):
        lanes: int = Decision(values=(3, 128))
        finish: str = Decision(values=("matte", "gloss"))
        compute: Packed | Stub = Decision(CORES, width=lanes)

        @derived
        def rows(self) -> int:
            calls.append("rows")
            return 2 if isinstance(self.compute, Stub) else 1  # reads the forced selector

        # Its admission reads ``rows``, which reads the forced ``compute``.
        depth: int = Decision(domain=domain(accepts=within("depth"), limit=rows))

    first = commit(design_space(Fit()), {"lanes": 128})
    assert forced(first) == {"compute": "stub"}  # found, so the next change is bounded by it
    base = commit(first, {"depth": 1})
    assert forced(base) == {"compute": "stub"} and base.depth == 1
    calls.clear()
    glossy = commit(base, {"finish": "gloss"})
    # The change does not reach ``compute``'s verdict: ``depth``'s admission is kept.
    assert glossy.depth == 1 and calls == []
    narrow = commit(base, {"lanes": 3})
    # It does: ``depth`` is admitted again, and ``rows`` derived again under it.
    assert narrow.depth == 1 and forced(narrow) == {"compute": "packed"}
    assert calls and set(calls) == {"depth", "rows"}
    fresh = commit(commit(design_space(Fit()), {"lanes": 3}), {"depth": 1})
    assert inspection.explain(narrow, Fit.depth) == inspection.explain(fresh, Fit.depth)
    assert forced(narrow) == forced(fresh)


def test_an_admission_that_assumed_a_forced_value_is_reused_only_while_that_value_stands() -> None:
    calls: list[str] = []

    def within(*, candidate: int, limit: int) -> bool:
        calls.append("depth")
        return candidate <= limit

    class Fit(Space):
        lanes: int = Decision(values=(2, 3, 100, 128))
        finish: str = Decision(values=("matte", "gloss"))
        # Two lanes leave both cores viable, three force packed, 100 and 128 force stub.
        compute: Packed | Stub = Decision(CORES, width=lanes)

        @derived
        def rows(self) -> int:
            return 2 if isinstance(self.compute, Stub) else 1  # reads the forced selector

        depth: int = Decision(domain=domain(accepts=within, limit=rows))

    first = commit(design_space(Fit()), {"lanes": 128})
    assert forced(first) == {"compute": "stub"}
    base = commit(first, {"depth": 1})
    assert forced(base) == {"compute": "stub"}

    # The change reaches ``compute``'s verdict, and the successor forces the same core:
    # the admission of ``depth``, taken under the base's forced stub, stands.
    same = commit(base, {"lanes": 100})
    assert forced(same) == {"compute": "stub"}
    calls.clear()
    glossy = commit(same, {"finish": "gloss"})
    assert calls == [] and glossy.depth == 1  # reused, its callback not called again
    built = recomputed(glossy)
    assert inspection.explain(glossy, Fit.depth) == inspection.explain(built, Fit.depth)

    # Here the successor forces the packed core: the stub the admission assumed is gone.
    # ``depth`` of 2 was admitted under the base's stub (two rows) and stands, as a
    # configuration built from nothing admits it. A change that does not reach ``compute``
    # admits it again, under the packed core (one row), and refuses it, as that
    # configuration built from nothing does.
    two = commit(first, {"depth": 2})
    packed = commit(two, {"lanes": 3})
    assert forced(packed) == {"compute": "packed"} and packed.depth == 2
    calls.clear()
    report = packed.try_with_choices({Fit.finish: "gloss"})
    assert calls == ["depth"]  # admitted again
    want = recomputed(packed).try_with_choices({Fit.finish: "gloss"})
    assert not report.accepted and not want.accepted
    refusals = [
        {item.owner: item.result for item in each.outcomes if item.status == "refused"}
        for each in (report, want)
    ]
    assert refusals[0] == refusals[1] and list(refusals[0]) == ["depth"]
