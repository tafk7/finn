# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One reason reached by two routes is reported once.

Explicit required inputs and aliases may reach the same finding more than once. Findings are frozen
values, so identical ones are kept once where answers are merged; findings that
differ in anything, their owner included, are all kept.
"""

from __future__ import annotations

from finn.core.reporting.findings import Finding, FindingKind
from finn.core.space import (
    Available,
    Decision,
    Param,
    Space,
    constraint,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.core.space.results import (
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_view,
    merged_findings,
)


def blocker(owner: str, message: str = "no choice has set this decision") -> Finding:
    return Finding(FindingKind.BLOCKER, "decision-unassigned", owner, message)


def refusal(owner: str) -> Finding:
    return Finding(FindingKind.REJECTION, "too-large", owner, "area 50 exceeds 30")


def test_merged_findings_keeps_each_identical_finding_once_and_every_distinct_one() -> None:
    same = blocker("kitchen.finish")
    other_owner = blocker("dining.finish")
    other_message = blocker("kitchen.finish", "choose a finish")
    merged = merged_findings(
        (
            Unresolved((same, other_owner)),
            Unresolved((same,)),
            Unresolved(
                (
                    Finding(
                        FindingKind.BLOCKER,
                        "decision-unassigned",
                        "kitchen.finish",
                        "no choice has set this decision",
                    ),
                )
            ),
            Unresolved((other_message,)),
        )
    )
    # First-seen order (each answer already orders its own findings); an equal
    # finding built separately is the same reason.
    assert merged == (other_owner, same, other_message)
    # Distinct causes make distinct findings, even under one owner and code.
    caused = Finding(FindingKind.REJECTION, "blocked", "total", "blocked", causes=(same,))
    uncaused = Finding(FindingKind.REJECTION, "blocked", "total", "blocked")
    assert merged_findings((Rejected((caused,)), Rejected((uncaused, caused)))) == (
        caused,
        uncaused,
    )


def test_the_view_reducer_preserves_its_local_findings() -> None:
    kitchen = blocker("kitchen.finish")
    waiting: ViewAssessment[int] = assess_view(Unresolved((kitchen,)), validity=Available(True))
    assert waiting.result == Unresolved((kitchen,))
    refused: ViewAssessment[int] = assess_view(
        Rejected((refusal("kitchen.small_enough"),)), validity=Available(True)
    )
    assert refused.result == Rejected((refusal("kitchen.small_enough"),))


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def small_enough(self) -> bool | Rejected:
        return True if self.area <= 30 else reject("too-large", f"area {self.area} exceeds 30")

    @view
    def cost(self) -> int:
        return self.area * self.finish


class Hall(Space):
    kitchen = Room(area=12)
    dining = Room(area=16)

    @view
    def total(self) -> int:
        return self.kitchen.cost + self.dining.cost


def test_a_waiting_view_lists_its_outputs_blocker_once() -> None:
    base = design_space(Hall())
    waiting = base.inspect(Hall.total).result
    assert isinstance(waiting, Unresolved)
    # The Space is valid (open choices are no pieces); the output halts at its first read.
    assert [(f.owner, f.code) for f in waiting.findings] == [
        ("kitchen.finish", "decision-unassigned"),
    ]
    half = base.with_choices({Hall.kitchen.finish: 2}).inspect(Hall.total).result
    assert isinstance(half, Unresolved)
    assert [f.owner for f in half.findings] == ["dining.finish"]


def test_a_refusal_through_output_and_validity_is_reported_once_per_owner() -> None:
    class Big(Space):
        kitchen = Room(area=50)
        dining = Room(area=40)

        @view
        def total(self) -> int:
            return self.kitchen.cost + self.dining.cost

    point = design_space(Big()).with_choices({Big.kitchen.finish: 1, Big.dining.finish: 1})
    assert isinstance(point.inspect(Big.total).result, Rejected)
    assert point.inspect(Big.total).output_result is None
    refused = inspection.validity(point).result
    assert isinstance(refused, Rejected)
    assert [(f.owner, f.message) for f in refused.findings] == [
        ("dining.small_enough", "area 40 exceeds 30"),
        ("kitchen.small_enough", "area 50 exceeds 30"),
    ]


def test_a_computation_reading_two_aliases_of_one_blocked_value_lists_it_once() -> None:
    class Twice(Space):
        width: int = Decision(values=(2, 4))
        left = Room(area=width)
        right = Room(area=width)

        @derived(first=left.area, second=right.area)
        def combined(*, first: int, second: int) -> int:
            return first + second

        both = view(combined)  # a derived output with two arguments

    # Both arguments collapse onto the one open Decision; the output merges them.
    waiting = design_space(Twice()).inspect(Twice.both).output_result
    assert isinstance(waiting, Unresolved)
    assert [(f.owner, f.code) for f in waiting.findings] == [("width", "decision-unassigned")]
