# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""One reason reached by two routes is reported once.

A view that reads a value and also requires it, a refusal reached through a
view's output and an obligation, and a computation reading two aliases of one
blocked value all merge the same finding more than once. Findings are frozen
values, so identical ones are kept once where answers are merged; findings that
differ in anything, their owner included, are all kept.
"""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Space,
    View,
    constraint,
    design_space,
    reject,
    view,
)
from finn.core.space.results import (
    Finding,
    FindingKind,
    Rejected,
    Unresolved,
    ViewAssessment,
    assess_view,
    merged_findings,
)


def blocker(owner: str, message: str = "decision requires a commitment") -> Finding:
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
                        "decision requires a commitment",
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


def test_the_view_reducer_reports_a_reason_from_output_and_obligation_once() -> None:
    kitchen, dining = blocker("kitchen.finish"), blocker("dining.finish")
    waiting: ViewAssessment[int] = assess_view(
        Unresolved((kitchen,)),
        owner="total",
        constraints={"kitchen.cost": Unresolved((kitchen,)), "dining.cost": Unresolved((dining,))},
    )
    assert waiting.accepted_result == Unresolved((dining, kitchen))
    # Obligation results keep their own findings; only the merged result de-duplicates.
    assert waiting.constraints.results["kitchen.cost"] == Unresolved((kitchen,))

    refused: ViewAssessment[int] = assess_view(
        Rejected((refusal("kitchen.small_enough"),)),
        owner="total",
        constraints={
            "kitchen.cost": Rejected((refusal("kitchen.small_enough"),)),
            "dining.cost": Rejected((refusal("dining.small_enough"),)),
        },
    )
    assert refused.accepted_result == Rejected(
        (refusal("dining.small_enough"), refusal("kitchen.small_enough"))
    )


class Room(Space):
    area: int = Param()
    finish: int = Decision(values=(1, 2, 3))

    @constraint
    def small_enough(self) -> bool | Rejected:
        return True if self.area <= 30 else reject("too-large", f"area {self.area} exceeds 30")

    @view(requires=(small_enough,))
    def cost(self) -> int:
        return self.area * self.finish


class Hall(Space):
    kitchen = Room(area=12)
    dining = Room(area=16)

    @view(requires=(kitchen.cost, dining.cost))  # read and required
    def total(self) -> int:
        return self.kitchen.cost + self.dining.cost


def test_a_view_that_reads_and_requires_a_value_lists_its_blocker_once() -> None:
    base = design_space(Hall())
    waiting = base.inspect(Hall.total).accepted_result
    assert isinstance(waiting, Unresolved)
    assert [(f.owner, f.code) for f in waiting.findings] == [
        ("dining.finish", "decision-unassigned"),
        ("kitchen.finish", "decision-unassigned"),
    ]
    half = base.with_choices({Hall.kitchen.finish: 2}).inspect(Hall.total).accepted_result
    assert isinstance(half, Unresolved)
    assert [f.owner for f in half.findings] == ["dining.finish"]


def test_a_refusal_through_output_and_obligation_is_reported_once_per_owner() -> None:
    class Big(Space):
        kitchen = Room(area=50)
        dining = Room(area=40)

        @view(requires=(kitchen.cost, dining.cost))
        def total(self) -> int:
            return self.kitchen.cost + self.dining.cost

    point = design_space(Big()).with_choices({Big.kitchen.finish: 1, Big.dining.finish: 1})
    refused = point.inspect(Big.total).accepted_result
    assert isinstance(refused, Rejected)
    # The output halts at kitchen; both obligations refuse: two owners, once each.
    assert [(f.owner, f.message) for f in refused.findings] == [
        ("dining.small_enough", "area 40 exceeds 30"),
        ("kitchen.small_enough", "area 50 exceeds 30"),
    ]


def test_a_computation_reading_two_aliases_of_one_blocked_value_lists_it_once() -> None:
    class Twice(Space):
        width: int = Decision(values=(2, 4))
        left = Room(area=width)
        right = Room(area=width)

        both = View(left.area + right.area)  # a derived output with two arguments

    # Both arguments collapse onto the one open Decision; the output merges them.
    waiting = design_space(Twice()).inspect(Twice.both).output_result
    assert isinstance(waiting, Unresolved)
    assert [(f.owner, f.code) for f in waiting.findings] == [("width", "decision-unassigned")]
