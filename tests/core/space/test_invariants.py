# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Concrete stability laws for fixed facts and compatible additional choices."""

from finn.core.space import Available, Decision, Param, Space, Unresolved, constraint, derived, view


def test_settled_values_and_acceptance_survive_unrelated_added_choices() -> None:
    calls: list[int] = []

    class Family(Space):
        extent = Param(int)
        lanes = Decision(int, values=(1, 2))
        style = Decision(str, values=("small", "fast"))

        @derived
        def cycles(self) -> int:
            value = self.extent // self.lanes
            calls.append(value)
            return value

        @constraint
        def supported(self) -> bool:
            return self.lanes <= self.extent

        @view(constraints=(supported,))
        def physical(self) -> int:
            return self.cycles

    base = Family(extent=8)
    assert isinstance(base.query(Family.cycles), Unresolved)
    assert calls == []
    first = base.with_choices(lanes=2)
    before = first.physical.inspect()
    second = first.with_choices(style="fast")
    assert second.query(Family.cycles) == Available(4)
    assert second.physical.inspect() == before
    assert first.cycles == 4 and calls == [4, 4]
    assert isinstance(first.query(Family.style), Unresolved)
    # Replacement is allowed to change an already-settled computation.
    assert second.with_choices(lanes=1).cycles == 8
