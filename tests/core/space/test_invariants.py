# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Concrete stability laws for fixed facts and compatible additional choices."""

from finn.core.space import (
    Available,
    Decision,
    Param,
    Space,
    Unresolved,
    constraint,
    derived,
    design_space,
    view,
)


def test_settled_values_and_acceptance_survive_unrelated_added_choices() -> None:
    calls: list[int] = []

    class Example(Space):
        extent: int = Param()
        lanes: int = Decision(values=(1, 2))
        style: str = Decision(values=("small", "fast"))

        @derived
        def cycles(self) -> int:
            value = self.extent // self.lanes
            calls.append(value)
            return value

        @constraint
        def supported(self) -> bool:
            return self.lanes <= self.extent

        @view(requires=(supported,))
        def physical(self) -> int:
            return self.cycles

    base = design_space(Example(extent=8))
    assert isinstance(base.query(Example.cycles), Unresolved)
    assert calls == []
    first = base.with_choices(lanes=2)
    before = first.inspect(Example.physical)
    second = first.with_choices(style="fast")
    assert second.query(Example.cycles) == Available(4)
    assert second.inspect(Example.physical) == before
    # The successor keeps what its base evaluated that read no Decision the change touched.
    assert first.cycles == 4 and calls == [4]
    assert isinstance(first.query(Example.style), Unresolved)
    # Replacement is allowed to change an already-settled computation.
    assert second.with_choices(lanes=1).cycles == 8
