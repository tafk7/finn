# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Plain-Python source freezing, capture and restore, without graph APIs."""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Space,
    design_space,
    divisors_of,
    selections,
)


def test_freeze_restore_explore_capture_and_rebind() -> None:
    class Example(Space):
        extent: int = Param()
        factor: int = Decision(domain=divisors_of(extent))
        buffers: int = Decision(values=(1, 2))

    facts = {"extent": 12}
    base = design_space(Example(extent=facts["extent"]))
    facts["extent"] = 10
    initial = base.with_choices(factor=3).with_choices(buffers=1)
    captured = selections.capture(initial)
    checkpoint = selections.restore(base, captured)
    assert checkpoint.accepted and checkpoint.instance.extent == 12
    revised = checkpoint.instance.with_choices(
        checkpoint.instance.field(Example.buffers).clear(),
        factor=4,
    )
    alternative = selections.capture(revised)
    explored = selections.restore(base, alternative)
    assert explored.accepted and explored.instance.factor == 4
    assert initial.factor == 3 and initial.buffers == 1
    rebound = design_space(Example(extent=facts["extent"]))
    refused = selections.restore(rebound, selections.capture(explored.instance))
    assert not refused.accepted and refused.instance is rebound
    assert selections.capture(rebound).keys == ()
