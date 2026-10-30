# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Plain-Python source freezing, exploration and rebinding, without graph APIs."""

from __future__ import annotations

from finn.core.space import (
    Decision,
    Param,
    Space,
    design_space,
    divisors_of,
)


def test_freeze_explore_and_rebind() -> None:
    class Example(Space):
        extent: int = Param()
        factor: int = Decision(domain=divisors_of(extent))
        buffers: int = Decision(domain=(1, 2))

    facts = {"extent": 12}
    base = design_space(Example(extent=facts["extent"]))
    facts["extent"] = 10
    initial = base.with_choices(factor=3).with_choices(buffers=1)
    assert initial.extent == 12
    revised = initial.with_choices(initial.field(Example.buffers).clear(), factor=4)
    assert revised.factor == 4
    assert initial.factor == 3 and initial.buffers == 1
    rebound = design_space(Example(extent=facts["extent"]))
    refused = rebound.try_with_choices(factor=4)
    assert not refused.accepted and refused.instance is rebound
