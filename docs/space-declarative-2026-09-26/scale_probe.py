# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How much work one local edit costs in a graph of N design spaces.

The previous spike's probe (``space-design-graph-2026-09-26/scale_probe.py``),
ported to the declarative API: a pipeline of N stages is built as data (plain
Python nodes, joined by assignment in a loop and named by ``composite``); each
stage's input is assigned the previous stage's output. After a full
evaluation, one decision of the *last* stage is changed and the last width is
read again. Every snapshot has its own cache, so the unchanged prefix is
re-evaluated. Run from the FINN checkout with PYTHONPATH=src:tests.
"""

import time

from finn.core.space import (
    Decision,
    Param,
    Space,
    accepted,
    composite,
    design_space,
    inspection,
    view,
)


class Stage(Space):
    width_in: int = Param()
    growth: int = Decision(values=(0, 1, 2))

    @view
    def width_out(self) -> int:
        return self.width_in + self.growth


def pipeline(count: int) -> type[Space]:
    stages = [Stage(width_in=4), *(Stage() for _ in range(1, count))]
    for previous, current in zip(stages, stages[1:]):
        current.width_in = accepted(previous.width_out)
    return composite(f"Probe{count}", {f"s{index}": stage for index, stage in enumerate(stages)})


for count in (50, 200, 800):
    family = pipeline(count)
    start = time.perf_counter()
    point = design_space(family())
    prepared = time.perf_counter() - start
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices({handles[f"s{i}.growth"]: 1 for i in range(count)})
    last = getattr(point, f"s{count - 1}")
    start = time.perf_counter()
    assert last.width_out() == 4 + count
    full = time.perf_counter() - start
    before = point._state.work.callback_starts  # type: ignore[attr-defined]
    edited = point.with_choices({handles[f"s{count - 1}.growth"]: 2})
    start = time.perf_counter()
    assert getattr(edited, f"s{count - 1}").width_out() == 5 + count
    local = time.perf_counter() - start
    calls = edited._state.work.callback_starts  # type: ignore[attr-defined]
    print(
        f"N={count:4}  prepare {prepared * 1e3:8.1f} ms  full read {full * 1e3:7.1f} ms  "
        f"read after one local edit {local * 1e3:7.1f} ms  callbacks re-run {calls} "
        f"(first snapshot: {before})"
    )
