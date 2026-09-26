# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""How much work one local edit costs in a graph of N design spaces.

A pipeline of N stages is built as data; each stage's input is bound to the
previous stage's output. After a full evaluation, one decision of the *last*
stage is changed and the last width is read again. Every snapshot has its own
cache, so the unchanged prefix is re-evaluated.
Run from the FINN checkout on spike/space-design-graph with PYTHONPATH=src:tests.
"""

import time

from finn.core.space import Bind, Decision, Param, ScopeBuilder, Space, Subspace, inspection, view


class Stage(Space):
    width_in = Param(int)
    growth = Decision(int, values=(0, 1, 2))

    @view
    def width_out(self) -> int:
        return self.width_in + self.growth


def pipeline(count: int) -> type[Space]:
    builder = ScopeBuilder(Space, name=f"Probe{count}")
    stages = [builder.add("s0", Subspace(Stage, width_in=4))]
    for index in range(1, count):
        stages.append(builder.add(f"s{index}", Subspace(Stage)))
        builder.add(
            f"e{index}",
            Bind(stages[index].ref(Stage.width_in), stages[index - 1].accepted(Stage.width_out)),
        )
    return builder.finish()


for count in (50, 200, 800):
    family = pipeline(count)
    start = time.perf_counter()
    point = family()
    prepared = time.perf_counter() - start
    handles = {item.key: item.reference for item in inspection.decisions(point)}
    point = point.with_choices(*(point.field(handles[f"s{i}.growth"]).change(1) for i in range(count)))
    last = getattr(point, f"s{count - 1}")
    start = time.perf_counter()
    assert last.width_out() == 4 + count
    full = time.perf_counter() - start
    before = point._state.work.callback_starts  # type: ignore[attr-defined]
    edited = point.with_choices(point.field(handles[f"s{count - 1}.growth"]).change(2))
    start = time.perf_counter()
    assert getattr(edited, f"s{count - 1}").width_out() == 5 + count
    local = time.perf_counter() - start
    calls = edited._state.work.callback_starts  # type: ignore[attr-defined]
    print(
        f"N={count:4}  prepare {prepared * 1e3:8.1f} ms  full read {full * 1e3:7.1f} ms  "
        f"read after one local edit {local * 1e3:7.1f} ms  callbacks re-run {calls} "
        f"(first snapshot: {before})"
    )
