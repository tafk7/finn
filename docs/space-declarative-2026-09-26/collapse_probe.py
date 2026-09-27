# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""The effect of collapsing forwarding chains: node counts and evaluated work.

For MVAU (two configurations, reading ``structure``), the scale probe's
pipeline (reading the last stage's width) and a pipeline of composites that
forward their input to an inner stage, the same declaration is compiled with
and without collapse. Node and scope counts are identical by construction;
what changes is how many edges name an alias and how many nodes a read
evaluates (timing is the scale probe's job). Run from the FINN checkout with
PYTHONPATH=src:tests:deps/qonnx/src.
"""

from core.space._collapse_support import counts, open_space
from kernels.test_mvau_collapse import _open

from finn.core.space import Decision, Param, Space, accepted, composite, inspection, view


def mvau(case: str) -> None:
    def read(point: object) -> object:
        return point.structure.query()  # type: ignore[attr-defined]

    for collapsed in (False, True):
        result = counts(_open(case, collapsed=collapsed), read)
        label = "after " if collapsed else "before"
        print(f"MVAU {case:12} {label}  {result.row()}")


class Stage(Space):
    width_in: int = Param()
    growth: int = Decision(values=(0, 1, 2))

    @view
    def width_out(self) -> int:
        return self.width_in + self.growth


class Wrapped(Space):
    """A composite forwarding its input to an inner stage: a two-alias chain."""

    width_in: int = Param()
    inner = Stage(width_in=width_in)

    @view
    def width_out(self) -> int:
        return self.inner.width_out()


def pipeline(count: int, *, wrapped: bool) -> type[Space]:
    kind = Wrapped if wrapped else Stage
    stages = [kind(width_in=4), *(kind() for _ in range(1, count))]
    for previous, current in zip(stages, stages[1:]):
        current.width_in = accepted(previous.width_out)
    name = f"{'Wrapped' if wrapped else 'Probe'}{count}"
    return composite(name, {f"s{index}": stage for index, stage in enumerate(stages)})


def probe(count: int, *, wrapped: bool) -> None:
    family = pipeline(count, wrapped=wrapped)
    key = "inner.growth" if wrapped else "growth"

    def read(point: Space) -> object:
        return getattr(point, f"s{count - 1}").width_out()

    for collapsed in (False, True):
        point = open_space(family(), collapsed=collapsed)
        handles = {item.key: item.reference for item in inspection.decisions(point)}
        point = point.with_choices({handles[f"s{i}.{key}"]: 1 for i in range(count)})
        result = counts(point, read)
        label = "after " if collapsed else "before"
        name = "composites" if wrapped else "pipeline  "
        print(f"{name} N={count:4} {label}  {result.row()}")


mvau("external")
mvau("cyclic-fifo")
for count in (50, 200, 800):
    probe(count, wrapped=False)
for count in (50, 200):
    probe(count, wrapped=True)
