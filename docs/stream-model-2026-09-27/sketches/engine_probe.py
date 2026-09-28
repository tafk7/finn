# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Engine probe for DESIGN.md §8.4: can the proposed Stream be built on today's engine?

Toy families only (no kernels). Checks that:

1. ends present their own value through a per-input export and read only the
   stream's tensor fact (no cycle);
2. the stream derives a plan from ``Users(KEY)``;
3. a Decision over adapter candidate nodes can bind a candidate formal to that
   derived plan, and a candidate refuses itself when it cannot realize it;
4. a folding Decision's domain can depend on a value read through a reference
   input (the stream's tensor extent).

Run: PYTHONPATH=src python docs/stream-model-2026-09-27/sketches/engine_probe.py
"""

from __future__ import annotations

from finn.core.space import (
    Available,
    Decision,
    Param,
    Rejected,
    Space,
    Users,
    View,
    ViewKey,
    constraint,
    default_semantics,
    derived,
    design_space,
    domain,
    reject,
    view,
)

PRESENT = ViewKey("present", default_semantics(int))  # an end's lanes, standing in for a form


class Direct(Space):
    plan: str = Param()

    @constraint
    def applicable(self) -> bool | Rejected:
        return True if self.plan == "identity" else reject("direct", f"needs {self.plan}")

    @view(semantics=default_semantics(str), requires=(applicable,))
    def stage(self) -> str:
        return "wires"


class Dwc(Space):
    plan: str = Param()

    @constraint
    def applicable(self) -> bool | Rejected:
        return True if self.plan == "width" else reject("dwc", f"cannot realize {self.plan}")

    @view(semantics=default_semantics(str), requires=(applicable,))
    def stage(self) -> str:
        return "dwc"


class Stream(Space):
    extent: int = Param()  # the tensor fact
    ends = Users(PRESENT)

    @derived(semantics=default_semantics(str))
    def plan(self) -> str:
        lanes = [end.value for end in self.ends]
        return "identity" if len(set(lanes)) == 1 else "width"

    realization: Direct | Dwc = Decision(values={"direct": Direct(plan=plan), "dwc": Dwc(plan=plan)})
    stage = View(realization.stage)


class Producer(Space):
    out: Stream = Param()
    pe: int = Decision(
        domain=domain(
            accepts=lambda *, candidate, extent: extent % candidate == 0,
            candidates=lambda *, extent: [d for d in range(1, extent + 1) if extent % d == 0],
            extent=out.extent,
        )
    )

    @view(semantics=default_semantics(int))
    def lanes(self) -> int:
        return self.pe

    exports = {PRESENT: {out: lanes}}


class Consumer(Space):
    inp: Stream = Param()
    pe: int = Param()

    @view(semantics=default_semantics(int))
    def lanes(self) -> int:
        _ = self.inp.extent  # reads the tensor fact only
        return self.pe

    exports = {PRESENT: {inp: lanes}}


class Composite(Space):
    s = Stream(extent=8)
    a = Producer(out=s)
    b = Consumer(inp=s, pe=2)


if __name__ == "__main__":
    space = design_space(Composite())
    same = space.with_choices({Composite.a.pe: 2, Composite.s.realization: "direct"})
    assert same.s.plan == "identity" and same.s.stage == "wires"
    wider = space.with_choices({Composite.a.pe: 4, Composite.s.realization: "direct"})
    assert isinstance(wider.s.query(Stream.stage), Rejected)
    adapted = space.with_choices({Composite.a.pe: 4, Composite.s.realization: "dwc"})
    assert isinstance(adapted.s.query(Stream.stage), Available) and adapted.s.stage == "dwc"
    refused = space.try_with_choices({Composite.a.pe: 3, Composite.s.realization: "direct"})
    assert not refused.accepted  # 3 does not divide the stream's extent 8
    print("engine_probe: all checks passed")
