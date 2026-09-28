# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Stub kernels for case 5 and the rule tests: two cores with disjoint folds.

``ProtoKernel`` stands in for the K1 ``Kernel`` base: it declares a
``required()`` member (``schedule``) and a shared fact (``width``). The packed
stub owns ``pe`` and ``simd``; the other stub owns ``rows``. Each refuses what
it cannot build through its ``admission`` group (F9's one admission group).
"""

from __future__ import annotations

from refined import RequiredMeta, required

from finn.core.space import (
    ConstraintGroup,
    Decision,
    Param,
    Rejected,
    Space,
    constraint,
    derived,
    divisors_of,
    reject,
    view,
)


class ProtoKernel(Space, metaclass=RequiredMeta):
    """The stand-in base: every kernel defines its schedule; the width is a fact."""

    schedule: str = required()
    width: int = Param()


class PackedCore(ProtoKernel):
    pe: int = Decision(domain=divisors_of(ProtoKernel.width))
    simd: int = Decision(values=(1, 2))
    narrow_weights: bool = Param(default=False)

    @derived
    def schedule(self) -> str:
        return f"pe{self.pe}.simd{self.simd}"

    @view
    def cycles(self) -> int:
        return self.width // self.pe

    @constraint
    def fits(self) -> bool | Rejected:
        if self.width > 64:
            return reject("packed-width", "the packed stub takes at most 64 lanes")
        return True

    admission = ConstraintGroup(fits)


class StubCore(ProtoKernel):
    rows: int = Decision(values=(1, 2, 4))

    @derived
    def schedule(self) -> str:
        return f"rows{self.rows}"

    @view
    def cycles(self) -> int:
        return self.width * self.rows

    @constraint
    def even(self) -> bool | Rejected:
        if self.width % 2:
            return reject("stub-width", "the stub takes an even width")
        return True

    admission = ConstraintGroup(even)


class WideCore(ProtoKernel):
    """A third core: cycles typed ``str`` (a common member of another type)."""

    @derived
    def schedule(self) -> str:
        return "wide"

    @view
    def cycles(self) -> str:
        return "one"


class Unfinished(ProtoKernel):
    """Leaves ``schedule`` unmet: it cannot be placed."""

    @view
    def cycles(self) -> int:
        return 1


class Narrow(Space):
    """A plain family lacking ``width``: a candidate that cannot take the shared binding."""

    depth: int = Param(default=2)

    @view
    def cycles(self) -> int:
        return self.depth


__all__ = ["Narrow", "PackedCore", "ProtoKernel", "StubCore", "Unfinished", "WideCore"]
