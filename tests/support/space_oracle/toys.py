# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN's toy Spaces the oracle walks (``TOYS``), shared by its tests and the dump tool's
``toys`` target: the house (``core.space._toys_support``), the pipeline
(``support.space_pipeline``), a chain of stages (``support.space_cores``), the boundary
(``Boundary``) and the three shapes of a missing fact, each opened, with the values a walk
may give its root facts and the change the dump makes on it.

The boundary is the shape of S1's boundary (the collapse tests' ``Outer``,
``core.space.test_collapse``): a formal read from outside its scope while that scope is
absent. ``outside``'s width is bound to ``guarded.leaf.width``, and ``guarded`` is placed
only when ``enabled``; with ``enabled`` false, the answer for ``outside.width`` is
inapplicable (a formal bound by reference answers within its scope).

**The missing-fact shapes** (``WaitingMembership``, ``AcceptedWaiting``, ``ThreeDeep``).
Each opens with its optional fact (``Param(required=False)``) left unsupplied, and a walk
gives it in a facts change. The engine reads such a fact unresolved, so what reads it
waits: a choice's membership, a choice set meanwhile (accepted, waiting), a rule three
scopes down. The reference model reads an optional fact left out as ``None`` (milestone
T4c) and no longer has these shapes, so they are kept here: the engine answers them so
until T4c.

Engine-only: this module imports ``finn.core.space`` and the engine's own toys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from core.space._toys_support import House

from finn.core.space import (
    Decision,
    Param,
    Space,
    constraint,
    design_space,
    divisors_of,
    domain,
    view,
)
from support.space_cores import chain as stages
from support.space_pipeline import Pipeline


class BoundaryLeaf(Space):
    width: int = Param()
    lanes: int = Decision(values=(1, 2))

    @view
    def bits(self) -> int:
        return self.width * self.lanes


class BoundaryMiddle(Space):
    width: int = Param()
    leaf = BoundaryLeaf(width=width)  # forwards the enclosing formal


class Boundary(Space):
    """A formal read from outside its scope: ``outside.width`` reads ``guarded.leaf.width``,
    present only when ``enabled``."""

    width: int = Param()
    enabled: bool = Decision(values=(False, True))
    middle = BoundaryMiddle(width=width)  # a chain: middle.leaf.width -> middle.width -> width
    guarded = BoundaryMiddle(width=width, when=enabled)
    outside = BoundaryLeaf(width=guarded.leaf.width)


def _within(*, candidate: int, limit: int) -> bool:
    return 0 < candidate <= limit


class WaitingMembership(Space):
    """Membership waiting on a fact: ``size``'s domain is membership-only over the optional
    fact ``limit``. With ``limit`` left out, ``size`` lists nothing and is no piece of
    validity; set, it waits on ``limit``, and so does ``fits``, which reads it. Given,
    ``limit`` checks the value it waited on."""

    limit: int = Param(required=False)
    size: int = Decision(domain=domain(accepts=_within, limit=limit))
    lanes: int = Decision(values=(1, 2))

    @constraint
    def fits(self) -> bool:
        return self.size * self.lanes <= 8


class AcceptedWaiting(Space):
    """A choice accepted while its membership waits: ``pe``'s list is the divisors of the
    optional fact ``width``. With ``width`` left out the list waits (an unknown piece of
    validity), and ``pe = 3`` is accepted, waiting on ``width``. Given, ``width`` keeps the
    value (3, 6) or drops it (8)."""

    width: int = Param(required=False)
    pe: int = Decision(domain=divisors_of(width))


class DeepLeaf(Space):
    width: int = Param()
    pe: int = Decision(values=(1, 2, 4))

    @constraint
    def fits(self) -> bool:
        return self.pe <= self.width


class DeepMiddle(Space):
    width: int = Param()
    leaf = DeepLeaf(width=width)


class DeepOuter(Space):
    width: int = Param()
    middle = DeepMiddle(width=width)


class ThreeDeep(Space):
    """A rule three scopes down waiting on a fact: ``outer.middle.leaf.fits`` reads the
    optional fact ``width``, forwarded down as a formal. With ``width`` left out and
    ``pe`` set, the rule, and the root's validity, wait on ``width``; given, the rule
    holds (8) or breaks with ``pe`` standing (2)."""

    width: int = Param(required=False)
    outer = DeepOuter(width=width)


@dataclass(frozen=True)
class Toy:
    """A toy opened (``make``), the values a walk may give its root facts (``facts``, by
    key), and the change the dump makes on it (``change``, by key)."""

    make: Callable[[], Space]
    facts: Mapping[str, Sequence[object]] = field(default_factory=dict)
    change: Mapping[str, object] = field(default_factory=dict)


_STAGES = stages(6)

TOYS: dict[str, Toy] = {
    "house": Toy(
        lambda: design_space(House(budget=100)), {"budget": (60, 100, 400)}, {"kitchen.finish": 2}
    ),
    "pipeline": Toy(
        lambda: design_space(Pipeline(width=10)), {"width": (6, 10, 12)}, {"u0.compute": "packed"}
    ),
    "stages": Toy(lambda: design_space(_STAGES()), {}, {"s0.size": 2}),
    "boundary": Toy(
        lambda: design_space(Boundary(width=3)), {"width": (3, 4, 8)}, {"enabled": False}
    ),
    "waiting-membership": Toy(
        lambda: design_space(WaitingMembership()), {"limit": (2, 4, 8)}, {"size": 3}
    ),
    "accepted-waiting": Toy(
        lambda: design_space(AcceptedWaiting()), {"width": (3, 6, 8)}, {"pe": 3}
    ),
    "three-deep": Toy(
        lambda: design_space(ThreeDeep()), {"width": (1, 2, 8)}, {"outer.middle.leaf.pe": 4}
    ),
}
"""FINN's toys, by name."""

__all__ = [
    "TOYS",
    "AcceptedWaiting",
    "Boundary",
    "BoundaryLeaf",
    "BoundaryMiddle",
    "DeepLeaf",
    "DeepMiddle",
    "DeepOuter",
    "ThreeDeep",
    "Toy",
    "WaitingMembership",
]
