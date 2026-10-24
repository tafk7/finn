# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN's toy Spaces the oracle walks (``TOYS``), shared by its tests and the dump tool's
``toys`` target: the house (``core.space._toys_support``), the pipeline
(``support.space_pipeline``), a chain of stages (``support.space_cores``) and the boundary
(``Boundary``), each opened, with the values a walk may give its root facts and the change
the dump makes on it.

The boundary is the shape of S1's boundary (the collapse tests' ``Outer``,
``core.space.test_collapse``): a formal read from outside its scope while that scope is
absent. ``outside``'s width is bound to ``guarded.leaf.width``, and ``guarded`` is placed
only when ``enabled``; with ``enabled`` false, the answer for ``outside.width`` is
inapplicable (a formal bound by reference answers within its scope).

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
    design_space,
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
}
"""FINN's toys, by name."""

__all__ = [
    "TOYS",
    "Boundary",
    "BoundaryLeaf",
    "BoundaryMiddle",
    "Toy",
]
