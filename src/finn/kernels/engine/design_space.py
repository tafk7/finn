############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DesignSpace`` — a design space as data: axes + derived + predicates.

The design space computes the axis dependency order once at construction: a topological
sort over "axis A's guard/domain reads axis B" (declared via each axis's ``deps``,
since closures cannot be introspected reliably). A cycle is an authoring error, caught
here once (design-space-model.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .axis import Axis
from .derived import Derived
from .predicate import Predicate


class DesignSpaceError(ValueError):
    """Raised for a malformed schema (dependency cycle, unknown dep, dup name)."""


@dataclass(frozen=True)
class DesignSpace:
    axes: tuple[Axis, ...]
    derived: tuple[Derived, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    _order: tuple[Axis, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "_order", tuple(_topo_sort(self.axes)))

    @property
    def axis_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.axes)

    def ordered_axes(self) -> tuple[Axis, ...]:
        """Axes in dependency order (each axis after every axis it reads)."""
        return self._order


def _topo_sort(axes) -> list[Axis]:
    by_name: dict[str, Axis] = {}
    for a in axes:
        if a.name in by_name:
            raise DesignSpaceError(f"Duplicate axis name: {a.name!r}")
        by_name[a.name] = a

    for a in axes:
        for dep in a.deps:
            if dep not in by_name:
                raise DesignSpaceError(
                    f"Axis {a.name!r} declares dependency on unknown axis {dep!r}"
                )

    # Deterministic depth-first topological sort with cycle detection. Input
    # order is preserved among independent axes so the schema reads predictably.
    WHITE, GREY, BLACK = 0, 1, 2
    color = {a.name: WHITE for a in axes}
    order: list[Axis] = []

    def visit(name: str, stack: list[str]):
        color[name] = GREY
        stack.append(name)
        for dep in sorted(by_name[name].deps):
            if color[dep] == GREY:
                cycle = " -> ".join(stack[stack.index(dep):] + [dep])
                raise DesignSpaceError(f"Dependency cycle among axes: {cycle}")
            if color[dep] == WHITE:
                visit(dep, stack)
        stack.pop()
        color[name] = BLACK
        order.append(by_name[name])

    for a in axes:
        if color[a.name] == WHITE:
            visit(a.name, [])
    return order
