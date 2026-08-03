############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DesignSpace`` — a design space as data: axes + derived + predicates.

The design space computes the dependency order once at construction: a single
topological sort over one DAG of axes *and* deriveds. Edges are "node A's
guard/domain/compute reads node B", declared via each node's ``deps`` (closures
cannot be introspected reliably). Axes are fed first, so — since an axis only ever
reads other axes, never a derived — the axis suffix of the order is identical to
sorting axes alone; deriveds follow, ordered after the axes and other deriveds they
read. A cycle or an unknown/duplicate name is an authoring error, caught here once
(design-space-model.md §2).
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
    _ordered_axes: tuple[Axis, ...] = field(default=(), init=False, repr=False)
    _ordered_derived: tuple[Derived, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        order = _topo_sort(self.axes, self.derived)
        object.__setattr__(
            self, "_ordered_axes", tuple(n for n in order if isinstance(n, Axis))
        )
        object.__setattr__(
            self, "_ordered_derived", tuple(n for n in order if isinstance(n, Derived))
        )

    @property
    def axis_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.axes)

    def ordered_axes(self) -> tuple[Axis, ...]:
        """Axes in dependency order (each axis after every axis it reads)."""
        return self._ordered_axes

    def ordered_derived(self) -> tuple[Derived, ...]:
        """Deriveds in dependency order (each derived after every axis/derived it reads)."""
        return self._ordered_derived


def _topo_sort(axes, derived=()):
    """Topologically order axes and deriveds as one DAG.

    Axes are fed first, so — because an axis only ever reads other axes, never a
    derived — the axis members come out in exactly the order a standalone axis sort
    would produce (declaration order preserved among independent axes). Deriveds
    follow, each after the axes and deriveds its ``deps`` name. Duplicate names,
    unknown deps, and cycles all raise :class:`DesignSpaceError`.
    """
    nodes = list(axes) + list(derived)
    by_name: dict[str, object] = {}
    for n in nodes:
        if n.name in by_name:
            raise DesignSpaceError(f"Duplicate name: {n.name!r}")
        by_name[n.name] = n

    for n in nodes:
        for dep in n.deps:
            if dep not in by_name:
                raise DesignSpaceError(
                    f"{n.name!r} declares dependency on unknown name {dep!r}"
                )

    # Deterministic depth-first topological sort with cycle detection. Input
    # order is preserved among independent nodes so the schema reads predictably.
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n.name: WHITE for n in nodes}
    order: list[object] = []

    def visit(name: str, stack: list[str]):
        color[name] = GREY
        stack.append(name)
        for dep in sorted(by_name[name].deps):
            if color[dep] == GREY:
                cycle = " -> ".join(stack[stack.index(dep):] + [dep])
                raise DesignSpaceError(f"Dependency cycle: {cycle}")
            if color[dep] == WHITE:
                visit(dep, stack)
        stack.pop()
        color[name] = BLACK
        order.append(by_name[name])

    for n in nodes:
        if color[n.name] == WHITE:
            visit(n.name, [])
    return order
