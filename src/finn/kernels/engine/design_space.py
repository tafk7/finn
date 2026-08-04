############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``DesignSpace`` — a design space as data: axes + derived + predicates.

The dependency order is a single topological sort over one DAG of axes *and* deriveds.
Edges are "node A's guard/domain/compute reads node B", declared via each node's ``deps``
(closures cannot be introspected reliably). Axes are fed first, so — since an axis only ever
reads other axes, never a derived — the axis suffix of the order is identical to sorting axes
alone; deriveds follow, ordered after the axes and other deriveds they read. A cycle or an
unknown/duplicate name is an authoring error (design-space-model.md §2).

**Validation is a whole-space property, run on the FINALIZED design space — never on an
assembly fragment.** ``compile`` builds the full space in stages (the compute pool first, then
each parameters sub-schema folded in); a compute derived may legitimately declare a dep on a
name a *later* fragment introduces (e.g. ``accDataType`` → ``parameters.weights.datatype``).
At fragment-construction time that dep is indistinguishable from a typo — both are "unknown
name" — so validating a fragment cannot be both strict (catch typos) and composition-tolerant
(allow cross-pool forward refs). We therefore relocate *when* validation fires, not *whether*:
construction is cheap and unvalidated; the topo-sort + dep/cycle/dup check runs ONCE, lazily,
the first time the order is needed (any ``resolve``, or an explicit :meth:`finalize`), by which
point every pool is present. A genuine typo still raises ``DesignSpaceError`` — just at
finalize rather than at the fragment that introduced it. Deterministic, still before any point
is produced.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        # No topo-sort here: this may be an assembly FRAGMENT (the compute pool before the
        # parameters sub-schemas fold in) whose cross-pool deps are not yet satisfiable.
        # Ordering + validation run once on the finalized whole — see _ensure_ordered.

    def _ensure_ordered(self):
        """Compute (and memoize) the dependency order + validation over the COMPLETE DAG.

        Runs the topo-sort the first time the order is needed and caches it. The cache is a
        plain attribute set via ``object.__setattr__`` (the dataclass is frozen), deliberately
        NOT a field — it takes no part in equality/hash and never appears in ``repr``."""
        cached = getattr(self, "_order_cache", None)
        if cached is None:
            order = _topo_sort(self.axes, self.derived)
            cached = (
                tuple(n for n in order if isinstance(n, Axis)),
                tuple(n for n in order if isinstance(n, Derived)),
            )
            object.__setattr__(self, "_order_cache", cached)
        return cached

    def finalize(self) -> "DesignSpace":
        """Force validation + ordering now (rather than lazily at first resolve), returning
        self. ``compile`` calls this after folding every fragment, so a malformed COMPLETE
        space fails at compile time. Idempotent (the order is memoized)."""
        self._ensure_ordered()
        return self

    @property
    def axis_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.axes)

    def ordered_axes(self) -> tuple[Axis, ...]:
        """Axes in dependency order (each axis after every axis it reads)."""
        return self._ensure_ordered()[0]

    def ordered_derived(self) -> tuple[Derived, ...]:
        """Deriveds in dependency order (each derived after every axis/derived it reads)."""
        return self._ensure_ordered()[1]


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
