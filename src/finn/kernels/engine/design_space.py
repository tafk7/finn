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

That handles a dep that is absent *yet* (a forward ref, resolved by composition). The dual
case is a dep that may be absent *forever* on a legitimate path: the same storage pool is
resolved STANDALONE (via ``parameters_schema()``, where no composing op publishes a demand)
and COMPOSED into an op (where the demand exists and must order first). Neither ``deps``
(fails finalize when absent) nor silence (leaves the edge to fold order) is right, so nodes
also carry ``optional_deps`` — "order me after this name IF the space defines it". See
:class:`~finn.kernels.engine.derived.Derived`. The distinction is load-bearing for the
read-set audit: an undeclared read is a defect, a declared-optional-and-absent read is not.
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

    @classmethod
    def merge(cls, *spaces: "DesignSpace") -> "DesignSpace":
        """Union of assembly fragments into one unvalidated space.

        The ONE place fragments are unioned, so a future third pool cannot get the
        concatenation wrong. Order of the arguments carries NO meaning: every real
        ordering constraint is a declared ``deps``/``optional_deps`` edge resolved by the
        topo-sort. The result is deliberately NOT finalized — the caller finalizes once
        every fragment is folded in."""
        return cls(
            axes=tuple(a for s in spaces for a in s.axes),
            derived=tuple(d for s in spaces for d in s.derived),
            predicates=tuple(p for s in spaces for p in s.predicates),
        )

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
                # The two ways to get here are a typo and finalizing a FRAGMENT before its
                # sibling fragments are merged in. They are indistinguishable from inside
                # (that is why validation is deferred to the whole at all), so name both.
                # Fragment-ness is not a property we can mark on the object: the same
                # pool_space() output is a fragment under ParameterSource._source_subspace
                # and a COMPLETE space under parameters_schema().
                raise DesignSpaceError(
                    f"{n.name!r} declares dependency on unknown name {dep!r} — either a "
                    f"typo, or this space is an assembly FRAGMENT finalized before the "
                    f"fragment defining {dep!r} was merged in (see DesignSpace.merge). If "
                    f"{dep!r} is legitimately absent on some valid path, declare it in "
                    f"optional_deps rather than deps."
                )

    # An OPTIONAL dep is an edge only when the name is present. Absent is legal and
    # silent — that is the whole point: the same pool resolves standalone (key absent)
    # and composed (key present, must order first). Resolving them ONCE here keeps the
    # traversal below a plain lookup, and keeps `deps` strictly the typo-checked set.
    edges = {
        n.name: n.deps | {d for d in n.optional_deps if d in by_name} for n in nodes
    }

    # Deterministic depth-first topological sort with cycle detection. Input
    # order is preserved among independent nodes so the schema reads predictably.
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n.name: WHITE for n in nodes}
    order: list[object] = []

    def visit(name: str, stack: list[str]):
        color[name] = GREY
        stack.append(name)
        for dep in sorted(edges[name]):
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
