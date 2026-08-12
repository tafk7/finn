############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``resolve`` — walk the dependent design space (design-space-model.md §2).

    resolve(schema, context, assignment, *, want=None) -> Point | Illegal(reasons)

Write the Attrs (node constants — assignment-or-default, no guard, no ordering);
walk axes in dependency order, skipping a guarded-out axis (it is ABSENT, not
defaulted); take the value from the assignment or the axis default; reject a
value outside its domain; compute Derived after axes are fixed; run every
Predicate on the assembled point and collect all reasons.

Attrs go FIRST so an axis guard/domain or a derived may read one. They cannot read
anything themselves, so no ordering question arises among them.

``resolve`` is pure over (schema, context, assignment). Collecting *all* predicate
reasons powers the explain-style diagnostics the model emphasises, so the eager
path deliberately does not short-circuit.

``want`` makes evaluation DEMAND-DRIVEN: given a set of names, only their transitive
read-closure is computed, and only the predicates whose own closure was fully
evaluated are run. This is for callers that need one value (a port's dtype, a single
width) and today pay for all 26 deriveds. ``want=None`` is the default and keeps the
eager semantics EXACTLY — a partial point would otherwise silently answer a question
it had not actually checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from .context import Context
from .point import Illegal, Point
from .design_space import DesignSpace

Result = Union[Point, Illegal]


def resolve(
    schema: DesignSpace,
    context: Context,
    assignment: Mapping | None = None,
    *,
    want: frozenset[str] | None = None,
) -> Result:
    assignment = dict(assignment or {})
    point: dict = {}

    if want is not None:
        return _resolve_wanted(schema, context, assignment, frozenset(want))

    # 0. Attrs — node CONSTANTS, written FIRST so every axis guard/domain and every derived
    #    can read one. No guard, no point-dependent domain, no ordering: an attr is fixed
    #    before resolution begins (engine/attr.py).
    illegal = _write_attrs(schema, context, assignment, point)
    if illegal is not None:
        return illegal

    # 1. Axes, in dependency order.
    for axis in schema.ordered_axes():
        if not axis.exists(_view(point)):
            # Guard is false: the axis is absent. An assignment that pins an
            # absent axis is a caller error — surface it rather than silently
            # dropping it (this catches e.g. ram_style set under mem_mode=embedded).
            if axis.name in assignment:
                return Illegal(
                    [
                        f"{axis.name} = {assignment[axis.name]!r} was assigned but the "
                        f"axis is absent (its guard is false for this point)"
                    ]
                )
            continue

        dom = axis.domain(_view(point), context)
        if axis.name in assignment:
            val = assignment[axis.name]
        else:
            val = axis.default(_view(point), context)

        if val not in dom:
            return Illegal([f"{axis.name} = {val!r} not in {dom}"])

        point[axis.name] = val

    # 2. Derived, in dependency order (a derived may read another derived).
    for d in schema.ordered_derived():
        point[d.name] = d.compute(_view(point), context)

    # 3. Predicates: collect every reason (None == legal).
    reasons = []
    for pred in schema.predicates:
        reason = pred.check(_view(point), context)
        if reason is not None:
            reasons.append(_with_origin(reason, pred))
    if reasons:
        return Illegal(reasons)

    return Point(point)


def _resolve_wanted(
    schema: DesignSpace, context: Context, assignment: dict, want: frozenset[str]
) -> Result:
    """Evaluate only what ``want`` transitively needs.

    Axes are still walked in dependency order and still guarded/defaulted/domain-checked
    exactly as in the eager path — the only difference is WHICH entries are visited.

    Two deliberate conservatisms:

    * **All axes in the closure are computed, none skipped by cleverness.** An axis's guard
      may read another axis, so pruning axes by anything other than the declared closure
      risks a guard reading an absent key.
    * **A predicate runs only if its ENTIRE closure was evaluated.** A rule we cannot fully
      evaluate is simply not run — never assumed to pass. So a ``want`` result can be
      Illegal (the rules it could check rejected), but "not Illegal" here means "nothing
      checkable objected", which is strictly weaker than the eager path's verdict. That is
      why ``want`` is opt-in and never the default.
    """
    needed = _closure(schema, want)
    point: dict = {}

    # Attrs are constants: cheap, unordered, and readable by anything. Writing only the ones
    # in the closure keeps the demand-driven promise (compute nothing unasked-for).
    illegal = _write_attrs(
        schema, context, assignment, point, only=lambda name: name in needed
    )
    if illegal is not None:
        return illegal

    for axis in schema.ordered_axes():
        if axis.name not in needed:
            continue
        if not axis.exists(_view(point)):
            if axis.name in assignment:
                return Illegal(
                    [
                        f"{axis.name} = {assignment[axis.name]!r} was assigned but the "
                        f"axis is absent (its guard is false for this point)"
                    ]
                )
            continue
        dom = axis.domain(_view(point), context)
        val = (
            assignment[axis.name]
            if axis.name in assignment
            else axis.default(_view(point), context)
        )
        if val not in dom:
            return Illegal([f"{axis.name} = {val!r} not in {dom}"])
        point[axis.name] = val

    for d in schema.ordered_derived():
        if d.name in needed:
            point[d.name] = d.compute(_view(point), context)

    # A guarded-out axis is legitimately absent, so "evaluated" is what we visited, which is
    # `needed` minus anything a guard excluded.
    evaluated = set(point)
    reasons = []
    for pred in schema.predicates:
        if not _closure(schema, pred.deps | pred.optional_deps) <= evaluated:
            continue  # cannot decide from what we computed — do not guess
        reason = pred.check(_view(point), context)
        if reason is not None:
            reasons.append(_with_origin(reason, pred))
    return Illegal(reasons) if reasons else Point(point)


def _write_attrs(
    schema: DesignSpace, context: Context, assignment: dict, point: dict, *, only=None
) -> Illegal | None:
    """Write the node CONSTANTS onto ``point``: assignment-or-default, domain-checked.

    Returns an :class:`Illegal` on an out-of-domain value, else ``None``. Deliberately much
    simpler than the axis loop — no guard (an attr is never absent), no point-dependent domain
    (its membership test is fixed), no ordering (it reads nothing). That simplicity IS the
    category: everything the axis loop does beyond this exists to serve a CHOICE.

    ``only`` filters by name for the demand-driven path."""
    for a in schema.attrs:
        if only is not None and not only(a.name):
            continue
        val = assignment[a.name] if a.name in assignment else a.value(context)
        if val not in a.domain:
            return Illegal([f"{a.name} = {val!r} not in {a.domain}"])
        point[a.name] = val
    return None


def _closure(schema: DesignSpace, roots) -> frozenset[str]:
    """The transitive read-closure of ``roots`` over declared deps.

    Optional deps are included WHEN PRESENT: if the space defines the name, the reader may
    genuinely read it, so it must be computed first. An optional dep the space does not
    define contributes nothing, which is exactly its meaning.

    Attrs are in ``by_name`` because a derived may NAME one (``narrow_weights`` reads
    ``mlo_max_iter``); omitting them would silently drop that entry from the closure, and the
    demand-driven walk would then read a key it never wrote. They extend no walk — an attr's
    deps are empty, so it is always a leaf."""
    by_name = {
        n.name: n
        for n in tuple(schema.attrs) + tuple(schema.axes) + tuple(schema.derived)
    }
    seen: set[str] = set()
    stack = [r for r in roots if r in by_name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        node = by_name[name]
        stack.extend(
            d for d in node.deps | node.optional_deps if d in by_name and d not in seen
        )
    return frozenset(seen)


def _with_origin(reason: str, pred) -> str:
    """Append a generated rule's provenance to its reason.

    A hand-written rule's description already names its file and intent; a GENERATED one
    (a divisibility check, a selection-guarded backend rule) reads as if it came from
    nowhere. Suffixing the origin is what turns "who wrote this rule?" into a lookup.
    Only generated entries carry an origin, so an authored rule's reason is unchanged."""
    return f"{reason} [{pred.origin}]" if pred.origin else reason


def _view(values: dict) -> Point:
    """A Point view over the partial assignment so guards/domains/derived read it
    with the same absent-key-raises semantics as a finished point."""
    return Point(values)
