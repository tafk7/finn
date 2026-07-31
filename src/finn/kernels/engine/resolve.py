############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``resolve`` — walk the dependent design space (design-space-model.md §2).

    resolve(schema, context, assignment) -> Point | Illegal(reasons)

Walk axes in dependency order; skip a guarded-out axis (it is ABSENT, not
defaulted); take the value from the assignment or the axis default; reject a
value outside its domain; compute Derived after axes are fixed; run every
Predicate on the assembled point and collect all reasons.

The build/configure split survives only as a performance structure: ``resolve``
is pure over (schema, context, assignment). Tiered/short-circuit evaluation is an
optional optimization we deliberately skip here — collecting *all* predicate
reasons powers the explain-style diagnostics the model emphasises.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from .context import Context
from .point import Illegal, Point
from .design_space import DesignSpace

Result = Union[Point, Illegal]


def resolve(schema: DesignSpace, context: Context, assignment: Mapping | None = None) -> Result:
    assignment = dict(assignment or {})
    point: dict = {}

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

    # 2. Derived, computed after every axis is fixed.
    for d in schema.derived:
        point[d.name] = d.compute(_view(point), context)

    # 3. Predicates: collect every reason (None == legal).
    reasons = []
    for pred in schema.predicates:
        reason = pred.check(_view(point), context)
        if reason is not None:
            reasons.append(reason)
    if reasons:
        return Illegal(reasons)

    return Point(point)


def _view(values: dict) -> Point:
    """A Point view over the partial assignment so guards/domains/derived read it
    with the same absent-key-raises semantics as a finished point."""
    return Point(values)
