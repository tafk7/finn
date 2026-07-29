############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Schema-axis → FINN-nodeattr-type mapping — the schema *is* the nodeattr registry.

FINN's ``get_nodeattr_types()`` must declare *every* attribute any pass might read or
write (``set_nodeattr`` asserts declared membership — the R12 obligation in
``consumer-surface-model.md``). Rather than hand-maintain that union with its
inconsistent FINN conventions, we derive it from the
:class:`~finn.kernels.engine.schema.Schema`: each free :class:`Axis` becomes one
nodeattr, typed and value-constrained by inspecting its domain. The kernel is the
source of truth — FINN's declared type/allowed-set/default all *flow from* the axis,
never hardcoded here.

The one wrinkle is **pool dispatch.** A pool member's axis (``resType``, ``ram_style``,
``pumpedCompute``) has its domain rewritten by ``pool_schema`` to dispatch on the
selected implementation (``space/implementation.py:_dispatch_domain``), so evaluating
it needs a point that pins the selection. We therefore probe each axis's domain across
the cartesian product of the *statically-resolvable* discrete axes (the selection roots
— ``implementation``, ``parameters.topology`` — plus simple flag axes) and union the
results. An axis that resolves under no probe is a point/context-dependent integer
quantity (``fixed``/``divisor`` — matrix dims, fold factors) and is typed ``"i"``.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from finn.kernels.space import PredicateDomain

# FINN AttributeProto member names (qonnx base.py): "i" int, "s" str, "ints" int list.
_ZERO: dict[str, Any] = {"i": 0, "s": "", "ints": []}
# Bound on the probe grid — the selection space is tiny (a few discrete axes); this
# only guards against a pathological schema, never trims a real one.
_MAX_PROBES = 512


def axis_nodeattr_types(schema) -> dict[str, tuple]:
    """Map every axis in ``schema`` to a FINN nodeattr type tuple.

    Returns ``{axis_name: (dtype, required, default[, allowed_values])}`` — the shape
    FINN's ``get_nodeattr_types`` expects. All axes are non-required (``False``): an
    unpinned axis resolves to its schema default, so FINN need never supply it.
    """
    probes = _probe_points(schema)
    return {axis.name: _axis_to_nodeattr(axis, probes) for axis in schema.ordered_axes()}


def _axis_to_nodeattr(axis, probes) -> tuple:
    dtype, allowed = _classify_domain(axis, probes)
    default = _safe_default(axis, dtype, probes)
    if allowed is not None:
        return (dtype, False, default, allowed)
    return (dtype, False, default)


def _resolve_domain(axis, point):
    """Evaluate an axis's domain against a candidate point, or None if it can't
    (reads a dim/context the probe doesn't supply, or dispatches to a non-owner)."""
    try:
        return axis.domain(point, None)
    except Exception:
        return None


def _probe_points(schema) -> list[dict]:
    """Candidate points pinning the discrete selection axes, so pool-dispatched axes
    resolve. Built from the cartesian product of every axis whose domain is a static
    non-empty ``frozenset`` (the roots + simple flags) — the axes other axes dispatch
    on. Always includes the empty point so a fully-static axis is probed too."""
    choices: list[list[tuple[str, Any]]] = []
    for axis in schema.ordered_axes():
        dom = _resolve_domain(axis, {})
        if isinstance(dom, frozenset) and dom:
            choices.append([(axis.name, v) for v in sorted(dom, key=repr)])

    points: list[dict] = [{}]
    for combo in product(*choices):
        points.append(dict(combo))
        if len(points) >= _MAX_PROBES:
            break
    return points


def _classify_domain(axis, probes) -> tuple[str, frozenset | None]:
    """(FINN dtype, allowed_values or None) for an axis, unioned over the probe grid.

    An enumerable ``frozenset`` domain yields ``allowed_values`` (enables FINN's
    ``resType``/``ram_style`` sweeps and value validation); a ``PredicateDomain``
    yields storage type only (non-enumerable values); a domain that resolves under no
    probe (``fixed``/``divisor`` — matrix dims, fold factors) is integer.
    """
    values: set = set()
    saw_str_predicate = False
    saw_list_predicate = False
    saw_frozenset = False

    for point in probes:
        dom = _resolve_domain(axis, point)
        if isinstance(dom, PredicateDomain):
            if "list" in dom.label:
                saw_list_predicate = True
            else:
                saw_str_predicate = True
        elif isinstance(dom, frozenset) and dom:
            saw_frozenset = True
            values |= dom

    if saw_list_predicate:
        return "ints", None
    if saw_str_predicate:
        # A non-list predicate domain (ActVal "int", mlo_max_iter "nonneg int") is a
        # scalar int stored as "i" with no membership constraint.
        return "i", None
    if saw_frozenset:
        if all(isinstance(v, str) for v in values):
            return "s", frozenset(values)
        # bool is an int subclass; {0,1} flags are ints, not strings.
        if all(isinstance(v, int) for v in values):
            return "i", frozenset(int(v) for v in values)

    # Resolved under no probe: point/context-dependent integer quantity.
    return "i", None


def _safe_default(axis, dtype: str, probes):
    """The axis's declared default, or a type-appropriate zero if it needs a point.

    A ``discrete``/``predicate`` default is a plain value; a context-derived default
    (``fixed_axis`` matrix dim, ``divisor`` fold) needs a point, so we fall back to the
    storage-type zero — the real value is baked as a geometry nodeattr and set
    explicitly, never read from this default."""
    for point in probes:
        try:
            d = axis.default(point, None)
        except Exception:
            continue
        if d is not None:
            return d
    return _ZERO[dtype]
