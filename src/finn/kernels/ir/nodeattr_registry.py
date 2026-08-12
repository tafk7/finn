############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""DesignSpace-axis → FINN-nodeattr-type mapping — the schema *is* the nodeattr registry.

FINN's ``get_nodeattr_types()`` must declare *every* attribute any pass might read or
write (``set_nodeattr`` asserts declared membership — the R12 obligation in
``consumer-surface-model.md``). Rather than hand-maintain that union with its
inconsistent FINN conventions, we derive it from the
:class:`~finn.kernels.engine.design_space.DesignSpace`: each free :class:`Axis` becomes one
nodeattr, typed and value-constrained by inspecting its domain. The kernel is the
source of truth — FINN's declared type/allowed-set/default all *flow from* the axis,
never hardcoded here.

The one wrinkle is **pool dispatch.** A pool member's axis (``resType``, ``ram_style``,
``pumpedCompute``) has its domain rewritten by ``pool_space`` to dispatch on the
selected implementation (``model/backend.py:_dispatch_domain``), so evaluating
it needs a point that pins the selection. We therefore probe each axis's domain across
the cartesian product of the *statically-resolvable* discrete axes (the selection roots
— ``implementation``, ``parameters.topology`` — plus simple flag axes) and union the
results. An axis that resolves under no probe is a point/context-dependent integer
quantity (``fixed``/``divisor`` — matrix dims, fold factors) and is typed ``"i"``.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from finn.kernels.engine.axis import PredicateDomain
from finn.kernels.engine.ordered_parameter import OrderedParameter
from finn.kernels.engine.point import AbsentAxisError

# FINN AttributeProto member names (qonnx base.py): "i" int, "s" str, "ints" int list.
_ZERO: dict[str, Any] = {"i": 0, "s": "", "ints": []}
# Bound on the probe grid — the selection space is tiny (a few discrete axes); this
# only guards against a pathological schema, never trims a real one.
_MAX_PROBES = 512


def axis_nodeattr_types_for(schema, realized=None) -> dict[str, tuple]:
    """The nodeattr specs for a node, with each DOMAIN taken from the selected realization.

    F3. ``axis_nodeattr_types`` publishes the pool UNION of domains, so a DSP node advertises
    ``resType in {lut, dsp}`` and ``set_nodeattr("resType", "lut")`` succeeds at the write,
    failing later at whichever getter first resolves a point. The union is right for one
    question and wrong for the other, and the two were conflated:

    * the NAME SET is a union — ``get_nodeattr_types`` must declare every name the node could
      carry, or re-specializing onto another member would hit an undeclared attribute;
    * each published DOMAIN is a fact about the COMMITTED realization, so it comes from
      ``realized``.

    ``realized`` is an already-per-realization space (from
    :meth:`~finn.kernels.model.cell.Kernel.space_for`). Its axis domains are plain VALUES —
    there is no sibling to dispatch to — so they are read directly, with no probe grid.
    ``None`` means the node is unspecialized: there is no selection to read, so every spec
    falls back to the union, which is both the honest answer and the only one that leaves an
    unspecialized node writable.
    """
    union = axis_nodeattr_types(schema)
    if realized is None:
        return union

    from finn.kernels.engine.point import Point

    empty = Point({})
    out = dict(union)
    for axis in realized.ordered_axes():
        try:
            dom = axis.domain(empty, None)
        except (ValueError, KeyError, AbsentAxisError):
            # A context-dependent domain (a divisor over a matrix dim) cannot resolve without
            # a point — the same case the union path types as a bare int. Keep its spec.
            continue
        dtype, allowed = _classify_resolved(dom)
        if dtype is None:
            continue
        default = _realized_default(axis, empty, dtype)
        out[axis.name] = (
            (dtype, False, default, allowed) if allowed is not None else (dtype, False, default)
        )
    return out


def _classify_resolved(dom) -> tuple[str | None, frozenset | None]:
    """(FINN dtype, allowed_values) for an ALREADY-RESOLVED domain value.

    The union path's :func:`_classify_domain` accumulates across a probe grid because a
    dispatched domain needs a point that pins the selection. A realized domain is just a
    value, so this is the same classification with the grid removed. ``(None, None)`` means
    "no opinion" — leave the union's spec alone."""
    if isinstance(dom, PredicateDomain):
        return _predicate_storage_type(dom.label), None
    if isinstance(dom, frozenset) and dom:
        if all(isinstance(v, str) for v in dom):
            return "s", frozenset(dom)
        # bool is an int subclass; {0,1} flags are ints, not strings.
        if all(isinstance(v, int) for v in dom):
            return "i", frozenset(int(v) for v in dom)
        return None, None
    if isinstance(dom, OrderedParameter):
        # A fold dial's divisors. Enumerable in principle, but publishing them as
        # `allowed_values` would pin the node to ONE geometry's divisor set, and geometry is
        # a tensor fact that a reshape upstream can change. Type only.
        return "i", None
    return None, None


def _realized_default(axis, point, dtype: str):
    """The axis's own default, or the storage-type zero when it needs a point/context."""
    try:
        value = axis.default(point, None)
    except (ValueError, KeyError, AbsentAxisError):
        return _ZERO[dtype]
    return _ZERO[dtype] if value is None else value


def axis_nodeattr_types(schema) -> dict[str, tuple]:
    """Map every axis AND attr in ``schema`` to a FINN nodeattr type tuple.

    Returns ``{name: (dtype, required, default[, allowed_values])}`` — the shape
    FINN's ``get_nodeattr_types`` expects. All entries are non-required (``False``): an
    unpinned axis or attr resolves to its schema default, so FINN need never supply it.

    Both kinds appear because both are node-owned values the frontend may bake and resolve
    reads back — the nodeattr surface is "what the node carries", which is not the same
    question as "what is a choice". An :class:`~finn.kernels.engine.attr.Attr` types DIRECTLY
    from its ``label``/``default`` and needs no probe grid: its domain cannot depend on a
    point, which is precisely what the probe grid exists to work around for a pool-dispatched
    axis.
    """
    probes = _probe_points(schema)
    types = {axis.name: _axis_to_nodeattr(axis, probes) for axis in schema.ordered_axes()}
    types.update({a.name: _attr_to_nodeattr(a) for a in schema.attrs})
    return types


def _attr_to_nodeattr(a) -> tuple:
    """An attr's nodeattr spec, read straight off the declaration.

    No probe grid, no domain resolution, no `_safe_default` fallback — an attr's default is
    either a plain value or a ``(context) -> value`` callable, and in the latter case the
    registry has no Context, so the storage-type zero is the honest answer (the real value is
    supplied per node, exactly as for a context-derived axis default)."""
    dtype = _predicate_storage_type(a.label)
    default = a.default
    if callable(default):
        default = _ZERO[dtype]
    return (dtype, False, default)


def _axis_to_nodeattr(axis, probes) -> tuple:
    dtype, allowed = _classify_domain(axis, probes)
    default = _safe_default(axis, dtype, probes)
    if allowed is not None:
        return (dtype, False, default, allowed)
    return (dtype, False, default)


def _predicate_storage_type(label: str) -> str:
    """FINN storage type for a membership-test domain, from its LABEL.

    A non-enumerable domain carries no values to inspect, so the label is the only signal —
    ``"list[int]"`` is an int list, anything else a scalar int. Shared by the axis path
    (:func:`_classify_domain`, via a resolved ``PredicateDomain``) and the attr path, so the
    two cannot classify the same label differently."""
    return "ints" if "list" in label else "i"


def _resolve_domain(axis, point):
    """Evaluate an axis's domain against a candidate point, or None if it can't
    (reads a dim/context the probe doesn't supply, or dispatches to a non-owner)."""
    try:
        return axis.domain(point, None)
    except (ValueError, KeyError, AbsentAxisError):
        # The legitimate "this axis can't resolve for this probe" signals (a dim/context the
        # probe omits, a dispatch to a non-owner). Any OTHER exception is a kernel bug that
        # must propagate (INV5), not read as an unprobeable axis.
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
            if _predicate_storage_type(dom.label) == "ints":
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
    # (both branches above go through _predicate_storage_type's rule — see there)
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
        except (ValueError, KeyError, AbsentAxisError):
            # A context-derived default (matrix dim, fold) needs a point this probe lacks —
            # skip it. Any OTHER exception is a kernel bug that must propagate (INV5).
            continue
        if d is not None:
            return d
    return _ZERO[dtype]
