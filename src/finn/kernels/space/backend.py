############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Backend`` — a self-contained backend realization bundle, and
``pool_schema`` — the assembler that lowers a pool of bundles + op-level shared
elements into the flat :class:`Schema` the existing ``resolve`` consumes.

This is the SELECTION half of the composability thesis (design-space-model.md
§1.2.1/§1.2.2): the op has one root ``implementation`` axis whose domain is the
pool of buildable designs (HLS, RTL soft-vec, RTL DSP58-packed for MVAU). Each
:class:`Backend` owns its axes/derived/predicates/feasibility/sources in one
place; ``pool_schema`` merges them so that ONLY the selected bundle's contributions
are active for a given point. Adding a backend is then purely additive — declare
one bundle, pass it in the pool, edit nothing else.

Selection (sum — pick one impl) is distinct from COMPOSITION (product — an impl
co-exists with sub-kernels, e.g. weight delivery). Composition is deliberately NOT
built here: per design-space-model.md §5 it rides the existing ``Derived`` primitive
(a ``Derived`` whose ``compute`` returns a resolved sub-``Point``), so the engine
needs no new type for it. This module builds selection only and leaves the
composition seam open.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .axis import Axis, discrete_axis
from .derived import Derived
from .artifacts import Artifacts
from .predicate import Predicate
from .schema import Schema


# The point key under which pool_schema exposes the selected bundle's source list.
# Reserved: a bundle may not declare a derived of this name.
SOURCES_KEY = "sources"


def _feasible_ok(_point, _context) -> None:
    return None


@dataclass(frozen=True)
class Backend:
    """One buildable realization of an op — a self-contained bundle.

    Attributes:
        name: the pool-member identity (the value of the root ``implementation``
            axis when this bundle is selected).
        feasible: this bundle's OWN device/dtype gate, ``(point, context) ->
            reason | None``; None means feasible. Wrapped by ``pool_schema`` into a
            predicate that fires only when this bundle is selected.
        axes: axes this bundle introduces (guarded on selection by the assembler).
        derived: quantities this bundle computes (present only when selected).
        predicates: this bundle's OWN legality checks (fire only when selected).
        sources: the RTL/HLS source files this bundle owns (declared association;
            overlaps between bundles surface non-separation — handoff §2b).
        emit: this bundle's hermetic codegen, ``(point, context) -> Artifacts``, or
            None if emit is not yet implemented for this backend. Dispatched by
            :func:`emit_point`. Reads the resolved ``point`` + frozen ``context``
            (which carries initializer VALUES) — never the graph.
        stream: this bundle's STREAM folding, ``{interface_name -> [StreamFold, ...]}`` —
            a list positional over the op interface's ``block`` dims: ``stream[iface][i]``
            folds ``block[iface][i]``. Each entry is ``1`` (unfolded), a bare axis name
            (``"SIMD"`` — the string IS the dial declaration), or a
            :class:`~finn.kernels.space.tiling.TileExpr`
            (``derive("PE")/param("TH")``). Impl-owned by construction: STREAM folding IS
            the RTL realization of the op's block structure — there is NO block field here,
            so an impl cannot change the math (kernelop-tensor-block-stream.md §5). An
            interface absent from the map is unfolded (stream = 1 element/cycle).
    """

    name: str
    feasible: Callable[[Any, Any], str | None] = _feasible_ok
    axes: tuple[Axis, ...] = ()
    derived: tuple[Derived, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    sources: tuple[str, ...] = ()
    emit: Callable[[Any, Any], "Artifacts"] | None = None
    stream: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "stream", dict(self.stream))


class EmitError(ValueError):
    """Raised when emit is requested for a point whose bundle has no emit, or whose
    implementation is not in the pool."""


def emit_point(pool, point, context, *, root: str = "implementation") -> Artifacts:
    """Dispatch codegen for a resolved ``point`` to its selected bundle's ``emit``.

    Looks up the pool member named by ``point[root]`` and calls its ``emit(point,
    context)``. ``root`` is the pool's selection axis — ``"implementation"`` for a
    compute pool, ``"parameters.topology"`` for the composed parameters pool. Raises
    :class:`EmitError` if that bundle has no emit yet, or if the point's selection is
    not a pool member.
    """
    impl = point[root]
    by_name = {b.name: b for b in pool}
    bundle = by_name.get(impl)
    if bundle is None:
        raise EmitError(
            f"{root} {impl!r} is not in the pool (have {sorted(by_name)})"
        )
    if bundle.emit is None:
        raise EmitError(f"emit not implemented for {root} {impl!r}")
    return bundle.emit(point, context)


class PoolError(ValueError):
    """Raised for a malformed pool (empty, duplicate names, sibling coupling)."""


def pool_schema(
    root_name: str,
    shared_axes: tuple[Axis, ...],
    shared_derived: tuple[Derived, ...],
    shared_predicates: tuple[Predicate, ...],
    pool: tuple[Backend, ...],
    *,
    sources_key: str = SOURCES_KEY,
) -> Schema:
    """Assemble op-level shared elements + a pool of bundles into a ``Schema``.

    The root ``implementation`` axis selects one bundle. Bundle contributions are
    merged by name and dispatched on the selected impl, so a resolved point carries
    only the selected bundle's axes/derived, and only its predicates + feasibility
    fire. ``resolve`` is unchanged — this produces the flat structure it already
    walks.

    ``sources_key`` is the point key under which the selected bundle's source list is
    exposed (default ``"sources"``). A SECONDARY pool composed into the same op schema
    (e.g. the ``parameters`` pool via :func:`compose`) passes a namespaced key
    (``"parameters.sources"``) so the two pools' source lists never collide.
    """
    if not pool:
        raise PoolError("pool must contain at least one Backend")

    names = [b.name for b in pool]
    if len(names) != len(set(names)):
        raise PoolError(f"duplicate implementation names in pool: {names}")

    _check_no_sibling_coupling(root_name, shared_axes, pool)
    _check_no_derived_shadowing(shared_derived, pool, sources_key)

    root = discrete_axis(root_name, frozenset(names), names[0])

    merged_axes = _merge_axes(root_name, pool)
    merged_derived = _merge_derived(root_name, pool)
    sources_derived = _sources_derived(root_name, pool, sources_key)
    wrapped_predicates = _wrap_predicates(root_name, pool)

    return Schema(
        axes=(root,) + tuple(shared_axes) + tuple(merged_axes),
        derived=tuple(shared_derived) + tuple(merged_derived) + (sources_derived,),
        predicates=tuple(shared_predicates) + tuple(wrapped_predicates),
    )


def compose(op_schema: Schema, sub_schema: Schema, *, guard=None) -> Schema:
    """Merge a secondary (namespaced) pool schema into an op schema — the COMPOSITION
    of two selection pools into one design space.

    Both arguments are ``pool_schema`` results. ``sub_schema``'s axes/derived/
    predicates are appended to ``op_schema``'s; because the sub-pool authored its keys
    namespaced (``parameters.*``) and used a distinct ``sources_key``, there is no
    name collision and ``resolve`` walks the union unchanged. This is the mechanism the
    engine-mapping proofs validated (``tests/test_composition_mapping.py``): composition
    (product) is a plain schema union of two selection pools — no new primitive.

    ``guard`` (optional): a ``(point) -> bool`` existence predicate applied to the
    sub-pool's ROOT axis, so a param-free op can compose the pool yet omit it entirely
    (the MHA difference-in-kind). Omit for an always-present subsystem.

    NOTE: cross-coordinate couplings (a derived/predicate that reads BOTH schemas'
    fields — e.g. memstream depth = f(compute fold)) are NOT added here; the composing
    op appends them to the op schema before calling ``compose``, where both coordinate
    surfaces are in scope.
    """
    sub_axes = sub_schema.axes
    sub_derived = sub_schema.derived
    if guard is not None:
        sub_axes = tuple(_guard_axis(a, guard) if _is_root(a, sub_schema) else a for a in sub_axes)
        # When the pool is guarded out its root axis is absent, so its derived (which
        # read the root) must not run — wrap them to no-op (None) when guarded out.
        sub_derived = tuple(_guard_derived(d, guard) for d in sub_derived)
    return Schema(
        axes=tuple(op_schema.axes) + tuple(sub_axes),
        derived=tuple(op_schema.derived) + tuple(sub_derived),
        predicates=tuple(op_schema.predicates) + tuple(sub_schema.predicates),
    )


def _is_root(axis: Axis, schema: Schema) -> bool:
    # The root selection axis is the first axis pool_schema emits.
    return schema.axes and axis.name == schema.axes[0].name


def _guard_axis(axis: Axis, guard) -> Axis:
    """Wrap an axis with an additional existence guard (AND with its own)."""
    from dataclasses import replace

    own = axis.exists

    def exists(point, _own=own, _guard=guard):
        return _guard(point) and _own(point)

    return replace(axis, exists=exists)


def _guard_derived(derived: Derived, guard) -> Derived:
    """Wrap a derived so it computes only when the pool is present (guard true); when
    the pool is guarded out its axes are absent, so the derived would raise — return
    None instead (the pool contributes nothing to the point)."""
    own = derived.compute

    def compute(point, context, _own=own, _guard=guard):
        return _own(point, context) if _guard(point) else None

    return Derived(derived.name, compute)


def _check_no_sibling_coupling(root_name, shared_axes, pool) -> None:
    """A pool member may not depend on a sibling pool member's axis. Sibling
    coupling would break the additive property (adding a 4th bundle could change a
    sibling's resolution). A bundle axis may depend on the root, on op-level shared
    axes, on its OWN bundle's axes, or on any derived — never on another bundle's
    axis."""
    shared_names = {a.name for a in shared_axes} | {root_name}
    for bundle in pool:
        own = {a.name for a in bundle.axes}
        allowed = shared_names | own
        for axis in bundle.axes:
            for dep in axis.deps:
                # A dep that names another bundle's axis (present in some sibling,
                # absent from allowed) is illegal sibling coupling. Deps on derived
                # or shared/own axes are fine and simply aren't flagged here.
                for sibling in pool:
                    if sibling.name == bundle.name:
                        continue
                    sibling_axes = {a.name for a in sibling.axes}
                    if dep in sibling_axes and dep not in allowed:
                        raise PoolError(
                            f"implementation {bundle.name!r} axis {axis.name!r} "
                            f"depends on sibling {sibling.name!r}'s axis {dep!r} — "
                            f"pool members must not couple to siblings"
                        )


def _check_no_derived_shadowing(shared_derived, pool, sources_key=SOURCES_KEY) -> None:
    """A bundle's derived must not shadow an op-level shared derived or the reserved
    ``sources`` key. ``Schema`` only dedups *axis* names, so a colliding derived would
    silently let one definition win with no diagnostic — breaking the "additive, can't
    perturb others" guarantee. (Bundle derived sharing a name ACROSS bundles is fine
    and intentional — that is the per-impl dispatch merge.)"""
    shared_names = {d.name for d in shared_derived}
    for bundle in pool:
        for d in bundle.derived:
            if d.name == sources_key:
                raise PoolError(
                    f"implementation {bundle.name!r} declares a derived named "
                    f"{sources_key!r}, which is reserved by pool_schema"
                )
            if d.name in shared_names:
                raise PoolError(
                    f"implementation {bundle.name!r} derived {d.name!r} shadows an "
                    f"op-level shared derived of the same name"
                )


def _owners_by_axis(pool) -> dict[str, list[Backend]]:
    owners: dict[str, list[Backend]] = {}
    for bundle in pool:
        for axis in bundle.axes:
            owners.setdefault(axis.name, []).append(bundle)
    return owners


def _merge_axes(root_name, pool) -> list[Axis]:
    """Merge bundle axes by name into single axes that dispatch on the selected
    impl. Existence is `impl in owners`; domain/default dispatch to the owning
    bundle's axis for the current impl. Deps gain the root axis (existence reads
    it)."""
    owners = _owners_by_axis(pool)
    merged: list[Axis] = []
    for axis_name, owning in owners.items():
        owner_names = frozenset(b.name for b in owning)
        # Map impl-name -> the contributing Axis object for that impl.
        by_impl = {b.name: _axis_of(b, axis_name) for b in owning}

        # Union of the underlying axes' declared deps, plus the root (existence and
        # dispatch both read the selected impl). Exclude the axis's own name.
        dep_union: set[str] = set()
        for ax in by_impl.values():
            dep_union |= set(ax.deps)
        dep_union.add(root_name)
        dep_union.discard(axis_name)

        merged.append(
            Axis(
                name=axis_name,
                domain=_dispatch_domain(root_name, by_impl),
                default=_dispatch_default(root_name, by_impl),
                exists=_dispatch_exists(root_name, owner_names, by_impl),
                deps=frozenset(dep_union),
            )
        )
    return merged


def _axis_of(bundle, axis_name) -> Axis:
    for axis in bundle.axes:
        if axis.name == axis_name:
            return axis
    raise PoolError(f"{bundle.name!r} has no axis {axis_name!r}")  # unreachable


def _dispatch_exists(root_name, owner_names, by_impl):
    def exists(point, _root=root_name, _owners=owner_names, _by=by_impl):
        impl = point.get(_root)
        if impl not in _owners:
            return False
        # Respect the owning axis's own guard, if any (e.g. a bundle axis that only
        # exists under a further condition). Its guard reads the same point.
        return _by[impl].exists(point)

    return exists


def _dispatch_domain(root_name, by_impl):
    def domain(point, context, _root=root_name, _by=by_impl):
        return _by[point[_root]].domain(point, context)

    return domain


def _dispatch_default(root_name, by_impl):
    def default(point, context, _root=root_name, _by=by_impl):
        return _by[point[_root]].default(point, context)

    return default


def _merge_derived(root_name, pool) -> list[Derived]:
    """Merge bundle derived by name; compute returns the owning bundle's value when
    its impl is selected, else None (present-but-None — matches the pre-restructure
    behaviour where e.g. SEGMENTLEN is None on HLS)."""
    owners: dict[str, list[Backend]] = {}
    for bundle in pool:
        for d in bundle.derived:
            owners.setdefault(d.name, []).append(bundle)

    merged: list[Derived] = []
    for name, owning in owners.items():
        by_impl = {b.name: _derived_of(b, name) for b in owning}
        merged.append(Derived(name, _dispatch_compute(root_name, by_impl)))
    return merged


def _derived_of(bundle, name) -> Derived:
    for d in bundle.derived:
        if d.name == name:
            return d
    raise PoolError(f"{bundle.name!r} has no derived {name!r}")  # unreachable


def _dispatch_compute(root_name, by_impl):
    def compute(point, context, _root=root_name, _by=by_impl):
        d = _by.get(point[_root])
        return d.compute(point, context) if d is not None else None

    return compute


def _sources_derived(root_name, pool, sources_key=SOURCES_KEY) -> Derived:
    """A derived exposing the selected bundle's source-file list on the point, so a
    resolved Point carries ``r.sources`` (or the namespaced key for a secondary pool).
    Overlaps between bundles are visible here (handoff §2b)."""
    by_impl = {b.name: b.sources for b in pool}

    def compute(point, _context, _root=root_name, _by=by_impl):
        return _by[point[_root]]

    return Derived(sources_key, compute)


def _wrap_predicates(root_name, pool) -> list[Predicate]:
    """Each bundle's predicates fire only when that bundle is selected; each
    bundle's ``feasible`` becomes a predicate guarded on selection."""
    wrapped: list[Predicate] = []
    for bundle in pool:
        wrapped.append(_feasible_predicate(root_name, bundle))
        for pred in bundle.predicates:
            wrapped.append(_guarded_predicate(root_name, bundle.name, pred))
    return wrapped


def _feasible_predicate(root_name, bundle) -> Predicate:
    def check(point, context, _root=root_name, _name=bundle.name, _fn=bundle.feasible):
        if point.get(_root) != _name:
            return None
        return _fn(point, context)

    return Predicate(check=check, description=f"{bundle.name} feasibility")


def _guarded_predicate(root_name, impl_name, pred) -> Predicate:
    def check(point, context, _root=root_name, _name=impl_name, _p=pred):
        if point.get(_root) != _name:
            return None
        return _p.check(point, context)

    return Predicate(check=check, description=pred.describe())
