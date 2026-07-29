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

from ..engine.axis import Axis, discrete_axis
from ..engine.derived import Derived
from .artifacts import Artifacts, RtlModule
from ..engine.predicate import Predicate
from ..engine.schema import Schema


# The point key under which pool_schema exposes the selected bundle's source list.
# Reserved: a bundle may not declare a derived of this name.
SOURCES_KEY = "sources"

# The compute pool's selection axis: the root axis whose value names the selected
# `Backend`. On a KERNEL node this is the realization axis (its nodeattr); on a
# CLASSIC node the same-named `backend` nodeattr is the family-membership token —
# distinct concepts that never coexist on one node.
BACKEND_AXIS = "backend"


def _feasible_ok(_point, _context) -> None:
    return None


@dataclass(frozen=True)
class Backend:
    """One buildable realization of an op — a self-contained bundle.

    Attributes:
        name: the pool-member identity (the value of the root ``implementation``
            axis when this bundle is selected).
        language: this bundle's realization language — ``"hls"`` or ``"rtl"`` — a static,
            1:1-with-the-backend identity fact (a microarch is HLS or RTL by construction).
            Read BARE-NODE by taxonomy routing (``is_hls_node``/``is_rtl_node`` via
            ``kernel_hw_language``) to classify a resolved kernel node without a Point, and
            read off the SELECTED bundle (``kernel.selected_backend(point).language``) by any
            point reader. A STATIC FIELD only — NOT re-projected onto the point as a derived
            (F5). ``None`` only for a bundle that emits no HDL of its own (the ``embedded``
            delivery topology — params baked into the compute core, ``emit=None``).
        rtl_core_module: for an RTL compute bundle, the per-core wrapper module the emitted
            top instantiates (→ ``$MODULE_NAME_COMPUTE_CORE$``); static backend identity, read
            off the selected bundle by emit (``kernel.selected_backend(point).rtl_core_module``),
            NOT re-projected onto the point (F5). ``None`` for a non-RTL bundle.
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
        consumes: this bundle's per-parameter-interface CONSUMPTION MODES,
            ``{interface_name -> frozenset[str]}`` over ``{"constant", "stream"}``. The
            consumption mode is how the compute core takes a parameter interface:
            ``constant`` (baked into the core — an HLS ROM/``params.h``; no port) or
            ``stream`` (an AXIS port a delivery block feeds). It restricts which delivery
            topologies are legal for that interface (a topology carries a mode; the domain
            keeps only topologies whose mode this backend consumes). An interface ABSENT
            from the map is PERMISSIVE (both modes) — so a backend that declares nothing
            regresses nothing. "embedded" is not a delivery topology but the ``constant``
            mode (consumption-mode-delivery.md). A flat second per-interface dict beside
            ``stream`` for now; a future InterfaceSchema may group them.
        mode: for a DELIVERY-pool member (a storage topology), the CONSUMPTION MODE it
            presents — ``"constant"`` (baked into the core, no port) or ``"stream"`` (an
            AXIS port). A topology CARRIES its own mode here, so the generic delivery guard
            reads ``topology.mode`` vs a compute backend's ``consumes[iface]`` — both
            ``Backend`` fields — with no string→mode side-table and no knowledge of the
            topology's identity string. ``None`` for a compute-pool member (it has no
            delivery mode; it CONSUMES modes via ``consumes``).
        schema: an OPTIONAL reference to this bundle's typed template contract
            (:class:`~finn.kernels.space.artifacts.RtlModule`). The schema is OWNED by the
            template (defined next to it, 1:1); ``Backend`` only REFERENCES it, so N
            backends emitting one template share one schema object (softvec + packed both
            point at ``_V_WRAPPER_SCHEMA``) — expressing the N:1 by shared reference rather
            than per-backend copies that could diverge. A backend *reader* (a validator,
            build-manifest tool, or Arc-3 Integrator) reads ``backend.schema`` for the
            typed contract without duplicating it. ``None`` for a backend with no
            ``$SLOT$`` template — an HLS core (free-form ``#define`` text, not typed slots)
            or a pure-wiring delivery cell. Does NOT change ``bind``/render: it is a
            reference for readers, not a new render path.
    """

    name: str
    language: str | None = None
    rtl_core_module: str | None = None
    feasible: Callable[[Any, Any], str | None] = _feasible_ok
    axes: tuple[Axis, ...] = ()
    derived: tuple[Derived, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    sources: tuple[str, ...] = ()
    emit: Callable[[Any, Any], "Artifacts"] | None = None
    stream: Mapping[str, Any] = field(default_factory=dict)
    consumes: Mapping[str, frozenset[str]] = field(default_factory=dict)
    mode: str | None = None
    schema: RtlModule | None = None

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "stream", dict(self.stream))
        object.__setattr__(
            self, "consumes", {k: frozenset(v) for k, v in dict(self.consumes).items()}
        )


class EmitError(ValueError):
    """Raised when emit is requested for a point whose bundle has no emit, or whose
    implementation is not in the pool."""


def emit_point(pool, point, context, *, root: str = BACKEND_AXIS) -> Artifacts:
    """Dispatch codegen for a resolved ``point`` to its selected bundle's ``emit``.

    Looks up the pool member named by ``point[root]`` and calls its ``emit(point,
    context)``. ``root`` is the pool's selection axis — ``"backend"`` for a
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
    unspecialized_sentinel: bool = False,
) -> Schema:
    """Assemble op-level shared elements + a pool of bundles into a ``Schema``.

    The root ``implementation`` axis selects one bundle. Bundle contributions are
    merged by name and dispatched on the selected impl, so a resolved point carries
    only the selected bundle's axes/derived, and only its predicates + feasibility
    fire. ``resolve`` is unchanged — this produces the flat structure it already
    walks.

    ``sources_key`` is the point key under which the selected bundle's source list is
    exposed (default ``"sources"``). A SECONDARY pool folded into the same op schema (e.g.
    the ``parameters`` delivery pool via a
    :class:`~finn.kernels.space.backend_interface.Interface`) passes a namespaced key
    (``"parameters.sources"``) so the two pools' source lists never collide.

    The backend IDENTITY fields ``language``/``rtl_core_module`` are STATIC FIELDS on the
    :class:`Backend` (read bare-node by routing, and off the selected bundle by emit via
    :meth:`~finn.kernels.space.kernel.Kernel.selected_backend`). They are deliberately NOT
    re-projected onto the point as deriveds — one fact, one home.

    ``unspecialized_sentinel`` makes the root selection axis default to ``""`` — the
    UNSPECIALIZED sentinel (no backend committed) — instead of the first pool member. True
    ONLY for the COMPUTE ``implementation`` root, where "which backend" is an explicit
    Selection act (Seam B) and an unpinned node must read as unspecialized (F1). False (the
    default) for a DELIVERY ``topology`` pool, whose first-member default (``embedded``) is a
    genuine, always-legal fallback that production resolution relies on.
    """
    if not pool:
        raise PoolError("pool must contain at least one Backend")

    names = [b.name for b in pool]
    if len(names) != len(set(names)):
        raise PoolError(f"duplicate implementation names in pool: {names}")

    _check_no_sibling_coupling(root_name, shared_axes, pool)
    _check_no_derived_shadowing(shared_derived, pool, sources_key)

    # The compute ``implementation`` root defaults to "" — the UNSPECIALIZED sentinel (no
    # backend committed). "" is deliberately NOT in the domain frozenset, so an unpinned node
    # that resolves through the impl-DEPENDENT path yields Illegal, which the kernel getters
    # turn into a clean "unspecialized" raise. A node is SPECIALIZED once ``implementation``
    # is a real pool member; the ONE predicate reading that is ``routing.is_specialized``. A
    # delivery ``topology`` pool keeps the first-member default (a genuine legal fallback).
    root_default = "" if unspecialized_sentinel else names[0]
    root = discrete_axis(root_name, frozenset(names), root_default)

    merged_axes = _merge_axes(root_name, pool)
    merged_derived = _merge_derived(root_name, pool)
    sources_derived = _field_derived(root_name, pool, "sources", key=sources_key)
    wrapped_predicates = _wrap_predicates(root_name, pool)

    return Schema(
        axes=(root,) + tuple(shared_axes) + tuple(merged_axes),
        derived=(
            tuple(shared_derived)
            + tuple(merged_derived)
            + (sources_derived,)
        ),
        predicates=tuple(shared_predicates) + tuple(wrapped_predicates),
    )


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
    """A bundle's derived must not shadow an op-level shared derived or the pool_schema
    reserved ``sources`` projection key. ``Schema`` only dedups *axis* names, so a colliding
    derived would silently let one definition win with no diagnostic — breaking the
    "additive, can't perturb others" guarantee. (Bundle derived sharing a name ACROSS
    bundles is fine and intentional — that is the per-impl dispatch merge.)"""
    shared_names = {d.name for d in shared_derived}
    reserved = {sources_key}
    for bundle in pool:
        for d in bundle.derived:
            if d.name in reserved:
                raise PoolError(
                    f"implementation {bundle.name!r} declares a derived named "
                    f"{d.name!r}, which is reserved by pool_schema"
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


def _field_derived(root_name, pool, field_name, *, key=None) -> Derived:
    """A derived projecting a selected bundle's static FIELD onto the point, so a resolved
    Point carries it as a plain value. Used for the ``sources`` projection (a resolved Point
    carries ``r.sources``, or the namespaced key for a secondary pool): ``sources`` is a
    genuine per-point list, not a static identity field. A non-owning bundle whose field is
    its default (``None``) surfaces present-but-None. ``key`` overrides the point key (used
    to namespace ``sources`` for a secondary pool); defaults to ``field_name``."""
    by_impl = {b.name: getattr(b, field_name) for b in pool}

    def compute(point, _context, _root=root_name, _by=by_impl):
        return _by[point[_root]]

    return Derived(key or field_name, compute)


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
