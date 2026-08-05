############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Backend`` — a self-contained backend realization bundle, and
``pool_space`` — the assembler that lowers a pool of bundles + op-level shared
elements into the flat :class:`DesignSpace` the existing ``resolve`` consumes.

This is the SELECTION half of the composability thesis (design-space-model.md
§1.2.1/§1.2.2): the op has one root ``implementation`` axis whose domain is the
pool of buildable designs (HLS, RTL soft-vec, RTL DSP58-packed for MVAU). Each
:class:`Backend` owns its axes/derived/predicates/feasibility/sources in one
place; ``pool_space`` merges them so that ONLY the selected bundle's contributions
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
from ..engine.design_space import DesignSpace


# The point key under which pool_space exposes the selected bundle's source list.
# Reserved: a bundle may not declare a derived of this name.
SOURCES_KEY = "sources"

# The compute pool's selection axis: the root axis whose value names the selected
# `Backend`. On a KERNEL node this is the realization axis (its nodeattr); on a
# CLASSIC node the same-named `backend` nodeattr is the family-membership token —
# distinct concepts that never coexist on one node.
BACKEND_AXIS = "backend"


@dataclass(frozen=True)
class Interface:
    """One backend's realization of ONE op port — the per-(backend, port) home of every
    fact a backend declares about a port, in one object instead of parallel ``{iface -> …}``
    maps on :class:`Backend`.

    Attributes:
        stream: the BLOCK→STREAM fold list for this port — positional over the op
            interface's ``block`` dims (``stream[i]`` folds ``block[i]``). Each entry is
            ``1`` (unfolded), a bare axis name (``"SIMD"`` — the string IS the dial
            declaration), or a :class:`~finn.kernels.model.tiling.TileExpr`. Empty = the
            port is unfolded (1 element/cycle). Impl-owned: there is no block field here, so
            an impl cannot change the op math (kernelop-tensor-block-stream.md §5).
        mem_modes: the SET of memory-realization modes this backend accepts for a parameter
            port, a ``frozenset`` over ``{"embedded", "decoupled"}``. ``embedded`` = baked
            into the core (an HLS ROM/``params.h``, no port); ``decoupled`` = an AXIS port a
            separate memory backend feeds. Presence of an iface here is the authoritative
            "param port" signal for this backend. Restricts which delivery topologies are
            legal (a topology carries a mode; the delivery guard keeps only topologies whose
            mode this backend accepts). ``None`` = PERMISSIVE (both modes) — a port that
            declares nothing regresses nothing. Meaningful only for a delivered-parameter port.
            INPUT-only (an OUTPUT port has no exogenous memory realization).
        accepted_dtypes: the GATE (exogenous) half of the datatype model — this backend's
            declared datatype SUPPORT for an INPUT port, either a
            :class:`~finn.kernels.engine.datatype_support.DatatypeSupport` (category +
            bitwidth range) or a custom callable ``(dt) -> reason | None``. Compiled by
            ``pool_space`` into a guarded feasibility predicate (fires only when this
            backend is selected). ``None`` = unconstrained. The union of the pool's declared
            support is what ``can_infer_from`` accepts — a new backend widens it with no op
            edit. INPUT-only.
        derived_dtype: the DERIVATION (endogenous) half — a
            :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` declaring the dtype this
            backend PRODUCES for an OUTPUT port (the stream-width bitwidth, when it differs
            from the raw graph dtype — e.g. MVAU's accumulator-as-output under
            ``noActivation``). Resolved by ``resolve_datatype_spec`` against the point. ``None``
            = the port carries its graph tensor dtype. OUTPUT-only.

    Direction-exclusivity (``accepted_dtypes``/``mem_modes`` on INPUT only, ``derived_dtype``
    on OUTPUT only) is enforced at pool assembly by
    :meth:`~finn.kernels.model.kernel.Kernel._check_port_direction`, where the op schema
    supplies each port's direction. Internal-register derivations (accumulator, narrowed
    weight — no port) live on :attr:`Backend.derived_dtypes`, not here.
    """

    stream: tuple = ()
    mem_modes: frozenset[str] | None = None
    accepted_dtypes: Any | None = None
    derived_dtype: Any | None = None

    def __post_init__(self):
        object.__setattr__(self, "stream", tuple(self.stream))
        if self.mem_modes is not None:
            object.__setattr__(self, "mem_modes", frozenset(self.mem_modes))


def ports_from(
    *,
    stream: Mapping[str, Any] | None = None,
    mem_modes: Mapping[str, Any] | None = None,
    accepted_dtypes: Mapping[str, Any] | None = None,
    derived_dtype: Mapping[str, Any] | None = None,
) -> dict[str, Interface]:
    """Assemble a ``{iface -> Interface}`` ports map from per-facet maps. An ergonomic
    constructor that keeps SHARED facts (e.g. the pool-wide ``COMPUTE_STREAM`` fold map)
    expressible once while the storage stays one :class:`Interface` per port. A fact absent
    for an interface takes the :class:`Interface` default."""
    stream = stream or {}
    mem_modes = mem_modes or {}
    accepted_dtypes = accepted_dtypes or {}
    derived_dtype = derived_dtype or {}
    names = set(stream) | set(mem_modes) | set(accepted_dtypes) | set(derived_dtype)
    return {
        n: Interface(
            stream=stream.get(n, ()),
            mem_modes=mem_modes.get(n),
            accepted_dtypes=accepted_dtypes.get(n),
            derived_dtype=derived_dtype.get(n),
        )
        for n in names
    }


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
        axes: axes this bundle introduces (guarded on selection by the assembler).
        derived: quantities this bundle computes (present only when selected).
        predicates: this bundle's OWN legality checks (fire only when selected).
        sources: the RTL/HLS source files this bundle owns (declared association;
            overlaps between bundles surface non-separation — handoff §2b).
        emit: this bundle's hermetic codegen, ``(point, context) -> Artifacts``, or
            None if emit is not yet implemented for this backend. Dispatched by
            :func:`emit_point`. Reads the resolved ``point`` + frozen ``context``
            (which carries initializer VALUES) — never the graph.
        ports: this bundle's per-op-port realization facts, ``{interface_name ->
            Interface}``. Each :class:`Interface` bundles what this backend declares about
            one port — its STREAM fold, memory-realization modes, datatype SUPPORT, and
            datatype DERIVATION — in one object rather than parallel ``{iface -> …}`` maps.
            An interface absent from ``ports`` takes every :class:`Interface` default
            (unfolded, permissive, unconstrained). Build ergonomically with
            :func:`ports_from` when a fact (e.g. a pool-wide fold map) is shared.
        mem_mode: for a DELIVERY-pool member (a storage topology), the single memory-
            realization mode it PRESENTS — ``"embedded"`` (baked into the core, no port) or
            ``"decoupled"`` (an AXIS port). A topology CARRIES its own mode here, so the
            generic delivery guard reads ``topology.mem_mode`` vs a compute backend's
            ``mem_modes[iface]`` — both ``Backend`` fields — with no string→mode side-table
            and no knowledge of the topology's identity string. ``None`` for a compute-pool
            member (it has no delivery mode; it ACCEPTS modes via ``mem_modes``).
        schema: an OPTIONAL reference to this bundle's typed template contract
            (:class:`~finn.kernels.model.artifacts.RtlModule`). The schema is OWNED by the
            template (defined next to it, 1:1); ``Backend`` only REFERENCES it, so N
            backends emitting one template share one schema object (softvec + packed both
            point at ``_V_WRAPPER_SCHEMA``) — expressing the N:1 by shared reference rather
            than per-backend copies that could diverge. A backend *reader* (a validator,
            build-manifest tool, or Arc-3 Integrator) reads ``backend.schema`` for the
            typed contract without duplicating it. ``None`` for a backend with no
            ``$SLOT$`` template — an HLS core (free-form ``#define`` text, not typed slots)
            or a pure-wiring delivery cell. Does NOT change ``bind``/render: it is a
            reference for readers, not a new render path.
        derived_dtypes: this backend's INTERNAL-REGISTER datatype derivations —
            ``{register_name -> DatatypeSpec}`` for a produced dtype that has NO port
            (accumulator ``accDataType``, narrowed weight ``weightDataType``). ``pool_space``
            merges each into the resolved point under its register name (dispatched on
            selection, exactly like the generic ``derived``), so emit reads
            ``point.accDataType`` unchanged. The endogenous mirror of ``Interface.derived_dtype``
            for a quantity with no interface — the OUTPUT port's produced dtype lives on the
            port, an internal register's here. Both speak one
            :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` vocabulary.
    """

    name: str
    language: str | None = None
    rtl_core_module: str | None = None
    axes: tuple[Axis, ...] = ()
    derived: tuple[Derived, ...] = ()
    predicates: tuple[Predicate, ...] = ()
    sources: tuple[str, ...] = ()
    emit: Callable[[Any, Any], "Artifacts"] | None = None
    ports: Mapping[str, Interface] = field(default_factory=dict)
    mem_mode: str | None = None
    schema: RtlModule | None = None
    derived_dtypes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "ports", dict(self.ports))
        object.__setattr__(self, "derived_dtypes", dict(self.derived_dtypes))

    def stream_of(self, iface: str) -> tuple:
        """The BLOCK→STREAM fold this backend declares for ``iface`` (``()`` if absent)."""
        port = self.ports.get(iface)
        return port.stream if port is not None else ()

    def mem_modes_of(self, iface: str) -> frozenset[str] | None:
        """The memory-realization modes this backend accepts for ``iface`` (``None`` = permissive)."""
        port = self.ports.get(iface)
        return port.mem_modes if port is not None else None

    @property
    def stream(self) -> dict[str, tuple]:
        """The per-interface fold map, projected from ``ports`` (the tiling engine + seam
        builder read this as one ``{iface -> folds}`` dict). Only folding interfaces appear."""
        return {n: p.stream for n, p in self.ports.items() if p.stream}

    @property
    def mem_modes(self) -> dict[str, frozenset[str]]:
        """The per-interface memory-realization-mode map, projected from ``ports`` (the delivery
        guard reads this). Only interfaces that declare a mode appear; absent ⇒ permissive."""
        return {n: p.mem_modes for n, p in self.ports.items() if p.mem_modes is not None}


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


def pool_space(
    root_name: str,
    shared_axes: tuple[Axis, ...],
    shared_derived: tuple[Derived, ...],
    shared_predicates: tuple[Predicate, ...],
    pool: tuple[Backend, ...],
    *,
    sources_key: str = SOURCES_KEY,
    unspecialized_sentinel: bool = False,
) -> DesignSpace:
    """Assemble op-level shared elements + a pool of bundles into a ``DesignSpace``.

    The root ``implementation`` axis selects one bundle. Bundle contributions are
    merged by name and dispatched on the selected impl, so a resolved point carries
    only the selected bundle's axes/derived, and only its predicates + feasibility
    fire. ``resolve`` is unchanged — this produces the flat structure it already
    walks.

    ``sources_key`` is the point key under which the selected bundle's source list is
    exposed (default ``"sources"``). A SECONDARY pool folded into the same op schema (e.g.
    the ``parameters`` source pool via a
    :class:`~finn.kernels.model.parameter_source.ParameterSource`) passes a namespaced key
    (``"parameters.sources"``) so the two pools' source lists never collide.

    The backend IDENTITY fields ``language``/``rtl_core_module`` are STATIC FIELDS on the
    :class:`Backend` (read bare-node by routing, and off the selected bundle by emit via
    :meth:`~finn.kernels.model.kernel.Kernel.selected_backend`). They are deliberately NOT
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
    dtype_register_derived = _merge_derived_dtypes(root_name, pool)
    sources_derived = _field_derived(root_name, pool, "sources", key=sources_key)
    wrapped_predicates = _wrap_predicates(root_name, pool)

    return DesignSpace(
        axes=(root,) + tuple(shared_axes) + tuple(merged_axes),
        derived=(
            tuple(shared_derived)
            + tuple(merged_derived)
            + tuple(dtype_register_derived)
            + (sources_derived,)
        ),
        predicates=tuple(shared_predicates) + tuple(wrapped_predicates),
    )


def _check_no_sibling_coupling(root_name, shared_axes, pool) -> None:
    """A pool member may not depend on a sibling pool member's axis. Sibling
    coupling would break the additive property (adding a 4th bundle could change a
    sibling's resolution). A bundle axis or derived may depend on the root, on op-level
    shared axes, on its OWN bundle's axes, or on any derived — never on another bundle's
    axis.

    Covers a bundle's AXES and its DERIVED. Predicates carry no deps yet (engine hone Task
    3.1); when they do they join this same walk — the entry kind is irrelevant to the
    property, only the dep is."""
    shared_names = {a.name for a in shared_axes} | {root_name}
    for bundle in pool:
        own = {a.name for a in bundle.axes}
        allowed = shared_names | own
        # A sibling's axis names, keyed by the sibling that owns them.
        siblings = {s.name: {a.name for a in s.axes} for s in pool if s.name != bundle.name}

        for kind, nodes in (("axis", bundle.axes), ("derived", bundle.derived)):
            for node in nodes:
                # Both required and optional deps couple: an optional dep is a real edge
                # wherever the name exists, and a sibling's axis exists in the merged space.
                for dep in node.deps | node.optional_deps:
                    if dep in allowed:
                        continue
                    for sibling_name, sibling_axes in siblings.items():
                        if dep in sibling_axes:
                            raise PoolError(
                                f"implementation {bundle.name!r} {kind} {node.name!r} "
                                f"depends on sibling {sibling_name!r}'s axis {dep!r} — "
                                f"pool members must not couple to siblings"
                            )


def _check_no_derived_shadowing(shared_derived, pool, sources_key=SOURCES_KEY) -> None:
    """A bundle's derived must not shadow an op-level shared derived or the pool_space
    reserved ``sources`` projection key. ``DesignSpace`` only dedups *axis* names, so a colliding
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
                    f"{d.name!r}, which is reserved by pool_space"
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
        optional_union: set[str] = set()
        for ax in by_impl.values():
            dep_union |= set(ax.deps)
            optional_union |= set(ax.optional_deps)
        dep_union.add(root_name)
        dep_union.discard(axis_name)
        # A name REQUIRED by any owning impl is required on the merge, so drop it from the
        # optional set (declaring both is contradictory and rejected at construction).
        optional_union -= dep_union
        optional_union.discard(axis_name)

        merged.append(
            Axis(
                name=axis_name,
                domain=_dispatch_domain(root_name, by_impl),
                default=_dispatch_default(root_name, by_impl),
                exists=_dispatch_exists(root_name, owner_names, by_impl),
                deps=frozenset(dep_union),
                optional_deps=frozenset(optional_union),
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
        # Union each owning bundle's declared deps onto the merged derived (mirrors the axis
        # and dtype-register merges), so a tiling-generated derived carrying deps — e.g.
        # stream_width.out, whose output-dtype spec reads the ParamDatatype key — keeps its ordering
        # constraint through the merge. Same-name bundle deriveds normally declare identical
        # deps; the union is the safe superset.
        deps = frozenset().union(*(d.deps for d in by_impl.values()))
        # optional_deps merge the same way, with one wrinkle: if ANY owning bundle declares
        # a name as REQUIRED, the merged node requires it (the union above already has it),
        # so it must not also appear as optional — that pairing is rejected as contradictory.
        optional = frozenset().union(*(d.optional_deps for d in by_impl.values())) - deps
        merged.append(
            Derived(
                name,
                _dispatch_compute(root_name, by_impl),
                deps=deps,
                optional_deps=optional,
            )
        )
    return merged


def _merge_derived_dtypes(root_name, pool) -> list[Derived]:
    """Merge each backend's INTERNAL-REGISTER dtype specs (``Backend.derived_dtypes``) into
    name-keyed :class:`Derived`\\ s that resolve the owning bundle's
    :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` when its impl is selected, else
    ``None`` (present-but-None, matching the generic ``derived`` merge). The register name is
    the fallback tensor for a ``None``/``VALUE_OPTIMIZED`` spec — an internal register has no
    port, so the resolver only reads Context through it when the spec asks."""
    owners: dict[str, list[Backend]] = {}
    for bundle in pool:
        for name in bundle.derived_dtypes:
            owners.setdefault(name, []).append(bundle)

    from ..engine.datatype_spec import RegisterSpec

    merged: list[Derived] = []
    for name, owning in owners.items():
        by_impl = {b.name: b.derived_dtypes[name] for b in owning}
        # A register may wrap its spec in a RegisterSpec to declare derived deps (e.g.
        # accDataType reads the storage owner's ParamDatatype). Union the deps across
        # owning impls — mirrors _merge_axes' dep union — so the topo-sort orders the merged
        # register after whatever any owning impl reads. Bare specs contribute no deps.
        deps: set[str] = set()
        for spec in by_impl.values():
            if isinstance(spec, RegisterSpec):
                deps |= spec.deps
        merged.append(
            Derived(name, _dispatch_dtype_compute(root_name, name, by_impl), deps=frozenset(deps))
        )
    return merged


def _dispatch_dtype_compute(root_name, register_name, by_impl):
    from ..engine.datatype_spec import resolve_datatype_spec

    def compute(point, context, _root=root_name, _name=register_name, _by=by_impl):
        spec = _by.get(point[_root], _MISSING)
        if spec is _MISSING:
            return None
        return resolve_datatype_spec(spec, iface=_name, point=point, context=context)

    return compute


_MISSING = object()


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
    """Each bundle's predicates fire only when that bundle is selected. Device/dtype
    feasibility is just a predicate — there is no separate ``feasible`` mechanism (one
    concept, one home). Each port's declared datatype ``dtypes`` compiles through the SAME
    :func:`~finn.kernels.engine.constraints.compile_constraint` path as every other
    constraint (as a :class:`~finn.kernels.engine.constraints.DatatypeConstraint`), gaining
    the optional-port skip, then wraps in the selection guard — so the pool's UNION of
    supported datatypes is what a frontend claim accepts with no per-op logic and no
    special-cased branch."""
    from ..engine.constraints import DatatypeConstraint, compile_constraint

    wrapped: list[Predicate] = []
    for bundle in pool:
        for pred in bundle.predicates:
            wrapped.append(_guarded_predicate(root_name, bundle.name, pred))
        for iface, port in bundle.ports.items():
            if port.accepted_dtypes is not None:
                support_pred = compile_constraint(DatatypeConstraint(iface, port.accepted_dtypes))
                wrapped.append(_guarded_predicate(root_name, bundle.name, support_pred))
    return wrapped


def _guarded_predicate(root_name, impl_name, pred) -> Predicate:
    def check(point, context, _root=root_name, _name=impl_name, _p=pred):
        if point.get(_root) != _name:
            return None
        return _p.check(point, context)

    return Predicate(check=check, description=pred.describe())
