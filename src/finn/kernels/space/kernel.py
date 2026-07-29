############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Kernel`` — the WHAT-owning op node, and the Tier-3 (estimate-only) surface
it projects from a resolved :class:`~finn.kernels.space.point.Point`
(kernelop-tensor-block-stream.md §3, §7; consumer-surface-model.md Tier 0-3).

The op declares its **interfaces** (identity + role — the ONNX-facing arity, the
BLOCK/reduction structure) and a **pool** of Backends. Each Backend owns
its **stream tiling** (the BLOCK->STREAM lowering) — so a *normal* shape resolves with
no backend, while a *folded* shape needs a resolved point that names the selected impl
and its fold dials.

This module builds ONLY the estimate-only surface: the port-indexed normal/folded
shapes + stream widths, and a rough ``get_exp_cycles`` (``prod(stream_cycles)`` — the
monotone throughput floor that drives folding search). No emit, no codegen, no Vivado
(those are Tier-4). Cost is a defaultable op-level derived; a Backend may
override it (not exercised here).

SCOPE (increment 1): folding is a LAST-AXIS reshape of a DATA interface — the common
case shared by LayerNorm/elementwise/MVU activation+output. A PARAM interface whose
stream width is not a tensor-axis fold (MVU's weight port ``WSIMD=PE*SIMD/TH``) resolves
its *width* fine via the tiling evaluator, but its folded *shape* is not a plain
last-axis reshape; asking for that shape raises loudly rather than faking a reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..engine.context import Context
from .backend import BACKEND_AXIS, Backend, pool_schema
from .backend_interface import backend_interface_for
from ..engine.point import Illegal, Point
from .ports import Direction
from ..engine.resolve import resolve
from ..engine.schema import Schema
from .tiling import TileError, generate_tiling, stream_width_key as _stream_width_key


class KernelError(ValueError):
    """Raised for an ill-formed Kernel query (unknown interface index, a folded-shape
    request on a non-last-axis PARAM port, or a stream dial that does not divide)."""


@dataclass(frozen=True)
class InterfaceSchema:
    """One op-side interface — identity + DIRECTION + BLOCK structure (the math), NOT stream.

    The op declares the arity, the direction (which node slot), and how the block segments
    this tensor (``block`` — which dims a calc-state quantum spans). The selected Backend
    owns how the block is folded into a stream (its ``stream`` map). Block folding is the
    math → op-owned; stream folding is the realization → impl-owned. An impl cannot change
    the block: it has no field to express one (the ownership split, made structural).

    NO SEMANTIC ROLE. An interface is neutral: whether it is a weight/parameter feed or a
    dataflow activation is NOT declared — it EMERGES from graph context at resolve time
    (initializer attached → parameter; a producer → dataflow; else → boundary), read by the
    derived/predicate closures via ``ctx.initializer`` (resolution-phases.md: a
    Context-reading fact is phase-3, never a phase-1 declared field). Only DIRECTION (IN/OUT)
    is declared — a node-slot fact known from the op's math alone.

    Attributes:
        name: the interface key — ALSO the Context tensor name (shape + datatype). Matched
            against a Backend's ``stream`` map.
        direction: IN or OUT — which side of the node this interface sits on. A declare-time
            structural fact (the node slot), stored explicitly (not derived from list
            position — no positional rule survives inp/weights/out vs inp/out/indices).
        block: per-tensor-dim BLOCK extents, positional over the tensor's dims. Each is
            ``FULL`` (the whole dim sits in one block), ``1`` (iterate one at a time), an
            int size, or a ``derive(...)`` expr (a bounded / cross-interface block, e.g. a
            conv window). NO reduce/free tag — reduction is emergent from the math, not
            declared (matches brainsmith's block_tiling). The impl's ``stream[name][i]``
            folds ``block[i]``, positionally.
        dtype_source: the point key whose DataType supplies this interface's stream-width
            bitwidth, when it differs from the raw tensor dtype. MVAU's ``out`` sets
            ``"outputDataType"`` (= the accumulator type under ``noActivation``). None ⇒
            the tensor dtype.

        optional: whether this is an OPTIONAL node input (0-or-1). An ONNX-invariant
            identity fact (like ``direction``) — the opset says the slot may be absent (MVU
            thresholds, a bias, MaxPool indices). Its PRESENCE for a given node is EMERGENT
            from Context (its tensor exists), read at projection time by
            :meth:`Kernel.present_interfaces` — the same declared-slot / emergent-existence
            split as role emergence. A required interface (default ``False``) is always
            present. (Variadic 0-to-N is a later additive generalization of the same rule:
            present interfaces come from Context, not the declared list.)

    ``index`` is DERIVED, not stored: position among same-direction peers in the kernel's
    interface list (0=first, 1=…).
    """

    name: str
    direction: "Direction"
    block: tuple = ()
    dtype_source: str | None = None
    optional: bool = False

    def __post_init__(self):
        object.__setattr__(self, "block", tuple(self.block))

    @property
    def tensor(self) -> str:
        """The Context tensor name — equals ``name`` (the two were always the same key)."""
        return self.name


@dataclass(frozen=True)
class KernelSchema:
    """The Kernel's immutable, ONNX-invariant IDENTITY half — grouped into one named
    object so the identity ⊥ realization spine is structural in the code, not just
    conceptual (kernel-layers.md §2).

    Holds only what is true regardless of HOW the op is built: its ``name``, its
    ``interfaces`` (the inter-node communication contract — direction + BLOCK structure),
    and the op-level shared design space (``op_axes``/``op_derived``/``op_predicates`` —
    folding-independent geometry, datatype rules, legality), plus a rough op-level
    ``cost_model`` default. Every field is independently optional: a minimal op declares
    only ``name`` + ``interfaces`` (the whole space then comes from the pool's tiling).

    EXCLUDES the realization half — the pool of Backends and the delivered parameters
    (weight delivery, memory) are held by the :class:`Kernel` alongside this, never inside
    it. A ``KernelSchema`` compiles (with the pool) down to the flat resolve
    :class:`~finn.kernels.space.schema.Schema` via :meth:`Kernel.schema`."""

    name: str
    interfaces: tuple[InterfaceSchema, ...]
    op_axes: tuple = ()
    op_derived: tuple = ()
    op_predicates: tuple = ()
    cost_model: Any = None  # (point, context) -> int; None => the rough op-level default

    def __post_init__(self):
        object.__setattr__(self, "interfaces", tuple(self.interfaces))


@dataclass(frozen=True)
class Kernel:
    """A hardware kernel op: a :class:`KernelSchema` (identity) + a pool of Backends
    (realizations) + declared ``delivered_parameters`` (weight delivery, memory).

    ``identity`` is the ONNX-invariant half (name, interfaces, op-level axes/derived/
    predicates, rough cost). ``pool`` is the flat list of Backends; each owns its tiling,
    feasibility, sources, emit. ``delivered_parameters`` are the op's per-interface
    :class:`~finn.kernels.space.delivery.DeliveredParam` declarations, each lowered by a
    :class:`~finn.kernels.space.backend_interface.Interface` into the demand stage +
    guarded delivery pool. :meth:`schema` assembles all into the flat resolve ``Schema``;
    :meth:`configure` resolves a point; the getters project from it. The identity fields
    are exposed as read-only properties (``name``/``interfaces``/``op_axes``/… delegate to
    ``identity``) so callers read them off the Kernel unchanged.
    """

    identity: KernelSchema
    pool: tuple[Backend, ...]
    delivered_parameters: tuple = ()  # DeliveredParam per delivered interface; wired generically
    _tiling_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "pool", tuple(self.pool))
        object.__setattr__(self, "delivered_parameters", tuple(self.delivered_parameters))
        object.__setattr__(self, "_tiling_cache", {})

    # -- identity passthroughs: read the KernelSchema fields off the Kernel ---

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def interfaces(self) -> tuple[InterfaceSchema, ...]:
        return self.identity.interfaces

    @property
    def op_axes(self) -> tuple:
        return self.identity.op_axes

    @property
    def op_derived(self) -> tuple:
        return self.identity.op_derived

    @property
    def op_predicates(self) -> tuple:
        return self.identity.op_predicates

    @property
    def cost_model(self):
        return self.identity.cost_model

    # -- schema / resolve ---------------------------------------------------

    def _generated(self, impl: Backend):
        """The tiling engine's generated fragments + fold map for one Backend,
        derived from its ``stream`` map joined against the op interfaces' ``block``.
        Memoized per Kernel by impl name."""
        cache = self._tiling_cache
        got = cache.get(impl.name)
        if got is None:
            got = generate_tiling(self.interfaces, dict(impl.stream))
            cache[impl.name] = got
        return got

    def _augmented_pool(self) -> tuple[Backend, ...]:
        """Each Backend with the tiling-engine-generated axes/divisibility
        predicates appended to its OWN axes/predicates (so ``pool_schema`` dispatches
        them on selection), and the generated stream-width deriveds. The impl's declared
        tiling map is the single source; the fold dials, their ranges, the divisibility,
        and the widths are all derived here — not hand-written on the op."""
        from dataclasses import replace

        out = []
        for impl in self.pool:
            gen = self._generated(impl)
            out.append(
                replace(
                    impl,
                    axes=tuple(impl.axes) + gen.axes,
                    derived=tuple(impl.derived) + gen.derived,
                    predicates=tuple(impl.predicates) + gen.predicates,
                )
            )
        return tuple(out)

    def schema(self) -> Schema:
        """The full design space: the identity's op-level shared elements + the
        backend pool (each backend augmented with its tiling-engine-derived fold dials
        / divisibility / widths), plus the delivered parameters' realization sub-schemas.
        ``op_derived``/``op_predicates`` are PURE identity — the cross-coordinate memory
        couplings that once lived here relocated into the parameters pool, and the
        compute→memory demand crosses the seam owned by a ``Interface``."""
        op = pool_schema(
            BACKEND_AXIS,
            tuple(self.op_axes),
            tuple(self.op_derived),
            tuple(self.op_predicates),
            self._augmented_pool(),
            unspecialized_sentinel=True,  # compute root: "" = no backend committed (F1)
        )
        # DELIVERED PARAMETERS: the generic compute→delivery wiring, OWNED by a
        # Interface per delivered interface — the realization-side per-port object
        # that holds the DEMAND stage + guarded delivery sub-schema (design pitch §2). Its
        # to_subschemas() folds into the op schema in supply-waterfall order; the seam has
        # one owner and the waterfall is structural (its declared two-root deps) rather than
        # list-position. Reads the compute pool's `consumes` + each topology's `mode` — no
        # op-specific logic here. Namespaced keys (`parameters.*`) + distinct sources_key
        # mean the union never collides, so resolve walks it unchanged.
        for dp in self.delivered_parameters:
            for sub in backend_interface_for(dp, self.pool).to_subschemas():
                op = Schema(
                    axes=tuple(op.axes) + tuple(sub.axes),
                    derived=tuple(op.derived) + tuple(sub.derived),
                    predicates=tuple(op.predicates) + tuple(sub.predicates),
                )
        return op

    def configure(self, context: Context, assignment: Mapping | None = None):
        """Resolve a design point (or an Illegal). Thin wrapper over ``resolve``."""
        return resolve(self.schema(), context, assignment)

    def has_feasible_point(self, context: Context) -> bool:
        """Whether ANY pool member yields a legal :class:`Point` for this Context — the
        POOL-FEASIBILITY query (F2). Trials each backend with only its ``implementation``
        pinned (unpinned folding axes take defaults — the specialized-at-default-fold
        semantics); True if at least one resolves. Reuses the engine ``resolve`` unchanged.

        This is the SAME query resolve's ``PerNodePolicy(first_feasible)`` selector will use,
        so infer's claim check and resolve's selection converge. A node whose datatypes
        disqualify it from EVERY backend (e.g. float32 where only integer is feasible) has no
        feasible point, so ``can_infer_from`` can delegate to this rather than encoding a
        backend fact in the frontend."""
        schema = self.schema()
        for impl in self.pool:
            try:
                result = resolve(schema, context, {BACKEND_AXIS: impl.name})
            except Exception:  # noqa: BLE001
                # A backend feasibility check that raises on THIS context (e.g. a
                # device-family probe that needs an fpgapart the trial context omits) is not
                # feasible here — treat it as "no point for this backend", not a hard error.
                # A backend that IS feasible resolves cleanly; the pool needs only ONE.
                continue
            if isinstance(result, Point):
                return True
        return False

    def op_schema(self) -> Schema:
        """The op-level, impl-INDEPENDENT subschema: the identity's shared axes/derived/
        predicates ONLY, with NO pool (no ``implementation`` root axis, no per-backend fold
        dials). It resolves the folding-independent facts — the datatype contract
        (``accDataType``/``weightDataType``/``outputDataType``) and geometry — with NO
        committed backend. Used to publish output datatypes on an UNSPECIALIZED node, where
        the full :meth:`schema` would fabricate/require a backend selection (F1)."""
        return Schema(
            axes=tuple(self.op_axes),
            derived=tuple(self.op_derived),
            predicates=tuple(self.op_predicates),
        )

    def configure_op(self, context: Context, assignment: Mapping | None = None):
        """Resolve an op-level (impl-independent) point over :meth:`op_schema`. The
        datatype/geometry deriveds do not depend on the selected backend, so this succeeds on
        an unspecialized node."""
        return resolve(self.op_schema(), context, assignment)

    # -- interface lookup ---------------------------------------------------

    def present_interfaces(self, context: Context) -> tuple[InterfaceSchema, ...]:
        """The interfaces PRESENT for this node: all required ones, plus each optional one
        whose Context tensor exists. Presence is emergent (phase-3) — an ``optional=True``
        interface with no tensor in ``context`` is absent (the node did not wire that slot),
        so Context-reading loops (cost, any all-interface projection) skip it rather than
        ``KeyError``-ing on its shape. Required interfaces are always present."""
        out = []
        for i in self.interfaces:
            if i.optional and i.name not in context.shapes:
                continue
            out.append(i)
        return tuple(out)

    def inputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.interfaces if i.direction == Direction.IN)

    def outputs(self) -> tuple[InterfaceSchema, ...]:
        return tuple(i for i in self.interfaces if i.direction == Direction.OUT)

    def _input(self, ind: int) -> InterfaceSchema:
        ins = self.inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"input index {ind} out of range (have {len(ins)})")
        return ins[ind]

    def _output(self, ind: int) -> InterfaceSchema:
        outs = self.outputs()
        if ind < 0 or ind >= len(outs):
            raise KernelError(f"output index {ind} out of range (have {len(outs)})")
        return outs[ind]

    # -- Tier-3 getters: normal (TENSOR) shapes — no backend needed ---------

    def get_normal_input_shape(self, context: Context, ind: int = 0) -> tuple[int, ...]:
        return tuple(context.tensor_shape(self._input(ind).tensor))

    def get_normal_output_shape(self, context: Context, ind: int = 0) -> tuple[int, ...]:
        return tuple(context.tensor_shape(self._output(ind).tensor))

    def get_input_datatype(self, context: Context, ind: int = 0):
        return context.tensor_datatype(self._input(ind).tensor)

    def get_output_datatype(self, context: Context, ind: int = 0):
        return context.tensor_datatype(self._output(ind).tensor)

    # -- Tier-3 getters: folded (STREAM) shapes — need a resolved point -----

    def get_folded_input_shape(self, point: Point, context: Context, ind: int = 0):
        return self._folded_shape(self._input(ind), point, context)

    def get_folded_output_shape(self, point: Point, context: Context, ind: int = 0):
        return self._folded_shape(self._output(ind), point, context)

    def get_instream_width(self, point: Point, context: Context, ind: int = 0) -> int:
        return self._stream_width(self._input(ind), point, context)

    def get_outstream_width(self, point: Point, context: Context, ind: int = 0) -> int:
        return self._stream_width(self._output(ind), point, context)

    # -- rough cost: prod(stream_cycles) over the interfaces ----------------

    def get_exp_cycles(self, point: Point, context: Context) -> int:
        """PLACEHOLDER cost — NOT accurate. Returns the max over interfaces of each
        interface's stream-cycle count (``prod(tensor) / stream_elems``): monotone in the
        fold dials (so SetFolding still converges), but it does NOT model the nested
        cross-interface coupling real cost needs — e.g. MVU re-traverses the weight block
        once per input vector, so true cost is ``nf·sf·n_vecs`` while this floor gives only
        ``max(nf·sf, ...)`` (undercounts whenever n_vecs>1, i.e. conv-as-matmul). A proper
        cost model (nested block traversal, pipeline fill/drain, per-impl overrides) is a
        FUTURE PASS; cost modelling is deliberately ignored for now. ``cost_model`` remains
        an op-level escape hatch if an op needs a real number before then."""
        if self.cost_model is not None:
            return int(self.cost_model(point, context))
        cycles = 1
        for iface in self.present_interfaces(context):
            n = _prod(context.tensor_shape(iface.tensor))
            elems = self._stream_elems(iface, point)
            if elems <= 0 or n % elems != 0:
                # A partial last stream is a real cycle; round up.
                cycles = max(cycles, -(-n // max(elems, 1)))
            else:
                cycles = max(cycles, n // elems)
        return int(cycles)

    # -- internals ----------------------------------------------------------

    def selected_backend(self, point: Point) -> Backend:
        """The :class:`Backend` pool member the resolved ``point`` selected. Public accessor
        for reading a backend's STATIC identity fields (``language``/``rtl_core_module``) off
        the selection — those fields live on the ``Backend``, not re-projected onto the point
        (F5). Emit reads ``kernel.selected_backend(point).rtl_core_module``."""
        return self._selected(point)

    def _selected(self, point: Point) -> Backend:
        """The pool member named by the resolved ``backend`` axis."""
        impl_name = point[BACKEND_AXIS]
        by_name = {b.name: b for b in self.pool}
        bundle = by_name.get(impl_name)
        if bundle is None:
            raise KernelError(
                f"resolved implementation {impl_name!r} is not in the pool "
                f"(have {sorted(by_name)})"
            )
        return bundle

    def _stream_elems(self, iface: InterfaceSchema, point: Point) -> int:
        """Elements/cycle for this interface = the generated stream-width expression for
        the selected impl (1 if the impl declares no tiling for it)."""
        gen = self._generated(self._selected(point))
        expr = gen.width_exprs.get(iface.name)
        if expr is None:
            return 1
        try:
            return int(expr.eval(point))
        except TileError as exc:
            raise KernelError(f"interface {iface.name!r} tiling: {exc}") from exc

    def _stream_width(self, iface: InterfaceSchema, point: Point, context: Context) -> int:
        """Stream width in bits for this interface. Reads the per-interface
        ``stream_width.<iface>`` derived the tiling engine produced on the point — the
        SAME value emit reads, so the getter (FINN's contract) and emit share one produced
        quantity instead of a recompute-vs-precompute pair. Falls back to recomputing from
        ``width_exprs`` when the interface has no generated width key (an impl that declares
        no stream for it — the key is absent from the point)."""
        key = _stream_width_key(iface.name)
        if key in point:
            return int(point[key])
        # No generated width derived (impl declares no stream for this interface): the
        # stream is one element/cycle at the interface's dtype.
        elems = self._stream_elems(iface, point)
        if iface.dtype_source is not None:
            dt = point[iface.dtype_source]
        else:
            dt = context.tensor_datatype(iface.tensor)
        return elems * dt.bitwidth()

    def _folds_reshape(self, iface: InterfaceSchema, point: Point) -> bool:
        """Whether a folded SHAPE is a plain reshape for this interface under the selected
        impl. Derived from the stream folds (a fold whose width is a cross-interface expr
        ⇒ not a plain reshape). True when the impl declares no stream for the interface."""
        gen = self._generated(self._selected(point))
        return gen.reshapes.get(iface.name, True)

    def _folded_shape(self, iface: InterfaceSchema, point: Point, context: Context):
        """Fold each dim the impl's stream map folds: for each position ``(dim_index,
        elems_expr)``, split that tensor dim into ``(extent // elems, elems)``. The engine's
        ``fold_map`` names WHICH dims fold (any dim, not just the last), so a 2-D weight
        block ``(MW, MH)`` streamed ``[SIMD, PE]`` folds to ``(MW/SIMD, MH/PE, SIMD, PE)``.
        Raises when a fold width is a cross-interface expr (not a plain tensor-axis
        reshape)."""
        if not self._folds_reshape(iface, point):
            raise KernelError(
                f"interface {iface.name!r} does not fold a tensor axis (its stream WIDTH "
                f"resolves via get_*stream_width, but a folded SHAPE is not a plain "
                f"reshape for this port, e.g. a cross-interface weight width)"
            )
        normal = tuple(context.tensor_shape(iface.tensor))
        if not normal:
            raise KernelError(f"interface {iface.name!r} has no shape to fold")
        gen = self._generated(self._selected(point))
        fmap = gen.fold_map.get(iface.name)
        if fmap is None:
            return normal  # no stream for this interface ⇒ unfolded
        # Build the folded shape by expanding each folded position into (fold, elems).
        out: list[int] = []
        for dim_idx, elems_expr in fmap:
            extent = int(normal[dim_idx])
            if elems_expr is None:
                out.append(extent)
                continue
            elems = int(elems_expr.eval(point))
            if elems <= 0 or extent % elems != 0:
                raise KernelError(
                    f"interface {iface.name!r}: stream {elems} does not divide dim "
                    f"{dim_idx} = {extent} (illegal fold)"
                )
            out.extend((extent // elems, elems))
        return tuple(out)


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out
