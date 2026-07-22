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
BLOCK/reduction structure) and a **pool** of Implementations. Each Implementation owns
its **stream tiling** (the BLOCK->STREAM lowering) — so a *normal* shape resolves with
no backend, while a *folded* shape needs a resolved point that names the selected impl
and its fold dials.

This module builds ONLY the estimate-only surface: the port-indexed normal/folded
shapes + stream widths, and a rough ``get_exp_cycles`` (``prod(stream_cycles)`` — the
monotone throughput floor that drives folding search). No emit, no codegen, no Vivado
(those are Tier-4). Cost is a defaultable op-level derived; an Implementation may
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

from .context import Context
from .implementation import Implementation, compose, pool_schema
from .point import Illegal, Point
from .ports import Role
from .resolve import resolve
from .schema import Schema
from .tiling import TileError, generate_tiling


class KernelError(ValueError):
    """Raised for an ill-formed Kernel query (unknown interface index, a folded-shape
    request on a non-last-axis PARAM port, or a stream dial that does not divide)."""


@dataclass(frozen=True)
class Interface:
    """One op-side interface — identity + role, NOT tiling.

    The op declares the arity and semantics (which tensor, which direction, what role);
    the selected Implementation owns how it is folded into a stream (its ``tiling``).

    Attributes:
        name: the interface key, matched against an Implementation's ``tiling`` map.
        tensor: the Context tensor name carrying this interface's shape + datatype.
        direction: ``Direction.IN`` / ``Direction.OUT``.
        role: the port-taxonomy role (DATA_IN/DATA_OUT/WEIGHT_SINK/…). DATA/WEIGHT roles
            carry a folded tensor shape; others do not.
        index: disambiguates same-direction interfaces (the port index the FINN getters
            use — 0=activation, 1=weights).
        folds_last_axis: True (default) when the stream folds this interface's last tensor
            axis, so a folded SHAPE is a plain ``normal[:-1] + (fold, elems)`` reshape
            (DATA ports). False for a PARAM port whose stream WIDTH is a cross-interface
            expression not tied to its own tensor axes (MVU weight ``WSIMD=PE*SIMD/TH``):
            its width still resolves via the tiling evaluator, but a folded-shape request
            raises rather than fake a reshape. When the impl declares tiling as spec lists
            (Full/Fold/WidthOnly), the engine derives this per interface (any WidthOnly
            position ⇒ not a plain reshape); this field is then the fallback/legacy path.
        dtype_source: the point key whose DataType supplies this interface's stream-width
            bitwidth, when it differs from the raw tensor dtype. MVAU's ``out`` sets
            ``"outputDataType"`` (= the accumulator type under ``noActivation``), so the
            generated stream width matches the emit-side derived. None ⇒ the tensor dtype.
    """

    name: str
    tensor: str
    direction: Any  # Direction
    role: Role
    index: int = 0
    folds_last_axis: bool = True
    dtype_source: str | None = None


@dataclass(frozen=True)
class Kernel:
    """A hardware kernel op: interfaces + op-level design space + an implementation pool.

    ``op_axes``/``op_derived``/``op_predicates`` are the op-level shared elements (the
    ONNX-invariant space every impl resolves against — folding-independent geometry,
    datatype rules, legality). ``pool`` is the flat list of Implementations; each owns
    its tiling, feasibility, sources, emit. :meth:`schema` assembles them via
    ``pool_schema``; :meth:`configure` resolves a point; the getters project from it.
    """

    name: str
    interfaces: tuple[Interface, ...]
    pool: tuple[Implementation, ...]
    op_axes: tuple = ()
    op_derived: tuple = ()
    op_predicates: tuple = ()
    cost_model: Any = None  # (point, context) -> int; None => the rough op-level default
    sub_schemas: tuple = ()  # secondary pools (e.g. parameters) composed via `compose`
    _by_name: Mapping[str, Interface] = field(default_factory=dict, init=False, repr=False)
    _tiling_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "pool", tuple(self.pool))
        object.__setattr__(self, "sub_schemas", tuple(self.sub_schemas))
        object.__setattr__(self, "_by_name", {i.name: i for i in self.interfaces})
        object.__setattr__(self, "_tiling_cache", {})

    # -- schema / resolve ---------------------------------------------------

    def _generated(self, impl: Implementation):
        """The tiling engine's generated fragments + fold map for one Implementation,
        derived from its ``tiling`` spec map. Memoized per Kernel by impl name."""
        cache = self._tiling_cache
        got = cache.get(impl.name)
        if got is None:
            got = generate_tiling(self.interfaces, dict(impl.tiling))
            cache[impl.name] = got
        return got

    def _augmented_pool(self) -> tuple[Implementation, ...]:
        """Each Implementation with the tiling-engine-generated axes/divisibility
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
        """The full design space: op-level shared elements + the implementation pool
        (each impl augmented with its tiling-engine-derived fold dials / divisibility /
        widths), plus any composed secondary pools (``sub_schemas`` — e.g. the parameters
        pool). ``op_derived``/``op_predicates`` already include the cross-coordinate
        couplings (those that read BOTH the compute fold and a sub-pool's topology),
        appended by the op before composition."""
        op = pool_schema(
            "implementation",
            tuple(self.op_axes),
            tuple(self.op_derived),
            tuple(self.op_predicates),
            self._augmented_pool(),
        )
        for sub in self.sub_schemas:
            op = compose(op, sub)
        return op

    def configure(self, context: Context, assignment: Mapping | None = None):
        """Resolve a design point (or an Illegal). Thin wrapper over ``resolve``."""
        return resolve(self.schema(), context, assignment)

    # -- interface lookup ---------------------------------------------------

    def inputs(self) -> tuple[Interface, ...]:
        from .ports import Direction

        return tuple(i for i in self.interfaces if i.direction == Direction.IN)

    def outputs(self) -> tuple[Interface, ...]:
        from .ports import Direction

        return tuple(i for i in self.interfaces if i.direction == Direction.OUT)

    def _input(self, ind: int) -> Interface:
        ins = self.inputs()
        if ind < 0 or ind >= len(ins):
            raise KernelError(f"input index {ind} out of range (have {len(ins)})")
        return ins[ind]

    def _output(self, ind: int) -> Interface:
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
        """Expected cycles for a resolved point.

        If the op declares a ``cost_model`` (a ``(point, context) -> int``), use it — the
        op-level model that captures cross-interface coupling the generic floor cannot
        (e.g. MVU's reduction trip x output trip x TH, invisible to the per-interface
        stream shapes; kernelop-tensor-block-stream.md §2.2). Otherwise fall back to the
        rough default: the max over interfaces of that interface's stream-cycle count
        (``prod(normal_shape) / stream_elems``), monotone in the fold dials — the property
        SetFolding needs. A precise per-IMPL override is Tier-4 work."""
        if self.cost_model is not None:
            return int(self.cost_model(point, context))
        cycles = 1
        for iface in self.interfaces:
            # A PARAM port that does not fold its own tensor axes has no per-output stream-
            # cycle count; it does not bound the streaming throughput floor (skip it).
            if not iface.folds_last_axis:
                continue
            n = _prod(context.tensor_shape(iface.tensor))
            elems = self._stream_elems(iface, point)
            if elems <= 0 or n % elems != 0:
                # A partial last stream is a real cycle; round up.
                cycles = max(cycles, -(-n // max(elems, 1)))
            else:
                cycles = max(cycles, n // elems)
        return int(cycles)

    # -- internals ----------------------------------------------------------

    def _selected(self, point: Point) -> Implementation:
        """The pool member named by the resolved ``implementation`` axis."""
        impl_name = point["implementation"]
        by_name = {b.name: b for b in self.pool}
        bundle = by_name.get(impl_name)
        if bundle is None:
            raise KernelError(
                f"resolved implementation {impl_name!r} is not in the pool "
                f"(have {sorted(by_name)})"
            )
        return bundle

    def _stream_elems(self, iface: Interface, point: Point) -> int:
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

    def _stream_width(self, iface: Interface, point: Point, context: Context) -> int:
        """Stream width in bits = elements/cycle * bitwidth(dtype). The dtype is the
        interface's declared ``dtype_source`` (a point key, e.g. ``outputDataType``) when
        set — so a derived output type is honored — else the raw tensor dtype."""
        elems = self._stream_elems(iface, point)
        if iface.dtype_source is not None:
            dt = point[iface.dtype_source]
        else:
            dt = context.tensor_datatype(iface.tensor)
        return elems * dt.bitwidth()

    def _folds_reshape(self, iface: Interface, point: Point) -> bool:
        """Whether a folded SHAPE is a plain reshape for this interface under the selected
        impl. Derived from the tiling specs (any WidthOnly position ⇒ not a reshape);
        falls back to the legacy ``folds_last_axis`` flag when the impl declares no
        tiling for the interface."""
        gen = self._generated(self._selected(point))
        if iface.name in gen.reshapes:
            return gen.reshapes[iface.name]
        return iface.folds_last_axis

    def _folded_shape(self, iface: Interface, point: Point, context: Context):
        """Fold the dims the impl's tiling map binds to a dial: for each folded position
        ``(dim_index, dial)``, split that tensor dim into ``(extent // elems, elems)``.
        The engine's ``fold_map`` names WHICH dim each dial folds (not just the last).
        Raises for a WidthOnly interface (the MVU weight port — width resolves via
        ``get_*stream_width`` but a folded SHAPE is not a plain reshape)."""
        if not self._folds_reshape(iface, point):
            raise KernelError(
                f"interface {iface.name!r} does not fold a tensor axis (its stream WIDTH "
                f"resolves via get_*stream_width, but a folded SHAPE is not a plain "
                f"reshape for this port, e.g. MVU weight WSIMD)"
            )
        normal = tuple(context.tensor_shape(iface.tensor))
        if not normal:
            raise KernelError(f"interface {iface.name!r} has no shape to fold")
        gen = self._generated(self._selected(point))
        fmap = gen.fold_map.get(iface.name)
        if fmap is None:
            return normal  # no tiling for this interface ⇒ unfolded
        # Build the folded shape by expanding each folded position into (fold, elems).
        out: list[int] = []
        for dim_idx, dial in fmap:
            extent = int(normal[dim_idx])
            if dial is None:
                out.append(extent)
                continue
            elems = int(point[dial])
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
