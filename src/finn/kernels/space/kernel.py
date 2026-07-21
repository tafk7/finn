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
from .tiling import TileError, eval_entry


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
            raises rather than fake a reshape.
    """

    name: str
    tensor: str
    direction: Any  # Direction
    role: Role
    index: int = 0
    folds_last_axis: bool = True


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

    def __post_init__(self):
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "pool", tuple(self.pool))
        object.__setattr__(self, "sub_schemas", tuple(self.sub_schemas))
        object.__setattr__(self, "_by_name", {i.name: i for i in self.interfaces})

    # -- schema / resolve ---------------------------------------------------

    def schema(self) -> Schema:
        """The full design space: op-level shared elements + the implementation pool,
        plus any composed secondary pools (``sub_schemas`` — e.g. the parameters/weight-
        delivery pool). ``op_derived``/``op_predicates`` already include the cross-
        coordinate couplings (those that read BOTH the compute fold and a sub-pool's
        topology), appended by the op before composition, matching ``mvau_schema``."""
        op = pool_schema(
            "implementation",
            tuple(self.op_axes),
            tuple(self.op_derived),
            tuple(self.op_predicates),
            self.pool,
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
        iface = self._input(ind)
        return self._stream_elems(iface, point) * context.tensor_datatype(iface.tensor).bitwidth()

    def get_outstream_width(self, point: Point, context: Context, ind: int = 0) -> int:
        iface = self._output(ind)
        return self._stream_elems(iface, point) * context.tensor_datatype(iface.tensor).bitwidth()

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

    def _tiling_entry(self, iface: Interface, point: Point):
        """The selected Implementation's stream-tiling entry for this interface, or None
        (unfolded). Looks the impl up by the resolved ``implementation`` axis."""
        impl_name = point["implementation"]
        by_name = {b.name: b for b in self.pool}
        bundle = by_name.get(impl_name)
        if bundle is None:
            raise KernelError(
                f"resolved implementation {impl_name!r} is not in the pool "
                f"(have {sorted(by_name)})"
            )
        return bundle.tiling.get(iface.name)

    def _stream_elems(self, iface: Interface, point: Point) -> int:
        """Elements/cycle for this interface = the resolved stream dial (1 if unfolded)."""
        entry = self._tiling_entry(iface, point)
        if entry is None:
            return 1
        try:
            return eval_entry(entry, point)
        except TileError as exc:
            raise KernelError(f"interface {iface.name!r} tiling: {exc}") from exc

    def _folded_shape(self, iface: Interface, point: Point, context: Context):
        """Last-axis fold: ``normal[:-1] + (fold, elems)`` where ``elems`` is the stream
        dial and ``fold = last_dim / elems``. Raises if the dial does not divide the last
        axis (a real illegal fold, not a silent floor), or if the interface's width is not
        a last-axis fold at all (``folds_last_axis=False`` — the MVU weight port, whose
        WIDTH resolves via the evaluator but whose SHAPE is not a tensor-axis reshape)."""
        if not iface.folds_last_axis:
            raise KernelError(
                f"interface {iface.name!r} does not fold a tensor axis (folds_last_axis="
                f"False); its stream WIDTH resolves via get_*stream_width, but a folded "
                f"SHAPE is not a plain reshape for this port (e.g. MVU weight WSIMD)"
            )
        normal = tuple(context.tensor_shape(iface.tensor))
        elems = self._stream_elems(iface, point)
        if not normal:
            raise KernelError(f"interface {iface.name!r} has no shape to fold")
        last = normal[-1]
        if last % elems != 0:
            raise KernelError(
                f"interface {iface.name!r}: stream {elems} does not divide last dim {last} "
                f"(illegal fold)"
            )
        return normal[:-1] + (last // elems, elems)


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out
