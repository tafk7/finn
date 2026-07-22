############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Derived stream-tiling expressions — the BLOCK→STREAM lowering as introspectable
data (kernelop-tensor-block-stream.md §5.1, §6.2).

A stream-tiling entry is either a plain axis name (``"SIMD"``) or a composed
:class:`TileExpr`. An expr is a small typed AST — ``Ref`` (an axis/derived value on
the point), ``Const``, ``Param`` (a kernel_param), ``Mul``/``Div`` (integer folding
arithmetic), and ``BroadcastAware`` (the size-1 replicate exception) — evaluated
against a resolved :class:`~finn.kernels.space.point.Point`.

Why an AST and not an opaque closure: the engine orders resolution by *declared*
dependencies because "closures cannot be introspected reliably" (the exact reason
:class:`~finn.kernels.space.axis.Axis` carries an explicit ``deps`` frozenset). A
stream-tiling entry that reads ``PE``, ``SIMD`` and ``TH`` must expose those names so
the schema can order it and so an evaluator can check they are in scope. ``.deps()``
walks the tree and returns exactly the point keys the expr reads — never a guess.

These exprs live on an **Implementation**'s interface (they ARE the RTL translation),
so their operands are backend-local: ``mvau_rtl_tiled``'s weight port
``div(mul(Ref('PE'), Ref('SIMD')), Param('TH'))`` reads ``TH``, which exists only on
that bundle, so every dep is in scope wherever the expr is declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Union

from .point import Point

# A stream-tiling entry: a bare axis name, a plain int, or a composed expression.
TileEntry = Union[str, int, "TileExpr"]


class TileError(ValueError):
    """Raised when a stream-tiling expr cannot be evaluated (missing dep, non-integer
    fold, or a divisor that does not divide its dividend)."""


class TileExpr:
    """Base of the stream-tiling expression AST. Subclasses implement ``eval`` and
    ``deps``; instances are immutable, hashable value objects."""

    def eval(self, point: Point) -> int:
        raise NotImplementedError

    def deps(self) -> frozenset[str]:
        """The point keys this expr reads — for schema dependency ordering."""
        raise NotImplementedError

    # Operator sugar so authors can write derive("PE") * derive("SIMD") / param("TH").
    def __mul__(self, other: TileEntry) -> "Mul":
        return Mul(self, _coerce(other))

    def __rmul__(self, other: TileEntry) -> "Mul":
        return Mul(_coerce(other), self)

    def __floordiv__(self, other: TileEntry) -> "Div":
        return Div(self, _coerce(other))

    def __truediv__(self, other: TileEntry) -> "Div":
        # Folding arithmetic is integer-exact; `/` and `//` mean the same here (Div
        # raises if the division is not exact), so authors may write either.
        return Div(self, _coerce(other))


def _coerce(entry: TileEntry) -> TileExpr:
    if isinstance(entry, TileExpr):
        return entry
    if isinstance(entry, str):
        return Ref(entry)
    if isinstance(entry, int):
        return Const(entry)
    raise TileError(f"cannot use {entry!r} ({type(entry).__name__}) as a tiling expression")


@dataclass(frozen=True)
class Const(TileExpr):
    """A fixed integer (e.g. an unfolded singleton axis)."""

    value: int

    def eval(self, point: Point) -> int:
        return self.value

    def deps(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class Ref(TileExpr):
    """The value of an axis or derived on the point (e.g. ``Ref("PE")``)."""

    name: str

    def eval(self, point: Point) -> int:
        try:
            value = point[self.name]
        except Exception as exc:  # AbsentAxisError or unknown name
            raise TileError(
                f"tiling expr reads {self.name!r}, absent from the point "
                f"(guarded out, or not a declared axis/derived of this implementation)"
            ) from exc
        return _as_int(self.name, value)

    def deps(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class Param(TileExpr):
    """A kernel_param value read off the point (e.g. ``Param("TH")``). Distinct from
    :class:`Ref` only in intent — a structural op parameter rather than a folding axis
    — but resolved identically; both are point keys."""

    name: str

    def eval(self, point: Point) -> int:
        try:
            value = point[self.name]
        except Exception as exc:
            raise TileError(
                f"tiling expr reads kernel_param {self.name!r}, absent from the point"
            ) from exc
        return _as_int(self.name, value)

    def deps(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class Mul(TileExpr):
    """Integer product of two exprs (e.g. weight-lane width ``PE * SIMD``)."""

    left: TileExpr
    right: TileExpr

    def eval(self, point: Point) -> int:
        return self.left.eval(point) * self.right.eval(point)

    def deps(self) -> frozenset[str]:
        return self.left.deps() | self.right.deps()


@dataclass(frozen=True)
class Div(TileExpr):
    """Exact integer quotient (e.g. tiled weight width ``(PE*SIMD) / TH``). Raises if
    the divisor does not evenly divide — a folding count is never fractional."""

    dividend: TileExpr
    divisor: TileExpr

    def eval(self, point: Point) -> int:
        num = self.dividend.eval(point)
        den = self.divisor.eval(point)
        if den == 0:
            raise TileError("tiling expr divides by zero")
        if num % den != 0:
            raise TileError(
                f"tiling expr {num} / {den} is not an exact integer fold "
                f"(divisor must divide the dividend)"
            )
        return num // den

    def deps(self) -> frozenset[str]:
        return self.dividend.deps() | self.divisor.deps()


@dataclass(frozen=True)
class BroadcastAware(TileExpr):
    """The broadcast (size-1) folding exception: fold ``dim`` by ``inner`` normally,
    but when the folded axis is broadcast (its tensor extent is 1) DO NOT fold — the
    stream carries the single element replicated, not ``extent/inner`` (elementwise
    ``rhs``; matrixvectoractivation/elementwise_binary broadcast branch).

    ``extent`` is the point key carrying the folded axis's tensor length (e.g. the
    last-dim size). When ``point[extent] == 1`` the result is 1 (replicate); otherwise
    it is ``inner.eval(point)`` (the normal fold width)."""

    extent: str
    inner: TileExpr

    def eval(self, point: Point) -> int:
        try:
            length = _as_int(self.extent, point[self.extent])
        except Exception as exc:
            raise TileError(
                f"broadcast_aware reads extent {self.extent!r}, absent from the point"
            ) from exc
        if length == 1:
            return 1
        return self.inner.eval(point)

    def deps(self) -> frozenset[str]:
        return frozenset({self.extent}) | self.inner.deps()


# =============================================================================
# Authoring helpers — the surface an Implementation interface declares tiling with.
# =============================================================================


def derive(name: str) -> Ref:
    """Reference an axis/derived value on the point: ``derive("PE")``."""
    return Ref(name)


def param(name: str) -> Param:
    """Reference a kernel_param on the point: ``param("TH")``."""
    return Param(name)


def const(value: int) -> Const:
    """A fixed integer stream width."""
    return Const(value)


def broadcast_aware(extent: str, inner: TileEntry) -> BroadcastAware:
    """Fold by ``inner`` unless the axis is broadcast (``point[extent] == 1``), then 1."""
    return BroadcastAware(extent, _coerce(inner))


def eval_entry(entry: TileEntry, point: Point) -> int:
    """Evaluate one stream-tiling entry (bare name, int, or expr) against a point."""
    return _coerce(entry).eval(point)


def entry_deps(entry: TileEntry) -> frozenset[str]:
    """The point keys one stream-tiling entry reads (for schema dependency ordering)."""
    return _coerce(entry).deps()


def _as_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TileError(
            f"tiling expr expected an integer for {name!r}, got {value!r} "
            f"({type(value).__name__})"
        )
    return value


# =============================================================================
# The tiling ENGINE — per-dimension fold specs that GENERATE the design-space
# fragments (fold-dial axes, divisibility predicates, stream-width deriveds) and
# the dim↔dial fold map, from ONE declaration (kernelop-tensor-block-stream.md §5).
#
# An Implementation declares, per interface, a list of specs positional over the
# interface tensor's dims. The engine derives what every op used to hand-write four
# times (divisor_axis + divisibility predicate + width derived + implicit last-axis
# fold): the single authoritative statement is "this dial folds this dim".
# =============================================================================


class DimSpec:
    """Base of a per-dimension fold spec. Subclasses are immutable value objects."""


@dataclass(frozen=True)
class Full(DimSpec):
    """This dim passes through unfolded — one element/cycle in this position."""


@dataclass(frozen=True)
class Fold(DimSpec):
    """A fold dial folds THIS dim. The block dim it divides is the interface tensor's
    extent at this position (from the Context). This is the one authoritative binding
    from which the dial's divisor domain, its divisibility predicate, and the folded-
    shape reshape all derive."""

    dial: str


@dataclass(frozen=True)
class WidthOnly(DimSpec):
    """A stream WIDTH that is not a reshape of this interface's own tensor axes — a
    cross-interface expression (MVU weight port ``PE*SIMD`` or ``PE*SIMD/TH``). Its
    width resolves via the :class:`TileExpr` evaluator; a folded-SHAPE request raises
    (today's ``folds_last_axis=False``). It never sources a dial's range — the dials it
    references are declared elsewhere (as normal impl axes, e.g. ``TH``)."""

    expr: TileExpr

    def __init__(self, expr: TileEntry):
        object.__setattr__(self, "expr", _coerce(expr))


@dataclass(frozen=True)
class Broadcast(DimSpec):
    """The size-1 replicate exception: fold this dim by ``dial`` normally, but when the
    dim's tensor extent is 1 stream a single replicated element (elementwise ``rhs``).
    ``extent`` is the point key carrying that dim's length."""

    extent: str
    dial: str


# The stream-tiling map an Implementation declares: interface name -> spec list, OR a
# legacy bare entry (a name/int/TileExpr) that lowers to a last-axis Fold/WidthOnly.
TilingSpec = Union[list, tuple, str, int, TileExpr]


def _normalize_specs(entry) -> list[DimSpec]:
    """Lower a tiling-map value to a list of DimSpec. A list/tuple is taken as-is (each
    element a DimSpec). A bare legacy entry lowers to a single trailing spec: a plain
    axis name/int -> Fold on the last axis (its width IS that dial); a TileExpr ->
    WidthOnly (a cross-interface width). Leading dims are implicitly Full."""
    if isinstance(entry, (list, tuple)):
        specs = list(entry)
        for s in specs:
            if not isinstance(s, DimSpec):
                raise TileError(
                    f"tiling spec list may contain only DimSpec (Full/Fold/WidthOnly/"
                    f"Broadcast), got {s!r} ({type(s).__name__})"
                )
        return specs
    if isinstance(entry, str):
        return [Fold(entry)]
    if isinstance(entry, int):
        # A literal fold width with no named dial — a constant single-position fold.
        return [WidthOnly(Const(entry))]
    if isinstance(entry, TileExpr):
        return [WidthOnly(entry)]
    raise TileError(
        f"cannot interpret tiling entry {entry!r} ({type(entry).__name__})"
    )


def _stream_width_expr(specs: list[DimSpec]) -> TileExpr:
    """The elements/cycle width for an interface = product of every folded position's
    stream dial (Full contributes 1). Fold -> Ref(dial); Broadcast -> BroadcastAware;
    WidthOnly -> its expr. This is what ``get_*stream_width`` evaluates."""
    width: TileExpr = Const(1)
    for s in specs:
        if isinstance(s, Full):
            continue
        elif isinstance(s, Fold):
            width = Mul(width, Ref(s.dial))
        elif isinstance(s, Broadcast):
            width = Mul(width, BroadcastAware(s.extent, Ref(s.dial)))
        elif isinstance(s, WidthOnly):
            width = Mul(width, s.expr)
        else:
            raise TileError(f"unknown DimSpec {s!r}")
    return width


def folds_a_tensor_axis(specs: list[DimSpec]) -> bool:
    """True when the interface's stream is a reshape of its OWN tensor axes (every spec
    is Full/Fold/Broadcast) — a folded SHAPE is a plain reshape. False when any position
    is WidthOnly (a cross-interface width; folded-shape must raise)."""
    return all(not isinstance(s, WidthOnly) for s in specs)


@dataclass(frozen=True)
class GeneratedTiling:
    """The schema fragments + fold map the engine derives from one impl's tiling map.

    Attributes:
        axes/derived/predicates: fragments to append to the Implementation's own before
            ``pool_schema`` merges them (so they dispatch on the selected impl).
        width_exprs: ``{interface_name: TileExpr}`` — the elements/cycle width, used by
            the Kernel getters (``_stream_elems``).
        fold_map: ``{interface_name: [(dim_index, dial | None)]}`` — the authoritative
            dim↔dial binding the facade uses to fold the correct axis; None = unfolded.
        reshapes: ``{interface_name: bool}`` — whether a folded SHAPE is a plain reshape
            (all Full/Fold/Broadcast) or must raise (any WidthOnly).
    """

    axes: tuple
    derived: tuple
    predicates: tuple
    width_exprs: dict[str, TileExpr]
    fold_map: dict[str, list]
    reshapes: dict[str, bool]


def generate_tiling(interfaces, tiling: dict) -> GeneratedTiling:
    """Derive design-space fragments + the fold map from one Implementation's tiling.

    ``interfaces`` is the Kernel's interface tuple (for tensor names, roles, dtype
    sources); ``tiling`` is ``{interface_name: TilingSpec}``. Generates, for insertion
    into the schema:

      * a ``divisor_axis`` per fold DIAL — domain = divisors(GCD of every block dim the
        dial folds, across interfaces). Only ``Fold`` appearances source the range; a
        dial that appears only inside ``WidthOnly`` has no range source and must be a
        declared impl axis (raises here if it is not resolvable that way — see below).
      * a divisibility ``Predicate`` per ``(dial, block-dim)`` fold.
      * a stream-width ``Derived`` named ``instream_width``/``outstream_width`` — ONLY
        for a single-DATA_IN / single-DATA_OUT op (what emit reads). Multi-input ops
        rely on the port-indexed getters and get no named derived (avoids collisions).

    The block dim a ``Fold`` divides is the interface tensor's extent at that position,
    read at resolve time from the Context via a ``fixed_axis``-style domain closure — so
    the generated dial's domain is context-dependent exactly like a hand-written
    ``divisor_axis``.
    """
    from .axis import Axis
    from .derived import Derived
    from .predicate import Predicate
    from .ports import Direction, Role

    by_name = {i.name: i for i in interfaces}

    width_exprs: dict[str, TileExpr] = {}
    fold_map: dict[str, list] = {}
    reshapes: dict[str, bool] = {}
    # dial -> list of (interface_name, dim_index) where a plain Fold binds it. ONLY plain
    # Fold specs source a dial's range + divisibility. A Broadcast dim (extent may be 1)
    # does NOT — a broadcast operand must not shrink the dial's domain to {1}; its own
    # divisibility is handled by the broadcast-aware width (1 when broadcast).
    dial_folds: dict[str, list[tuple[str, int]]] = {}

    for iface_name, entry in tiling.items():
        if iface_name not in by_name:
            raise TileError(
                f"tiling names interface {iface_name!r} not in the kernel "
                f"(have {sorted(by_name)})"
            )
        specs = _normalize_specs(entry)
        width_exprs[iface_name] = _stream_width_expr(specs)
        reshapes[iface_name] = folds_a_tensor_axis(specs)
        fmap: list = []
        for dim_idx, s in enumerate(specs):
            if isinstance(s, Fold):
                fmap.append((dim_idx, s.dial))
                dial_folds.setdefault(s.dial, []).append((iface_name, dim_idx))
            elif isinstance(s, Broadcast):
                # Folds the dim for SHAPE purposes, but does not constrain the dial range.
                fmap.append((dim_idx, s.dial))
            else:
                fmap.append((dim_idx, None))
        fold_map[iface_name] = fmap

    # A dial referenced by a Broadcast must also be a real Fold somewhere (its range
    # source); otherwise it has no domain. (A dial only inside WidthOnly is likewise
    # unsourced — declare it as a normal impl axis, e.g. TH.)
    for iface_name, entry in tiling.items():
        for s in _normalize_specs(entry):
            if isinstance(s, Broadcast) and s.dial not in dial_folds:
                raise TileError(
                    f"tiling dial {s.dial!r} is used only in a Broadcast (interface "
                    f"{iface_name!r}) and never as a plain Fold — it has no range source. "
                    f"Fold it on a non-broadcast interface, or declare it as an impl axis."
                )

    # -- fold-dial axes: divisor of the GCD of every block dim the dial FOLDS -------
    axes: list = []
    for dial, binds in dial_folds.items():
        axes.append(_fold_dial_axis(dial, binds, by_name))

    # -- divisibility predicates: block_dim % dial == 0, per plain Fold -------------
    predicates: list = []
    for dial, binds in dial_folds.items():
        for iface_name, dim_idx in binds:
            predicates.append(_divisibility_predicate(dial, by_name[iface_name], dim_idx))

    # -- stream-width deriveds (single DATA_IN / DATA_OUT only) --------------------
    derived: list = []
    data_ins = [i for i in interfaces if i.role == Role.DATA_IN]
    data_outs = [i for i in interfaces if i.role == Role.DATA_OUT]
    if len(data_ins) == 1 and data_ins[0].name in width_exprs:
        derived.append(
            _width_derived("instream_width", data_ins[0], width_exprs[data_ins[0].name])
        )
    if len(data_outs) == 1 and data_outs[0].name in width_exprs:
        derived.append(
            _width_derived("outstream_width", data_outs[0], width_exprs[data_outs[0].name])
        )

    return GeneratedTiling(
        axes=tuple(axes),
        derived=tuple(derived),
        predicates=tuple(predicates),
        width_exprs=width_exprs,
        fold_map=fold_map,
        reshapes=reshapes,
    )


def _block_dim(iface, dim_idx: int, context) -> int:
    """The interface tensor's extent at ``dim_idx`` (the block dim a fold divides).
    Supports negative-style indexing implicitly via Python list indexing."""
    shape = tuple(context.tensor_shape(iface.tensor))
    return int(shape[dim_idx])


def _fold_dial_axis(dial: str, binds, by_name):
    """A ``divisor_axis`` for ``dial``: domain = divisors of the GCD of every block dim
    it folds. Built directly (not via the axis factory) so the domain closure reads the
    Context at resolve time and the deps include ``implementation`` implicitly via the
    pool merge."""
    from .axis import Axis
    from finn.kernels.primitives.ordered_parameter import OrderedParameter

    def _gcd_block(context) -> int:
        g = 0
        for iface_name, dim_idx in binds:
            g = gcd(g, _block_dim(by_name[iface_name], dim_idx, context))
        return g

    def domain(p, ctx, _dial=dial):
        n = _gcd_block(ctx)
        divisors = tuple(d for d in range(1, n + 1) if n % d == 0)
        return OrderedParameter(_dial, divisors)

    def default(p, ctx):
        return 1

    return Axis(name=dial, domain=domain, default=default)


def _divisibility_predicate(dial: str, iface, dim_idx: int):
    from .predicate import Predicate

    def check(point, context, _dial=dial, _iface=iface, _idx=dim_idx):
        block = _block_dim(_iface, _idx, context)
        val = point.get(_dial)
        if val is None:
            return None  # dial guarded out under this impl — nothing to check
        if block % val != 0:
            return f"{_iface.tensor} dim {_idx} = {block} not divisible by {_dial} = {val}"
        return None

    return Predicate(check=check, description=f"{iface.tensor}[{dim_idx}] % {dial} == 0")


def _width_derived(name: str, iface, width_expr: TileExpr):
    """A stream-width ``Derived`` = fold_width * bitwidth(dtype_source). The dtype is the
    interface's declared ``dtype_source`` (a derived name, e.g. ``outputDataType``) when
    set, else the raw tensor dtype. Names it ``instream_width``/``outstream_width`` — the
    keys emit reads off the point."""
    from .derived import Derived

    dtype_source = getattr(iface, "dtype_source", None)

    def compute(point, context, _expr=width_expr, _iface=iface, _src=dtype_source):
        elems = _expr.eval(point)
        if _src is not None:
            dt = point[_src]
        else:
            dt = context.tensor_datatype(_iface.tensor)
        return int(elems) * dt.bitwidth()

    return Derived(name, compute)
