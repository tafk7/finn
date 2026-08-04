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
:class:`TileExpr`. An expr is a small typed AST — ``Ref`` (an axis/derived/kernel_param
value on the point), ``Const``, ``Mul``/``Div`` (integer folding arithmetic), and
``BroadcastAware`` (the size-1 replicate exception) — evaluated against a resolved
:class:`~finn.kernels.engine.point.Point`.

Why an AST and not an opaque closure: the engine orders resolution by *declared*
dependencies because "closures cannot be introspected reliably" (the exact reason
:class:`~finn.kernels.engine.axis.Axis` carries an explicit ``deps`` frozenset). A
stream-tiling entry that reads ``PE``, ``SIMD`` and ``TH`` must expose those names so
the schema can order it and so an evaluator can check they are in scope. ``.deps()``
walks the tree and returns exactly the point keys the expr reads — never a guess.

These exprs live on a **Backend**'s interface (they ARE the RTL translation),
so their operands are backend-local: ``mvau_rtl_tiled``'s weight port
``div(mul(Ref('PE'), Ref('SIMD')), param('TH'))`` reads ``TH``, which exists only on
that bundle, so every dep is in scope wherever the expr is declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Union

from ..engine.point import Point

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
# Authoring helpers — the surface a Backend interface declares tiling with.
# =============================================================================


def derive(name: str) -> Ref:
    """Reference an axis/derived value on the point: ``derive("PE")``."""
    return Ref(name)


def param(name: str) -> Ref:
    """Reference a kernel_param on the point: ``param("TH")``. A kernel_param is a point
    key resolved identically to any axis/derived, so this returns a :class:`Ref` — the
    helper survives to let call sites read structural-parameter intent."""
    return Ref(name)


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
# A Backend declares, per interface, a list of specs positional over the
# interface tensor's dims. The engine derives what every op used to hand-write four
# times (divisor_axis + divisibility predicate + width derived + implicit last-axis
# fold): the single authoritative statement is "this dial folds this dim".
# =============================================================================


# -- BLOCK extents (op-owned) -------------------------------------------------
# An op interface declares, per tensor dim, a BLOCK extent: how much of the dim sits in
# one calc-state quantum. NO reduce/free tag — reduction is emergent from the math, not
# declared (matches brainsmith's block_tiling). The tokens:
#   FULL     — the whole tensor dim is in one block (brainsmith FULL_DIM)
#   1        — iterate one at a time (unblocked)
#   int > 1  — a bounded block (e.g. a conv window size)
#   TileExpr — a derived/cross-interface block extent


class _FullType:
    """Singleton sentinel: the whole tensor dim sits in one block."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "FULL"


FULL = _FullType()

# A block extent (op-side): FULL sentinel, an int, or a derived expr.
BlockExtent = Union["_FullType", int, TileExpr]
# A stream fold (impl-side): 1 (unfolded), a dial name, an int width, or an expr.
StreamFold = Union[str, int, TileExpr]


def _block_extent(extent: BlockExtent, iface, dim_idx: int, context) -> int:
    """Resolve a block extent to a concrete int against the Context. FULL → the tensor
    dim; an int → itself; a TileExpr → evaluated (rare, cross-interface block).

    A FULL/expr extent needs the Context; a context-less probe (the nodeattr registry
    enumerating static axes with ``context=None``) legitimately cannot resolve it — raise a
    NARROW ``ValueError`` (the "can't resolve for this probe" signal) rather than letting
    ``None.tensor_shape`` surface as an ``AttributeError`` that reads as a kernel bug (T0.1)."""
    if extent is FULL:
        if context is None:
            raise ValueError(
                f"block extent FULL for {iface.tensor!r} dim {dim_idx} needs a Context "
                f"(this axis is context-dependent; a context-less probe cannot resolve it)"
            )
        return int(tuple(context.tensor_shape(iface.tensor))[dim_idx])
    if isinstance(extent, TileExpr):
        if context is None:
            raise ValueError(
                f"block extent expr for {iface.tensor!r} dim {dim_idx} needs a Context "
                f"(context-dependent; a context-less probe cannot resolve it)"
            )
        return int(extent.eval_ctx(context)) if hasattr(extent, "eval_ctx") else int(extent.eval(context))
    return int(extent)


def _is_fold_dial(fold: StreamFold) -> str | None:
    """The dial name if this stream fold is a bare axis name (a genuine fold that sources
    a dial range), else None (``1``/int width/expr — folds shape but not a range source)."""
    return fold if isinstance(fold, str) else None


def _fold_elems_expr(fold: StreamFold) -> TileExpr:
    """The elements/cycle expression for one stream-fold entry: ``1``→Const(1); a dial
    name→Ref; an int→Const; a TileExpr→itself."""
    if isinstance(fold, str):
        return Ref(fold)
    if isinstance(fold, int):
        return Const(fold)
    if isinstance(fold, TileExpr):
        return fold
    raise TileError(f"bad stream fold {fold!r} ({type(fold).__name__})")


@dataclass(frozen=True)
class GeneratedTiling:
    """The schema fragments + fold map the engine derives from one impl's ``stream`` map
    joined against the op interfaces' ``block``.

    Attributes:
        axes/derived/predicates: fragments to append to the Backend's own before
            ``pool_space`` merges them (so they dispatch on the selected impl).
        width_exprs: ``{interface_name: TileExpr}`` — the elements/cycle width, used by
            the Kernel getters (``_stream_elems``).
        fold_map: ``{interface_name: [(dim_index, elems_expr | None)]}`` — the folded
            positions and their stream-element expressions; None = unfolded pass-through.
        reshapes: ``{interface_name: bool}`` — whether a folded SHAPE is a plain reshape
            of the interface's own tensor axes (all folds are named dials / 1) or must
            raise (a fold whose width is a cross-interface expr, e.g. the MVU weight port
            declared as an expr rather than per-dim dials).
    """

    axes: tuple
    derived: tuple
    predicates: tuple
    width_exprs: dict[str, TileExpr]
    fold_map: dict[str, list]
    reshapes: dict[str, bool]


def generate_tiling(interfaces, stream: dict, derived_dtypes: dict | None = None) -> GeneratedTiling:
    """Derive design-space fragments + the fold map from one Backend's ``stream``
    map joined against the op ``interfaces`` block structure.

    ``interfaces`` is the Kernel's interface tuple — each carries the op-owned ``block``
    (extents per tensor dim). ``stream`` is ``{interface_name: [StreamFold, ...]}`` —
    positional over the SAME dims: ``stream[iface][i]`` folds ``block[iface][i]``.
    ``derived_dtypes`` is ``{interface_name: DatatypeSpec}`` — the selected backend's
    per-OUTPUT-port produced dtype (:attr:`Backend.Interface.derived_dtype`); an interface
    absent (or ``None``) folds its raw graph dtype. Generates, for insertion into the schema:

      * a ``divisor_axis`` per fold DIAL — domain = divisors(GCD of every BLOCK extent the
        dial folds, across interfaces). Only bare-dial folds source a range; an int width
        or a cross-interface expr does not.
      * a divisibility ``Predicate`` per ``(dial, block-dim)`` fold.
      * a stream-width ``Derived`` keyed ``stream_width.<iface>`` per interface that folds
        (what emit + the getters read). One per interface — no singular-stream assumption,
        no arity guard; a multi-input op simply gets one width key per input.
    """
    by_name = {i.name: i for i in interfaces}

    width_exprs: dict[str, TileExpr] = {}
    fold_map: dict[str, list] = {}
    reshapes: dict[str, bool] = {}
    # dial -> [(interface_name, dim_index)] where a bare-dial fold binds it. Only these
    # source a dial's range + divisibility.
    dial_folds: dict[str, list[tuple[str, int]]] = {}

    for iface_name, folds in stream.items():
        if iface_name not in by_name:
            raise TileError(
                f"stream names interface {iface_name!r} not in the kernel "
                f"(have {sorted(by_name)})"
            )
        iface = by_name[iface_name]
        folds = list(folds)
        # A folds-list may be shorter than the block (leading dims default to unfolded).
        block = list(iface.block) if iface.block else [FULL] * len(folds)
        if len(folds) > len(block):
            raise TileError(
                f"interface {iface_name!r}: stream has {len(folds)} entries but block has "
                f"{len(block)} dims"
            )
        # Left-pad folds with 1 (unfolded) so positions align to the block's trailing dims.
        folds = [1] * (len(block) - len(folds)) + folds

        width: TileExpr = Const(1)
        fmap: list = []
        plain_reshape = True
        for dim_idx, fold in enumerate(folds):
            if fold == 1:
                fmap.append((dim_idx, None))
                continue
            elems = _fold_elems_expr(fold)
            width = Mul(width, elems)
            fmap.append((dim_idx, elems))
            dial = _is_fold_dial(fold)
            if dial is not None:
                dial_folds.setdefault(dial, []).append((iface_name, dim_idx))
            elif isinstance(fold, BroadcastAware):
                # A broadcast-aware fold is a per-dim fold of THIS interface's own axis
                # (streams 1 when broadcast, else the inner dial) — it reshapes. Its dial
                # range is sourced by a plain Fold elsewhere (a broadcast dim, extent maybe
                # 1, must not shrink the domain), so it is NOT added to dial_folds.
                pass
            elif isinstance(fold, TileExpr):
                # A general cross-interface width expr (e.g. the MVU weight PE*SIMD/TH) is
                # not a reshape of this interface's own tensor axes -> folded-shape raises.
                plain_reshape = False
        width_exprs[iface_name] = width
        fold_map[iface_name] = fmap
        reshapes[iface_name] = plain_reshape

    # -- fold-dial axes: divisor of the GCD of every BLOCK extent the dial folds ----
    axes: list = []
    for dial, binds in dial_folds.items():
        axes.append(_fold_dial_axis(dial, binds, by_name))

    # -- divisibility predicates: block_extent % dial == 0, per bare-dial fold ------
    predicates: list = []
    for dial, binds in dial_folds.items():
        for iface_name, dim_idx in binds:
            predicates.append(_divisibility_predicate(dial, by_name[iface_name], dim_idx))

    # -- stream-width deriveds: one PER INTERFACE (no arity guard) -----------------
    # Every interface with a fold expr publishes its own ``stream_width.<iface>`` key.
    # There is no "which one is THE instream/outstream" decision — the singular-stream
    # assumption (and its role filter + ``len==1`` guard) is dissolved, not relocated
    # (resolution-phases.md §4). Emit and the getter both read the per-interface key.
    derived_dtypes = derived_dtypes or {}
    derived: list = []
    for iface_name in stream:
        if iface_name in width_exprs:
            derived.append(
                _width_derived(
                    by_name[iface_name],
                    width_exprs[iface_name],
                    derived_dtypes.get(iface_name),
                )
            )

    return GeneratedTiling(
        axes=tuple(axes),
        derived=tuple(derived),
        predicates=tuple(predicates),
        width_exprs=width_exprs,
        fold_map=fold_map,
        reshapes=reshapes,
    )


def _fold_dial_axis(dial: str, binds, by_name):
    """A ``divisor_axis`` for ``dial``: domain = divisors of the GCD of every BLOCK extent
    it folds. Built directly so the domain closure reads the Context at resolve time; deps
    include ``implementation`` implicitly via the pool merge."""
    from ..engine.axis import Axis
    from ..engine.ordered_parameter import OrderedParameter

    def _gcd_block(context) -> int:
        g = 0
        for iface_name, dim_idx in binds:
            iface = by_name[iface_name]
            extent = iface.block[dim_idx] if iface.block else FULL
            g = gcd(g, _block_extent(extent, iface, dim_idx, context))
        return g

    def domain(p, ctx, _dial=dial):
        n = _gcd_block(ctx)
        divisors = tuple(d for d in range(1, n + 1) if n % d == 0)
        return OrderedParameter(_dial, divisors)

    def default(p, ctx):
        return 1

    return Axis(name=dial, domain=domain, default=default)


def _divisibility_predicate(dial: str, iface, dim_idx: int):
    from ..engine.predicate import Predicate

    def check(point, context, _dial=dial, _iface=iface, _idx=dim_idx):
        extent = _iface.block[_idx] if _iface.block else FULL
        block = _block_extent(extent, _iface, _idx, context)
        val = point.get(_dial)
        if val is None:
            return None  # dial guarded out under this impl — nothing to check
        if block % val != 0:
            return f"{_iface.tensor} block dim {_idx} = {block} not divisible by {_dial} = {val}"
        return None

    return Predicate(check=check, description=f"{iface.tensor} block[{dim_idx}] % {dial} == 0")


def stream_width_key(iface_name: str) -> str:
    """The point key under which an interface's resolved stream width (bits/cycle) is
    published. One PER INTERFACE (``stream_width.<iface>``, dotted like ``parameters.*``) —
    replacing the old singular ``instream_width``/``outstream_width`` pair, so there is no
    "which one is THE instream" arity decision. Both emit (subscript read) and the FINN
    getter path read this one produced value."""
    return f"stream_width.{iface_name}"


def _width_derived(iface, width_expr: TileExpr, derived_dtype=None):
    """A per-interface stream-width ``Derived`` = fold_width * bitwidth(dtype), keyed
    ``stream_width.<iface>``. The dtype is the selected backend's declared ``derived_dtype``
    :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` for this port (resolved against
    the point), or the raw graph tensor dtype when the backend declares none (``None``).

    When the port's spec is a :class:`~finn.kernels.engine.datatype_spec.RegisterSpec`, its
    declared ``deps`` flow onto this width derived — so a port whose produced dtype transitively
    reads another derived (e.g. MVAU's OUTPUT under no-activation resolves the accumulator,
    which reads the storage owner's ``storageDataType``) is ordered after that derived by the
    unified topo-sort. A bare spec declares no deps and the width derived resolves freely."""
    from ..engine.derived import Derived
    from ..engine.datatype_spec import RegisterSpec, resolve_datatype_spec

    deps = derived_dtype.deps if isinstance(derived_dtype, RegisterSpec) else frozenset()

    def compute(point, context, _expr=width_expr, _iface=iface, _spec=derived_dtype):
        elems = _expr.eval(point)
        dt = resolve_datatype_spec(
            _spec, iface=_iface.tensor, point=point, context=context
        )
        return int(elems) * dt.bitwidth()

    return Derived(stream_width_key(iface.name), compute, deps=deps)
