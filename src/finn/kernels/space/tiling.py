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
from typing import Union

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
