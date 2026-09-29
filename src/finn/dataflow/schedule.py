# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's iteration over named indices, and the traversal each port presents.

An ``Index`` names one dimension of an operation (``m``, ``n``, ``k``). Index
arithmetic builds the affine expressions a port reads its tensor by
(``oh * S + kh * D``); a plain index is the expression of coefficient one.

A ``Schedule`` is a kernel's iteration: each index's extent, its fold (the
lanes it spreads over each beat; one when unfolded) and the order of the beats,
outer to inner. A folded index ``i`` of extent ``E`` and fold ``F`` walks
``E / F`` beats, each carrying ``F`` lanes, at the position ``i_beat * F +
i_lane``. An index split into several temporal parts (a tile) is several
indices, joined in the expression a port reads by (``mt * T + t``).

``present`` projects the schedule through one port's expressions into the
``Traversal`` it presents; it is the single derivation rule, so ports that
share a schedule agree by construction.

- Each index's beat part becomes a beat loop, in the schedule's order; one the
  port does not read steps by zero, a replay.
- ``lanes`` orders the lane parts a port carries, outer first (field zero is
  the innermost): the hardware's field convention. A folded index the port does
  not carry must not move its position: a broadcast.
- ``reduces`` lists the indices whose beats a port is presented *after* (an
  output closing a reduction), ``holds`` those it is presented *before* (an
  operand held while they run). Either way the port does not step through
  them, so they must not move its position.

A port may read a *view* of its tensor: another row-major shape of the same
size, whose flat positions are the tensor's. That is how a dense datapath
reads a depthwise operand ``(M, K, N)`` as ``(M, K * N)``.

``closing`` is the marker a reduction ends with: the reduced indices must be
the innermost beats.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from finn.dataflow.traversal import LevelEnd, Loop, Traversal, axis_strides


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class Refused(ValueError):
    """A schedule or expression that the requested beat sequence cannot be derived from."""


@dataclass(frozen=True, order=True)
class Index:
    """One named dimension of an operation."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("an index has a nonempty name")

    def __mul__(self, coefficient: int) -> Affine:
        return Affine.of(self) * coefficient

    __rmul__ = __mul__

    def __add__(self, other: Index | Affine) -> Affine:
        return Affine.of(self) + other

    __radd__ = __add__

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Affine:
    """A sum of indices, each times a nonnegative integer coefficient."""

    terms: tuple[tuple[Index, int], ...]

    @classmethod
    def of(cls, value: Index | Affine) -> Affine:
        if isinstance(value, Affine):
            return value
        if isinstance(value, Index):
            return cls(((value, 1),))
        raise TypeError("an index expression is built from Index values")

    def __post_init__(self) -> None:
        merged: dict[Index, int] = {}
        for index, coefficient in self.terms:
            if not isinstance(index, Index):
                raise TypeError("an index expression is built from Index values")
            if type(coefficient) is not int or coefficient < 0:
                raise ValueError("index coefficients are nonnegative integers")
            merged[index] = merged.get(index, 0) + coefficient
        object.__setattr__(self, "terms", tuple(sorted((i, c) for i, c in merged.items() if c)))

    def coefficient(self, index: Index) -> int:
        return dict(self.terms).get(index, 0)

    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(index for index, _ in self.terms)

    def __mul__(self, coefficient: int) -> Affine:
        if type(coefficient) is not int:
            return NotImplemented
        return Affine(tuple((index, c * coefficient) for index, c in self.terms))

    __rmul__ = __mul__

    def __add__(self, other: Index | Affine) -> Affine:
        return Affine((*self.terms, *Affine.of(other).terms))

    __radd__ = __add__

    def __repr__(self) -> str:
        return " + ".join(f"{i!r}" if c == 1 else f"{i!r}*{c}" for i, c in self.terms) or "0"


@dataclass(frozen=True, init=False)
class Schedule:
    """Each index's extent and fold, and the beats' order, outer to inner."""

    extents: tuple[tuple[Index, int], ...]
    folds: tuple[tuple[Index, int], ...]

    def __init__(
        self,
        extents: Mapping[Index, int],
        folds: Mapping[Index, int] | None = None,
        beats: Sequence[Index] | None = None,
    ) -> None:
        order = tuple(extents) if beats is None else tuple(beats)
        if sorted(order) != sorted(extents) or len(set(order)) != len(order):
            raise ValueError(f"the beats {list(order)} order each index {list(extents)} once")
        folded = dict(folds or {})
        for index in folded:
            if index not in extents:
                raise ValueError(f"{index!r} is folded but has no extent")
        for index in order:
            extent, fold = extents[index], folded.get(index, 1)
            _positive(extent, f"{index!r}'s extent")
            _positive(fold, f"{index!r}'s fold")
            if extent % fold:
                raise ValueError(f"a fold of {fold} does not divide {index!r}'s extent {extent}")
        object.__setattr__(self, "extents", tuple((index, extents[index]) for index in order))
        object.__setattr__(
            self, "folds", tuple((index, folded[index]) for index in order if index in folded)
        )

    @property
    def beats(self) -> tuple[Index, ...]:
        """The indices in beat order, outer to inner."""
        return tuple(index for index, _ in self.extents)

    def extent(self, index: Index) -> int:
        return dict(self.extents)[index]

    def fold(self, index: Index) -> int:
        """The lanes ``index`` spreads over each beat; one when unfolded."""
        if index not in dict(self.extents):
            raise KeyError(index)
        return dict(self.folds).get(index, 1)

    def steps(self, index: Index) -> int:
        """The beats ``index`` walks: its extent over its fold."""
        return self.extent(index) // self.fold(index)

    @property
    def beat_count(self) -> int:
        return prod(self.steps(index) for index in self.beats)

    def present(
        self,
        shape: Sequence[int],
        index: Sequence[Index | Affine],
        *,
        lanes: Sequence[Index] = (),
        reduces: Sequence[Index] = (),
        holds: Sequence[Index] = (),
        view: Sequence[int] | None = None,
    ) -> Traversal:
        """The traversal a port reading ``shape`` at ``index`` presents.

        ``index`` gives each axis of ``view`` (``shape`` itself by default) as
        an expression over this schedule's indices.
        """
        shape = tuple(shape)
        viewed = shape if view is None else tuple(view)
        for extent in (*shape, *viewed):
            _positive(extent, "tensor extent")
        if prod(viewed) != prod(shape):
            raise Refused(f"a {shape} tensor cannot be viewed as {viewed}")
        axes = tuple(Affine.of(axis) for axis in index)
        if len(axes) != len(viewed):
            raise Refused(f"{len(axes)} expressions for a rank-{len(viewed)} tensor")
        known = set(self.beats)
        for axis in axes:
            for used in axis.indices:
                if used not in known:
                    raise Refused(f"{used!r} is not an index of the schedule")
        for named in (*lanes, *reduces, *holds):
            if named not in known:
                raise Refused(f"{named!r} is not an index of the schedule")
        strides = axis_strides(viewed)

        def stride(i: Index) -> int:
            return sum(axis.coefficient(i) * strides[j] for j, axis in enumerate(axes))

        for i in self.beats:
            if self.fold(i) > 1 and i not in lanes and stride(i):
                raise Refused(f"{i!r}'s lanes move the position; carry them as a field")
        dropped = (*reduces, *holds)
        for i in dropped:
            if stride(i):
                word = "reduced" if i in reduces else "held"
                raise Refused(f"{i!r} moves the position, so it cannot be {word} away")
        beats = [
            Loop(self.steps(i), self.fold(i) * stride(i)) for i in self.beats if i not in dropped
        ]
        fields = [Loop(self.fold(i), stride(i)) for i in lanes]
        try:
            return Traversal(shape, beats, fields)
        except ValueError as error:
            raise Refused(str(error)) from error

    def closing(self, reduces: Sequence[Index]) -> LevelEnd:
        """The marker ending each reduction: the reduced indices must be the innermost beats."""
        order = self.beats
        suffix = order[len(order) - len(reduces) :] if reduces else ()
        if not reduces or sorted(suffix) != sorted(reduces):
            raise Refused(f"the reduced {list(reduces)} are not the innermost of {list(order)}")
        return LevelEnd(prod(self.steps(index) for index in reduces))


__all__ = ["Affine", "Index", "Refused", "Schedule"]
