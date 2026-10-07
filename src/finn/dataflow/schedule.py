# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's iteration over named indices, and the traversal each port presents.

An ``Index`` names one dimension of an operation (``m``, ``n``, ``k``). Index
arithmetic builds the affine expressions a port reads its tensor by
(``oh * S + kh * D``); a plain index is the expression of coefficient one.

A ``Schedule`` is a kernel's iteration: each index's extent, its folding
factor (the lanes it spreads over each beat; one when it has none) and the
order of the beats, outer to inner. An index ``i`` of extent ``E`` and folding
factor ``F`` walks ``E / F`` beats (its fold, ``steps``), each carrying ``F``
lanes, at the position ``i_beat * F + i_lane``. An index split into several
temporal parts (a tile) is several indices, joined in the expression a port
reads by (``mt * T + t``).

``present`` projects the schedule through one port's expressions into the
``Traversal`` it presents; it is the single derivation rule, so ports that
share a schedule agree by construction.

- Each index's beat part becomes a beat loop, in the schedule's order; one the
  port does not read steps by zero, a replay.
- ``lanes`` orders the lane parts a port carries, outer first (lane zero is
  the innermost): the hardware's convention. An index with lanes that the
  port does not carry must not move its position: a broadcast.
- ``reduces`` lists the indices whose beats a port is presented *after* (an
  output closing a reduction), ``holds`` those it is presented *before* (an
  operand held while they run). Either way the port does not step through
  them, so they must not move its position.

A port may read a *view* of its tensor: another row-major shape of the same
size, whose flat positions are the tensor's. That is how a dense datapath
reads a depthwise operand ``(M, K, N)`` as ``(M, K * N)``.

``closing`` is the marker a reduction ends with: the reduced indices must be
the innermost beats.

``beat_times`` is the same projection told as times: the schedule beat at which
each of a port's beats is presented, a reduced index's on its last step and a
held one's on its first. A ``Pace`` is a port's schedule with its ``reduces`` and
``holds``: its beat times and its kernel's beats a frame, which a channel's ends
state with their contracts so that a FIFO between them can be sized.

``bind_extents`` gives the indices their extents from the tensors the ports
read (each an ``Access``), so a kernel states which index addresses which
axis and never writes an extent getter:

- An axis addressed by a plain index gives that index the axis's extent; an
  index given two different extents (by one port or two) is refused.
- An axis addressed by any other expression (a sliding window
  ``oh * S + kh``) binds nothing: its indices take their extents from another
  axis or from the author (``extents``), and it is checked to stay inside the
  axis.
- A reshaped access binds nothing (its view's extents are its indices'), and
  is checked: its indices bound elsewhere, its view the tensor's size.
- An index nothing binds and the author does not give is refused.

An index bound from an axis walks exactly that axis, so a port whose axes are
all bound (or read through a checked view) presents every position of its
tensor: a tensor wider than its kernel's other ports disagrees on an index and
is refused, not silently left partly unread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from finn.dataflow.traversal import LevelEnd, Loop, Traversal, axis_strides, require_positive


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
    """Each index's extent and folding factor, and the beat order, outer to inner."""

    extents: tuple[tuple[Index, int], ...]
    factors: tuple[tuple[Index, int], ...]

    def __init__(
        self,
        extents: Mapping[Index, int],
        factors: Mapping[Index, int] | None = None,
        order: Sequence[Index] | None = None,
    ) -> None:
        order = tuple(extents) if order is None else tuple(order)
        if sorted(order) != sorted(extents) or len(set(order)) != len(order):
            raise ValueError(f"the beat order {list(order)} names each index {list(extents)} once")
        given = dict(factors or {})
        for index in given:
            if index not in extents:
                raise ValueError(f"{index!r} has a folding factor but no extent")
        for index in order:
            extent, factor = extents[index], given.get(index, 1)
            require_positive(extent, f"{index!r}'s extent")
            require_positive(factor, f"{index!r}'s folding factor")
            if extent % factor:
                raise ValueError(
                    f"a folding factor of {factor} does not divide {index!r}'s extent {extent}"
                )
        object.__setattr__(self, "extents", tuple((index, extents[index]) for index in order))
        object.__setattr__(
            self, "factors", tuple((index, given[index]) for index in order if index in given)
        )

    @property
    def order(self) -> tuple[Index, ...]:
        """The beat order: the indices, outer to inner."""
        return tuple(index for index, _ in self.extents)

    def extent(self, index: Index) -> int:
        """``index``'s extent; ``Refused`` when it is not an index of the schedule."""
        extents = dict(self.extents)
        if index not in extents:
            raise Refused(f"{index!r} is not an index of the schedule")
        return extents[index]

    def factor(self, index: Index) -> int:
        """The lanes ``index`` spreads over each beat, its folding factor; one when it has none.

        ``Refused`` when ``index`` is not an index of the schedule."""
        if index not in dict(self.extents):
            raise Refused(f"{index!r} is not an index of the schedule")
        return dict(self.factors).get(index, 1)

    def steps(self, index: Index) -> int:
        """The beats ``index`` walks, its fold: its extent over its folding factor."""
        return self.extent(index) // self.factor(index)

    @property
    def beat_count(self) -> int:
        return prod(self.steps(index) for index in self.order)

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
            require_positive(extent, "tensor extent")
        if prod(viewed) != prod(shape):
            raise Refused(f"a {shape} tensor cannot be viewed as {viewed}")
        axes = tuple(Affine.of(axis) for axis in index)
        if len(axes) != len(viewed):
            raise Refused(f"{len(axes)} expressions for a rank-{len(viewed)} tensor")
        known = set(self.order)
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

        for i in self.order:
            if self.factor(i) > 1 and i not in lanes and stride(i):
                raise Refused(f"{i!r}'s lanes move the position; carry them as lanes")
        dropped = (*reduces, *holds)
        for i in dropped:
            if stride(i):
                word = "reduced" if i in reduces else "held"
                raise Refused(f"{i!r} moves the position, so it cannot be {word} away")
        beats = [
            Loop(self.steps(i), self.factor(i) * stride(i)) for i in self.order if i not in dropped
        ]
        lane_loops = [Loop(self.factor(i), stride(i)) for i in lanes]
        try:
            return Traversal(shape, beats, lane_loops)
        except ValueError as error:
            raise Refused(str(error)) from error

    def beat_times(
        self, *, reduces: Sequence[Index] = (), holds: Sequence[Index] = ()
    ) -> tuple[int, ...]:
        """The schedule beat at which each beat of a port is presented, in its order.

        The port is ``present``'s with the same ``reduces`` and ``holds``: it steps
        through every other index, one beat each schedule beat that reads it. A
        reduced index's beat is presented on its last step (when the reduction
        closes), a held one's on its first (before the run). A port reading every
        beat is ``0, 1, …, beat_count - 1``; an output closing a reduction of ``s``
        steps innermost is ``s - 1, 2s - 1, …``. Times count schedule beats from the
        frame's first: one a clock cycle when the kernel runs at its rate (K10),
        before its pipeline's constant latency.
        """
        known = set(self.order)
        for named in (*reduces, *holds):
            if named not in known:
                raise Refused(f"{named!r} is not an index of the schedule")
        if set(reduces) & set(holds):
            raise Refused("an index is reduced or held, not both")
        # Each index's weight in the beat count: the beats of every index inside it.
        weight: dict[Index, int] = {}
        inner = 1
        for index in reversed(self.order):
            weight[index], inner = inner, inner * self.steps(index)
        times = [sum((self.steps(index) - 1) * weight[index] for index in reduces)]
        for index in self.order:
            if index in reduces or index in holds:
                continue
            step = weight[index]
            times = [time + beat * step for time in times for beat in range(self.steps(index))]
        return tuple(times)

    def closing(self, reduces: Sequence[Index]) -> LevelEnd:
        """The marker ending each reduction: the reduced indices must be the innermost beats."""
        order = self.order
        suffix = order[len(order) - len(reduces) :] if reduces else ()
        if not reduces or sorted(suffix) != sorted(reduces):
            raise Refused(f"the reduced {list(reduces)} are not the innermost of {list(order)}")
        return LevelEnd(prod(self.steps(index) for index in reduces))


@dataclass(frozen=True)
class Pace:
    """When a port presents its beats: its kernel's ``schedule`` and the indices the
    port is presented after (``reduces``) or before (``holds``), as it presents them.

    ``times`` is each beat's schedule beat (``Schedule.beat_times``), ``span`` the
    beats its kernel takes a frame: a port's beats lie within its span, and the rest
    of a period is its kernel's idle time.
    """

    schedule: Schedule
    reduces: tuple[Index, ...] = ()
    holds: tuple[Index, ...] = ()

    @property
    def times(self) -> tuple[int, ...]:
        return self.schedule.beat_times(reduces=self.reduces, holds=self.holds)

    @property
    def span(self) -> int:
        return self.schedule.beat_count


@dataclass(frozen=True)
class Access:
    """One port's read of a tensor: its ``shape`` and the expression of each axis.

    ``name`` names the port in refusals. A ``reshaped`` access reads a
    row-major view of ``shape`` whose axes are ``index``, each as long as its
    index's extent (``Schedule.present``'s ``view``).
    """

    name: str
    shape: tuple[int, ...]
    index: tuple[Index | Affine, ...]
    reshaped: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("an access has a nonempty name")
        shape, index = tuple(self.shape), tuple(self.index)
        for extent in shape:
            require_positive(extent, f"{self.name}'s tensor extent")
        for axis in index:
            if not isinstance(axis, (Index, Affine)):
                raise TypeError("an access's index is built from Index values")
        if type(self.reshaped) is not bool:
            raise TypeError("reshaped is a bool")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "index", index)


def _plain(axis: Index | Affine) -> Index | None:
    """The index an axis is addressed by alone (coefficient one), if any."""
    terms = Affine.of(axis).terms
    return terms[0][0] if len(terms) == 1 and terms[0][1] == 1 else None


def bind_extents(
    accesses: Sequence[Access], extents: Mapping[Index, int] | None = None
) -> dict[Index, int]:
    """Each index's extent, from the axes it alone addresses and the ``extents`` given.

    Raises ``Refused`` naming the access and axis: a rank mismatch, an index
    given two extents, an index without one, a windowed axis reaching past its
    extent, or a view of another size.
    """
    bound: dict[Index, int] = {}
    origin: dict[Index, str] = {}
    for index, extent in (extents or {}).items():
        if not isinstance(index, Index):
            raise TypeError("extents are given to Index values")
        if type(extent) is not int or extent < 1:
            raise Refused(f"the extent given to {index!r} must be a positive integer")
        bound[index], origin[index] = extent, "given"
    for access in accesses:
        if access.reshaped:
            if any(_plain(axis) is None for axis in access.index):
                raise Refused(f"{access.name}: a reshaped port reads plain indices")
        elif len(access.index) != len(access.shape):
            raise Refused(
                f"{access.name}: {len(access.index)} indices for a rank-{len(access.shape)} tensor"
            )
    # Binding: every plain axis of every access that is not a view.
    for access in accesses:
        if access.reshaped:
            continue
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            plain = _plain(expression)
            if plain is None:
                continue
            here = f"{access.name} axis {axis}"
            known = bound.setdefault(plain, extent)
            origin.setdefault(plain, here)
            if known != extent:
                raise Refused(f"{plain!r} is {known} ({origin[plain]}) and {extent} ({here})")
    # Checking: every index has an extent, windows stay inside, views keep the size.
    for access in accesses:
        for axis, expression in enumerate(access.index):
            for index in Affine.of(expression).indices:
                if index not in bound:
                    raise Refused(
                        f"{access.name} axis {axis}: {index!r} has no extent "
                        "(no axis addresses it alone and none is given)"
                    )
        if access.reshaped:
            view = tuple(bound[Affine.of(axis).indices[0]] for axis in access.index)
            if prod(view) != prod(access.shape):
                raise Refused(f"{access.name}: a {access.shape} tensor cannot be viewed as {view}")
            continue
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            if _plain(expression) is None:
                terms = Affine.of(expression).terms
                reach = sum(c * (bound[i] - 1) for i, c in terms)
                if reach >= extent:
                    raise Refused(
                        f"{access.name} axis {axis}: {expression!r} reaches {reach}, "
                        f"beyond extent {extent}"
                    )
    return bound


__all__ = ["Access", "Affine", "Index", "Pace", "Refused", "Schedule", "bind_extents"]
