# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's iteration over named indices, and the traversal each port presents.

An ``Index`` names one dimension of an operation (``m``, ``n``, ``k``). Index
arithmetic builds the affine expressions a port reads its tensor by
(``oh * S + kh * D``); a plain index is the expression of coefficient one.

A ``Schedule`` is a kernel's iteration: each index's extent, its folding
factor (the lanes it spreads over each iteration; one when it has none) and the
iteration order, outer to inner. An index ``i`` of extent ``E`` and folding
factor ``F`` takes ``E / F`` iterations (its fold, ``fold``), each carrying ``F``
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

A port may read a *reshape* of its tensor: another row-major shape of the same
size, whose flat positions are the tensor's. That is how a dense datapath
reads a depthwise operand ``(M, K, N)`` as ``(M, K * N)``.

``closing`` is the marker of a reduction's last index value: the reduced indices must be
one contiguous block of the iteration order. With nothing inside the block it closes on
the reduction's last beat (``LevelEnd``); with indices inside, it holds across their
beats (``LevelDone``).

``beat_times`` is the same projection told as times: the iteration at which
each of a port's beats is presented, a reduced index's on its last index value and a
held one's on its first. A ``Pace`` is a port's schedule with its ``reduces`` and
``holds``: its beat times and its kernel's iterations a frame, which the ports on a
channel state with their contracts so that a FIFO between them can be sized.

``bind_extents`` gives the indices their extents from the tensors the ports
read (each an ``Access``), so a kernel states which index addresses which
axis and never writes an extent getter:

- An axis addressed by a plain index gives that index the axis's extent; an
  index given two different extents (by one port or two) is refused.
- An axis addressed by any other expression (a sliding window
  ``oh * S + kh``) binds nothing: its indices take their extents from another
  axis or from the author (``extents``), and it is checked to stay inside the
  axis.
- A reshaped access binds nothing (its reshape's extents are its indices'), and
  is checked: its indices bound elsewhere, its reshape the tensor's size.
- An index nothing binds and the author does not give is refused.

An index bound from an axis walks exactly that axis, so a port whose axes are
all bound (or read through a checked reshape) presents every position of its
tensor: a tensor wider than its kernel's other ports disagrees on an index and
is refused, not silently left partly unread.

Over the bound extents, ``schedule_over`` builds a kernel's schedule (its order and
folding factors) and ``span`` the shape an output's expressions span: a kernel
states its extents, its schedule and its outputs' shapes by these three, from its
ports' accesses, and holds none of their computation itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow.traversal import (
    LevelDone,
    LevelEnd,
    Loop,
    Traversal,
    axis_strides,
    level_done,
    require_positive,
)


class Refused(Refusal):
    """A schedule or expression that the requested beat sequence cannot be derived from:
    one rejection, ``code`` by ``owner``, the rule's check (``Schedule.present``,
    ``bind_extents``), its message the error's."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),), message)


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
    """Each index's extent and folding factor, and the iteration order, outer to inner."""

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
            raise ValueError(
                f"the iteration order {list(order)} names each index {list(extents)} once"
            )
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
        """The iteration order: the indices, outer to inner."""
        return tuple(index for index, _ in self.extents)

    def _known(self, *indices: Index) -> None:
        """``Refused`` naming the first of ``indices`` that is not an index of the schedule."""
        known = set(self.order)
        for index in indices:
            if index not in known:
                raise Refused(
                    "index-unknown", "Schedule", f"{index!r} is not an index of the schedule"
                )

    def extent(self, index: Index) -> int:
        """``index``'s extent; ``Refused`` when it is not an index of the schedule."""
        self._known(index)
        return dict(self.extents)[index]

    def factor(self, index: Index) -> int:
        """The lanes ``index`` spreads over each iteration, its folding factor; one when it has
        none.

        ``Refused`` when ``index`` is not an index of the schedule."""
        self._known(index)
        return dict(self.factors).get(index, 1)

    def fold(self, index: Index) -> int:
        """The iterations ``index`` takes, its fold: its extent over its folding factor."""
        return self.extent(index) // self.factor(index)

    @property
    def iterations(self) -> int:
        return prod(self.fold(index) for index in self.order)

    def present(
        self,
        shape: Sequence[int],
        index: Sequence[Index | Affine],
        *,
        lanes: Sequence[Index] = (),
        reduces: Sequence[Index] = (),
        holds: Sequence[Index] = (),
        reshape: Sequence[int] | None = None,
    ) -> Traversal:
        """The traversal a port reading ``shape`` at ``index`` presents.

        ``index`` gives each axis of ``reshape`` (``shape`` itself by default) as
        an expression over this schedule's indices.
        """
        shape = tuple(shape)
        reshaped = shape if reshape is None else tuple(reshape)
        for extent in (*shape, *reshaped):
            require_positive(extent, "tensor extent")
        if prod(reshaped) != prod(shape):
            raise Refused(
                "reshape-size-mismatch",
                "Schedule.present",
                f"a {shape} tensor cannot be reshaped to {reshaped}",
            )
        axes = tuple(Affine.of(axis) for axis in index)
        if len(axes) != len(reshaped):
            raise Refused(
                "rank-mismatch",
                "Schedule.present",
                f"{len(axes)} expressions for a rank-{len(reshaped)} tensor",
            )
        self._known(*(used for axis in axes for used in axis.indices), *lanes, *reduces, *holds)
        strides = axis_strides(reshaped)

        def stride(i: Index) -> int:
            return sum(axis.coefficient(i) * strides[j] for j, axis in enumerate(axes))

        for i in self.order:
            if self.factor(i) > 1 and i not in lanes and stride(i):
                raise Refused(
                    "lanes-move-position",
                    "Schedule.present",
                    f"{i!r}'s lanes move the position; carry them as lanes",
                )
        dropped = (*reduces, *holds)
        for i in dropped:
            if stride(i):
                word = "reduced" if i in reduces else "held"
                raise Refused(
                    "dropped-index-moves",
                    "Schedule.present",
                    f"{i!r} moves the position, so it cannot be {word} away",
                )
        beats = [
            Loop(self.fold(i), self.factor(i) * stride(i)) for i in self.order if i not in dropped
        ]
        lane_loops = [Loop(self.factor(i), stride(i)) for i in lanes]
        try:
            return Traversal(shape, beats, lane_loops)
        except ValueError as error:
            raise Refused("traversal-invalid", "Schedule.present", str(error)) from error

    def beat_times(
        self, *, reduces: Sequence[Index] = (), holds: Sequence[Index] = ()
    ) -> tuple[int, ...]:
        """The iteration at which each beat of a port is presented, in its order.

        The port is ``present``'s with the same ``reduces`` and ``holds``: it steps
        through every other index, one beat each iteration that reads it. A
        reduced index's beat is presented on its last index value (when the reduction
        closes), a held one's on its first (before the run). A port reading every
        beat is ``0, 1, …, iterations - 1``; an output closing a reduction of ``s``
        index values innermost is ``s - 1, 2s - 1, …``. Times count iterations from the
        frame's first: one a clock cycle when the kernel runs at its rate,
        before its pipeline's constant latency.
        """
        self._known(*reduces, *holds)
        if set(reduces) & set(holds):
            raise Refused(
                "reduced-and-held", "Schedule.beat_times", "an index is reduced or held, not both"
            )
        # Each index's weight in the iteration count: the folds of every index inside it.
        weight: dict[Index, int] = {}
        inner = 1
        for index in reversed(self.order):
            weight[index], inner = inner, inner * self.fold(index)
        times = [sum((self.fold(index) - 1) * weight[index] for index in reduces)]
        for index in self.order:
            if index in reduces or index in holds:
                continue
            stride = weight[index]
            times = [time + value * stride for time in times for value in range(self.fold(index))]
        return tuple(times)

    def closing(self, reduces: Sequence[Index]) -> LevelEnd | LevelDone:
        """The marker of each reduction's last index value, in iterations: asserted exactly on
        the iterations where every reduced index is at its last index value.

        The reduced indices must be one contiguous block of the iteration order, each named
        once. The indices inside the block (after it in the order) are spanned: the
        marker holds across their beats, the ``inner``, the product of their folds; its
        ``beats`` is the inner times the block's folds (``level_done``). With nothing
        inside the block inner is one beat and the marker is ``LevelEnd(beats)``, the
        reduction closing on its last beat.

        ``Refused`` (``reduction-not-innermost``) where nothing is reduced, an index is
        named twice or is not the schedule's, or an index that is not reduced lies between
        two reduced ones.

        The positions inside the block are one partial result each: the reduction holds
        that many accumulators. That is the kernel's fact, not the marker's.
        """
        order = self.order
        reduced = tuple(reduces)
        if not reduced or len(set(reduced)) != len(reduced) or not set(reduced) <= set(order):
            raise Refused(
                "reduction-not-innermost",
                "Schedule.closing",
                f"the reduced {list(reduces)} are not the innermost of {list(order)}",
            )
        places = sorted(order.index(index) for index in reduced)
        first, last = places[0], places[-1]
        between = [index for index in order[first : last + 1] if index not in reduced]
        if between:
            raise Refused(
                "reduction-not-innermost",
                "Schedule.closing",
                f"the reduced {list(reduces)} are not one block of {list(order)}: "
                f"{between} lie between them",
            )
        inner = prod(self.fold(index) for index in order[last + 1 :])
        return level_done(inner * prod(self.fold(index) for index in reduced), inner)


@dataclass(frozen=True)
class Pace:
    """When a port presents its beats: its kernel's ``schedule`` and the indices the
    port is presented after (``reduces``) or before (``holds``), as it presents them.

    ``times`` is each beat's iteration (``Schedule.beat_times``), ``span`` the
    iterations its kernel takes a frame: a port's beats lie within its span, and the rest
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
        return self.schedule.iterations


@dataclass(frozen=True)
class Access:
    """One port's read of a tensor: its ``shape`` and the expression of each axis.

    ``name`` names the port in refusals. A ``reshaped`` access reads a
    row-major reshape of ``shape`` whose axes are ``index``, each as long as its
    index's extent (``Schedule.present``'s ``reshape``).
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


def plain_index(axis: Index | Affine) -> Index | None:
    """The index an axis is addressed by alone (coefficient one: ``m`` or ``1 * m``), if
    any: what a reshaped port's every axis is, and what binds an axis's extent."""
    terms = Affine.of(axis).terms
    return terms[0][0] if len(terms) == 1 and terms[0][1] == 1 else None


def bind_extents(
    accesses: Sequence[Access], extents: Mapping[Index, int] | None = None
) -> dict[Index, int]:
    """Each index's extent, from the axes it alone addresses and the ``extents`` given.

    Raises ``Refused`` naming the access and axis: a rank mismatch, an index
    given two extents, an index without one, a windowed axis reaching past its
    extent, or a reshape of another size.
    """
    bound: dict[Index, int] = {}
    origin: dict[Index, str] = {}
    for index, extent in (extents or {}).items():
        if not isinstance(index, Index):
            raise TypeError("extents are given to Index values")
        if type(extent) is not int or extent < 1:
            raise Refused(
                "extent-invalid",
                "bind_extents",
                f"the extent given to {index!r} must be a positive integer",
            )
        bound[index], origin[index] = extent, "given"
    for access in accesses:
        if access.reshaped:
            if any(plain_index(axis) is None for axis in access.index):
                raise Refused(
                    "reshape-not-plain",
                    "bind_extents",
                    f"{access.name}: a reshaped port reads plain indices",
                )
        elif len(access.index) != len(access.shape):
            raise Refused(
                "rank-mismatch",
                "bind_extents",
                f"{access.name}: {len(access.index)} indices for a rank-{len(access.shape)} tensor",
            )
    # Binding: every plain axis of every access that is not a reshape.
    for access in accesses:
        if access.reshaped:
            continue
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            plain = plain_index(expression)
            if plain is None:
                continue
            here = f"{access.name} axis {axis}"
            known = bound.setdefault(plain, extent)
            origin.setdefault(plain, here)
            if known != extent:
                raise Refused(
                    "extent-conflict",
                    "bind_extents",
                    f"{plain!r} is {known} ({origin[plain]}) and {extent} ({here})",
                )
    # Checking: every index has an extent, windows stay inside, views keep the size.
    for access in accesses:
        for axis, expression in enumerate(access.index):
            for index in Affine.of(expression).indices:
                if index not in bound:
                    raise Refused(
                        "extent-unbound",
                        "bind_extents",
                        f"{access.name} axis {axis}: {index!r} has no extent "
                        "(no axis addresses it alone and none is given)",
                    )
        if access.reshaped:
            reshape = tuple(bound[Affine.of(axis).indices[0]] for axis in access.index)
            if prod(reshape) != prod(access.shape):
                raise Refused(
                    "reshape-size-mismatch",
                    "bind_extents",
                    f"{access.name}: a {access.shape} tensor cannot be reshaped to {reshape}",
                )
            continue
        for axis, (extent, expression) in enumerate(zip(access.shape, access.index)):
            if plain_index(expression) is None:
                terms = Affine.of(expression).terms
                reach = sum(c * (bound[i] - 1) for i, c in terms)
                if reach >= extent:
                    raise Refused(
                        "window-out-of-extent",
                        "bind_extents",
                        f"{access.name} axis {axis}: {expression!r} reaches {reach}, "
                        f"beyond extent {extent}",
                    )
    return bound


def schedule_over(
    extents: Mapping[Index, int],
    order: Sequence[Index],
    factors: Mapping[Index, int] | None = None,
) -> Schedule:
    """The schedule in ``order`` (outer to inner) over the bound ``extents``
    (``bind_extents``), with folding ``factors``.

    Raises ``Refused``: ``extent-unbound`` where an index of ``order`` has no extent,
    ``schedule-invalid`` where ``Schedule`` refuses the order or a factor (an index named
    twice, a factor that does not divide its extent), with its message.
    """
    missing = [index for index in order if index not in extents]
    if missing:
        raise Refused("extent-unbound", "schedule_over", f"{missing} are bound by no placed port")
    try:
        return Schedule({index: extents[index] for index in order}, factors, order)
    except ValueError as error:
        raise Refused("schedule-invalid", "schedule_over", str(error)) from error


def span(index: Sequence[Index | Affine], extents: Mapping[Index, int]) -> tuple[int, ...]:
    """The shape ``index`` spans over the bound ``extents``: an axis read at ``i`` spans
    ``i``'s extent, one at ``oh * OW + ow`` the reach of its terms, plus one. How a kernel
    states an output's shape, from the extents its inputs bind.

    Raises ``Refused`` (``extent-unbound``) at the first axis with an index that has no
    extent.
    """
    shape: list[int] = []
    for axis in index:
        terms = Affine.of(axis).terms
        missing = [i for i, _ in terms if i not in extents]
        if missing:
            raise Refused("extent-unbound", "span", f"{missing} are bound by no placed port")
        shape.append(1 + sum(c * (extents[i] - 1) for i, c in terms))
    return tuple(shape)


__all__ = [
    "Access",
    "Affine",
    "Index",
    "Pace",
    "Refused",
    "Schedule",
    "bind_extents",
    "plain_index",
    "schedule_over",
    "span",
]
