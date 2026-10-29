# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical beat order as a loop nest over a row-major tensor.

A ``Traversal`` walks an operand of ``shape`` with two loop nests, both listed
outer to inner. Each iteration of ``beat_loops`` is one beat; each iteration of
``lane_loops`` is one lane of that beat, lane zero first (least significant).
A loop advances the operand's flat row-major index by ``stride`` elements per
index value; a stride of zero repeats (replays) positions. Tiles, chunked tiles,
transposes, sliding windows and replay are all ordinary loop nests.

Construction canonicalizes the nests, so two traversals are equal exactly when
they present the same positions in the same beats and lanes. ``classify``
compares two traversals of one operand and names the adaptation a mismatch
needs: free lane wiring, a loop-nest reorder (``Reorder``: its frame, loop
extents and strides; replay, and windows that overlap or skip within the frame,
included), a width conversion, a lane regroup, or none at all.

A ``BeatSequence`` is what one side of a channel presents of the tensor it
carries: its traversal per pass, whether the pass repeats (``Repetition``), and
the marker rules it offers or requires: a level closing (``LevelEnd``, on its last
beat) or a level on its last index value (``LevelDone``, held across the levels inside it;
``level_done`` gives the one form of either). A side presenting one of several sets a
pass, chosen at run time by a selector stream, is ``Selected``.

``unreplayed`` is the boundary rule: the receiver of a channel realizes its own replay
and window, while whole-pass
repetition stays part of the stream; ``one_pass`` strips the whole-pass repetition.
``stored`` is what a cyclic source holds for its consumer, one pass of the
consumer's order without its replay (and the frame for a window that reads a position
twice), and ``stored_depth`` its size.

A port whose lanes cut its element order where no loop ends (four lanes against
rows of six) is no ``Traversal``: its form is a lane group (``Grouped``), an order
and a lane count. ``grouped`` gives the one form of an order at a lane count, the
``Traversal`` ``regrouped`` gives where the lanes close whole loops, a ``Grouped``
otherwise; a beat sequence's form is either.

Words and tensors: ``pack`` presents an integer operand as one raw word per beat,
and ``unpack``, its inverse, reads the operand back from the words; both at the
flat offsets ``offsets`` gives.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod

Position = tuple[int, ...]


def require_positive(value: int, name: str) -> None:
    """Raise ``ValueError`` unless ``value``, named ``name``, is a positive ``int``."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, order=True)
class Loop:
    extent: int
    stride: int

    def __post_init__(self) -> None:
        require_positive(self.extent, "loop extent")
        if type(self.stride) is not int or self.stride < 0:
            raise ValueError("a loop stride is a nonnegative number of elements")


def _canonical(loops: Sequence[Loop]) -> tuple[Loop, ...]:
    """Drop unit loops and merge an outer loop into a contiguous inner one."""
    merged: list[Loop] = []
    for loop in loops:
        if loop.extent == 1:
            continue
        if merged and merged[-1].stride == loop.extent * loop.stride:
            outer = merged.pop()
            loop = Loop(outer.extent * loop.extent, loop.stride)
        merged.append(loop)
    return tuple(merged)


def _offsets(loops: Sequence[Loop]) -> Iterator[int]:
    if not loops:
        yield 0
        return
    head, rest = loops[0], loops[1:]
    inner = tuple(_offsets(rest))
    for index in range(head.extent):
        for offset in inner:
            yield index * head.stride + offset


def axis_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Row-major element strides of each axis, outer first."""
    return tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))


AxisLoop = tuple[int | None, int, int]
"""(operand axis or None for a replay loop, extent, stride along that axis)."""


@dataclass(frozen=True, init=False)
class Traversal:
    shape: tuple[int, ...]
    beat_loops: tuple[Loop, ...]
    lane_loops: tuple[Loop, ...]

    def __init__(
        self, shape: Sequence[int], beat_loops: Sequence[Loop], lane_loops: Sequence[Loop]
    ) -> None:
        shape = tuple(shape)
        if not shape:
            raise ValueError("a traversal walks an operand of rank at least one")
        for extent in shape:
            require_positive(extent, "operand extent")
        size = prod(shape)
        for loop in (*beat_loops, *lane_loops):
            if not isinstance(loop, Loop):
                raise TypeError("traversal loops are Loop values")
            if loop.stride and (loop.extent - 1) * loop.stride >= size:
                raise ValueError(f"{loop} leaves an operand of {size} elements")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "beat_loops", _canonical(beat_loops))
        object.__setattr__(self, "lane_loops", _canonical(lane_loops))
        reach = sum(
            (loop.extent - 1) * loop.stride for loop in (*self.beat_loops, *self.lane_loops)
        )
        if reach >= size:
            raise ValueError("the traversal addresses positions outside the operand")

    @classmethod
    def over(
        cls, shape: Sequence[int], beats: Sequence[AxisLoop], lanes: Sequence[AxisLoop]
    ) -> Traversal:
        """Build from (axis, extent, stride) loops; axis None is a replay loop."""
        strides = axis_strides(tuple(shape))

        def loops(axes: Sequence[AxisLoop]) -> tuple[Loop, ...]:
            return tuple(
                Loop(extent, 0 if axis is None else stride * strides[axis])
                for axis, extent, stride in axes
            )

        return cls(shape, loops(beats), loops(lanes))

    @property
    def lanes(self) -> int:
        return prod(loop.extent for loop in self.lane_loops)

    @property
    def beats(self) -> int:
        return prod(loop.extent for loop in self.beat_loops)

    @property
    def row_major(self) -> bool:
        """Whether it presents each position once, in the operand's row-major order: beat
        after beat, lane zero first, as a flat buffer of the tensor holds them."""
        lanes, beats = self.lanes, self.beats
        if lanes * beats != prod(self.shape):
            return False
        return self == Traversal(self.shape, (Loop(beats, lanes),), (Loop(lanes, 1),))

    def position(self, beat: int, lane: int) -> Position:
        flat = 0
        for loops, index in ((self.beat_loops, beat), (self.lane_loops, lane)):
            for loop in reversed(loops):
                index, digit = divmod(index, loop.extent)
                flat += digit * loop.stride
        position = []
        for extent in reversed(self.shape):
            flat, digit = divmod(flat, extent)
            position.append(digit)
        return tuple(reversed(position))

    def positions(self) -> Iterator[tuple[Position, ...]]:
        for beat in range(self.beats):
            yield tuple(self.position(beat, lane) for lane in range(self.lanes))

    def repeated(self, count: int) -> Traversal:
        """The whole pass presented ``count`` times."""
        require_positive(count, "count")
        return Traversal(self.shape, (Loop(count, 0), *self.beat_loops), self.lane_loops)

    def replayed(self, count: int, *, inner_beats: int) -> Traversal:
        """Present every consecutive group of ``inner_beats`` beats ``count`` times."""
        require_positive(count, "count")
        outer, inner = _split_at(self.beat_loops, inner_beats)
        return Traversal(self.shape, (*outer, Loop(count, 0), *inner), self.lane_loops)


def vector_major(shape: Sequence[int], lanes: int) -> Traversal:
    """FINN's default order: row-major, the innermost axis split into ``lanes`` lanes."""
    shape = tuple(shape)
    require_positive(lanes, "lanes")
    if shape[-1] % lanes:
        raise ValueError("lanes must divide the innermost extent")
    last = len(shape) - 1
    beats = [(axis, extent, 1) for axis, extent in enumerate(shape[:-1])]
    return Traversal.over(shape, (*beats, (last, shape[-1] // lanes, lanes)), ((last, lanes, 1),))


def tile(rows: int, cols: int, pe: int, simd: int) -> Traversal:
    """A (rows, cols) matrix in pe x simd tiles: row folds, then column folds; SIMD fastest."""
    if rows % pe or cols % simd:
        raise ValueError("PE must divide rows and SIMD must divide cols")
    return Traversal.over(
        (rows, cols),
        ((0, rows // pe, pe), (1, cols // simd, simd)),
        ((0, pe, 1), (1, simd, 1)),
    )


def _split(loop: Loop, boundary: int) -> tuple[Loop, ...] | None:
    """Split one nonzero-stride loop at a stride boundary strictly inside it."""
    low, high = loop.stride, loop.stride * loop.extent
    if not loop.stride or not low < boundary < high:
        return (loop,)
    if boundary % low or loop.extent % (boundary // low):
        return None
    inner = boundary // low
    return (Loop(loop.extent // inner, boundary), Loop(inner, loop.stride))


def _refine(loops: Sequence[Loop], boundaries: set[int]) -> tuple[Loop, ...] | None:
    refined: list[Loop] = []
    for loop in loops:
        pieces: tuple[Loop, ...] | None = (loop,)
        for boundary in sorted(boundaries, reverse=True):
            next_pieces: list[Loop] = []
            for piece in pieces or ():
                split = _split(piece, boundary)
                if split is None:
                    return None
                next_pieces.extend(split)
            pieces = tuple(next_pieces)
        refined.extend(pieces or ())
    return tuple(refined)


def _common_refinement(
    first: Sequence[Loop], second: Sequence[Loop]
) -> tuple[tuple[Loop, ...], tuple[Loop, ...]] | None:
    boundaries = {
        value
        for loop in (*first, *second)
        if loop.stride
        for value in (loop.stride, loop.stride * loop.extent)
    }
    a, b = _refine(first, boundaries), _refine(second, boundaries)
    return None if a is None or b is None else (a, b)


def regrouped(form: Traversal | Grouped, lanes: int) -> Traversal:
    """The same element sequence, ``lanes`` elements a beat: a width conversion. The
    lanes must close whole loops of the order (``ValueError`` otherwise: ``grouped``
    gives a lane group where they divide a pass)."""
    beats, lane_loops = _split_at(_order(form), lanes)
    return Traversal(form.shape, beats, lane_loops)


def _leading_replay(loops: Sequence[Loop]) -> int:
    """How many of ``loops``, outermost first, have stride zero: a whole-pass repetition."""
    return next((index for index, loop in enumerate(loops) if loop.stride), len(loops))


def _elements(loops: Sequence[Loop]) -> int:
    return prod(loop.extent for loop in loops)


@dataclass(frozen=True, init=False)
class Grouped:
    """A stream whose lanes group its element order across its loops: ``lanes`` elements
    a beat of ``order``, a loop nest over a row-major tensor of ``shape`` (outer to inner,
    canonical as a ``Traversal``'s). Beat ``b``, lane ``l`` is element ``b * lanes + l``
    of the order.

    It is the form of a port whose beats cut its order where no loop ends: four lanes
    against rows of six. Where ``lanes`` elements close whole innermost loops the group is
    aligned, and the form is that ``Traversal`` (``regrouped``), never a ``Grouped``: the
    constructor refuses an aligned group, so ``==`` stays the equality of positions.

    The lane count divides the elements of one pass, the order without its outermost
    stride-0 loops (``one_pass``): a beat never holds elements of two passes of a
    repetition. It may hold elements of two presentations of a replay inside the pass.
    """

    shape: tuple[int, ...]
    order: tuple[Loop, ...]
    lanes: int

    def __init__(self, shape: Sequence[int], order: Sequence[Loop], lanes: int) -> None:
        walk = Traversal(shape, order, ())
        require_positive(lanes, "lanes")
        loops = walk.beat_loops
        each = _elements(loops[_leading_replay(loops) :])
        if each % lanes:
            raise ValueError(f"{lanes} lanes do not divide the {each} elements of a pass")
        try:
            _split_at(loops, lanes)
        except ValueError:
            pass
        else:
            raise ValueError(f"{lanes} lanes close whole loops: an aligned group is a Traversal")
        object.__setattr__(self, "shape", walk.shape)
        object.__setattr__(self, "order", loops)
        object.__setattr__(self, "lanes", lanes)

    @property
    def beats(self) -> int:
        return _elements(self.order) // self.lanes

    @property
    def row_major(self) -> bool:
        """Never: a row-major order is one loop, which any dividing lane count splits."""
        return False

    def position(self, beat: int, lane: int) -> Position:
        index, flat = beat * self.lanes + lane, 0
        for loop in reversed(self.order):
            index, digit = divmod(index, loop.extent)
            flat += digit * loop.stride
        position = []
        for extent in reversed(self.shape):
            flat, digit = divmod(flat, extent)
            position.append(digit)
        return tuple(reversed(position))

    def positions(self) -> Iterator[tuple[Position, ...]]:
        for beat in range(self.beats):
            yield tuple(self.position(beat, lane) for lane in range(self.lanes))

    def repeated(self, count: int) -> Grouped:
        """The whole pass presented ``count`` times."""
        require_positive(count, "count")
        return Grouped(self.shape, (Loop(count, 0), *self.order), self.lanes)


def _order(form: Traversal | Grouped) -> tuple[Loop, ...]:
    """``form``'s element order, beat after beat, lane zero first: its loops, flattened."""
    if isinstance(form, Grouped):
        return form.order
    return _canonical((*form.beat_loops, *form.lane_loops))


def grouped(form: Traversal | Grouped, lanes: int) -> Traversal | Grouped:
    """The same element order, ``lanes`` elements a beat: ``regrouped``'s ``Traversal``
    where the lanes close whole loops of the order, else a ``Grouped`` where they divide
    the elements of a pass. Every other count is ``regrouped``'s refusal, by the same
    class and message: not positive, more than the pass, or no divisor of a pass's
    elements ("does not align")."""
    try:
        return regrouped(form, lanes)
    except ValueError:
        loops = _order(form)
        if type(lanes) is not int or lanes < 1:
            raise
        if _elements(loops[_leading_replay(loops) :]) % lanes:
            raise
        return Grouped(form.shape, loops, lanes)


def _split_at(loops: Sequence[Loop], inner_beats: int) -> tuple[tuple[Loop, ...], tuple[Loop, ...]]:
    require_positive(inner_beats, "inner_beats")
    outer: list[Loop] = list(loops)
    inner: list[Loop] = []
    remaining = inner_beats
    while remaining > 1:
        if not outer:
            raise ValueError("inner_beats exceeds the pass")
        loop = outer.pop()
        if loop.extent <= remaining:
            if remaining % loop.extent:
                raise ValueError("inner_beats does not align with the loop nest")
            inner.insert(0, loop)
            remaining //= loop.extent
            continue
        if loop.extent % remaining:
            raise ValueError("inner_beats does not align with the loop nest")
        outer.append(Loop(loop.extent // remaining, loop.stride * remaining))
        inner.insert(0, Loop(remaining, loop.stride))
        remaining = 1
    return tuple(outer), tuple(inner)


class Adaptation(Enum):
    """What must sit between a producer and a consumer of one operand."""

    IDENTITY = "identity"
    LANE_PERMUTATION = "lane_permutation"  # free: wires only
    REORDER = "reorder"  # a buffered loop-nest reorder: replay and windows included
    WIDTH_CONVERSION = "width_conversion"  # same element order, different lanes
    LANE_REGROUP = "lane_regroup"  # the lane axis changes (a banked transpose)
    INCOMPATIBLE = "incompatible"  # positions never presented, or an order no reorder reaches


@dataclass(frozen=True)
class Reorder:
    """A loop-nest reorder: per frame of ``frame_beats`` input beats, emit the beat
    at ``sum(index[i] * coefs[i])`` for the nested ``dims`` (outer first).

    A beat may be emitted more than once (replay, overlapping windows) or not at
    all (a window's stride passing it)."""

    frame_beats: int
    dims: tuple[int, ...]
    coefs: tuple[int, ...]


@dataclass(frozen=True)
class Classification:
    adaptation: Adaptation
    detail: str = ""
    lane_permutation: tuple[int, ...] = ()
    reorder: Reorder | None = None


def classify(source: Traversal, sink: Traversal) -> Classification:
    """Name the adapter that turns ``source``'s sequence into ``sink``'s. Both are
    traversals: a lane group (``Grouped``) is refused (``TypeError``), since it joins
    another form by a width step only (``steps``)."""
    if not isinstance(source, Traversal) or not isinstance(sink, Traversal):
        raise TypeError("classify compares two traversals: a lane group joins by a width step")
    if source.shape != sink.shape:
        return Classification(Adaptation.INCOMPATIBLE, "different operand shapes")
    if source == sink:
        return Classification(Adaptation.IDENTITY)
    if source.beat_loops == sink.beat_loops and source.lanes == sink.lanes:
        offsets = list(_offsets(source.lane_loops))
        wanted = list(_offsets(sink.lane_loops))
        if Counter(offsets) == Counter(wanted):
            return Classification(
                Adaptation.LANE_PERMUTATION,
                "the same positions in each beat, in another lane order",
                lane_permutation=tuple(offsets.index(offset) for offset in wanted),
            )
    if source.lane_loops == sink.lane_loops:
        reorder = _reorder(source.beat_loops, sink.beat_loops)
        if reorder is not None:
            return Classification(
                Adaptation.REORDER, "a buffered loop-nest reorder", reorder=reorder
            )
    flat = (
        _canonical((*source.beat_loops, *source.lane_loops)),
        _canonical((*sink.beat_loops, *sink.lane_loops)),
    )
    if flat[0] == flat[1]:
        return Classification(
            Adaptation.WIDTH_CONVERSION, f"{source.lanes} lanes regrouped as {sink.lanes}"
        )
    refined = _common_refinement(*flat)
    if refined is not None and Counter(refined[0]) == Counter(refined[1]):
        return Classification(Adaptation.LANE_REGROUP, "the same positions under another lane axis")
    return Classification(Adaptation.INCOMPATIBLE, "the sequences present different positions")


def _reorder(source: Sequence[Loop], sink: Sequence[Loop]) -> Reorder | None:
    """The reorder from ``source``'s beat nest to ``sink``'s, if one exists.

    The loop matching (each sink loop one source loop, or a new replay) comes
    first and keeps its parameters; a nest that matches no loop-for-loop may
    still read the frame through digit vectors (``_stepped``): a window.
    """
    return _matched(source, sink) or _stepped(source, sink)


def _matched(source: Sequence[Loop], sink: Sequence[Loop]) -> Reorder | None:
    refined = _common_refinement(source, sink)
    if refined is None:
        return None
    produced, consumed = refined
    # Equal stride-zero loops can be both a shared repetition and an inner
    # replay. Match their multiplicities, not just membership by value.
    replays = Counter(consumed) - Counter(produced)
    if any(loop.stride != 0 for loop in replays) or Counter(consumed) - replays != Counter(
        produced
    ):
        return None
    produced, consumed = _unshared(produced, consumed)
    beat_stride: dict[Loop, list[int]] = {}
    for index, loop in enumerate(produced):
        beat_stride.setdefault(loop, []).append(prod(item.extent for item in produced[index + 1 :]))
    coefs = []
    for loop in consumed:
        strides = beat_stride.get(loop)
        coefs.append(strides.pop(0) if strides else 0)
    return Reorder(
        prod(loop.extent for loop in produced),
        tuple(loop.extent for loop in consumed),
        tuple(coefs),
    )


def _stepped(source: Sequence[Loop], sink: Sequence[Loop]) -> Reorder | None:
    """A reorder whose sink loops each step the source's frame by a digit vector.

    Each sink loop's stride is decomposed over the frame's loops, largest
    stride first, into a digit vector: one step of the sink loop moves the
    source's digits by it. The sink's beat at digits ``d`` is then the frame
    beat whose digits are ``sum(d[k] * vector[k])``, provided no digit leaves
    its loop: the sink's whole nest stays inside the frame's box. Two sink loops
    stepping one digit overlap (a sliding window); a digit value no sink beat
    reaches drops those beats (a strided window).

    The greedy decomposition is exact when each of the frame's moving loops
    strides past everything its finer loops reach, as a frame presenting each
    offset once in a compact order does (``vector_major``). Otherwise it may
    miss a decomposition and answer None: sound, not complete.
    """
    boundaries = {
        value
        for loop in (*source, *sink)
        if loop.stride
        for value in (loop.stride, loop.stride * loop.extent)
    }
    # Split only to find the shared outer loops; the frame is read through its
    # coarsest loops, since a split would put a carry where the sink has none.
    produced, consumed = (
        _canonical(nest)
        for nest in _unshared(
            _split_where_exact(source, boundaries), _split_where_exact(sink, boundaries)
        )
    )
    beat_stride = [
        prod(loop.extent for loop in produced[index + 1 :]) for index in range(len(produced))
    ]
    largest_first = sorted(range(len(produced)), key=lambda index: -produced[index].stride)
    reach = [0] * len(produced)
    coefs = []
    for loop in consumed:
        rest, coef = loop.stride, 0
        for index in largest_first:
            stride = produced[index].stride
            if not stride or not rest:
                continue
            digit = min(produced[index].extent - 1, rest // stride)
            rest -= digit * stride
            reach[index] += (loop.extent - 1) * digit
            coef += digit * beat_stride[index]
        if rest:
            return None
        coefs.append(coef)
    if any(reached >= loop.extent for reached, loop in zip(reach, produced)):
        return None
    # A frame read once per frame (the sink's nest empty) is a loop of one beat.
    return Reorder(
        prod(loop.extent for loop in produced),
        tuple(loop.extent for loop in consumed) or (1,),
        tuple(coefs) or (0,),
    )


def _split_where_exact(loops: Sequence[Loop], boundaries: set[int]) -> tuple[Loop, ...]:
    """``loops`` split at every boundary that splits a loop exactly; the others are kept."""
    refined: list[Loop] = []
    for loop in loops:
        pieces: tuple[Loop, ...] = (loop,)
        for boundary in sorted(boundaries, reverse=True):
            pieces = tuple(
                part for piece in pieces for part in (_split(piece, boundary) or (piece,))
            )
        refined.extend(pieces)
    return tuple(refined)


def _unshared(
    produced: Sequence[Loop], consumed: Sequence[Loop]
) -> tuple[tuple[Loop, ...], tuple[Loop, ...]]:
    """Both nests without the outer loops they share: what remains of ``produced`` is the frame."""
    shared = 0
    while shared < min(len(produced), len(consumed)) and produced[shared] == consumed[shared]:
        shared += 1
    return tuple(produced[shared:]), tuple(consumed[shared:])


@dataclass(frozen=True)
class LevelEnd:
    """A marker closing a loop level: asserted on the last beat of every ``beats`` beats.

    The level is named by the number of beats it spans, not by a loop's name:
    canonical traversals merge contiguous loops, and loop names do not cross
    kernels. On a beat sequence it must close whole innermost loops
    (``aligned``). A periodic pin (AXIS ``TLAST``, ``replay_buffer``'s
    ``olast``) and a loop-completion pin (``input_gen``'s ``olst[d]``) carry the
    same rule.
    """

    beats: int

    def __post_init__(self) -> None:
        require_positive(self.beats, "a marker's level")

    def asserted(self, beat: int) -> bool:
        return (beat + 1) % self.beats == 0

    @property
    def constant(self) -> bool:
        """A level of one beat closes on every beat: a constant, which every sequence
        carries."""
        return self.beats == 1

    def aligned(self, form: Traversal | Grouped) -> bool:
        """Whether the level closes whole innermost loops of ``form``: of its beat loops,
        or, for a lane group, of its order (every ``beats * lanes`` elements). A constant
        (``constant``) is aligned on every sequence."""
        if self.constant:
            return True
        if form.beats % self.beats:
            return False
        try:
            if isinstance(form, Grouped):
                _split_at(form.order, self.beats * form.lanes)
            else:
                _split_at(form.beat_loops, self.beats)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class LevelDone:
    """A marker holding while a loop level is on its last index value: asserted on the last
    ``inner`` beats of every ``beats`` beats.

    ``beats`` is the beats of the level and every level inside it, ``inner`` the beats of
    the levels inside it, so the level has ``beats // inner`` index values and its last one holds
    across ``inner`` beats. Like ``LevelEnd``, the levels are named by beats, not by loop
    names. On a beat sequence ``inner`` must close whole innermost loops, and ``beats``
    whole loops around them (``aligned``).

    It is ``input_gen``'s ``done[d]`` (level ``d`` on its last index value, held across the
    levels inside it), where ``LevelEnd`` is its ``olst[d]`` (``term[d]``: level ``d``
    and every level inside it completing on one beat). FINN's ``mvu_tiled`` ``input_gen``
    carries it on its ``odone[d]`` pin.

    Its canonical form has two edges, both ``LevelEnd`` and never this kind (the
    constructor refuses them; ``level_done`` gives the one form):

    - an inner level of one beat is the level closing on its last beat, ``LevelEnd(beats)``;
    - inner levels of the whole level hold on every beat, ``LevelEnd(1)``, the constant.
    """

    beats: int
    inner: int

    def __post_init__(self) -> None:
        require_positive(self.beats, "a marker's level")
        require_positive(self.inner, "a marker's inner levels")
        if self.beats % self.inner:
            raise ValueError(f"{self.inner} inner beats divide no level of {self.beats}")
        if self.inner == 1:
            raise ValueError(f"an inner level of one beat is LevelEnd({self.beats})")
        if self.inner == self.beats:
            raise ValueError("inner levels of the whole level hold on every beat: LevelEnd(1)")

    def asserted(self, beat: int) -> bool:
        return beat % self.beats >= self.beats - self.inner

    @property
    def constant(self) -> bool:
        """Never: a marker holding on every beat is ``LevelEnd(1)``."""
        return False

    def aligned(self, form: Traversal | Grouped) -> bool:
        """Whether ``inner`` closes whole innermost loops of ``form`` and ``beats`` whole
        loops around them: of its beat loops, or, for a lane group, of its order (every
        ``inner * lanes`` and ``beats * lanes`` elements)."""
        if form.beats % self.beats:
            return False
        try:
            for beats in (self.inner, self.beats):
                if isinstance(form, Grouped):
                    _split_at(form.order, beats * form.lanes)
                else:
                    _split_at(form.beat_loops, beats)
        except ValueError:
            return False
        return True


Marker = LevelEnd | LevelDone
"""A marker rule: a level closing (``LevelEnd``) or a level on its last index value
(``LevelDone``)."""


def level_done(beats: int, inner: int) -> LevelEnd | LevelDone:
    """The marker asserted on the last ``inner`` beats of every ``beats`` beats, in its one
    form: ``LevelEnd(beats)`` for an inner level of one beat, ``LevelEnd(1)`` when the inner
    levels span the
    whole level, a ``LevelDone`` otherwise. ``LevelDone``'s refusals (``ValueError``)
    otherwise: a count that is not positive, or inner beats dividing no level."""
    require_positive(beats, "a marker's level")
    require_positive(inner, "a marker's inner levels")
    if beats % inner == 0:
        if inner == 1:
            return LevelEnd(beats)
        if inner == beats:
            return LevelEnd(1)
    return LevelDone(beats, inner)


def _misaligned(rule: Marker) -> str:
    """Why ``rule`` is refused on a sequence whose loops it does not close."""
    if isinstance(rule, LevelDone):
        inner, beats = rule.inner, rule.beats
        return f"a marker on the last {inner} beats of every {beats} beats closes no loop level"
    return f"a marker every {rule.beats} beats closes no loop level"


class Repetition(Enum):
    """Pass correspondence of a producer: one pass, or its traversal repeated indefinitely."""

    ONCE = "once"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class BeatSequence:
    """What one channel side presents of its tensor: traversal, pass repetition and marker rules.

    For a producer the markers are guarantees; for a consumer, requirements. A marker is a
    level closing (``LevelEnd``) or a level on its last index value (``LevelDone``).
    """

    form: Traversal | Grouped
    repetition: Repetition = Repetition.ONCE
    markers: tuple[LevelEnd | LevelDone, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.form, (Traversal, Grouped)) or not isinstance(
            self.repetition, Repetition
        ):
            raise TypeError("a beat sequence has a Traversal or a Grouped, and a Repetition")
        object.__setattr__(self, "markers", tuple(self.markers))
        if not all(isinstance(rule, (LevelEnd, LevelDone)) for rule in self.markers):
            raise TypeError("marker rules are LevelEnd or LevelDone values")
        for rule in self.markers:
            if not rule.aligned(self.form):
                raise ValueError(_misaligned(rule))


@dataclass(frozen=True)
class Selected:
    """A selected pass: one side presenting one of ``sets`` sets a pass, the set chosen by
    a selector stream. ``sequence`` is what it presents of one set (shape
    ``sequence.form.shape``), a pass; over the stacked tensor ``(sets, *that shape)``,
    pass ``j`` presents set ``selection[j]``, an outer loop whose offset is the set's size
    times the selector index.

    - **Its positions are given a selection** (``offsets``, ``positions``): before run
      time they are a set of possible orders, not one, so it is no ``Traversal`` and is
      never equal to one. A constant selection presents exactly what
      ``sequence.form.repeated(passes)`` presents over that set: repetition is its
      constant case.
    - **The selector stream is named** (``selector``): the tensor whose indices, one a
      pass, select the sets. A tensor has one producer, so two sides naming the same
      tensor are selected by the same indices in the same order, and "the same selector
      stream" is equality of names. Two sides of the same sets under different names
      are not shared, whatever the indices turn out to be.
    - **The selector stream is a separate channel**, one index a pass, from the
      selector tensor's producer to the selected side's selector port. Its steps are any
      channel's: its port is projected from the kernel's schedule through the indices
      the selector tensor varies with, and from a once-presented selector stream the steps replay
      it.

    ``steps.steps`` joins two selected sides of the same sets through the same selector
    stream (``shares``) by the steps between their ``sequence``s, inside one set: the
    selection loop is outermost and shared. Any other pairing with a selected side is
    refused (``steps.Unshared``, ``selection-not-shared``).
    """

    sequence: BeatSequence
    sets: int
    selector: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, BeatSequence):
            raise TypeError("a selected pass presents a BeatSequence of one set")
        require_positive(self.sets, "sets")
        if not isinstance(self.selector, str) or not self.selector:
            raise ValueError("a selector stream has a nonempty name")

    @property
    def shape(self) -> tuple[int, ...]:
        """The stacked tensor's shape: the sets, then one set's."""
        return (self.sets, *self.sequence.form.shape)

    @property
    def set_size(self) -> int:
        """The elements of one set: the offset between two sets of the stacked tensor."""
        return prod(self.sequence.form.shape)

    def shares(self, other: Selected) -> bool:
        """Whether ``other`` selects from the same sets (the same stacked shape) through the
        same selector stream (the same name)."""
        return self.selector == other.selector and self.shape == other.shape

    def offsets(self, selection: Sequence[int]) -> npt.NDArray[np.int64]:
        """The flat offset in the stacked tensor of every element presented over
        ``selection``, one pass a selector index, beat by beat, lane zero first."""
        chosen = tuple(selection)
        for index in chosen:
            if type(index) is not int or not 0 <= index < self.sets:
                raise ValueError(f"a selector index is one of {self.sets} sets, not {index!r}")
        inner = offsets(self.sequence.form)
        if not chosen:
            return np.zeros(0, dtype=np.int64)
        return (np.asarray(chosen, dtype=np.int64)[:, None] * self.set_size + inner).ravel()

    def positions(self, selection: Sequence[int]) -> Iterator[tuple[Position, ...]]:
        """Each beat's positions in the stacked tensor over ``selection``, lane zero first."""
        lanes = self.sequence.form.lanes
        flat = self.offsets(selection).reshape(-1, lanes)
        for beat in flat.tolist():
            yield tuple(
                tuple(int(digit) for digit in np.unravel_index(offset, self.shape))
                for offset in beat
            )


def each_once(form: Traversal | Grouped) -> bool:
    """Whether ``form`` presents each position of its tensor once a pass, its stride-0
    loops (repetition, replay) aside: its moving loops are a mixed radix over the tensor,
    each stride the product of the extents of those finer than it."""
    loops = form.order if isinstance(form, Grouped) else (*form.beat_loops, *form.lane_loops)
    moving = sorted((loop for loop in loops if loop.stride), key=lambda loop: loop.stride)
    reach = 1
    for loop in moving:
        if loop.stride != reach:
            return False
        reach *= loop.extent
    return reach == prod(form.shape)


def unreplayed(form: Traversal | Grouped) -> Traversal | Grouped:
    """``form`` without what its receiver realizes: its replay (its stride-0 beat loops
    inside a moving loop), and its window, where it reads a position twice or not at all.

    Outermost stride-0 loops repeat the whole pass and stay: that repetition is
    part of a stream, while replay is realized by the receiver. A window (a
    sliding window's overlap, or a stride that passes positions) is the receiver's
    too (its channel's reorder): without its replay, a ``form`` that does not
    present each position once (``each_once``) is the window's reorder frame, each position
    once in row-major order at ``form``'s lanes, where those are the tensor's
    innermost elements (``vector_major``); otherwise it stays as it is.

    A lane group (``Grouped``) is its order's, one element a beat, grouped again at its
    lanes (``grouped``): every stride-0 loop of its order inside a moving loop is its
    replay, and its window is the window's reorder frame, each position once, row-major. Where its
    lanes divide no pass of
    what is left, it is refused (``ValueError``).
    """
    if isinstance(form, Grouped):
        found = unreplayed(Traversal(form.shape, form.order, ()))
        assert isinstance(found, Traversal)  # a traversal's is a traversal
        return _grouped_again(found, form.lanes)
    loops = form.beat_loops
    moving = _leading_replay(loops)
    once = _replay_dropped(form)
    lanes = form.lane_loops
    contiguous = not lanes or (len(lanes) == 1 and lanes[0].stride == 1)
    if each_once(once) or not contiguous or form.shape[-1] % form.lanes:
        return once
    frame = vector_major(form.shape, form.lanes)
    return Traversal(form.shape, (*loops[:moving], *frame.beat_loops), frame.lane_loops)


def _replay_dropped(form: Traversal | Grouped) -> Traversal | Grouped:
    """``form`` without its replay, its window kept: its stride-0 beat loops inside a
    moving loop dropped (a lane group's, every stride-0 loop of its order inside one)."""
    loops = form.order if isinstance(form, Grouped) else form.beat_loops
    moving = _leading_replay(loops)
    kept = (*loops[:moving], *(loop for loop in loops[moving:] if loop.stride))
    if isinstance(form, Grouped):
        return _grouped_again(Traversal(form.shape, kept, ()), form.lanes)
    return Traversal(form.shape, kept, form.lane_loops)


def _grouped_again(found: Traversal, lanes: int) -> Traversal | Grouped:
    """``found``, one element a beat, at a lane group's ``lanes`` (``grouped``); refused
    where they divide no pass of it."""
    try:
        return grouped(found, lanes)
    except ValueError as error:
        raise ValueError(
            f"{lanes} lanes group no pass of the order without its replay and window, "
            f"{found.beats} elements: {error}"
        ) from error


def one_pass(form: Traversal | Grouped) -> Traversal | Grouped:
    """The traversal without its outermost stride-0 loops: one pass of a repetition.
    A lane group's pass is its order's, at the same lanes (they divide a pass)."""
    if isinstance(form, Grouped):
        return Grouped(form.shape, form.order[_leading_replay(form.order) :], form.lanes)
    loops = form.beat_loops
    return Traversal(form.shape, loops[_leading_replay(loops) :], form.lane_loops)


def passes(form: Traversal | Grouped) -> int:
    """How many times ``form`` presents its whole pass (its ``one_pass``) a frame."""
    return form.beats // one_pass(form).beats


def stored(form: Traversal | Grouped) -> Traversal | Grouped:
    """What a cyclic source stores for a consumer reading ``form``: one pass of the
    consumer's order without what the consumer's side realizes, at ``form``'s lanes.

    - Its replay is never stored.
    - A window that reads no position twice (one that only passes positions, or none)
      stores what it reads, each once: ``one_pass`` of ``form`` without its replay.
    - A window that reads a position twice stores the window's reorder frame, as the boundary rule
    reads
      it: ``one_pass`` of ``unreplayed``.

    The steps from it to ``form`` derive the rest: a reorder with replay or the window's
    reads, from a source presenting it cyclically, once for each of the consumer's
    passes (``steps``, ``steps.presented``). Where the window also passes positions, the
    frame holds positions the consumer never reads, and the reorder drops them. Its size
    is ``stored_depth``.
    """
    once = one_pass(_replay_dropped(form))
    taken = offsets(once)
    if len(np.unique(taken)) == len(taken):
        return once
    return one_pass(unreplayed(form))


def stored_depth(form: Traversal | Grouped) -> int:
    """The memory a cyclic source holds for a consumer reading ``form``, in **elements**:
    the beats times the lanes of ``stored(form)``. A memory, not a live window, so it is
    no field of a chain's ``steps.Cost``."""
    found = stored(form)
    return found.beats * found.lanes


def pack(
    form: Traversal | Grouped, integers: Sequence[int] | npt.NDArray[np.int64], bits: int
) -> tuple[int, ...]:
    """Pack an integer operand of ``form.shape``, given as its row-major ``integers``, into
    one raw word per beat, lane zero lowest: each lane the low ``bits`` bits of its
    integer's two's complement.

    Each beat's lanes are read at their flat offsets (``position`` before it is
    unravelled), so no position is formed and no nested operand walked. Integers that
    fit in 64 bits are packed with numpy, at any lane and word width; an operand with
    one that does not is packed one Python integer at a time. Both give the same words.
    """
    require_positive(bits, "bits")
    size = prod(form.shape)
    if len(integers) != size:
        raise ValueError(
            f"an operand of shape {form.shape} has {size} integers, not {len(integers)}"
        )
    try:
        flat = np.asarray(integers, dtype=np.int64)
    except OverflowError:
        return _pack_exact(form, integers, bits)
    lanes = flat[offsets(form).reshape(form.beats, form.lanes)]
    # Bit k of every lane, lane zero's bits first and each lane's least significant
    # first; an arithmetic shift by at most 63 reads the sign above an int64.
    planes = np.empty((*lanes.shape, bits), dtype=np.uint8)
    for k in range(bits):
        planes[..., k] = (lanes >> min(k, 63)) & 1
    raw = np.packbits(planes.reshape(len(lanes), -1), axis=1, bitorder="little")
    width = raw.shape[1]
    if width <= 8:
        padded = np.zeros((len(raw), 8), dtype=np.uint8)
        padded[:, :width] = raw
        return tuple(padded.view("<u8").ravel().tolist())
    data = raw.tobytes()
    return tuple(
        int.from_bytes(data[start : start + width], "little")
        for start in range(0, len(data), width)
    )


def offsets(form: Traversal | Grouped) -> npt.NDArray[np.int64]:
    """The flat row-major offset of every element ``form`` presents, beat by beat, lane
    zero first: where ``pack`` reads the operand, and ``unpack`` writes it. A lane group's
    are its order's."""
    if isinstance(form, Grouped):
        return _offset_array(form.order)
    return (_offset_array(form.beat_loops)[:, None] + _offset_array(form.lane_loops)).ravel()


def unpack(
    form: Traversal | Grouped, words: Sequence[int], bits: int, *, signed: bool
) -> npt.NDArray[np.int64]:
    """The integer operand of ``form.shape`` that ``words`` present, one raw word per beat
    as ``pack`` makes them: each lane the low ``bits`` bits of its word, lane zero lowest,
    sign-extended when ``signed``.

    ``pack``'s inverse on every operand of ``bits``-bit integers:
    ``pack(form, unpack(form, words, bits, signed=s).ravel(), bits) == words`` whenever
    ``unpack`` accepts ``words``. It refuses (``ValueError``) words that are no such
    operand's: another count than ``form.beats``, a word wider than its lanes, an element
    ``form`` never presents, and an element presented twice (a replay) with two values.
    Lanes of up to 63 bits, or 64 signed: an int64 holds them.
    """
    require_positive(bits, "bits")
    if bits > 64 or (bits == 64 and not signed):
        raise ValueError(f"{bits}-bit {'signed' if signed else 'unsigned'} lanes exceed an int64")
    if len(words) != form.beats:
        raise ValueError(f"{form.beats} beats, not {len(words)} words")
    width = form.lanes * bits
    nbytes = (width + 7) // 8
    for beat, word in enumerate(words):
        if word < 0 or word >> width:
            raise ValueError(f"word {beat}, {word:#x}, is wider than its {width} bits")
    raw = np.frombuffer(b"".join(int(word).to_bytes(nbytes, "little") for word in words), np.uint8)
    planes = np.unpackbits(raw.reshape(len(words), nbytes), axis=1, bitorder="little")
    lanes = planes[:, :width].reshape(len(words) * form.lanes, bits).astype(np.uint64)
    unsigned = (lanes << np.arange(bits, dtype=np.uint64)).sum(axis=1, dtype=np.uint64)
    values = unsigned.view(np.int64)  # two's complement at 64 bits; below, extended here
    if signed and bits < 64:
        values = values - ((values >> (bits - 1)) << bits)
    at = offsets(form)
    size = prod(form.shape)
    order = np.argsort(at, kind="stable")
    again = np.flatnonzero(
        (at[order][1:] == at[order][:-1]) & (values[order][1:] != values[order][:-1])
    )
    if again.size:
        first, second = order[again[0]], order[again[0] + 1]
        position = tuple(int(i) for i in np.unravel_index(at[first], form.shape))
        raise ValueError(
            f"element {position} is presented twice with two values: "
            f"{values[first]} at beat {first // form.lanes} lane {first % form.lanes}, "
            f"{values[second]} at beat {second // form.lanes} lane {second % form.lanes}"
        )
    operand = np.zeros(size, dtype=np.int64)
    seen = np.zeros(size, dtype=bool)
    operand[at] = values
    seen[at] = True
    if not seen.all():
        missing = int(np.flatnonzero(~seen)[0])
        position = tuple(int(i) for i in np.unravel_index(missing, form.shape))
        raise ValueError(f"the traversal presents no value of element {position}")
    return operand.reshape(form.shape)


def _offset_array(loops: Sequence[Loop]) -> npt.NDArray[np.int64]:
    """``_offsets`` as an array."""
    found = np.zeros(1, dtype=np.int64)
    for loop in loops:
        steps = np.arange(loop.extent, dtype=np.int64) * loop.stride
        found = (found[:, None] + steps).ravel()
    return found


def _pack_exact(
    form: Traversal | Grouped, integers: Sequence[int] | npt.NDArray[np.int64], bits: int
) -> tuple[int, ...]:
    """``pack`` one Python integer at a time, for integers beyond 64 bits."""
    mask = (1 << bits) - 1
    words = []
    for beat in offsets(form).reshape(form.beats, form.lanes).tolist():
        word = 0
        for lane, offset in enumerate(beat):
            word |= (integers[offset] & mask) << (lane * bits)
        words.append(word)
    return tuple(words)


__all__ = [
    "Adaptation",
    "AxisLoop",
    "BeatSequence",
    "Classification",
    "Grouped",
    "LevelDone",
    "LevelEnd",
    "Loop",
    "Marker",
    "Position",
    "Reorder",
    "Repetition",
    "Selected",
    "Traversal",
    "axis_strides",
    "classify",
    "each_once",
    "grouped",
    "level_done",
    "offsets",
    "pack",
    "passes",
    "one_pass",
    "regrouped",
    "require_positive",
    "stored",
    "stored_depth",
    "tile",
    "unpack",
    "unreplayed",
    "vector_major",
]
