# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical stream order as a loop nest over a row-major tensor.

A ``Traversal`` walks an operand of ``shape`` with two loop nests, both listed
outer to inner. Each iteration of ``beat_loops`` is one beat; each iteration of
``lane_loops`` is one lane of that beat, lane zero first (least significant).
A loop advances the operand's flat row-major index by ``stride`` elements per
step; a stride of zero repeats (replays) positions. Tiles, chunked tiles,
transposes, sliding windows and replay are all ordinary loop nests.

Construction canonicalizes the nests, so two traversals are equal exactly when
they present the same positions in the same beats and lanes. ``classify``
compares two traversals of one operand and names the adapter a mismatch needs:
free lane wiring, a loop-nest reorder with its ``input_gen`` parameters, a width
conversion, a lane regroup, or none at all.

A ``BeatSequence`` (the canon's name, without its Regions) is what one end of
a stream presents of the tensor it carries: its traversal per pass, whether
the pass repeats (``Repetition``), and the marker rules it offers or requires.
``unreplayed`` is the boundary rule: the receiver of a stream realizes its own
replay, while whole-pass repetition stays part of the interface; ``period``
strips the whole-pass repetition.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod

Position = tuple[int, ...]


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, order=True)
class Loop:
    extent: int
    stride: int

    def __post_init__(self) -> None:
        _positive(self.extent, "loop extent")
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


AxisStep = tuple[int | None, int, int]
"""(operand axis or None for a replay loop, extent, step along that axis)."""


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
            _positive(extent, "operand extent")
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
        cls, shape: Sequence[int], beats: Sequence[AxisStep], lanes: Sequence[AxisStep]
    ) -> Traversal:
        """Build from (axis, extent, step) loops; axis None is a replay loop."""
        strides = axis_strides(tuple(shape))

        def loops(steps: Sequence[AxisStep]) -> tuple[Loop, ...]:
            return tuple(
                Loop(extent, 0 if axis is None else step * strides[axis])
                for axis, extent, step in steps
            )

        return cls(shape, loops(beats), loops(lanes))

    @property
    def lanes(self) -> int:
        return prod(loop.extent for loop in self.lane_loops)

    @property
    def beats(self) -> int:
        return prod(loop.extent for loop in self.beat_loops)

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
        _positive(count, "count")
        return Traversal(self.shape, (Loop(count, 0), *self.beat_loops), self.lane_loops)

    def replayed(self, count: int, *, inner_beats: int) -> Traversal:
        """Present every consecutive group of ``inner_beats`` beats ``count`` times."""
        _positive(count, "count")
        outer, inner = _split_at(self.beat_loops, inner_beats)
        return Traversal(self.shape, (*outer, Loop(count, 0), *inner), self.lane_loops)


def vector_major(shape: Sequence[int], lanes: int) -> Traversal:
    """FINN's default order: row-major, the innermost axis split into ``lanes`` lanes."""
    shape = tuple(shape)
    _positive(lanes, "lanes")
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


def regrouped(form: Traversal, lanes: int) -> Traversal:
    """The same element sequence, ``lanes`` elements a beat: a width conversion."""
    flat = _canonical((*form.beat_loops, *form.lane_loops))
    beats, lane_loops = _split_at(flat, lanes)
    return Traversal(form.shape, beats, lane_loops)


def _split_at(loops: Sequence[Loop], inner_beats: int) -> tuple[tuple[Loop, ...], tuple[Loop, ...]]:
    _positive(inner_beats, "inner_beats")
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
    REORDER = "reorder"  # buffered loop-nest reorder or replay: input_gen / outer shuffle
    WIDTH_CONVERSION = "width_conversion"  # same element order, different lanes: DWC
    LANE_REGROUP = "lane_regroup"  # the lane axis changes: inner shuffle (banked transpose)
    INCOMPATIBLE = "incompatible"  # different positions, or no loop-nest relation


@dataclass(frozen=True)
class Reorder:
    """``input_gen`` parameters: per frame of ``frame_beats`` input beats, emit the
    beat at ``sum(index[i] * coefs[i])`` for the nested ``dims`` (outer first)."""

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
    """Name the adapter that turns ``source``'s sequence into ``sink``'s."""
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
    refined = _common_refinement(source, sink)
    if refined is None:
        return None
    produced, consumed = refined
    replays = [loop for loop in consumed if loop.stride == 0 and loop not in produced]
    if Counter(consumed) - Counter(replays) != Counter(produced):
        return None
    shared = 0
    while shared < min(len(produced), len(consumed)) and produced[shared] == consumed[shared]:
        shared += 1
    produced, consumed = produced[shared:], consumed[shared:]
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
        _positive(self.beats, "a marker's level")

    def asserted(self, beat: int) -> bool:
        return (beat + 1) % self.beats == 0

    def aligned(self, form: Traversal) -> bool:
        """Whether the level closes whole innermost loops of ``form``."""
        if form.beats % self.beats:
            return False
        try:
            _split_at(form.beat_loops, self.beats)
        except ValueError:
            return False
        return True


class Repetition(Enum):
    """Pass correspondence of a producer: one pass, or its traversal repeated indefinitely."""

    ONCE = "once"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class BeatSequence:
    """One stream end's view of its tensor: traversal, pass repetition and marker rules.

    For a producer the markers are guarantees; for a consumer, requirements.
    """

    form: Traversal
    repetition: Repetition = Repetition.ONCE
    markers: tuple[LevelEnd, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.form, Traversal) or not isinstance(self.repetition, Repetition):
            raise TypeError("a beat sequence has a Traversal and a Repetition")
        object.__setattr__(self, "markers", tuple(self.markers))
        if not all(isinstance(rule, LevelEnd) for rule in self.markers):
            raise TypeError("marker rules are LevelEnd values")
        for rule in self.markers:
            if not rule.aligned(self.form):
                raise ValueError(f"a marker every {rule.beats} beats closes no loop level")


def unreplayed(form: Traversal) -> Traversal:
    """``form`` without replay: its stride-0 beat loops inside a moving loop.

    Outermost stride-0 loops repeat the whole pass and stay: that repetition is
    part of an interface, while replay is realized by the receiver.
    """
    loops = form.beat_loops
    moving = next((index for index, loop in enumerate(loops) if loop.stride), len(loops))
    kept = (*loops[:moving], *(loop for loop in loops[moving:] if loop.stride))
    return Traversal(form.shape, kept, form.lane_loops)


def period(form: Traversal) -> Traversal:
    """The traversal without its outermost stride-0 loops: one period of a repetition."""
    loops = form.beat_loops
    moving = next((index for index, loop in enumerate(loops) if loop.stride), len(loops))
    return Traversal(form.shape, loops[moving:], form.lane_loops)


def pack(form: Traversal, values: object, bits: int) -> tuple[int, ...]:
    """Pack an integer operand of ``form.shape`` into one raw word per beat, lane zero lowest."""
    _positive(bits, "bits")
    _check_shape(values, form.shape)
    mask = (1 << bits) - 1

    def lookup(position: Position) -> int:
        item = values
        for index in position:
            assert isinstance(item, Sequence)
            item = item[index]
        if type(item) is not int:
            raise ValueError(f"operand position {position} is not an integer")
        return item

    return tuple(
        sum((lookup(position) & mask) << (lane * bits) for lane, position in enumerate(beat))
        for beat in form.positions()
    )


def _check_shape(values: object, shape: tuple[int, ...]) -> None:
    if not shape:
        return
    if not isinstance(values, Sequence) or len(values) != shape[0]:
        raise ValueError(f"operand must have shape {shape}")
    for item in values:
        _check_shape(item, shape[1:])


__all__ = [
    "Adaptation",
    "AxisStep",
    "BeatSequence",
    "Classification",
    "LevelEnd",
    "Loop",
    "Position",
    "Reorder",
    "Repetition",
    "Traversal",
    "axis_strides",
    "classify",
    "pack",
    "period",
    "regrouped",
    "tile",
    "unreplayed",
    "vector_major",
]
