# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical beat order as a loop nest over a row-major tensor.

A ``Traversal`` walks an operand of ``shape`` with two loop nests, both listed
outer to inner. Each iteration of ``beat_loops`` is one beat; each iteration of
``lane_loops`` is one lane of that beat, lane zero first (least significant).
A loop advances the operand's flat row-major index by ``stride`` elements per
step; a stride of zero repeats (replays) positions. Tiles, chunked tiles,
transposes, sliding windows and replay are all ordinary loop nests.

Construction canonicalizes the nests, so two traversals are equal exactly when
they present the same positions in the same beats and lanes. ``classify``
compares two traversals of one operand and names the adaptation a mismatch
needs: free lane wiring, a loop-nest reorder (``Reorder``: its frame, loop
extents and strides), a width conversion, a lane regroup, or none at all.

A ``BeatSequence`` is what one end of a channel presents of the tensor it
carries: its traversal per pass, whether the pass repeats (``Repetition``), and
the marker rules it offers or requires. ``unreplayed`` is the boundary rule: the
receiver of a channel realizes its own replay, while whole-pass repetition stays
part of the interface; ``period`` strips the whole-pass repetition.

Words and tensors: ``pack`` presents an integer operand as one raw word per beat,
and ``unpack``, its inverse, reads the operand back from the words; both at the
flat offsets ``offsets`` gives.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod

import numpy as np
import numpy.typing as npt

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


def regrouped(form: Traversal, lanes: int) -> Traversal:
    """The same element sequence, ``lanes`` elements a beat: a width conversion."""
    flat = _canonical((*form.beat_loops, *form.lane_loops))
    beats, lane_loops = _split_at(flat, lanes)
    return Traversal(form.shape, beats, lane_loops)


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
    REORDER = "reorder"  # a buffered loop-nest reorder, or replay
    WIDTH_CONVERSION = "width_conversion"  # same element order, different lanes
    LANE_REGROUP = "lane_regroup"  # the lane axis changes (a banked transpose)
    INCOMPATIBLE = "incompatible"  # different positions, or no loop-nest relation


@dataclass(frozen=True)
class Reorder:
    """A loop-nest reorder: per frame of ``frame_beats`` input beats, emit the beat
    at ``sum(index[i] * coefs[i])`` for the nested ``dims`` (outer first)."""

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
        require_positive(self.beats, "a marker's level")

    def asserted(self, beat: int) -> bool:
        return (beat + 1) % self.beats == 0

    @property
    def constant(self) -> bool:
        """A level of one beat closes on every beat: a constant, which every sequence
        carries."""
        return self.beats == 1

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
    """One channel end's view of its tensor: traversal, pass repetition and marker rules.

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


def passes(form: Traversal) -> int:
    """How many times ``form`` presents its whole pass (its ``period``) a frame."""
    return form.beats // period(form).beats


def pack(
    form: Traversal, integers: Sequence[int] | npt.NDArray[np.int64], bits: int
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
    lanes = flat[_offset_array(form.beat_loops)[:, None] + _offset_array(form.lane_loops)]
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


def offsets(form: Traversal) -> npt.NDArray[np.int64]:
    """The flat row-major offset of every element ``form`` presents, beat by beat, lane
    zero first: where ``pack`` reads the operand, and ``unpack`` writes it."""
    return (_offset_array(form.beat_loops)[:, None] + _offset_array(form.lane_loops)).ravel()


def unpack(
    form: Traversal, words: Sequence[int], bits: int, *, signed: bool
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
    form: Traversal, integers: Sequence[int] | npt.NDArray[np.int64], bits: int
) -> tuple[int, ...]:
    """``pack`` one Python integer at a time, for integers beyond 64 bits."""
    mask = (1 << bits) - 1
    lanes = tuple(zip(_offsets(form.lane_loops), range(0, form.lanes * bits, bits)))
    words = []
    for beat in _offsets(form.beat_loops):
        word = 0
        for lane, shift in lanes:
            word |= (integers[beat + lane] & mask) << shift
        words.append(word)
    return tuple(words)


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
    "offsets",
    "pack",
    "passes",
    "period",
    "regrouped",
    "require_positive",
    "tile",
    "unpack",
    "unreplayed",
    "vector_major",
]
