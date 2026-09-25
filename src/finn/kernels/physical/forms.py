# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Logical beat forms: which operand positions each beat carries, and in what order.

A form describes one pass of a stream over an operand of ``shape``. Beat ``n``
field ``f`` carries ``position(n, f)``; field zero is the least-significant field
of the physical word. Forms compare by value. Two streams carry the same
sequence when their forms are equal, so a connection can be checked without
enumerating positions; ``positions()`` enumerates them for packing and tests.

Forms deliberately contain no schedule, timing or physical bit placement. They
correspond to the canon ``BeatSequence`` (``elements_per_beat``, ``beat_count``
and ``beat``) for streams whose sequence has a named construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from finn.core.space import ValueSemantics

Position = tuple[int, ...]


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class BeatForm:
    """Base of all forms: ``lanes`` fields per beat, ``beats`` beats per pass."""

    @property
    def lanes(self) -> int:
        raise NotImplementedError

    @property
    def beats(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def position(self, beat: int, field: int) -> Position:
        raise NotImplementedError

    def positions(self) -> Iterator[tuple[Position, ...]]:
        for beat in range(self.beats):
            yield tuple(self.position(beat, field) for field in range(self.lanes))


@dataclass(frozen=True)
class Fold(BeatForm):
    """A vector of ``extent`` elements, ``width`` consecutive elements per beat."""

    extent: int
    width: int

    def __post_init__(self) -> None:
        _positive(self.extent, "extent")
        _positive(self.width, "width")
        if self.extent % self.width:
            raise ValueError("a fold's width must divide its extent")

    @property
    def lanes(self) -> int:
        return self.width

    @property
    def beats(self) -> int:
        return self.extent // self.width

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.extent,)

    def position(self, beat: int, field: int) -> Position:
        return (beat * self.width + field,)


@dataclass(frozen=True)
class Tile(BeatForm):
    """A (rows, cols) matrix in ``pe x simd`` tiles: row folds outer, column folds inner.

    Field ``p * simd + s`` carries ``(nf * pe + p, sf * simd + s)``: SIMD varies
    fastest within a beat. This is the MVAU/VVAU weight tile order.
    """

    rows: int
    cols: int
    pe: int
    simd: int

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "pe", "simd"):
            _positive(getattr(self, name), name)
        if self.rows % self.pe or self.cols % self.simd:
            raise ValueError("PE must divide rows and SIMD must divide cols")

    @property
    def lanes(self) -> int:
        return self.pe * self.simd

    @property
    def beats(self) -> int:
        return (self.rows // self.pe) * (self.cols // self.simd)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, self.cols)

    def position(self, beat: int, field: int) -> Position:
        row_fold, col_fold = divmod(beat, self.cols // self.simd)
        p, s = divmod(field, self.simd)
        return (row_fold * self.pe + p, col_fold * self.simd + s)


@dataclass(frozen=True)
class Repeat(BeatForm):
    """The whole ``form`` sequence presented ``count`` times; positions repeat."""

    form: BeatForm
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.form, BeatForm):
            raise TypeError("Repeat wraps a BeatForm")
        _positive(self.count, "count")

    @property
    def lanes(self) -> int:
        return self.form.lanes

    @property
    def beats(self) -> int:
        return self.form.beats * self.count

    @property
    def shape(self) -> tuple[int, ...]:
        return self.form.shape

    def position(self, beat: int, field: int) -> Position:
        return self.form.position(beat % self.form.beats, field)


@dataclass(frozen=True)
class Batch(BeatForm):
    """``count`` distinct instances of ``form``, outermost; positions gain a leading index."""

    form: BeatForm
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.form, BeatForm):
            raise TypeError("Batch wraps a BeatForm")
        _positive(self.count, "count")

    @property
    def lanes(self) -> int:
        return self.form.lanes

    @property
    def beats(self) -> int:
        return self.form.beats * self.count

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.count, *self.form.shape)

    def position(self, beat: int, field: int) -> Position:
        index, inner = divmod(beat, self.form.beats)
        return (index, *self.form.position(inner, field))


class Repetition(Enum):
    """Pass correspondence of a producer: one pass, or its form repeated indefinitely.

    A cyclic producer of form ``F`` satisfies a consumer pass of ``F`` or
    ``Repeat(F, k)``; the consumer's pass length is a whole number of periods.
    """

    ONCE = "once"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class Every:
    """A marker asserted on every ``period``-th beat of a pass (the last of each group)."""

    period: int

    def __post_init__(self) -> None:
        _positive(self.period, "marker period")

    def asserted(self, beat: int) -> bool:
        return (beat + 1) % self.period == 0


def pack(form: BeatForm, values: object, bits: int) -> tuple[int, ...]:
    """Pack an integer operand of ``form.shape`` into one raw word per beat.

    Fields are ``bits`` wide, field zero least significant; values are stored
    in two's complement. Range admission belongs to the caller's dtype policy.
    """
    _positive(bits, "bits")
    mask = (1 << bits) - 1

    def lookup(position: Position) -> int:
        item = values
        for index in position:
            if not isinstance(item, Sequence) or not 0 <= index < len(item):
                raise ValueError(f"operand has no position {position}; expected {form.shape}")
            item = item[index]
        if type(item) is not int:
            raise ValueError(f"operand position {position} is not an integer")
        return item

    _check_shape(values, form.shape)
    return tuple(
        sum((lookup(position) & mask) << (field * bits) for field, position in enumerate(beat))
        for beat in form.positions()
    )


def _check_shape(values: object, shape: tuple[int, ...]) -> None:
    if not shape:
        return
    if not isinstance(values, Sequence) or len(values) != shape[0]:
        raise ValueError(f"operand must have shape {shape}")
    for item in values:
        _check_shape(item, shape[1:])


BEAT_FORM: ValueSemantics[BeatForm] = ValueSemantics(
    BeatForm,
    "beat form",
    lambda value: isinstance(value, BeatForm),
    lambda left, right: left == right,
    lambda value: value,
)


__all__ = [
    "BEAT_FORM",
    "Batch",
    "BeatForm",
    "Every",
    "Fold",
    "Position",
    "Repeat",
    "Repetition",
    "Tile",
    "pack",
]
