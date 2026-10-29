# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Position maps: which position of which tensor each position of a tensor reads.

A ``PositionMap`` is what a pure data-movement operator does to positions, as a
fact: no data and no time. For each position of the tensor a reader reads (its
``shape``) it names one of the map's ``sources`` and a flat row-major offset in
it, or the map's ``fill``, a constant.

The map is a set of disjoint ``Piece`` values. A piece is a box of the reader's
positions (per axis, a start and an extent), the source it reads, and a strided
body: the box's own row-major index is factored into ``modes`` (each a ``Loop``
of extent and stride, outer first), and a position reads the source at ``base``
plus each mode's digit times its stride. A stride of zero repeats (a broadcast);
two modes stepping one source axis overlap (a window). A position no piece
covers reads the fill; a map whose pieces cover every position has none.

- The *strided* map is one piece over the whole shape, base zero: Reshape,
  Transpose, Squeeze, a broadcast, Im2Col and DepthToSpace are all one.
- Slice is one piece over a smaller reader, whose base is where it starts.
- Pad is one piece in the middle of a larger reader; the rest is the fill. A
  negative pad crops: it is the Slice it equals.
- Concat is one piece a source. Which source a consumer's beat comes from (the
  merge) is a later stage of the channel, not an operation on maps.

**Equality is equality of positions.** Every map is built canonical, so two maps
are ``==`` (and hash alike) exactly when they read the same source position, or
the fill, at every position of the same reader from the same sources:

- a piece's modes drop unit modes and merge contiguous ones (as a
  ``Traversal``'s loops do);
- the fill is ``None`` exactly when the pieces cover every position; one given
  although they do is dropped;
- the pieces are ordered by their start, and a map of several pieces is cut into
  boxes anew from what it reads (``_partition``): the reader is cut along its
  outermost axis where the source read (or the fill) changes, then each part
  likewise, and a part reading one source is one piece where its reads are
  strided. So the boxes depend on the positions read alone, not on how the map was
  spelled: two adjacent pieces that read one source as one strided box become one.

``reads`` is the enumeration: each reader position's source and offset, by NumPy.
Every composition is checked against it.

- ``through`` composes a strided map with a traversal of its reader (what a
  consumer's port presents) into a traversal of its source presenting the same
  elements in the same beats and lanes. Each loop is split where it crosses a
  mode, and the result is compared with the enumeration.
- ``chain`` composes two maps (one read through the other) by enumerating the
  two in turn and cutting the result into strided pieces.
- ``factor_fills`` splits a map with fills, read by a strided map, into an
  in-order ``Insertion`` of the fills (no buffer) and the strided map over the
  tensor with the fills in place: Pad's hardware.
- ``factor_drop`` splits a map that reads part of its source, read by a strided
  map, into an in-order ``Drop`` of the positions it skips (no buffer) and the
  strided map over what remains: Slice's hardware.

Composition is exact or it refuses (``Unmappable``, one finding with a code):
never approximated.

The constructors take numbers, not graph nodes, each with its operator's ONNX
or qonnx semantics: ``identity``, ``reshape``, ``transpose``, ``squeeze``,
``broadcast``, ``im2col`` and ``padded_im2col``, ``depth_to_space``, ``pad``,
``sliced`` and ``concat``. The fill is an integer, because FINN pads integer
tensors; a Pad constant that is not integral is refused.

No map exists for a read whose position comes from run-time data (a gather by
indices): no constructor builds one, and a caller that meets such a read refuses
it with the finding code ``data-dependent-read`` (``DATA_DEPENDENT_READ``).
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
import numpy.typing as npt

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow.traversal import (
    Loop,
    Traversal,
    _canonical,
    _offset_array,
    axis_strides,
    offsets,
)

Offsets = npt.NDArray[np.int64]

DATA_DEPENDENT_READ = "data-dependent-read"
"""The finding code for a read whose position comes from run-time data: no map exists."""

MAX_PIECES = 256
"""The most pieces a position map may have (``too-many-pieces`` above it).

A map of more pieces than this is no movement a channel realizes as insert, drop and
merge stages: each piece is a stage's run, and a read that changes source or stride this
often (an interleaving of two sources element by element, say) is a shuffle of single
elements. It is refused, not approximated; the canonical cut stops as soon as it passes
the bound rather than building every piece.
"""


class Unmappable(Refusal):
    """A position map that cannot be built, or a composition that is no map or traversal:
    one rejection, ``code`` by ``owner``, its message the error's."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),), message)


def _is_int(value: object) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class Piece:
    """A box of reader positions (``start``, ``extent`` per axis) reading source ``source``.

    The box's row-major index is factored by ``modes`` (outer first, canonical); the
    position at box index ``i`` reads the source's flat offset ``base`` plus each
    mode's digit of ``i`` times its stride.
    """

    start: tuple[int, ...]
    extent: tuple[int, ...]
    source: int
    base: int
    modes: tuple[Loop, ...]

    def __post_init__(self) -> None:
        start, extent, modes = tuple(self.start), tuple(self.extent), tuple(self.modes)
        if len(start) != len(extent):
            raise Unmappable("box-invalid", "Piece", "a box has a start and an extent per axis")
        if not all(_is_int(v) and v >= 0 for v in start) or not all(
            _is_int(v) and v >= 1 for v in extent
        ):
            raise Unmappable(
                "box-invalid", "Piece", f"a box starts at {start} >= 0 and spans {extent} >= 1"
            )
        if not _is_int(self.source) or self.source < 0:
            raise Unmappable("piece-invalid", "Piece", "a piece's source is an index >= 0")
        if not _is_int(self.base) or self.base < 0:
            raise Unmappable("piece-invalid", "Piece", "a piece's base is an offset >= 0")
        if not all(isinstance(mode, Loop) for mode in modes):
            raise TypeError("a piece's modes are Loop values")
        modes = _canonical(modes)
        if prod(mode.extent for mode in modes) != prod(extent):
            raise Unmappable(
                "modes-mismatch",
                "Piece",
                f"modes {modes} do not factor a box of {prod(extent)} positions",
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "modes", modes)

    @property
    def size(self) -> int:
        return prod(self.extent)

    @property
    def reach(self) -> int:
        """The highest source offset the piece reads."""
        return self.base + sum((mode.extent - 1) * mode.stride for mode in self.modes)

    def body(self) -> Offsets:
        """The source offset of each position of the box, in the box's row-major order."""
        return self.base + _offset_array(self.modes)


@dataclass(frozen=True)
class PositionMap:
    """For each position of ``shape``, a position of one of ``sources``, or ``fill``.

    ``pieces`` are disjoint boxes inside ``shape``, each reading inside its source.
    ``fill`` is an integer (FINN pads integer tensors), required where some position
    is in no piece and dropped where none is. Built canonical (the module's
    docstring): ``==`` is equality of the positions read.
    """

    shape: tuple[int, ...]
    sources: tuple[tuple[int, ...], ...]
    pieces: tuple[Piece, ...]
    fill: int | None = None

    def __post_init__(self) -> None:
        owner = "PositionMap"
        shape = tuple(self.shape)
        sources = tuple(tuple(source) for source in self.sources)
        pieces = tuple(self.pieces)
        for extent in (*shape, *(extent for source in sources for extent in source)):
            if not _is_int(extent) or extent < 1:
                raise Unmappable("shape-invalid", owner, "tensor extents are positive integers")
        if not all(isinstance(piece, Piece) for piece in pieces):
            raise TypeError("a position map's pieces are Piece values")
        if self.fill is not None and not _is_int(self.fill):
            raise Unmappable(
                "fill-not-integral",
                owner,
                f"the fill {self.fill!r} is no integer: FINN pads integer tensors",
            )
        for piece in pieces:
            if len(piece.start) != len(shape) or any(
                start + extent > bound
                for start, extent, bound in zip(piece.start, piece.extent, shape)
            ):
                raise Unmappable(
                    "piece-outside-reader",
                    owner,
                    f"a box at {piece.start} of {piece.extent} leaves the reader {shape}",
                )
            if piece.source >= len(sources):
                raise Unmappable(
                    "source-unknown", owner, f"source {piece.source} of {len(sources)}"
                )
            if piece.reach >= prod(sources[piece.source]):
                raise Unmappable(
                    "read-outside-source",
                    owner,
                    f"a piece reads offset {piece.reach} of source {sources[piece.source]}",
                )
        if len(pieces) > 1:
            label, value = _paint(shape, pieces, owner)
            pieces = tuple(_partition(label, value, owner))
        covered = sum(piece.size for piece in pieces)
        fill = self.fill
        if covered == prod(shape):
            fill = None
        elif fill is None:
            raise Unmappable(
                "fill-missing", owner, "positions no piece covers read a fill, and none is given"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "pieces", tuple(sorted(pieces, key=lambda piece: piece.start)))
        object.__setattr__(self, "fill", fill)

    @property
    def strided(self) -> bool:
        """Whether it is one piece over the whole shape at base zero: a strided map."""
        return (
            len(self.pieces) == 1
            and self.pieces[0].extent == self.shape
            and self.pieces[0].base == 0
        )


def _box_offsets(shape: Sequence[int], start: Sequence[int], extent: Sequence[int]) -> Offsets:
    """The flat row-major offsets in ``shape`` of a box, in the box's row-major order."""
    found = np.zeros(1, dtype=np.int64)
    for stride, low, count in zip(axis_strides(tuple(shape)), start, extent):
        found = (found[:, None] + np.arange(low, low + count, dtype=np.int64) * stride).ravel()
    return found


def _paint(shape: tuple[int, ...], pieces: Sequence[Piece], owner: str) -> tuple[Offsets, Offsets]:
    """Each reader position's source and offset (both -1 where no piece covers it), as
    arrays of ``shape``; ``Unmappable`` where two pieces overlap."""
    size = prod(shape)
    label = np.full(size, -1, dtype=np.int64)
    value = np.full(size, -1, dtype=np.int64)
    for piece in pieces:
        at = _box_offsets(shape, piece.start, piece.extent)
        if (label[at] >= 0).any():
            raise Unmappable("pieces-overlap", owner, f"the box at {piece.start} overlaps")
        label[at] = piece.source
        value[at] = piece.body()
    return label.reshape(shape), value.reshape(shape)


def reads(pmap: PositionMap) -> tuple[Offsets, Offsets]:
    """The enumeration: for each position of ``pmap.shape`` in row-major order, the source
    it reads and its flat offset there; both -1 where it reads the fill."""
    label, value = _paint(pmap.shape, pmap.pieces, "reads")
    return label.ravel(), value.ravel()


def strided_modes(values: Offsets) -> tuple[int, tuple[Loop, ...]] | None:
    """``values`` as ``base + sum(digit * stride)`` over row-major modes, if it is one: the
    base and the modes (``Loop`` extent and stride, outermost first), or None where it is
    no such sequence (a stride that goes backwards, or rows that do not repeat the first).

    The innermost mode's stride is the first increment, its extent the first index where
    that stride stops holding; every row of that extent must repeat it, and the rows'
    first values are factored the same way, outward. Each level checks every value,
    so the factoring found is exact, and canonical: each extent is the largest.
    """
    base = int(values[0])
    rest = values - base
    modes: list[Loop] = []
    while len(rest) > 1:
        stride = int(rest[1])
        if stride < 0:
            return None
        ramp = np.arange(len(rest), dtype=np.int64) * stride
        off = np.flatnonzero(rest != ramp)
        extent = int(off[0]) if off.size else len(rest)
        if len(rest) % extent:
            return None
        rows = rest.reshape(-1, extent)
        if not np.array_equal(rows - rows[:, :1], np.broadcast_to(ramp[:extent], rows.shape)):
            return None
        modes.insert(0, Loop(extent, stride))
        rest = rows[:, 0]
    return base, tuple(modes)


def _cuts(region: Offsets, *, steps: bool) -> tuple[int, list[int]]:
    """Where to cut ``region``: its outermost axis along which it changes (``steps``: along
    which its increment changes), at each index where it does, anywhere across the others.
    With no such axis, the outermost axis of more than one index, at every index."""
    for axis in range(region.ndim):
        if region.shape[axis] < 2:
            continue
        others = tuple(other for other in range(region.ndim) if other != axis)
        increment = np.diff(region, axis=axis)
        changed = increment != (np.take(increment, [0], axis=axis) if steps else 0)
        at = np.flatnonzero(changed.any(axis=others))
        if at.size:
            return axis, [int(index) + 1 for index in at]
    axis = next(axis for axis, extent in enumerate(region.shape) if extent > 1)
    return axis, list(range(1, region.shape[axis]))


def _partition(label: Offsets, value: Offsets, owner: str) -> list[Piece]:
    """The canonical pieces of a map that reads source ``label`` at offset ``value`` (both
    arrays of the reader's shape; label -1 for the fill).

    A box reading the fill throughout is no piece; a box reading one source in a
    strided order is one. Any other box is cut (``_cuts``): where the source changes,
    or, reading one source, where its increment changes; each part is then cut likewise.
    The cuts depend on what is read alone, so the pieces do. More than ``MAX_PIECES``
    pieces is refused (``too-many-pieces``) as soon as the bound is passed.
    """
    found: list[Piece] = []

    def visit(low: tuple[int, ...], high: tuple[int, ...]) -> None:
        region = tuple(slice(a, b) for a, b in zip(low, high))
        sources, at = label[region], value[region]
        first = int(sources.flat[0])
        if (sources == first).all():
            if first < 0:
                return
            strided = strided_modes(at.ravel())
            if strided is not None:
                extent = tuple(b - a for a, b in zip(low, high))
                found.append(Piece(low, extent, first, strided[0], strided[1]))
                if len(found) > MAX_PIECES:
                    raise Unmappable(
                        "too-many-pieces",
                        owner,
                        f"a map of at least {len(found)} pieces exceeds the bound of {MAX_PIECES}",
                    )
                return
            axis, cuts = _cuts(at, steps=True)
        else:
            axis, cuts = _cuts(sources, steps=False)
        bounds = [low[axis], *(low[axis] + cut for cut in cuts), high[axis]]
        for a, b in zip(bounds[:-1], bounds[1:]):
            visit((*low[:axis], a, *low[axis + 1 :]), (*high[:axis], b, *high[axis + 1 :]))

    visit((0,) * label.ndim, label.shape)
    return found


def _apply(piece: Piece, flat: Offsets) -> Offsets:
    """The source offsets of a covering piece at the reader's flat offsets ``flat``."""
    found = np.full(flat.shape, piece.base, dtype=np.int64)
    inner = 1
    for mode in reversed(piece.modes):
        found += (flat // inner) % mode.extent * mode.stride
        inner *= mode.extent
    return found


# --- composition with a traversal ------------------------------------------------------


def _through_loop(loop: Loop, modes: Sequence[Loop]) -> tuple[Loop, ...] | None:
    """``loop``, over the reader's flat offsets, as loops over the source's, outer first.

    While stepping ``loop`` carries out of no mode, it is one loop. Otherwise it must
    step a single mode by a divisor of the mode's extent, so the loop splits where the
    mode carries into the next: the inner part walks that mode, the outer part goes on
    from the next. Anything else is no loop nest (None).
    """
    if loop.stride == 0:
        return (loop,)
    extents = tuple(mode.extent for mode in modes)
    reading = axis_strides(extents)
    size = prod(extents)
    found: list[Loop] = []  # inner first
    count, stride = loop.extent, loop.stride
    while count > 1:
        if stride >= size:
            return None
        digits = [stride // inner % extent for inner, extent in zip(reading, extents)]
        if all((count - 1) * digit < extent for digit, extent in zip(digits, extents)):
            found.append(Loop(count, sum(d * mode.stride for d, mode in zip(digits, modes))))
            break
        moving = [axis for axis, digit in enumerate(digits) if digit]
        if len(moving) != 1:
            return None
        axis = moving[0]
        digit, extent = digits[axis], extents[axis]
        if extent % digit or count % (extent // digit):
            return None
        inner_count = extent // digit
        found.append(Loop(inner_count, digit * modes[axis].stride))
        count //= inner_count
        stride = reading[axis] * extent
    return tuple(reversed(found))


def _through_loops(loops: Sequence[Loop], modes: Sequence[Loop], owner: str) -> list[Loop]:
    composed: list[Loop] = []
    for loop in loops:
        split = _through_loop(loop, modes)
        if split is None:
            raise Unmappable(
                "loop-straddles-mode",
                owner,
                f"{loop} carries across the modes {tuple(modes)}: no loop nest",
            )
        composed.extend(split)
    return composed


def _strided_piece(pmap: PositionMap, owner: str) -> Piece:
    if len(pmap.pieces) != 1 or pmap.fill is not None:
        raise Unmappable(
            "not-one-piece",
            owner,
            f"a map of {len(pmap.pieces)} pieces{' and a fill' if pmap.fill is not None else ''}"
            " is no strided map",
        )
    piece = pmap.pieces[0]
    if piece.base:
        raise Unmappable(
            "map-offset",
            owner,
            f"the map starts at offset {piece.base} of its source; a traversal starts at 0",
        )
    return piece


def through(pmap: PositionMap, form: Traversal) -> Traversal:
    """``form``, a traversal of ``pmap``'s reader, as a traversal of its source.

    The result presents the same elements in the same beats and lanes. ``pmap`` must
    be strided (``not-one-piece``, ``map-offset`` otherwise). A loop that carries
    across the map's modes (``loop-straddles-mode``), or loops whose sum carries
    (``carry-between-loops``, found against the enumeration), is refused.
    """
    owner = "through"
    if form.shape != pmap.shape:
        raise Unmappable(
            "shape-mismatch", owner, f"the traversal walks {form.shape}, the map reads {pmap.shape}"
        )
    piece = _strided_piece(pmap, owner)
    source = pmap.sources[piece.source]
    if not source:
        raise Unmappable(
            "source-rank-zero", owner, "a traversal walks a tensor of rank one or more"
        )
    beats = _through_loops(form.beat_loops, piece.modes, owner)
    lanes = _through_loops(form.lane_loops, piece.modes, owner)
    try:
        composed = Traversal(source, beats, lanes)
    except ValueError as error:
        raise Unmappable("carry-between-loops", owner, str(error)) from error
    if not np.array_equal(offsets(composed), _apply(piece, offsets(form))):
        raise Unmappable(
            "carry-between-loops",
            owner,
            "the loops' strides carry into one another: the composition is no loop nest",
        )
    return composed


# --- composition of two maps -----------------------------------------------------------


def chain(outer: PositionMap, inner: PositionMap, source: int = 0) -> PositionMap:
    """``inner`` with its source ``source`` read through ``outer``: one map from
    ``inner``'s reader to ``outer``'s sources (in ``source``'s place among ``inner``'s).

    The two maps are enumerated in turn (``reads``) and the result is cut into its
    canonical strided pieces, so it is exact by construction. It has one fill: two
    different fills read are refused (``fills-differ``).
    """
    owner = "chain"
    if not _is_int(source) or not 0 <= source < len(inner.sources):
        raise Unmappable("source-unknown", owner, f"source {source} of {len(inner.sources)}")
    if inner.sources[source] != outer.shape:
        raise Unmappable(
            "shape-mismatch",
            owner,
            f"source {source} of the inner map is {inner.sources[source]}, "
            f"the outer map reads {outer.shape}",
        )
    added = len(outer.sources) - 1
    sources = (*inner.sources[:source], *outer.sources, *inner.sources[source + 1 :])
    label, value = reads(inner)
    outer_label, outer_value = reads(outer)
    through_outer = label == source
    read = value[through_outer]
    outer_fill_read = bool((outer_label[read] < 0).any())
    label = np.where(label > source, label + added, label)
    label[through_outer] = np.where(outer_label[read] < 0, -1, outer_label[read] + source)
    value[through_outer] = outer_value[read]
    fill = inner.fill
    if outer_fill_read:
        if fill is not None and fill != outer.fill:
            raise Unmappable(
                "fills-differ", owner, f"positions read the fills {fill} and {outer.fill}"
            )
        fill = outer.fill
    pieces = _partition(label.reshape(inner.shape), value.reshape(inner.shape), owner)
    return PositionMap(inner.shape, sources, tuple(pieces), fill)


# --- Pad's and Slice's factorizations --------------------------------------------------


@dataclass(frozen=True)
class Insertion:
    """A map with fills that a producer walking its reader in increasing order realizes
    by inserting the fills into its source's row-major stream, moving nothing.

    ``fills`` has one source, and its covered positions, in row-major order, read that
    source's offsets 0, 1, 2, ... each once (``insertion-out-of-order`` otherwise).
    """

    fills: PositionMap

    def __post_init__(self) -> None:
        owner = "Insertion"
        if not isinstance(self.fills, PositionMap):
            raise TypeError("an insertion is a PositionMap")
        if len(self.fills.sources) != 1:
            raise Unmappable(
                "not-an-insertion", owner, "an insertion moves the elements of one source"
            )
        source, offset = reads(self.fills)
        if not np.array_equal(
            offset[source >= 0], np.arange(prod(self.fills.sources[0]), dtype=np.int64)
        ):
            raise Unmappable(
                "insertion-out-of-order",
                owner,
                "the covered positions do not read the source once each, in its row-major order",
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor with the fills in place."""
        return self.fills.shape

    @property
    def count(self) -> int:
        """How many fills are inserted a pass."""
        return prod(self.fills.shape) - sum(piece.size for piece in self.fills.pieces)

    def inserted(self) -> npt.NDArray[np.bool_]:
        """For each position of ``shape``, in row-major order: whether it is a fill."""
        return reads(self.fills)[0] < 0


@dataclass(frozen=True)
class Drop:
    """A map that a producer walking its source in increasing order realizes by skipping
    the positions it does not read, moving nothing.

    ``kept`` has one source and no fill, and its positions, in row-major order, read
    that source at strictly increasing offsets (``drop-out-of-order`` otherwise).
    """

    kept: PositionMap

    def __post_init__(self) -> None:
        owner = "Drop"
        if not isinstance(self.kept, PositionMap):
            raise TypeError("a drop is a PositionMap")
        if len(self.kept.sources) != 1 or self.kept.fill is not None:
            raise Unmappable(
                "not-a-drop", owner, "a drop keeps positions of one source and inserts none"
            )
        offset = reads(self.kept)[1]
        if not (np.diff(offset) > 0).all():
            raise Unmappable(
                "drop-out-of-order",
                owner,
                "the positions kept are not read once each, in the source's row-major order",
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor the producer walks."""
        return self.kept.sources[0]

    @property
    def count(self) -> int:
        """How many positions are skipped a pass."""
        return prod(self.shape) - prod(self.kept.shape)

    def dropped(self) -> npt.NDArray[np.bool_]:
        """For each position of ``shape``, in row-major order: whether it is skipped."""
        skipped = np.ones(prod(self.shape), dtype=bool)
        skipped[reads(self.kept)[1]] = False
        return skipped


def _read_by(window: PositionMap, tensor: tuple[int, ...], owner: str) -> None:
    if window.sources != (tensor,):
        raise Unmappable(
            "shape-mismatch", owner, f"the window reads {window.sources}, not {tensor}"
        )
    _strided_piece(window, owner)


def factor_fills(fills: PositionMap, window: PositionMap) -> tuple[Insertion, PositionMap]:
    """``window`` reading ``fills``' reader, factored: the in-order insertion of the fills
    before a producer walking ``fills``' reader in increasing order (no buffer), then
    the strided ``window`` over that tensor, fills in place.

    ``window`` is the strided map (a window, for Pad before Im2Col). It must be
    strided over ``fills.shape`` (``shape-mismatch``,
    ``not-one-piece``, ``map-offset``); ``fills`` must be an ``Insertion``.
    """
    _read_by(window, fills.shape, "factor_fills")
    return Insertion(fills), window


def factor_drop(kept: PositionMap, window: PositionMap) -> tuple[Drop, PositionMap]:
    """``window`` reading ``kept``'s reader, factored: the in-order drop of the source
    positions ``kept`` does not read, by a producer walking the source in increasing
    order (no buffer), then the strided ``window`` over what remains.

    ``window`` is the strided map (a window, for Pad before Im2Col). It must be
    strided over ``kept.shape`` (``shape-mismatch``,
    ``not-one-piece``, ``map-offset``); ``kept`` must be a ``Drop``.
    """
    _read_by(window, kept.shape, "factor_drop")
    return Drop(kept), window


# --- constructors ----------------------------------------------------------------------


def _ints(values: Sequence[int], owner: str, what: str) -> tuple[int, ...]:
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError as error:
        raise Unmappable("not-integers", owner, f"{what} are integers: {values}") from error


def _shape(values: Sequence[int], owner: str) -> tuple[int, ...]:
    shape = _ints(values, owner, "extents")
    if any(extent < 1 for extent in shape):
        raise Unmappable("shape-invalid", owner, f"tensor extents are positive: {shape}")
    return shape


def _axis(axis: int, rank: int, owner: str) -> int:
    axis = operator.index(axis)
    if not -rank <= axis < rank:
        raise Unmappable("axis-invalid", owner, f"axis {axis} of a rank-{rank} tensor")
    return axis % rank


def _whole(shape: tuple[int, ...], source: tuple[int, ...], modes: Sequence[Loop]) -> PositionMap:
    """The strided map of ``modes`` from ``shape`` onto ``source``."""
    return PositionMap(shape, (source,), (Piece((0,) * len(shape), shape, 0, 0, tuple(modes)),))


def identity(shape: Sequence[int]) -> PositionMap:
    """Each position reads itself."""
    shape = _shape(shape, "identity")
    return _whole(shape, shape, (Loop(prod(shape), 1),))


def reshape(source: Sequence[int], shape: Sequence[int]) -> PositionMap:
    """ONNX Reshape to ``shape`` (resolved: no 0 or -1): the flat offsets are the source's."""
    source, shape = _shape(source, "reshape"), _shape(shape, "reshape")
    if prod(source) != prod(shape):
        raise Unmappable("size-mismatch", "reshape", f"{source} cannot be reshaped to {shape}")
    return _whole(shape, source, (Loop(prod(shape), 1),))


def transpose(source: Sequence[int], perm: Sequence[int] | None = None) -> PositionMap:
    """ONNX Transpose: output axis ``i`` is input axis ``perm[i]``; reversed by default."""
    source = _shape(source, "transpose")
    perm = tuple(reversed(range(len(source)))) if perm is None else _ints(perm, "transpose", "perm")
    if sorted(perm) != list(range(len(source))):
        raise Unmappable(
            "perm-invalid", "transpose", f"{perm} is no permutation of {len(source)} axes"
        )
    strides = axis_strides(source)
    return _whole(
        tuple(source[axis] for axis in perm),
        source,
        tuple(Loop(source[axis], strides[axis]) for axis in perm),
    )


def squeeze(source: Sequence[int], axes: Sequence[int] | None = None) -> PositionMap:
    """ONNX Squeeze: drop ``axes`` (every axis of extent one by default), each of extent one."""
    source = _shape(source, "squeeze")
    if axes is None:
        dropped = {axis for axis, extent in enumerate(source) if extent == 1}
    else:
        dropped = {_axis(axis, len(source), "squeeze") for axis in axes}
    if any(source[axis] != 1 for axis in dropped):
        raise Unmappable(
            "squeeze-not-unit", "squeeze", f"axes {sorted(dropped)} of {source} are not all one"
        )
    shape = tuple(extent for axis, extent in enumerate(source) if axis not in dropped)
    return _whole(shape, source, (Loop(prod(shape), 1),))


def broadcast(source: Sequence[int], shape: Sequence[int]) -> PositionMap:
    """NumPy broadcasting, as ONNX Expand: the reader is ``source`` and ``shape``
    broadcast together (trailing axes aligned, an extent of one repeated)."""
    source, shape = _shape(source, "broadcast"), _shape(shape, "broadcast")
    rank = max(len(source), len(shape))
    padded = (1,) * (rank - len(source)) + source
    other = (1,) * (rank - len(shape)) + shape
    reader = []
    for mine, theirs in zip(padded, other):
        if mine != theirs and 1 not in (mine, theirs):
            raise Unmappable(
                "broadcast-incompatible", "broadcast", f"{source} and {shape} do not broadcast"
            )
        reader.append(max(mine, theirs))
    strides = axis_strides(padded)
    return _whole(
        tuple(reader),
        source,
        tuple(
            Loop(extent, 0 if mine == 1 else stride)
            for extent, mine, stride in zip(reader, padded, strides)
        ),
    )


def _pair(values: Sequence[int], owner: str, what: str) -> tuple[int, int]:
    pair = _ints(values, owner, what)
    if len(pair) != 2 or min(pair) < 1:
        raise Unmappable("window-invalid", owner, f"a window's {what} is two positive integers")
    return pair[0], pair[1]


def im2col(
    source: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int] = (1, 1),
    dilation: Sequence[int] = (1, 1),
) -> PositionMap:
    """qonnx's Im2Col without padding: an NHWC image ``(N, H, W, C)`` read as
    ``(N, OH, OW, KH * KW * C)``, each output pixel's patch ``(kh, kw, c)``, ``c``
    innermost. A padded Im2Col is ``padded_im2col``."""
    owner = "im2col"
    image = _shape(source, owner)
    if len(image) != 4:
        raise Unmappable("rank-mismatch", owner, f"Im2Col reads an NHWC image, not {image}")
    (batch, h, w, c), (kh, kw) = image, _pair(kernel, owner, "kernel")
    (sh, sw), (dh, dw) = _pair(stride, owner, "stride"), _pair(dilation, owner, "dilation")
    oh, ow = (h - dh * (kh - 1) - 1) // sh + 1, (w - dw * (kw - 1) - 1) // sw + 1
    if min(h - dh * (kh - 1), w - dw * (kw - 1)) < 1:
        raise Unmappable(
            "window-larger-than-image", owner, f"a {kh}x{kw} window does not fit {image}"
        )
    modes = (
        Loop(batch, h * w * c),
        Loop(oh, sh * w * c),
        Loop(ow, sw * c),
        Loop(kh, dh * w * c),
        Loop(kw, dw * c),
        Loop(c, 1),
    )
    return _whole((batch, oh, ow, kh * kw * c), image, modes)


def padded_im2col(
    source: Sequence[int],
    kernel: Sequence[int],
    stride: Sequence[int] = (1, 1),
    dilation: Sequence[int] = (1, 1),
    pads: Sequence[int] = (0, 0, 0, 0),
    pad_value: int | float = 0,
) -> tuple[PositionMap, PositionMap]:
    """qonnx's Im2Col with ``pads`` ``(H_begin, W_begin, H_end, W_end)``: Pad, then the
    window. Returns the two maps in that order: ``pad``'s, and ``im2col``'s over the
    padded image (what ``factor_fills`` takes)."""
    hb, wb, he, we = _ints(pads, "padded_im2col", "pads")
    if min(hb, wb, he, we) < 0:
        raise Unmappable("pads-negative", "padded_im2col", f"Im2Col pads are >= 0: {pads}")
    padding = pad(source, (0, hb, wb, 0, 0, he, we, 0), pad_value)
    return padding, im2col(padding.shape, kernel, stride, dilation)


def depth_to_space(
    source: Sequence[int], block: int, mode: str = "DCR", *, channels_last: bool = False
) -> PositionMap:
    """ONNX DepthToSpace with ``block`` and ``mode`` (``DCR`` or ``CRD``) on ``(N, C, H, W)``,
    as its Reshape, Transpose and Reshape. ``channels_last`` reads ``(..., H, W, C)``, the
    channel index split the same way, into ``(..., H * block, W * block, C')``."""
    owner = "depth_to_space"
    shape = _shape(source, owner)
    block = operator.index(block)
    if block < 1:
        raise Unmappable("block-invalid", owner, f"a block of {block}")
    if mode not in ("DCR", "CRD"):
        raise Unmappable("mode-unsupported", owner, f"DepthToSpace's mode is DCR or CRD: {mode}")
    if channels_last:
        if len(shape) < 3:
            raise Unmappable("rank-mismatch", owner, f"(..., H, W, C) expected, not {shape}")
        lead, (h, w, c) = shape[:-3], shape[-3:]
    else:
        if len(shape) != 4:
            raise Unmappable("rank-mismatch", owner, f"(N, C, H, W) expected, not {shape}")
        lead, (c, h, w) = shape[:1], shape[1:]
    if c % (block * block):
        raise Unmappable("depth-indivisible", owner, f"{c} channels in blocks of {block}x{block}")
    depth = c // (block * block)
    rank = len(lead)
    kept = tuple(range(rank))
    if channels_last:
        split = (h, w, block, block, depth) if mode == "DCR" else (h, w, depth, block, block)
        order = (0, 2, 1, 3, 4) if mode == "DCR" else (0, 3, 1, 4, 2)
        merged = (h * block, w * block, depth)
    else:
        split = (block, block, depth, h, w) if mode == "DCR" else (depth, block, block, h, w)
        order = (2, 3, 0, 4, 1) if mode == "DCR" else (0, 3, 1, 4, 2)
        merged = (depth, h * block, w * block)
    swap = transpose((*lead, *split), (*kept, *(rank + axis for axis in order)))
    return chain(
        reshape(shape, (*lead, *split)), chain(swap, reshape(swap.shape, (*lead, *merged)))
    )


def _integral(value: object, owner: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise Unmappable(
        "fill-not-integral",
        owner,
        f"the constant {value!r} is not an integer: FINN pads integer tensors",
    )


def pad(
    source: Sequence[int],
    pads: Sequence[int],
    constant: int | float = 0,
    mode: str = "constant",
) -> PositionMap:
    """ONNX Pad in constant mode: ``pads`` is every axis's begin amount, then every end
    amount. The constant must be an integer (or an integral float): FINN pads integer
    tensors. Other modes are refused.

    A negative amount crops: the map is the ``sliced`` crop, then the growth, chained,
    so a Pad that only crops equals (``==``) its Slice."""
    owner = "pad"
    if mode != "constant":
        raise Unmappable("pad-mode-unsupported", owner, f"Pad's {mode!r} mode reads no constant")
    shape = _shape(source, owner)
    amounts = _ints(pads, owner, "pads")
    rank = len(shape)
    if len(amounts) != 2 * rank:
        raise Unmappable("pads-invalid", owner, f"{rank} axes take {2 * rank} pads: {amounts}")
    fill = _integral(constant, owner)
    begins, ends = amounts[:rank], amounts[rank:]
    padded = tuple(size + begin + end for size, begin, end in zip(shape, begins, ends))
    if any(extent < 1 for extent in padded):
        raise Unmappable("pad-empty", owner, f"pads {amounts} leave nothing of {shape}")
    if any(amount < 0 for amount in amounts):
        kept = [size + min(begin, 0) + min(end, 0) for size, begin, end in zip(shape, begins, ends)]
        if any(extent < 1 for extent in kept):
            return PositionMap(padded, (shape,), (), fill)
        crop = sliced(
            shape,
            [max(-begin, 0) for begin in begins],
            [size - max(-end, 0) for size, end in zip(shape, ends)],
        )
        return chain(crop, pad(crop.shape, [max(amount, 0) for amount in amounts], fill))
    strides = axis_strides(shape)
    piece = Piece(
        begins, shape, 0, 0, tuple(Loop(size, stride) for size, stride in zip(shape, strides))
    )
    return PositionMap(padded, (shape,), (piece,), fill)


def sliced(
    source: Sequence[int],
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> PositionMap:
    """ONNX Slice with steps above zero (others refused): negative starts and ends count
    from the end, and both are clamped to the axis."""
    owner = "sliced"
    shape = _shape(source, owner)
    rank = len(shape)
    starts, ends = _ints(starts, owner, "starts"), _ints(ends, owner, "ends")
    axes = tuple(range(len(starts))) if axes is None else _ints(axes, owner, "axes")
    steps = (1,) * len(starts) if steps is None else _ints(steps, owner, "steps")
    if not len(starts) == len(ends) == len(axes) == len(steps):
        raise Unmappable("slice-invalid", owner, "one start, end and step an axis")
    axes = tuple(_axis(axis, rank, owner) for axis in axes)
    if len(set(axes)) != len(axes):
        raise Unmappable("slice-invalid", owner, f"axes {axes} repeat")
    if any(step < 1 for step in steps):
        raise Unmappable("slice-step-unsupported", owner, f"steps above zero only: {steps}")
    first, extent, step_of = [0] * rank, list(shape), [1] * rank
    for axis, begin, end, step in zip(axes, starts, ends, steps):
        size = shape[axis]
        begin = min(max(begin + size if begin < 0 else begin, 0), size)
        end = min(max(end + size if end < 0 else end, 0), size)
        count = max(0, -(-(end - begin) // step))
        if count < 1:
            raise Unmappable("slice-empty", owner, f"axis {axis} keeps no position")
        first[axis], extent[axis], step_of[axis] = begin, count, step
    strides = axis_strides(shape)
    reader = tuple(extent)
    piece = Piece(
        (0,) * rank,
        reader,
        0,
        sum(low * stride for low, stride in zip(first, strides)),
        tuple(Loop(count, step * stride) for count, step, stride in zip(extent, step_of, strides)),
    )
    return PositionMap(reader, (shape,), (piece,))


def concat(sources: Sequence[Sequence[int]], axis: int) -> PositionMap:
    """ONNX Concat along ``axis``: one piece a source, each in its own row-major order."""
    owner = "concat"
    shapes = tuple(_shape(shape, owner) for shape in sources)
    if not shapes:
        raise Unmappable("concat-mismatch", owner, "Concat joins at least one tensor")
    rank = len(shapes[0])
    axis = _axis(axis, rank, owner)
    if any(
        len(shape) != rank
        or shape[:axis] + shape[axis + 1 :] != shapes[0][:axis] + shapes[0][axis + 1 :]
        for shape in shapes
    ):
        raise Unmappable("concat-mismatch", owner, f"{shapes} differ off axis {axis}")
    pieces, at = [], 0
    for index, shape in enumerate(shapes):
        start = tuple(at if each == axis else 0 for each in range(rank))
        pieces.append(Piece(start, shape, index, 0, (Loop(prod(shape), 1),)))
        at += shape[axis]
    reader = tuple(at if each == axis else extent for each, extent in enumerate(shapes[0]))
    return PositionMap(reader, shapes, tuple(pieces))


__all__ = [
    "DATA_DEPENDENT_READ",
    "MAX_PIECES",
    "Drop",
    "Insertion",
    "Piece",
    "PositionMap",
    "Unmappable",
    "broadcast",
    "chain",
    "concat",
    "depth_to_space",
    "factor_drop",
    "factor_fills",
    "identity",
    "im2col",
    "pad",
    "padded_im2col",
    "reads",
    "reshape",
    "sliced",
    "squeeze",
    "strided_modes",
    "through",
    "transpose",
]
