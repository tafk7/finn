# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A reindex, one value from its producer to its readers: each test states its
geometry and the numbers it asserts, and checks what it derives by enumeration.

- **One reader through a strided map**: X (32, 16) from a one-lane producer, Y = Xᵀ read by
  a MatMul (M 16, K 32, N 16) at PE 4, SIMD 1 through ``transpose``: the route is
  ``steps(producer, composed)``, ``reorder -> markers``, its cost (497, 512, 1).
- **A fork** (two readers through the same map, or through two maps) routes each reader as
  it is alone (``Forked``): its cost sums the branches' live windows (497 twice, or 497 and
  a width step's 4), reads the producer's 512 beats once and takes the fewest lanes; a
  derivation of one reader's (``composed``, ``stage_cost``) is each branch's. Two branches
  reading a cyclic producer's pass once and twice are refused a cost.
- **Pad**: CNV-like 8 x 8 x 4 image, pad 1, a 3 x 3 window, MatMul N 8 at SIMD 4, PE 2:
  144 constants inserted at 4 lanes, then ``reorder -> markers`` (23 beats of 4 lanes held);
  and a Pad of two channels on four, inserted at 2 lanes after a width step.
- **Slice**: rows 1..6 of (8, 6) dropped as whole beats, then a transpose; rows and
  columns at steps 2 and 3, dropped at one lane.
- **Concat's merge**: two (4, 4, C) values joined along the channels: 8 + 4 at 4 lanes
  ``AAB``; at 2 lanes ``AAAABB``; 6 + 2 at 4 lanes straddles, merged at 2 lanes, then a
  width step. A Pad of one source before the Concat: (4, 4, 8) padded by 1 on H and W to
  (6, 6, 8), joined with a (6, 6, 4) value at 4 lanes, the Pad stated as that source's
  own stage; the same Pad as a fill in the merge's own map is refused. A Slice of one source
  before the Concat: (6, 6, 8) sliced to its inner (4, 4, 8), joined with a (4, 4, 4) value at
  4 lanes, the Drop stated as that source's own stage (160 positions dropped, cost (0, 88, 4)).
- **The identity**: a call the package's ``steps`` refuses (a single pass feeding a cyclic
  consumer) is refused the same by ``route``. The check over every ``steps.steps`` call
  the comparison harness recorded while FINN derived the chain of eight kernels and the
  geometries is retired with that harness; it is at the immutable baseline 1ac1616.
"""

from __future__ import annotations

from math import prod

import numpy as np
import numpy.typing as npt
import pytest

from dataflow.harness.checks import undelivered
from finn.core.reporting.findings import Refusal
from finn.dataflow.gemm import k, m, n
from finn.dataflow.maps import (
    Drop,
    Insertion,
    PositionMap,
    Unmappable,
    chain,
    concat,
    identity,
    pad,
    padded_im2col,
    reads,
    reshape,
    sliced,
    through,
    transpose,
)
from finn.dataflow.reindex import (
    FORK_PER_BRANCH,
    Forked,
    Merge,
    Reader,
    Reindex,
    Route,
    Unrouted,
    branches,
    composed,
    joined,
    merge_cost,
    merge_lanes,
    merge_stage_cost,
    merged,
    pattern,
    route,
    route_cost,
    stage_cost,
    stage_lanes,
)
from finn.dataflow.schedule import Schedule
from finn.dataflow.steps import Cost, Step, Steps, Unrepeatable, cost, steps
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    offsets,
    regrouped,
    vector_major,
)

Array = npt.NDArray[np.int64]


def codes(error: pytest.ExceptionInfo[Refusal]) -> list[str]:
    return [finding.code for finding in error.value.findings]


def row_major(shape: tuple[int, ...], lanes: int) -> Traversal:
    return Traversal(shape, (Loop(prod(shape) // lanes, lanes),), (Loop(lanes, 1),))


def one(reindex: Reindex) -> Route:
    """The route of a reindex of one reader (a ``Route``, not a fork's ``Forked``)."""
    found = route(reindex)
    assert isinstance(found, Route)
    return found


# --- one reader ------------------------------------------------------------------------

# X (32, 16) row-major at one lane; Y = Xᵀ (16, 32); a MatMul M 16, K 32, N 16 at PE 4.
X_SHAPE = (32, 16)


def transposed_matmul(pe: int = 4) -> tuple[PositionMap, BeatSequence, Traversal]:
    """The map from Y to X, the MatMul's port over Y with its marker, and that port."""
    schedule = Schedule({m: 16, n: 16, k: 32}, {n: pe}, (m, n, k))
    port = schedule.present((16, 32), (m, k), lanes=(k,))
    return transpose(X_SHAPE, (1, 0)), BeatSequence(port, markers=(schedule.closing((k,)),)), port


def test_one_reader_through_a_strided_map_is_the_steps_to_its_composed_order() -> None:
    pmap, stream, port = transposed_matmul()
    producer = BeatSequence(vector_major(X_SHAPE, 1))
    reindex = Reindex(producer, (Reader(pmap, stream),))
    order = composed(reindex)
    assert order == BeatSequence(through(pmap, port), markers=stream.markers)
    found = one(reindex)
    assert found == Route(Steps(), None, steps(producer, order))
    assert found.after.kinds == (Step.REORDER, Step.MARKERS)
    assert [hop.reorder for hop in found.after.hops if hop.reorder] == [
        Reorder(512, (16, 4, 32), (1, 0, 16))
    ]
    assert undelivered(found.after, producer, order) == ""
    assert stage_cost(reindex) is None
    assert route_cost(reindex) == cost(found.after, order) == Cost(497, 512, 1)


def test_one_reader_through_the_identity_is_the_steps_between_the_two_streams() -> None:
    pmap, stream, port = transposed_matmul()
    producer = BeatSequence(vector_major((16, 32), 4))
    reindex = Reindex(producer, (Reader(identity((16, 32)), stream),))
    assert composed(reindex) == stream
    assert one(reindex) == Route(Steps(), None, steps(producer, stream))
    assert route_cost(reindex) == cost(steps(producer, stream), stream)


def test_a_value_read_twice_through_the_same_map_routes_each_reader_as_alone() -> None:
    pmap, stream, _ = transposed_matmul()
    producer = BeatSequence(vector_major(X_SHAPE, 1))
    reader = Reader(pmap, stream)
    fork = Reindex(producer, (reader, reader))
    alone = Reindex(producer, (reader,))
    assert branches(fork) == (alone, alone)
    assert route(fork) == Forked((one(alone), one(alone)))
    # Each branch holds its own 497-element window; the fork reads the 512 beats once.
    assert route_cost(fork) == Cost(2 * 497, 512, 1)
    for derive in (composed, stage_cost):
        with pytest.raises(Unrouted) as error:
            derive(fork)
        assert codes(error) == [FORK_PER_BRANCH]
    assert composed(branches(fork)[0]) == composed(alone)


def test_a_value_read_through_two_maps_routes_each_reader_its_own_steps() -> None:
    pmap, stream, _ = transposed_matmul()
    producer = BeatSequence(vector_major(X_SHAPE, 1))
    plain = Reader(identity(X_SHAPE), BeatSequence(vector_major(X_SHAPE, 4)))
    fork = Reindex(producer, (Reader(pmap, stream), plain))
    found = route(fork)
    assert isinstance(found, Forked)
    first, second = found.routes
    assert first.after.kinds == (Step.REORDER, Step.MARKERS)
    assert second == route(Reindex(producer, (plain,)))
    assert second.after.describe() == "width_conversion"
    # 497 elements of the reorder and 4 of the width step (lcm(1, 4)); 512 beats once; 1 lane.
    assert route_cost(fork) == Cost(497 + 4, 512, 1)


def test_a_fork_whose_branches_read_a_cyclic_producer_unequally_is_refused_a_cost() -> None:
    # A cyclic (4,) at one lane: one reader takes its pass once (4 beats), the other twice
    # (8 beats). The fork sends every word to both, so one stream cannot serve both.
    form = vector_major((4,), 1)
    producer = BeatSequence(form, Repetition.CYCLIC)
    once = Reader(identity((4,)), BeatSequence(form, Repetition.CYCLIC))
    twice = Reader(identity((4,)), BeatSequence(form.repeated(2), Repetition.CYCLIC))
    assert [route_cost(branch).reads for branch in branches(Reindex(producer, (once, twice)))] == [
        4,
        8,
    ]
    with pytest.raises(Unrouted) as error:
        route_cost(Reindex(producer, (once, twice)))
    assert codes(error) == ["fork-reads-differ"]


def test_a_map_that_is_not_strided_without_a_stage_is_refused_by_its_code() -> None:
    # A Pad's map read directly: one piece in the middle and a fill, no strided map.
    padding = pad((4, 4), (1, 1, 1, 1))
    producer = BeatSequence(vector_major((4, 4), 1))
    reindex = Reindex(producer, (Reader(padding, BeatSequence(vector_major((6, 6), 1))),))
    with pytest.raises(Unmappable) as error:
        one(reindex)
    assert codes(error) == ["not-one-piece"]
    # A Slice's map read directly: it starts inside its source.
    crop = sliced((4, 4), (1,), (3,))
    reindex = Reindex(producer, (Reader(crop, BeatSequence(vector_major((2, 4), 1))),))
    with pytest.raises(Unmappable) as error:
        one(reindex)
    assert codes(error) == ["map-offset"]


def test_a_reader_and_a_reindex_refuse_what_reads_another_tensor() -> None:
    pmap, stream, _ = transposed_matmul()
    with pytest.raises(Unrouted) as error:
        Reader(pmap, BeatSequence(vector_major(X_SHAPE, 1)))
    assert codes(error) == ["shape-mismatch"]
    with pytest.raises(Unrouted) as error:
        Reader(pmap, stream, None, 1)
    assert codes(error) == ["source-unknown"]
    with pytest.raises(Unrouted) as error:
        Reindex(BeatSequence(vector_major((16, 32), 1)), (Reader(pmap, stream),))
    assert codes(error) == ["shape-mismatch"]
    with pytest.raises(Unrouted) as error:
        Reindex(BeatSequence(vector_major(X_SHAPE, 1)), ())
    assert codes(error) == ["no-reader"]
    padding = pad((4, 4), (1, 1, 1, 1))
    with pytest.raises(Unrouted) as error:
        Reader(identity((4, 4)), BeatSequence(vector_major((4, 4), 1)), padding)
    assert codes(error) == ["shape-mismatch"]
    joined_map = concat(((2, 2), (2, 2)), 1)
    with pytest.raises(Unrouted) as error:
        Reader(joined_map, BeatSequence(vector_major((2, 4), 1)), joined_map)
    assert codes(error) == ["stage-sources"]  # a stage reads one value
    with pytest.raises(Unrouted) as error:
        Reader(joined_map, BeatSequence(vector_major((2, 4), 1)), pad((2, 3), (0, 0, 0, 0)))
    assert codes(error) == ["shape-mismatch"]  # the stage gives its own source's tensor
    staged = Reader(joined_map, BeatSequence(vector_major((2, 4), 1)), pad((2, 2), (0, 0, 0, 0)))
    assert staged.tensor == (2, 2)


# --- Pad and Slice ---------------------------------------------------------------------


def delivered_through_stage(reindex: Reindex) -> tuple[Array, Array]:
    """What the reader reads, by enumeration of its whole map (``chain`` of the stage's map
    and its own), and what the route delivers: the composed order's positions over the
    stage's output, each read back through the stage. Both as producer offsets in the
    reader's order, -1 for a fill."""
    (reader,) = reindex.readers
    assert reader.at_producer is not None
    whole = chain(reader.at_producer, reader.map)
    label, offset = reads(whole)
    wanted = np.where(label < 0, -1, offset)[offsets(reader.stream.form)]
    found = one(reindex)
    stage_label, stage_offset = reads(reader.at_producer)
    at = offsets(composed(reindex).form)
    if isinstance(found.stage, Insertion):
        got = np.where(stage_label < 0, -1, stage_offset)[at]
    else:
        assert isinstance(found.stage, Drop)
        got = stage_offset[at]
    return wanted, got


def stage_streams(reindex: Reindex, lanes: int) -> tuple[BeatSequence, BeatSequence]:
    """The stage's input (the producer's tensor) and output, row-major at ``lanes``."""
    stage = one(reindex).stage
    assert stage is not None
    if isinstance(stage, Insertion):
        ends = stage.fills.sources[0], stage.shape
    else:
        ends = stage.shape, stage.kept.shape
    repetition = reindex.producer.repetition
    return (
        BeatSequence(row_major(ends[0], lanes), repetition),
        BeatSequence(row_major(ends[1], lanes), repetition),
    )


def padded_window(simd: int, pe: int) -> Reindex:
    """An (8, 8, 4) image at SIMD lanes, padded by one, read by a 3 x 3 window as the MatMul
    (M 64, K 36, N 8) at SIMD and PE: the Pad at the producer's side, the window after."""
    padding, window = padded_im2col((1, 8, 8, 4), (3, 3), pads=(1, 1, 1, 1))
    rows = window.shape[1] * window.shape[2]
    flat = chain(window, reshape(window.shape, (rows, 36)))
    schedule = Schedule({m: rows, n: 8, k: 36}, {n: pe, k: simd}, (m, n, k))
    port = schedule.present(flat.shape, (m, k), lanes=(k,))
    reader = Reader(flat, BeatSequence(port, markers=(schedule.closing((k,)),)), padding)
    return Reindex(BeatSequence(vector_major((1, 8, 8, 4), simd)), (reader,))


def test_a_padded_window_is_an_insertion_of_whole_pixels_then_a_reorder() -> None:
    reindex = padded_window(simd=4, pe=2)
    found = one(reindex)
    assert isinstance(found.stage, Insertion)
    assert found.stage.count == 144
    assert found.stage.shape == (1, 10, 10, 4)
    assert stage_lanes(found.stage, 4) == 4  # a padded pixel is one whole beat
    assert found.before == Steps()  # the producer presents the image row-major already
    assert found.after.describe() == "reorder -> markers"
    stage_in, stage_out = stage_streams(reindex, 4)
    assert undelivered(found.after, stage_out, composed(reindex)) == ""
    wanted, got = delivered_through_stage(reindex)
    assert np.array_equal(wanted, got)
    # The insertion: no buffer, the producer's 64 beats, 4 lanes; the window holds 23 beats
    # of 4 lanes (as the padded window measured beat by beat).
    assert stage_cost(reindex) == Cost(0, 64, 4)
    after = cost(found.after, composed(reindex))
    assert after.live_window == 23 * 4
    assert route_cost(reindex) == Cost(after.live_window, 64, min(4, after.narrowest_lanes))


def test_a_pad_of_two_channels_on_four_inserts_at_two_lanes_after_a_width_step() -> None:
    # (1, 2, 2, 4) at 4 lanes, two channels of fill after the four: a pixel of six is no
    # whole number of 4-lane beats, so the stage runs at 2, after a width conversion.
    padding = pad((1, 2, 2, 4), (0, 0, 0, 0, 0, 0, 0, 2), constant=-1)
    producer = BeatSequence(vector_major((1, 2, 2, 4), 4))
    reader = Reader(identity((1, 2, 2, 6)), BeatSequence(vector_major((1, 2, 2, 6), 2)), padding)
    reindex = Reindex(producer, (reader,))
    found = one(reindex)
    assert isinstance(found.stage, Insertion) and found.stage.count == 8
    assert found.before.describe() == "width_conversion"
    assert found.after == Steps()
    stage_in, stage_out = stage_streams(reindex, 2)
    assert undelivered(found.before, producer, stage_in) == ""
    assert composed(reindex) == stage_out
    assert np.array_equal(*delivered_through_stage(reindex))
    assert stage_cost(reindex) == Cost(0, 8, 2)
    # The width step holds lcm(4, 2) = 4 elements and reads the producer's 4 beats.
    assert route_cost(reindex) == Cost(4, 4, 2)


def test_a_slice_of_whole_rows_is_a_drop_then_the_transpose_after_it() -> None:
    crop = sliced((8, 6), (1,), (7,))  # rows 1..6 of 8
    producer = BeatSequence(vector_major((8, 6), 6))
    reader = Reader(transpose((6, 6)), BeatSequence(vector_major((6, 6), 1)), crop)
    reindex = Reindex(producer, (reader,))
    found = one(reindex)
    assert isinstance(found.stage, Drop) and found.stage.count == 12
    assert found.before == Steps()
    assert found.after.describe() == "width_conversion -> reorder"
    stage_in, stage_out = stage_streams(reindex, 6)
    assert undelivered(found.after, stage_out, composed(reindex)) == ""
    assert np.array_equal(*delivered_through_stage(reindex))
    assert stage_cost(reindex) == Cost(0, 8, 6)
    assert route_cost(reindex).narrowest_lanes == 1


def test_a_slice_at_steps_drops_at_one_lane() -> None:
    crop = sliced((8, 6), (1, 0), (7, 6), None, (2, 3))  # rows 1, 3, 5; columns 0, 3
    producer = BeatSequence(vector_major((8, 6), 6))
    reader = Reader(identity((3, 2)), BeatSequence(vector_major((3, 2), 2)), crop)
    reindex = Reindex(producer, (reader,))
    found = one(reindex)
    assert isinstance(found.stage, Drop) and found.stage.count == 42
    assert stage_lanes(found.stage, 6) == 1
    assert found.before.describe() == "width_conversion"
    assert found.after.describe() == "width_conversion"
    stage_in, stage_out = stage_streams(reindex, 1)
    assert undelivered(found.before, producer, stage_in) == ""
    assert undelivered(found.after, stage_out, composed(reindex)) == ""
    assert np.array_equal(*delivered_through_stage(reindex))
    assert stage_cost(reindex) == Cost(0, 48, 1)


def test_a_stage_that_is_out_of_order_or_a_window_that_is_not_strided_is_refused() -> None:
    # A pad of the transposed tensor reads the producer out of its row-major order.
    out_of_order = chain(transpose((2, 3), (1, 0)), pad((3, 2), (1, 0, 0, 0)))
    producer = BeatSequence(vector_major((2, 3), 1))
    reader = Reader(identity((4, 2)), BeatSequence(vector_major((4, 2), 1)), out_of_order)
    with pytest.raises(Unmappable) as error:
        one(Reindex(producer, (reader,)))
    assert codes(error) == ["insertion-out-of-order"]
    # A window that is itself a pad after the pad: not strided.
    padding = pad((2, 3), (1, 0, 0, 0))
    again = pad((3, 3), (0, 0, 1, 0))
    reader = Reader(again, BeatSequence(vector_major((4, 3), 1)), padding)
    with pytest.raises(Unmappable) as error:
        one(Reindex(producer, (reader,)))
    assert codes(error) == ["not-one-piece"]


def test_a_cyclic_producer_inserts_once_for_each_pass_its_reader_takes() -> None:
    # The Pad of two channels on four from a producer that loops over its tensor, read
    # three times a pass of its reader: the stage and the width step before it run three
    # times, so they read three times the producer's beats.
    padding = pad((1, 2, 2, 4), (0, 0, 0, 0, 0, 0, 0, 2))
    producer = BeatSequence(vector_major((1, 2, 2, 4), 4), Repetition.CYCLIC)
    thrice = BeatSequence(vector_major((1, 2, 2, 6), 2).repeated(3))
    reindex = Reindex(producer, (Reader(identity((1, 2, 2, 6)), thrice, padding),))
    found = one(reindex)
    assert found.before.describe() == "width_conversion"
    assert found.after == Steps()  # the loop presents the three passes
    assert stage_cost(reindex) == Cost(0, 3 * 8, 2)
    # The width step holds lcm(4, 2) = 4 elements and reads 3 x 4 producer beats.
    assert route_cost(reindex) == Cost(4, 12, 2)


# --- Concat's merge --------------------------------------------------------------------


def channel_merge(
    c1: int, c2: int, pe: int, lanes: tuple[int, int] | None = None, markers: tuple[int, ...] = ()
) -> Merge:
    """Two (4, 4, C) values joined along the channels, read row-major at ``pe`` lanes; each
    producer row-major at ``lanes`` (``pe`` by default)."""
    pmap = concat(((4, 4, c1), (4, 4, c2)), 2)
    stream = BeatSequence(
        vector_major((4, 4, c1 + c2), pe), markers=tuple(LevelEnd(beats) for beats in markers)
    )
    first, second = lanes or (pe, pe)
    return Merge(
        (
            Reindex(BeatSequence(vector_major((4, 4, c1), first)), (Reader(pmap, stream),)),
            Reindex(
                BeatSequence(vector_major((4, 4, c2), second)),
                (Reader(pmap, stream, None, 1),),
            ),
        )
    )


def merge_delivers(merge: Merge) -> bool:
    """Whether each value's composed order is the reader's beats from it, in order, and
    its route delivers it; by enumeration."""
    reader = joined(merge)
    lanes = merge_lanes(reader)
    label, offset = reads(reader.map)
    narrowed = offsets(regrouped(reader.stream.form, lanes))
    for index, reindex in enumerate(merge.reindexes):
        order = composed(reindex)
        wanted = offset[narrowed][label[narrowed] == index]
        if not np.array_equal(offsets(order.form), wanted):
            return False
        if undelivered(one(reindex).after, reindex.producer, order):
            return False
    return True


@pytest.mark.parametrize(
    ("c1", "c2", "pe", "beats", "expected"),
    [(8, 4, 4, (0, 0, 1), Cost(0, 48, 4)), (8, 4, 2, (0, 0, 0, 0, 1, 1), Cost(0, 96, 2))],
    ids=["C8-4-PE4", "C8-4-PE2"],
)
def test_a_merge_along_the_channels_takes_whole_beats_pixel_by_pixel(
    c1: int, c2: int, pe: int, beats: tuple[int, ...], expected: Cost
) -> None:
    merge = channel_merge(c1, c2, pe)
    assert merge_lanes(joined(merge)) == pe
    assert pattern(merge) == beats  # AAB, AAAABB: each pixel's beats, the first value's first
    assert "".join("AB"[source] for source in beats) == ("AAB" if pe == 4 else "AAAABB")
    for reindex in merge.reindexes:
        assert one(reindex) == Route(Steps(), None, Steps())  # each value as it comes
        assert composed(reindex).form == reindex.producer.form
    assert merged(merge) == Steps()
    assert merge_delivers(merge)
    assert merge_stage_cost(merge) == Cost(0, 16 * (c1 + c2) // pe, pe)
    assert merge_cost(merge) == expected


def test_a_merge_whose_beats_straddle_runs_at_two_lanes_then_widens() -> None:
    # 6 + 2 channels at 4 lanes: the reader's second beat of a pixel holds the first value's
    # last two channels and the second's two. At 2 lanes every beat is one value's: the
    # first value narrows from 6 lanes, the merge runs at 2, a width step widens to 4.
    merge = channel_merge(6, 2, 4, lanes=(6, 2), markers=(2,))
    reader = joined(merge)
    assert merge_lanes(reader) == 2
    assert pattern(merge) == (0, 0, 0, 1)
    first, second = merge.reindexes
    assert one(first).after.describe() == "width_conversion"
    assert one(second).after == Steps()
    assert merged(merge).describe() == "width_conversion -> markers"
    assert merge_delivers(merge)
    assert merge_stage_cost(merge) == Cost(0, 64, 2)
    # lcm(6, 2) = 6 and lcm(2, 4) = 4 elements held; 16 + 16 beats read; 2 lanes.
    assert merge_cost(merge) == Cost(10, 32, 2)


# (4, 4, 8) padded by one on H and W to (6, 6, 8), joined along the channels with a (6, 6, 4)
# value; the stage is the first source's own (its reader's ``at_producer``).
PAD_ONE = (1, 1, 0, 1, 1, 0)  # before each axis, then after each: H and W by one


def padded_source_merge(pe: int = 4) -> Merge:
    pmap = concat(((6, 6, 8), (6, 6, 4)), 2)
    stream = BeatSequence(vector_major((6, 6, 12), pe))
    padding = pad((4, 4, 8), PAD_ONE)
    return Merge(
        (
            Reindex(BeatSequence(vector_major((4, 4, 8), pe)), (Reader(pmap, stream, padding, 0),)),
            Reindex(BeatSequence(vector_major((6, 6, 4), pe)), (Reader(pmap, stream, None, 1),)),
        )
    )


def test_a_pad_of_one_source_before_a_concat_is_that_sources_stage_before_the_merge() -> None:
    merge = padded_source_merge()
    reader = joined(merge)
    assert merge_lanes(reader) == 4
    assert pattern(merge) == (0, 0, 1)  # AAB, pixel by pixel, over the padded 6 x 6 image
    first, second = merge.reindexes
    found = one(first)
    assert isinstance(found.stage, Insertion)
    assert found.stage.shape == (6, 6, 8)
    assert found.stage.count == 6 * 6 * 8 - 4 * 4 * 8  # the ring of pixels, 160 constants
    assert stage_lanes(found.stage, 4) == 4  # a padded pixel is two whole beats
    assert found.before == Steps() and found.after == Steps()
    assert one(second) == Route(Steps(), None, Steps())
    stage_in, stage_out = stage_streams(first, 4)
    assert composed(first) == stage_out  # the first value's beats of the reader, in order
    assert undelivered(found.after, stage_out, composed(first)) == ""
    assert merged(merge) == Steps()
    # By enumeration: the elements of the first value in each merged beat, read back through
    # its Pad, are those the whole chain (Pad, then Concat's source 0) reads, a fill as -1.
    label, offset = reads(reader.map)
    narrowed = offsets(regrouped(reader.stream.form, 4))
    padding = pad((4, 4, 8), PAD_ONE)
    whole_label, whole_offset = reads(chain(padding, reader.map, 0))
    wanted = np.where(whole_label < 0, -1, whole_offset)[narrowed][label[narrowed] == 0]
    stage_label, stage_offset = reads(padding)
    got = np.where(stage_label < 0, -1, stage_offset)[offsets(composed(first).form)]
    assert np.array_equal(wanted, got)
    assert (got == -1).sum() == 160
    # The second value has no stage and is delivered as it comes.
    assert np.array_equal(offset[narrowed][label[narrowed] == 1], offsets(composed(second).form))
    # The Pad's insertion reads the producer's 32 beats; the merge takes 36 x 3 beats.
    assert stage_cost(first) == Cost(0, 32, 4) and stage_cost(second) is None
    assert route_cost(first) == Cost(0, 32, 4)
    assert merge_stage_cost(merge) == Cost(0, 6 * 6 * 12 // 4, 4)
    assert merge_cost(merge) == Cost(0, 32 + 36, 4)


# (6, 6, 8) sliced to its inner (4, 4, 8) (rows and columns 1..4), joined along the channels
# with a (4, 4, 4) value; the stage is the first source's own drop (its ``at_producer``).
SLICE_INNER = ((1, 1), (5, 5))  # starts and ends on H and W


def sliced_source_merge(pe: int = 4) -> Merge:
    pmap = concat(((4, 4, 8), (4, 4, 4)), 2)
    stream = BeatSequence(vector_major((4, 4, 12), pe))
    crop = sliced((6, 6, 8), *SLICE_INNER)
    return Merge(
        (
            Reindex(BeatSequence(vector_major((6, 6, 8), pe)), (Reader(pmap, stream, crop, 0),)),
            Reindex(BeatSequence(vector_major((4, 4, 4), pe)), (Reader(pmap, stream, None, 1),)),
        )
    )


def test_a_slice_of_one_source_before_a_concat_is_that_sources_stage_before_the_merge() -> None:
    merge = sliced_source_merge()
    reader = joined(merge)
    assert merge_lanes(reader) == 4
    # Each of the 16 pixels is 8 channels of the first value (two beats) and 4 of the second.
    assert pattern(merge) == (0, 0, 1)
    first, second = merge.reindexes
    found = one(first)
    assert isinstance(found.stage, Drop)
    assert found.stage.shape == (6, 6, 8) and found.stage.kept.shape == (4, 4, 8)
    assert found.stage.count == 6 * 6 * 8 - 4 * 4 * 8 == 160  # the border pixels
    assert stage_lanes(found.stage, 4) == 4  # a dropped pixel is two whole beats
    assert found.before == Steps() and found.after == Steps()
    assert one(second) == Route(Steps(), None, Steps())
    stage_in, stage_out = stage_streams(first, 4)
    assert stage_in.form.beats == 72 and stage_out.form.beats == 32
    assert composed(first) == stage_out  # the first value's beats of the reader, in order
    assert undelivered(found.after, stage_out, composed(first)) == ""
    assert merged(merge) == Steps()
    # By enumeration, without the maps: the source offsets of the (6, 6, 8) value kept by the
    # slice, rows and columns 1..4, row-major; those are the elements the first value's beats
    # of the merged stream deliver, in order, once the stage's output is read through them.
    source = np.arange(6 * 6 * 8).reshape(6, 6, 8)
    kept = source[1:5, 1:5, :].ravel()
    assert kept.size == 4 * 4 * 8 and kept[0] == 1 * 48 + 1 * 8 and kept[-1] == 4 * 48 + 4 * 8 + 7
    delivered = offsets(composed(first).form)
    assert np.array_equal(delivered, np.arange(4 * 4 * 8))
    assert np.array_equal(reads(found.stage.kept)[1][delivered], kept)
    # The merged stream's beats in the pattern: the first value's, then the second's.
    label, offset = reads(reader.map)
    narrowed = offsets(regrouped(reader.stream.form, 4))
    assert np.array_equal(label[narrowed].reshape(-1, 4)[:, 0], np.tile([0, 0, 1], 16))
    assert np.array_equal(offset[narrowed][label[narrowed] == 0], np.arange(4 * 4 * 8))
    # The second value has no stage and is delivered as it comes.
    assert np.array_equal(offset[narrowed][label[narrowed] == 1], offsets(composed(second).form))
    # The drop reads the first producer's 72 beats (288 elements at 4 lanes), the second value
    # its own 16 (64 at 4 lanes); the merge itself passes 16 x 3 = 48 beats; all at 4 lanes.
    assert stage_cost(first) == Cost(0, 6 * 6 * 8 // 4, 4) and stage_cost(second) is None
    assert route_cost(first) == Cost(0, 72, 4)
    assert merge_stage_cost(merge) == Cost(0, 4 * 4 * 12 // 4, 4)
    assert route_cost(second) == Cost(0, 4 * 4 * 4 // 4, 4)
    assert merge_cost(merge) == Cost(0, 72 + 16, 4)


def test_a_pad_of_one_source_before_a_concat_at_narrower_lanes_widens_after_its_stage() -> None:
    # PE 2 over the 8 + 4 channels: the first value's producer presents 4 lanes, so the
    # stage runs at 4 (whole beats of the pixel's 8 channels) and a width step follows it.
    pmap = concat(((6, 6, 8), (6, 6, 4)), 2)
    stream = BeatSequence(vector_major((6, 6, 12), 2))
    first = Reindex(
        BeatSequence(vector_major((4, 4, 8), 4)),
        (Reader(pmap, stream, pad((4, 4, 8), PAD_ONE), 0),),
    )
    second = Reindex(BeatSequence(vector_major((6, 6, 4), 2)), (Reader(pmap, stream, None, 1),))
    merge = Merge((first, second))
    assert merge_lanes(joined(merge)) == 2
    found = one(first)
    assert isinstance(found.stage, Insertion)
    assert found.before == Steps()
    assert found.after.describe() == "width_conversion"
    stage_in, stage_out = stage_streams(first, 4)
    assert undelivered(found.after, stage_out, composed(first)) == ""
    assert composed(first).form.lanes == 2
    label, offset = reads(pmap)
    narrowed = offsets(regrouped(stream.form, 2))
    stage_label, stage_offset = reads(pad((4, 4, 8), PAD_ONE))
    got = np.where(stage_label < 0, -1, stage_offset)[offsets(composed(first).form)]
    whole_label, whole_offset = reads(chain(pad((4, 4, 8), PAD_ONE), pmap, 0))
    wanted = np.where(whole_label < 0, -1, whole_offset)[narrowed][label[narrowed] == 0]
    assert np.array_equal(wanted, got)
    assert merge_cost(merge).narrowest_lanes == 2


def test_a_pad_as_a_fill_in_the_merges_own_map_is_still_refused() -> None:
    # The composed map as one reader's map: a merge's reader states each source's stage, it
    # does not reach one through a fill.
    reached = chain(pad((4, 4, 8), PAD_ONE), concat(((6, 6, 8), (6, 6, 4)), 2), 0)
    stream = BeatSequence(vector_major((6, 6, 12), 4))
    merge = Merge(
        (
            Reindex(BeatSequence(vector_major((4, 4, 8), 4)), (Reader(reached, stream, None, 0),)),
            Reindex(BeatSequence(vector_major((6, 6, 4), 4)), (Reader(reached, stream, None, 1),)),
        )
    )
    with pytest.raises(Unrouted) as error:
        pattern(merge)
    assert codes(error) == ["merge-reads-a-fill"]
    with pytest.raises(Unrouted) as error:
        merge_cost(merge)
    assert codes(error) == ["merge-reads-a-fill"]


def test_a_merge_that_shares_no_reader_is_refused_and_a_forked_value_is_its_branch() -> None:
    merge = channel_merge(8, 4, 4)
    first, second = merge.reindexes
    with pytest.raises(Unrouted) as error:
        Merge((first,))
    assert codes(error) == ["merge-too-few"]
    other = channel_merge(8, 4, 2).reindexes[1]
    with pytest.raises(Unrouted) as error:
        Merge((first, other))
    assert codes(error) == ["merge-reader"]
    with pytest.raises(Unrouted) as error:
        Merge((second, first))
    assert codes(error) == ["merge-reader"]
    # The first value also read elsewhere: the merge prices its branch to the merge only.
    elsewhere = Reader(identity((4, 4, 8)), BeatSequence(vector_major((4, 4, 8), 8)))
    forked = Reindex(first.producer, (*first.readers, elsewhere))
    fork = Merge((forked, second))
    assert joined(fork) == joined(merge)
    assert merge_cost(fork) == merge_cost(merge)


def test_a_merge_of_a_reader_that_reads_a_fill_is_refused() -> None:
    joined_then_padded = chain(concat(((2, 2), (2, 2)), 1), pad((2, 4), (0, 0, 0, 2)))
    stream = BeatSequence(vector_major((2, 6), 2))
    reindexes = tuple(
        Reindex(
            BeatSequence(vector_major((2, 2), 2)), (Reader(joined_then_padded, stream, None, i),)
        )
        for i in range(2)
    )
    with pytest.raises(Unrouted) as error:
        pattern(Merge(reindexes))
    assert codes(error) == ["merge-reads-a-fill"]


# --- the identity ----------------------------------------------------------------------

# The check of every ``steps.steps`` call the comparison harness recorded (its Recorder and
# sources, translated from the retired line) is retired with that harness: it is
# ``test_the_identity_is_the_steps_on_every_call_the_harness_records`` at baseline 1ac1616.


def test_a_call_the_package_refuses_is_refused_the_same_through_the_identity() -> None:
    # A single pass cannot feed a cyclic consumer: the package's steps and the reindex's
    # route (and its cost) refuse alike, by the same class and message.
    producer = BeatSequence(vector_major((4, 4), 2))
    consumer = BeatSequence(vector_major((4, 4), 2), Repetition.CYCLIC)
    with pytest.raises(Unrepeatable) as expected:
        steps(producer, consumer)
    reindex = Reindex(producer, (Reader(identity((4, 4)), consumer),))
    for derive in (route, route_cost):
        with pytest.raises(Unrepeatable) as derived:
            derive(reindex)
        assert type(derived.value) is type(expected.value)
        assert str(derived.value) == str(expected.value)
