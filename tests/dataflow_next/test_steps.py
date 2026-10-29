# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The canonical steps between two beat sequences of one tensor: a channel's two streams."""

from __future__ import annotations

import pytest

from finn.dataflow_next.steps import Step, Unrealizable, presented, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    regrouped,
    tile,
    vector_major,
)

R, K, SIMD, NF = 3, 8, 2, 2
ROWS = vector_major((R, K), SIMD)
REPLAYED = BeatSequence(ROWS.replayed(NF, inner_beats=K // SIMD), markers=(LevelEnd(K // SIMD),))


def test_equal_presentations_and_lane_orders_need_nothing() -> None:
    assert not steps(BeatSequence(ROWS), BeatSequence(ROWS))
    # A lane permutation is wires, not a step.
    weights = tile(4, 4, 2, 2)
    swapped = Traversal(weights.shape, weights.beat_loops, tuple(reversed(weights.lane_loops)))
    assert steps(BeatSequence(weights), BeatSequence(swapped)).kinds == ()
    assert steps(BeatSequence(ROWS), BeatSequence(ROWS)).describe() == "direct"


def test_a_replay_with_a_frame_is_a_reorder_then_markers() -> None:
    found = steps(BeatSequence(ROWS), REPLAYED)
    assert found.kinds == (Step.REORDER, Step.MARKERS)
    reorder, markers = found.hops
    assert reorder.reorder is not None and 0 in reorder.reorder.coefs
    assert markers.sink.markers == (LevelEnd(K // SIMD),)
    # Offered markers the consumer requires need no step.
    offered = BeatSequence(REPLAYED.form, markers=REPLAYED.markers)
    assert not steps(offered, REPLAYED)


def test_lanes_and_order_decompose_through_the_other_side_s_lanes() -> None:
    wide = vector_major((R, K), 4)
    assert steps(BeatSequence(wide), BeatSequence(ROWS)).kinds == (Step.WIDTH,)
    assert steps(BeatSequence(wide), REPLAYED).kinds == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    # Row-major one a beat into a tile: reorder at one lane, then widen.
    ones = vector_major((4, 4), 1)
    assert steps(BeatSequence(ones), BeatSequence(tile(4, 4, 2, 2))).kinds == (
        Step.REORDER,
        Step.WIDTH,
    )


def test_a_lane_regroup_goes_through_the_common_lane_count() -> None:
    rows_as_lanes = Traversal.over((3, 4), ((1, 4, 1),), ((0, 3, 1),))
    found = steps(BeatSequence(rows_as_lanes), BeatSequence(vector_major((3, 4), 2)))
    assert found.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert found.hops[0].sink.form.lanes == 1


def test_a_cyclic_source_repeats_once_a_consumer_pass() -> None:
    period = tile(4, 4, 2, 2)
    cyclic = BeatSequence(period, Repetition.CYCLIC)
    assert presented(cyclic, BeatSequence(period.repeated(3))) == period.repeated(3)
    # Another lane count: a pass for each of the consumer's.
    assert steps(cyclic, BeatSequence(regrouped(period, 2).repeated(2))).kinds == (Step.WIDTH,)
    # A pass of half the tensor reads one pass of the source, the other half dropped: from
    # (4, 4) row by row at 4 lanes, one reorder reading the first two of four beats.
    half = Traversal.over((4, 4), ((0, 2, 1),), ((1, 4, 1),))
    rows = BeatSequence(vector_major((4, 4), 4), Repetition.CYCLIC)
    assert presented(rows, BeatSequence(half)) == vector_major((4, 4), 4)
    found = steps(rows, BeatSequence(half))
    assert [hop.reorder for hop in found.hops] == [Reorder(4, (2,), (1,))]
    # A source repeating its own pass twice feeds a consumer of whole multiples of it.
    with pytest.raises(Unrealizable, match="whole repetitions"):
        steps(BeatSequence(period.repeated(2), Repetition.CYCLIC), BeatSequence(period.repeated(3)))


def test_what_no_chain_repairs_is_unrealizable() -> None:
    with pytest.raises(Unrealizable, match="cyclic consumer"):
        steps(BeatSequence(ROWS), BeatSequence(ROWS, Repetition.CYCLIC))
    # A sink reading positions the source never presents.
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    with pytest.raises(Unrealizable, match="different positions"):
        steps(BeatSequence(half), BeatSequence(vector_major((4,), 2)))


def test_a_sink_reading_part_of_each_frame_is_a_reorder() -> None:
    # The first beat of each two: the frame's other beat is dropped.
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    found = steps(BeatSequence(vector_major((4,), 2)), BeatSequence(half))
    assert found.kinds == (Step.REORDER,)
    reorder = found.hops[0].reorder
    assert reorder is not None
    assert (reorder.frame_beats, reorder.dims, reorder.coefs) == (2, (1,), (0,))


def test_the_steps_split_at_the_transport_after_their_leading_width_conversions() -> None:
    wide = vector_major((R, K), 4)
    found = steps(BeatSequence(wide), REPLAYED)
    assert found.output.kinds == (Step.WIDTH,)
    assert found.input.kinds == (Step.REORDER, Step.MARKERS)
    assert found.output.hops[-1].sink == found.input.hops[0].source
    # A width conversion after a reorder stays on the consumer's side.
    ones = vector_major((4, 4), 1)
    after = steps(BeatSequence(ones), BeatSequence(tile(4, 4, 2, 2)))
    assert not after.output and after.input.kinds == (Step.REORDER, Step.WIDTH)
    # A width conversion alone is the producer's side; nothing is left after the transport.
    alone = steps(BeatSequence(wide), BeatSequence(ROWS))
    assert alone.output.kinds == (Step.WIDTH,) and not alone.input


def test_a_traversal_is_row_major_when_it_presents_a_flat_buffer_in_order() -> None:
    """Every position once, beat after beat, lane zero first: any lane count of the
    default order, and a tiling whose tiles are whole rows; not a tiling across rows,
    a transpose, a replay or a traversal that skips positions."""
    assert vector_major((4, 6), 3).row_major and vector_major((1, 784), 4).row_major
    assert tile(4, 6, 1, 6).row_major and Traversal((1,), (), ()).row_major
    assert not tile(4, 6, 2, 3).row_major
    assert not Traversal.over((3, 2), ((1, 2, 1), (0, 3, 1)), ()).row_major
    assert not ROWS.replayed(2, inner_beats=K // SIMD).row_major
    assert not Traversal((5,), (Loop(2, 2),), ()).row_major
