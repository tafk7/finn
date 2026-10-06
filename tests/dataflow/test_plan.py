# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A channel's plan: the canonical steps between two beat sequences of one tensor."""

from __future__ import annotations

import pytest

from finn.dataflow.plan import Step, Unrealizable, plan, presented
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
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
    assert not plan(BeatSequence(ROWS), BeatSequence(ROWS))
    # A lane permutation is wires, not a step.
    weights = tile(4, 4, 2, 2)
    swapped = Traversal(weights.shape, weights.beat_loops, tuple(reversed(weights.lane_loops)))
    assert plan(BeatSequence(weights), BeatSequence(swapped)).steps == ()
    assert plan(BeatSequence(ROWS), BeatSequence(ROWS)).describe() == "direct"


def test_a_replay_with_a_frame_is_a_reorder_then_markers() -> None:
    found = plan(BeatSequence(ROWS), REPLAYED)
    assert found.steps == (Step.REORDER, Step.MARKERS)
    reorder, markers = found.hops
    assert reorder.reorder is not None and 0 in reorder.reorder.coefs
    assert markers.sink.markers == (LevelEnd(K // SIMD),)
    # Offered markers the consumer requires need no step.
    offered = BeatSequence(REPLAYED.form, markers=REPLAYED.markers)
    assert not plan(offered, REPLAYED)


def test_lanes_and_order_decompose_through_the_other_side_s_lanes() -> None:
    wide = vector_major((R, K), 4)
    assert plan(BeatSequence(wide), BeatSequence(ROWS)).steps == (Step.WIDTH,)
    assert plan(BeatSequence(wide), REPLAYED).steps == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    # Row-major one a beat into a tile: reorder at one lane, then widen.
    ones = vector_major((4, 4), 1)
    assert plan(BeatSequence(ones), BeatSequence(tile(4, 4, 2, 2))).steps == (
        Step.REORDER,
        Step.WIDTH,
    )


def test_a_lane_regroup_goes_through_the_common_lane_count() -> None:
    rows_as_lanes = Traversal.over((3, 4), ((1, 4, 1),), ((0, 3, 1),))
    found = plan(BeatSequence(rows_as_lanes), BeatSequence(vector_major((3, 4), 2)))
    assert found.steps == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert found.hops[0].sink.form.lanes == 1


def test_a_cyclic_source_repeats_by_elements_into_its_consumer_s_pass() -> None:
    period = tile(4, 4, 2, 2)
    cyclic = BeatSequence(period, Repetition.CYCLIC)
    assert presented(cyclic, BeatSequence(period.repeated(3))) == period.repeated(3)
    # Another lane count: the pass holds the same elements.
    assert plan(cyclic, BeatSequence(regrouped(period, 2).repeated(2))).steps == (Step.WIDTH,)
    # A pass of half the tensor is no whole number of the source's passes.
    half = Traversal.over((4, 4), ((0, 2, 1),), ((1, 4, 1),))
    with pytest.raises(Unrealizable, match="whole repetitions"):
        plan(cyclic, BeatSequence(half))


def test_what_no_chain_repairs_is_unrealizable() -> None:
    with pytest.raises(Unrealizable, match="cyclic consumer"):
        plan(BeatSequence(ROWS), BeatSequence(ROWS, Repetition.CYCLIC))
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    with pytest.raises(Unrealizable):
        plan(BeatSequence(vector_major((4,), 2)), BeatSequence(half))
