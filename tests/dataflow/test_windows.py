# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A sliding window is a reorder: its sink loops step the source's frame by digit vectors.

A window reads its image through ``oh + kh``: two sink loops step one source
digit, so positions overlap, and a stride or dilation leaves some unread. Each
case is checked against the sequences alone: the sink's beat at digits ``d``
must present the positions of the source's frame beat ``sum(d[k] * coefs[k])``.
"""

from __future__ import annotations

import random
from itertools import product
from math import prod

import pytest

from finn.dataflow.plan import Step, Unrealizable, plan
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    LevelEnd,
    Loop,
    Reorder,
    Traversal,
    classify,
    vector_major,
)

oh, ow, kh, kw, c, f = (Index(name) for name in ("oh", "ow", "kh", "kw", "c", "f"))


def window(
    h: int, w: int, ch: int, simd: int, k: int = 3, stride: int = 1, dilation: int = 1
) -> Traversal:
    """A k x k window over an (h, w, ch) image, ``simd`` channels a beat, channels innermost."""
    out_h, out_w = (
        (h - dilation * (k - 1) - 1) // stride + 1,
        (w - dilation * (k - 1) - 1) // stride + 1,
    )
    schedule = Schedule(
        {oh: out_h, ow: out_w, kh: k, kw: k, c: ch}, factors={c: simd}, order=(oh, ow, kh, kw, c)
    )
    access = (oh * stride + kh * dilation, ow * stride + kw * dilation, c)
    return schedule.present((h, w, ch), access, lanes=(c,))


def reads_through(source: Traversal, sink: Traversal, reorder: Reorder) -> bool:
    """Whether every sink beat presents the source's frame beat its digits select."""
    produced = list(source.positions())
    wanted = list(sink.positions())
    per_frame = prod(reorder.dims)
    if len(wanted) % per_frame:
        return False
    for frame in range(len(wanted) // per_frame):
        for index, digits in enumerate(product(*(range(extent) for extent in reorder.dims))):
            beat = frame * reorder.frame_beats + sum(d * k for d, k in zip(digits, reorder.coefs))
            if produced[beat] != wanted[frame * per_frame + index]:
                return False
    return True


def test_a_three_tap_window_is_a_reorder_of_its_input() -> None:
    sink = Schedule({oh: 4, kh: 3}).present((6,), (oh + kh,))
    source = vector_major((6,), 1)
    verdict = classify(source, sink)
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder == Reorder(6, (4, 3), (1, 1))
    assert reads_through(source, sink, verdict.reorder)
    assert plan(BeatSequence(source), BeatSequence(sink)).steps == (Step.REORDER,)


@pytest.mark.parametrize(
    ("image", "simd", "expected"),
    (
        # One channel: input_gen's 2-D kernel, (OH, OW, KH, KW) over rows of W.
        ((4, 4, 1), 1, Reorder(16, (2, 2, 3, 3), (4, 1, 4, 1))),
        # Two channels a pixel, one a beat: each window row is KW * C' contiguous beats.
        ((4, 4, 2), 1, Reorder(32, (2, 2, 3, 6), (8, 2, 8, 1))),
        # All channels a beat: the same nest as one channel.
        ((4, 4, 2), 2, Reorder(16, (2, 2, 3, 3), (4, 1, 4, 1))),
        # CNV's first layer: 32 x 32 x 3, SIMD 3.
        ((32, 32, 3), 3, Reorder(1024, (30, 30, 3, 3), (32, 1, 32, 1))),
    ),
)
def test_a_three_by_three_window_is_a_reorder_of_its_image(
    image: tuple[int, int, int], simd: int, expected: Reorder
) -> None:
    source, sink = vector_major(image, simd), window(*image, simd)
    verdict = classify(source, sink)
    assert verdict.reorder == expected
    if prod(image) <= 64:
        assert reads_through(source, sink, expected)


@pytest.mark.parametrize(("stride", "dilation"), ((2, 1), (1, 2), (2, 2), (4, 1)))
def test_strided_and_dilated_windows_read_part_of_the_frame(stride: int, dilation: int) -> None:
    source, sink = vector_major((7, 7, 1), 1), window(7, 7, 1, 1, 3, stride, dilation)
    verdict = classify(source, sink)
    assert verdict.adaptation is Adaptation.REORDER and verdict.reorder is not None
    assert reads_through(source, sink, verdict.reorder)
    if stride == 4:
        # Row and column 3 are no window's: the reorder drops them.
        presented = {position for beat in sink.positions() for position in beat}
        assert len(presented) == 36


def test_a_window_replayed_per_output_fold_with_a_marker_per_window() -> None:
    # A MatMul over windows: each window twice (two output folds), TLAST per window.
    image, k, folds = (5, 5, 2), 3, 2
    schedule = Schedule(
        {oh: 3, ow: 3, f: folds, kh: k, kw: k, c: 2}, factors={c: 2}, order=(oh, ow, f, kh, kw, c)
    )
    sink = schedule.present(image, (oh + kh, ow + kw, c), lanes=(c,))
    found = plan(
        BeatSequence(vector_major(image, 2)), BeatSequence(sink, markers=(LevelEnd(k * k),))
    )
    assert found.describe() == "reorder -> markers"
    reorder = found.hops[0].reorder
    assert reorder == Reorder(25, (3, 3, 2, 3, 3), (5, 1, 0, 5, 1))
    assert reads_through(vector_major(image, 2), sink, reorder)


def test_a_window_over_another_lane_count_converts_width_first() -> None:
    found = plan(BeatSequence(vector_major((4, 4, 2), 2)), BeatSequence(window(4, 4, 2, 1)))
    assert found.steps == (Step.WIDTH, Step.REORDER)


def test_a_window_over_columns_first_steps_the_transposed_frame() -> None:
    columns = Traversal.over((4, 4), ((1, 4, 1), (0, 4, 1)), ())
    rows_of_windows = Schedule({oh: 4, ow: 2, kw: 3}, order=(oh, ow, kw)).present(
        (4, 4), (oh, ow + kw)
    )
    verdict = classify(columns, rows_of_windows)
    assert verdict.reorder == Reorder(16, (4, 2, 3), (1, 4, 4))
    assert reads_through(columns, rows_of_windows, verdict.reorder)


@pytest.mark.parametrize("seed", range(3))
def test_every_reorder_reads_through_its_frame(seed: int) -> None:
    # The lemma, sampled: random beat nests over 12 elements that share a lane nest.
    rng = random.Random(seed)

    def nest(count: int) -> list[Loop]:
        return [
            Loop(rng.choice((1, 2, 3, 4)), rng.choice((0, 1, 2, 3, 4, 6))) for _ in range(count)
        ]

    checked = 0
    for _ in range(3000):
        lanes = nest(rng.randint(0, 2))
        try:
            source = Traversal((12,), nest(rng.randint(0, 3)), lanes)
            sink = Traversal((12,), nest(rng.randint(0, 3)), lanes)
        except ValueError:
            continue
        verdict = classify(source, sink)
        if verdict.adaptation is not Adaptation.REORDER:
            continue
        reorder = verdict.reorder
        assert reorder is not None
        assert reads_through(source, sink, reorder), (source, sink, reorder)
        # input_gen's address rule: every beat the nest selects lies within its frame.
        reach = sum((extent - 1) * coef for extent, coef in zip(reorder.dims, reorder.coefs))
        assert reach < reorder.frame_beats, (source, sink, reorder)
        checked += 1
    assert checked >= 100, checked


def test_positions_the_source_never_presents_stay_unrealizable() -> None:
    top_rows = Traversal.over((4, 4, 1), ((0, 2, 1), (1, 4, 1)), ())
    with pytest.raises(Unrealizable, match="different positions"):
        plan(BeatSequence(top_rows), BeatSequence(window(4, 4, 1, 1)))
