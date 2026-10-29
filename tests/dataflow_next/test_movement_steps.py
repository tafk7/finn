# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Movement between kernels as position maps: the steps a transpose, a broadcast and a lane
regroup come to, each test stating its whole geometry and the numbers it asserts.

- **Transpose, then MatMul.** X (32, 16) is presented row-major by its producer; Y = Xᵀ
  (16, 32) is its position map ``transpose((32, 16), (1, 0))``; a MatMul with M 16, K 32,
  N 16 at (PE, SIMD) reads Y through its plain port ``(m, k)``, SIMD lanes along ``k``,
  composed with ``through``. Its marker closes every ``k`` pass. Five (producer lanes,
  PE, SIMD): the line's steps for the first three, and a chain through one lane for the two
  the line refuses; its live windows, in source beats.
- **Broadcast.** A (C,) operand, C = 8, broadcast over (H, W, C) = (4, 6, 8) and read by an
  element-wise port at PE 1, 2 and 8: the composed traversal is the one the line states by
  reading the trailing index; from a source presenting (C,) once the steps are one reorder.
- **Through one lane.** The two arrangements the line refuses, as a reorder between their
  regroupings to one lane a beat.

Every chain is checked by enumeration (``undelivered``) and against the offsets stated
directly from the MatMul's loop order.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
from dataflow_next.harness import translate
from dataflow_next.harness.checks import undelivered

import finn.dataflow.schedule as line_schedule
import finn.dataflow.steps as line_steps
import finn.dataflow.traversal as line_traversal
from finn.dataflow_next.buffers import live_window, slots
from finn.dataflow_next.gemm import k, m, n
from finn.dataflow_next.maps import broadcast, through, transpose
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import Cost, Step, Steps, cost, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    LevelEnd,
    Reorder,
    Traversal,
    classify,
    offsets,
    regrouped,
    unreplayed,
    vector_major,
)

Array = npt.NDArray[np.int64]

# X (I, J) = (32, 16), row-major; Y = Xᵀ (J, I) = (16, 32); the MatMul: M = J = 16 rows
# of Y, K = I = 32 its depth, N = 16 outputs.
ROWS, COLUMNS, OUTPUTS = 32, 16, 16


def _geometry(lanes: int, pe: int, simd: int) -> tuple[BeatSequence, BeatSequence, Traversal]:
    """The channel's two ends and the consumer's own port: X from its producer at ``lanes``
    lanes a beat, the MatMul's port over Y composed with the transpose (a traversal of X,
    marker closing each ``k`` pass), and the port over Y itself."""
    schedule = Schedule({m: COLUMNS, n: OUTPUTS, k: ROWS}, {n: pe, k: simd}, (m, n, k))
    port = schedule.present((COLUMNS, ROWS), (m, k), lanes=(k,))
    composed = through(transpose((ROWS, COLUMNS), (1, 0)), port)
    source = BeatSequence(vector_major((ROWS, COLUMNS), lanes))
    sink = BeatSequence(composed, markers=(schedule.closing((k,)),))
    return source, sink, port


def _read(pe: int) -> Array:
    """The offsets of X the MatMul reads, in its loop order ``(m, n, k)``: Y[m, k] is
    X[k, m], offset ``k * 16 + m``; the loop over the N / PE output folds repeats it."""
    row, _, depth = np.meshgrid(
        np.arange(COLUMNS), np.arange(OUTPUTS // pe), np.arange(ROWS), indexing="ij"
    )
    return np.asarray((depth * COLUMNS + row).ravel(), dtype=np.int64)


def _reorders(chain: Steps) -> list[Reorder]:
    return [hop.reorder for hop in chain.hops if hop.reorder is not None]


def _line_steps(source: BeatSequence, sink: BeatSequence) -> Steps:
    """What ``finn.dataflow.steps.steps`` answers for the same two sequences."""
    answer = line_steps.steps(translate.back(source), translate.back(sink))
    found = translate.forward(answer)
    assert isinstance(found, Steps)
    return found


# --- transpose, then MatMul ------------------------------------------------------------


@pytest.mark.parametrize(
    ("lanes", "pe", "simd", "kinds", "reorder"),
    [
        (1, 4, 1, (Step.REORDER, Step.MARKERS), Reorder(512, (16, 4, 32), (1, 0, 16))),
        (
            4,
            4,
            1,
            (Step.WIDTH, Step.REORDER, Step.MARKERS),
            Reorder(512, (16, 4, 32), (1, 0, 16)),
        ),
        (
            1,
            1,
            4,
            (Step.REORDER, Step.WIDTH, Step.MARKERS),
            Reorder(512, (16, 16, 32), (1, 0, 16)),
        ),
    ],
    ids=["lanes1-pe4-simd1", "lanes4-pe4-simd1", "lanes1-pe1-simd4"],
)
def test_a_transposed_matmul_is_the_lines_steps(
    lanes: int, pe: int, simd: int, kinds: tuple[Step, ...], reorder: Reorder
) -> None:
    """X (32, 16) from a producer at 1 or 4 lanes, the MatMul at PE 4 SIMD 1, or PE 1 SIMD 4:
    one reorder, a width conversion where the lanes differ, then the markers; the line's
    chain exactly (the line states the transposed port itself), the reorder 512 beats
    framed, 16 x 4 x 32 beats of the frame read at 1, 0 and 16."""
    source, sink, _ = _geometry(lanes, pe, simd)
    found = steps(source, sink)
    assert found.kinds == kinds
    assert _reorders(found) == [reorder]
    assert undelivered(found, source, sink) == ""
    assert np.array_equal(offsets(sink.form), _read(pe))
    assert sink.markers == (LevelEnd(ROWS // simd),)
    assert _line_steps(source, sink) == found


@pytest.mark.parametrize(
    ("lanes", "pe", "simd", "expected"),
    [(4, 4, 4, Cost(505, 128, 1)), (2, 4, 2, Cost(501, 256, 1))],
    ids=["lanes4-pe4-simd4", "lanes2-pe4-simd2"],
)
def test_a_transposed_matmul_the_line_refuses_goes_through_one_lane(
    lanes: int, pe: int, simd: int, expected: Cost
) -> None:
    """Producer lanes and SIMD equal to 4, or to 2 (PE 4 both): the line refuses, the
    chain is a width conversion to one lane, the same reorder as at one lane (512 beats
    framed, 16 x 4 x 32 read at 1, 0, 16), a width conversion to the MatMul's SIMD and
    the markers. Its narrowest lane count is one; its live window 505 or 501 elements
    and its reads 128 or 256 beats of the producer."""
    source, sink, _ = _geometry(lanes, pe, simd)
    with pytest.raises(line_steps.Unrealizable):
        _line_steps(source, sink)
    found = steps(source, sink)
    assert found.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH, Step.MARKERS)
    assert _reorders(found) == [Reorder(512, (16, 4, 32), (1, 0, 16))]
    first, reorder, last, _ = found.hops
    assert [form.form.lanes for form in (first.source, first.sink)] == [lanes, 1]
    assert [form.form.lanes for form in (reorder.source, reorder.sink)] == [1, 1]
    assert [form.form.lanes for form in (last.source, last.sink)] == [1, simd]
    assert cost(found, sink) == expected
    # By enumeration, and against the offsets stated from the MatMul's loops.
    assert undelivered(found, source, sink) == ""
    assert np.array_equal(offsets(sink.form), _read(pe))
    # The enumeration rejects a chain that carries another sink's positions.
    other, _, _ = _geometry(lanes, 2, simd)
    assert undelivered(found, source, BeatSequence(other.form)) != ""


def test_the_live_windows_of_a_transpose_at_one_lane_a_beat() -> None:
    """Producer 1 lane, PE 4, SIMD 1: the composed reorder holds 497 source beats; the
    transpose alone (X row-major to Y presented row-major, no replay) 496, and the replay
    the MatMul's port then asks of a row-major Y 32, so two stages hold 528 where one
    holds 497. Both sides at one lane a beat, so a window is in beats of X."""
    source, sink, _ = _geometry(1, 4, 1)
    x = vector_major((ROWS, COLUMNS), 1)
    y_rows = vector_major((COLUMNS, ROWS), 1)
    # The port over Y at one lane (no SIMD lanes), composed with the transpose.
    serial = Schedule({m: COLUMNS, n: OUTPUTS, k: ROWS}, {n: 4}, (m, n, k)).present(
        (COLUMNS, ROWS), (m, k)
    )
    composed = through(transpose((ROWS, COLUMNS), (1, 0)), serial)
    alone = through(transpose((ROWS, COLUMNS), (1, 0)), y_rows)
    assert alone == unreplayed(composed)
    assert live_window(slots(x, composed)) == 497
    assert live_window(slots(x, alone)) == 496
    assert live_window(slots(y_rows, serial)) == 32
    assert 497 < 496 + 32
    # The channel's chain holds the same 497 beats, one a beat.
    found = steps(source, sink)
    assert cost(found, sink) == Cost(497, 512, 1)
    assert undelivered(found, source, sink) == ""
    assert undelivered(found, source, BeatSequence(composed, markers=sink.markers)) == ""


# --- broadcast -------------------------------------------------------------------------

H, W, C = 4, 6, 8


@pytest.mark.parametrize(
    ("pe", "reorder"),
    [
        (1, Reorder(8, (24, 8), (0, 1))),
        (2, Reorder(4, (24, 4), (0, 1))),
        (8, Reorder(1, (24,), (0,))),
    ],
    ids=["pe1", "pe2", "pe8"],
)
def test_a_broadcast_operand_composes_to_the_trailing_index_and_is_one_reorder(
    pe: int, reorder: Reorder
) -> None:
    """(C,) = (8,) broadcast over (H, W, C) = (4, 6, 8), an element-wise port with PE lanes
    along ``c``, its schedule over ``(h, w, c)``."""
    h, w, c = Index("h"), Index("w"), Index("c")
    schedule = Schedule({h: H, w: W, c: C}, {c: pe}, (h, w, c))
    # The operand as the line states it: only the trailing index, the (C,) operand at (c,).
    trailing = schedule.present((C,), (c,), lanes=(c,))
    plain = schedule.present((H, W, C), (h, w, c), lanes=(c,))
    composed = through(broadcast((C,), (H, W, C)), plain)
    assert composed == trailing
    # The same, built with the line's own schedule, translated.
    lh, lw, lc = (line_schedule.Index(name) for name in "hwc")
    line = line_schedule.Schedule({lh: H, lw: W, lc: C}, {lc: pe}, (lh, lw, lc))
    assert translate.forward(line.present((C,), (lc,), lanes=(lc,))) == composed
    # Read each (h, w) pixel's channels again: the (C,) operand, H * W times over.
    assert np.array_equal(offsets(composed), np.tile(np.arange(C), H * W))
    source, sink = BeatSequence(vector_major((C,), pe)), BeatSequence(composed)
    found = steps(source, sink)
    assert found.kinds == (Step.REORDER,)
    assert _reorders(found) == [reorder]
    assert undelivered(found, source, sink) == ""
    assert _line_steps(source, sink) == found


# --- through one lane ------------------------------------------------------------------


@pytest.mark.parametrize(("lanes", "simd"), [(4, 4), (2, 2)], ids=["lanes4-simd4", "lanes2-simd2"])
def test_the_refused_arrangements_are_a_reorder_through_one_lane(lanes: int, simd: int) -> None:
    """X (32, 16) at 4 lanes with the MatMul at PE 4 SIMD 4, and at 2 lanes with SIMD 2:
    regrouped to one lane a beat, the producer's order and the MatMul's differ by a
    reorder, as the line's own ``classify`` says too, with the chain's reorder."""
    source, sink, _ = _geometry(lanes, 4, simd)
    low, target = regrouped(source.form, 1), regrouped(sink.form, 1)
    verdict = classify(low, target)
    assert verdict.adaptation.name == "REORDER"
    assert verdict.reorder == Reorder(512, (16, 4, 32), (1, 0, 16))
    assert (low.lanes, target.lanes) == (1, 1)
    # The line: at the arrangements' own lanes a refusal, at one lane a reorder.
    old = line_traversal.classify(translate.back(low), translate.back(target))
    assert old.adaptation.name == "REORDER"
    assert translate.forward(old.reorder) == verdict.reorder
    assert line_traversal.classify(
        translate.back(source.form), translate.back(sink.form)
    ).adaptation.name not in ("REORDER", "WIDTH_CONVERSION")
    # That reorder is the chain's.
    found = steps(source, sink)
    assert _reorders(found) == [verdict.reorder]
    assert found.hops[1].source.form == low and found.hops[1].sink.form == target
