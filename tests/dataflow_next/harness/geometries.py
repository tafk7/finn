# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Geometries the revision is meant to change, stated with ``finn.dataflow``'s own values,
so the harness asks both packages about them.

- **CNV's five 3 × 3 layers**, the windowed dot product's activation port from a
  row-major image, at three PE: the window the revision moves onto a channel.
- **A transpose before a dense dot product**, at several lane arrangements: the
  activation port reading ``Y = Xᵀ`` stated as a traversal of ``X``; one arrangement is
  refused by the line (the lane regroup), and so are the two on X (32, 16) with N 16 at
  (producer lanes, PE, SIMD) = (4, 4, 4) and (2, 4, 2), which the revision builds.
- **The 4 × 4 regroup**: equal lane counts on different axes.
- **The tiled dot product's three cases** (TH vectors share each weight tile):
  its activation replay and output reorder as derived steps, its weight stream from a
  memory looping over one matrix, which the line over-counts or refuses, and the marker of
  its reduced index k, which the line refuses (k is not the innermost).

Each case calls the line's functions as FINN's kernels do; the recorder takes the calls.
"""

from __future__ import annotations

from finn.dataflow.gemm import Window, c, k, kh, kw, m, n, oh, ow
from finn.dataflow.schedule import Index, Refused, Schedule
from finn.dataflow.steps import Unrealizable, steps
from finn.dataflow.traversal import (
    BeatSequence,
    Repetition,
    Traversal,
    period,
    regrouped,
    unreplayed,
    vector_major,
)

CNV = (
    (32, 32, 3, 64, 3),
    (30, 30, 64, 64, 32),
    (14, 14, 64, 128, 32),
    (12, 12, 128, 128, 32),
    (5, 5, 128, 256, 32),
)
"""CNV's 3 × 3 layers: image height and width, channels, outputs, SIMD."""


def windows() -> None:
    for height, width, channels, outputs, simd in CNV:
        for pe in (1, 4, 16):
            window = Window(image=(height, width), size=(3, 3))
            schedule = Schedule(
                {**window.extents, n: outputs, c: channels},
                {n: pe, c: simd},
                (oh, ow, n, kh, kw, c),
            )
            port = schedule.present((height * width, channels), window.signature.x, lanes=(c,))
            marker = schedule.closing((kh, kw, c))
            steps(
                BeatSequence(vector_major((height * width, channels), simd)),
                BeatSequence(port, markers=(marker,)),
            )
            unreplayed(port)


def transposes() -> None:
    """Y = Xᵀ, X (I, J) row-major; a dense dot product reads Y (M = J, K = I)."""
    for rows, columns, outputs, lanes, pe, simd in (
        (4, 6, 4, 2, 2, 2),
        (4, 6, 4, 1, 2, 2),
        (6, 4, 6, 2, 3, 1),
        (8, 8, 4, 4, 2, 4),
        (4, 4, 2, 2, 1, 2),
        (32, 16, 16, 4, 4, 4),
        (32, 16, 16, 2, 4, 2),
    ):
        schedule = Schedule({m: columns, n: outputs, k: rows}, {n: pe, k: simd}, (m, n, k))
        marker = schedule.closing((k,))
        # The port over Y, stated over X: Y[m, k] = X[k, m].
        port = Traversal.over(
            (rows, columns),
            ((1, columns, 1), (None, outputs // pe, 0), (0, rows // simd, simd)),
            ((0, simd, 1),),
        )
        try:
            steps(
                BeatSequence(vector_major((rows, columns), lanes)),
                BeatSequence(port, markers=(marker,)),
            )
        except Unrealizable:
            pass


def regroup() -> None:
    """4 × 4: four lanes along rows against four lanes along columns."""
    along_rows = vector_major((4, 4), 4)
    along_columns = Traversal.over((4, 4), ((1, 4, 1),), ((0, 4, 1),))
    for source, sink in ((along_rows, along_columns), (along_columns, along_rows)):
        try:
            steps(BeatSequence(source), BeatSequence(sink))
        except Unrealizable:
            pass


def tiled() -> None:
    """The core's schedule (g, n, k, th), v = g * TH + th, against the module's ports."""
    g, th = Index("g"), Index("th")
    for vectors, depth, outputs, pe, simd, tile in (
        (6, 12, 6, 3, 4, 3),
        (8, 16, 8, 4, 4, 2),
        (6, 12, 4, 2, 6, 3),
    ):
        core = Schedule(
            {g: vectors // tile, n: outputs, k: depth, th: tile}, {n: pe, k: simd}, (g, n, k, th)
        )
        x = core.present((vectors, depth), (g * tile + th, k), lanes=(k,))
        y = core.present((vectors, outputs), (g * tile + th, n), lanes=(n,), reduces=(k,))
        w = core.present((depth, outputs), (k, n), lanes=(n, k))
        try:
            core.closing((k,))
        except Refused:
            pass
        steps(BeatSequence(vector_major((vectors, depth), simd)), BeatSequence(x))
        steps(BeatSequence(y), BeatSequence(vector_major((vectors, outputs), pe)))
        try:
            port = regrouped(unreplayed(w), pe * simd // tile)
            steps(BeatSequence(period(port), Repetition.CYCLIC), BeatSequence(w))
        except (ValueError, Unrealizable):
            pass


CASES = {"windows": windows, "transposes": transposes, "regroup": regroup, "tiled": tiled}


def run() -> None:
    for case in CASES.values():
        case()
