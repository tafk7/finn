# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A set port as a projection of the kernel's schedule: the selector tensor a thresholding
kernel's set port reads, at the grain it varies with.

The schedule is (a, cb, cl) with ``c = cb * PE + cl`` and ``cl`` in lanes at PE. The data
port reads (rows, C) at ``(a, cb * PE + cl)``; a selector tensor reads only the indices it
varies with:

- **per beat**, ``S[a, cb]``: one index a beat, the sequence one index a beat gives today
  (``vector_major((beats,), 1)``), offset for offset; a selector stream presented once connects
  directly;
- **per row**, ``S[a]``: one index a row, replayed over the row's beats; from a selector
  presented once, one reorder (a frame of one beat, replayed over the row's ``C / PE``
  beats), or direct where a row is one beat;
- **per frame**, ``S[0]`` (``ZERO``): one index, replayed over every beat of the frame:
  one reorder of a frame of one beat;
- **per channel**, ``S[a, cb * PE + cl]`` without lanes: refused, ``lanes-move-position``,
  since one index a lane is not one index a beat.

Three (rows, C, PE) cases: (3, 8, 4) with 6 data beats, (4, 6, 2) with 12, (2, 4, 4) with 2.
Every reorder is also checked by enumeration (``undelivered``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dataflow.harness.checks import undelivered
from finn.dataflow.gemm import ZERO
from finn.dataflow.schedule import Index, Refused, Schedule
from finn.dataflow.steps import Step, steps
from finn.dataflow.traversal import (
    BeatSequence,
    Reorder,
    Traversal,
    offsets,
    vector_major,
)

a, cb, cl = Index("a"), Index("cb"), Index("cl")

a, cb, cl = Index("a"), Index("cb"), Index("cl")


@dataclass(frozen=True)
class Case:
    """One geometry and every number stated for it."""

    rows: int
    channels: int
    pe: int
    beats: int
    per_beat: tuple[int, int]  # the tensor S[a, cb]
    row_loops: tuple[tuple[int, int], ...]  # the beats of S[a], (extent, stride)
    row_reorder: Reorder | None  # from a selector stream presented once; None is direct
    frame_loops: tuple[tuple[int, int], ...]  # the beats of S[0]
    frame_reorder: Reorder

    def schedule(self) -> Schedule:
        return Schedule(
            {a: self.rows, cb: self.channels // self.pe, cl: self.pe}, {cl: self.pe}, (a, cb, cl)
        )


CASES = [
    Case(
        3,
        8,
        4,
        6,
        (3, 2),
        ((3, 1), (2, 0)),
        Reorder(1, (2,), (0,)),
        ((6, 0),),
        Reorder(1, (6,), (0,)),
    ),
    Case(
        4,
        6,
        2,
        12,
        (4, 3),
        ((4, 1), (3, 0)),
        Reorder(1, (3,), (0,)),
        ((12, 0),),
        Reorder(1, (12,), (0,)),
    ),
    Case(2, 4, 4, 2, (2, 1), ((2, 1),), None, ((2, 0),), Reorder(1, (2,), (0,))),
]
IDS = ["3-8-4", "4-6-2", "2-4-4"]


def _loops(form: Traversal) -> tuple[tuple[int, int], ...]:
    return tuple((loop.extent, loop.stride) for loop in form.beat_loops)


def _steps_from_once(form: Traversal, reorder: Reorder | None) -> None:
    """The steps from a selector stream presented once to ``form``, and by enumeration that they
    deliver it."""
    source = BeatSequence(vector_major(form.shape, 1))
    sink = BeatSequence(form)
    found = steps(source, sink)
    if reorder is None:
        assert not found and found.describe() == "direct"
    else:
        assert found.kinds == (Step.REORDER,)
        assert found.hops[0].reorder == reorder
    assert undelivered(found, source, sink) == ""


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_data_port_has_its_beats(case: Case) -> None:
    data = case.schedule().present((case.rows, case.channels), (a, cb * case.pe + cl), lanes=(cl,))
    assert data.beats == case.beats == case.rows * (case.channels // case.pe)
    assert data.lanes == case.pe


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_selector_a_beat_is_the_sequence_one_index_a_beat_gives_today(case: Case) -> None:
    """``S[a, cb]``: tensor (3, 2), (4, 3), (2, 1), one index a beat in order."""
    form = case.schedule().present(case.per_beat, (a, cb))
    today = vector_major((case.beats,), 1)
    assert form.shape == case.per_beat
    assert form.beats == case.beats == today.beats
    assert _loops(form) == ((case.beats, 1),)
    assert np.array_equal(offsets(form), offsets(today))
    assert offsets(form).tolist() == list(range(case.beats))
    _steps_from_once(form, None)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_selector_a_row_is_replayed_over_the_row(case: Case) -> None:
    """``S[a]``, tensor (rows,): loops (3, 1)(2, 0), (4, 1)(3, 0), (2, 1). Where a row is
    one beat (C = PE) it is today's sequence, and connects directly."""
    form = case.schedule().present((case.rows,), (a,))
    assert form.shape == (case.rows,)
    assert form.beats == case.beats
    assert _loops(form) == case.row_loops
    # One index a row, held for the row's C / PE beats.
    held = case.channels // case.pe
    assert offsets(form).tolist() == [row for row in range(case.rows) for _ in range(held)]
    same_as_today = np.array_equal(offsets(form), offsets(vector_major((case.beats,), 1)))
    assert same_as_today == (case.row_reorder is None)
    _steps_from_once(form, case.row_reorder)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_selector_a_frame_is_replayed_over_the_frame(case: Case) -> None:
    """``S[0]``, tensor (1,): loops (6, 0), (12, 0), (2, 0), one reorder of one beat."""
    form = case.schedule().present((1,), (ZERO,))
    assert form.shape == (1,)
    assert form.beats == case.beats
    assert _loops(form) == case.frame_loops
    assert offsets(form).tolist() == [0] * case.beats
    _steps_from_once(form, case.frame_reorder)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_a_selector_a_channel_is_refused(case: Case) -> None:
    """One index a lane is not one index a beat: ``cl`` moves the position, and is not in
    lanes."""
    with pytest.raises(Refused, match="lanes move the position") as error:
        case.schedule().present((case.rows, case.channels), (a, cb * case.pe + cl))
    assert [found.code for found in error.value.findings] == ["lanes-move-position"]
