# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The live-window measure: ``slots`` and ``live_window`` on hand cases, and ``live_window``
against a plain loop that holds the beats one at a time."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from finn.dataflow.buffers import Unslotted, live_window, slots
from finn.dataflow.traversal import Loop, Traversal, vector_major


def window_by_holding(wanted: Sequence[int]) -> int:
    """The live window as a buffer's fullest moment, found beat by beat: after each sink
    beat is served, the source has delivered up to the highest beat read so far and the
    buffer keeps every beat from the lowest one a later sink beat reads."""
    fullest = 1 if wanted else 0
    for served in range(len(wanted) - 1):
        delivered = max(wanted[: served + 1])
        oldest = min(wanted[served + 1 :])
        fullest = max(fullest, delivered - oldest + 1)
    return fullest


def refused(code: str, source: Traversal, sink: Traversal) -> None:
    with pytest.raises(Unslotted) as info:
        slots(source, sink)
    assert isinstance(info.value, ValueError)
    assert [finding.code for finding in info.value.findings] == [code]


def array(values: Sequence[int]) -> npt.NDArray[np.int64]:
    return np.array(values, dtype=np.int64)


# --- live_window -----------------------------------------------------------------------


def test_nothing_and_one_beat_need_no_more_than_their_own_count() -> None:
    assert live_window(array([])) == 0
    assert live_window(array([5])) == 1


@pytest.mark.parametrize("beats", [2, 5, 64])
def test_an_in_order_sink_holds_one_beat(beats: int) -> None:
    assert live_window(np.arange(beats, dtype=np.int64)) == 1


@pytest.mark.parametrize("beats", [2, 5, 64])
def test_a_full_reversal_holds_every_beat(beats: int) -> None:
    assert live_window(np.arange(beats, dtype=np.int64)[::-1].copy()) == beats


def test_a_replay_holds_the_beats_it_replays() -> None:
    assert live_window(array([0, 0, 0, 1, 1, 1])) == 1  # each beat repeated in place
    assert live_window(array([0, 1, 2, 0, 1, 2])) == 3  # a block of three, twice
    assert live_window(array([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])) == 3


def test_a_transpose_holds_the_beats_between_its_first_and_last_column_beat() -> None:
    # A (3, 4) matrix of beats, written row by row and read column by column: the sink reads
    # beats 0, 4, 8, 1, 5, 9, ... After beat 8 is read, beats 1 to 8 (eight beats) are held.
    source = vector_major((3, 4), 1)
    sink = Traversal.over((3, 4), ((1, 4, 1), (0, 3, 1)), ())
    found = slots(source, sink)
    assert found.tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    assert live_window(found) == 8


def test_the_measure_is_a_buffers_fullest_moment_on_random_orders() -> None:
    rng = random.Random(7)
    for _ in range(200):
        beats = rng.randint(2, 24)
        order = [rng.randrange(beats) for _ in range(beats)]  # repeats allowed
        assert live_window(array(order)) == window_by_holding(order)
    for _ in range(50):
        order = rng.sample(range(16), 16)
        assert live_window(array(order)) == window_by_holding(order)


# --- slots -----------------------------------------------------------------------------


def test_slots_of_identical_traversals_are_the_beats_in_order() -> None:
    form = vector_major((4, 6), 3)
    assert slots(form, form).tolist() == list(range(8))


def test_slots_name_a_beat_by_its_first_element() -> None:
    # The source presents a (4, 6) tensor at 3 lanes, row by row; the sink reads every row's
    # first half, then every row's second half.
    source = vector_major((4, 6), 3)
    sink = Traversal((4, 6), (Loop(2, 3), Loop(4, 6)), (Loop(3, 1),))
    assert slots(source, sink).tolist() == [0, 2, 4, 6, 1, 3, 5, 7]


def test_a_source_that_presents_a_beat_twice_is_read_at_its_first() -> None:
    source = vector_major((4,), 1).repeated(2)
    assert slots(source, vector_major((4,), 1)).tolist() == [0, 1, 2, 3]


def test_a_replaying_sink_reads_the_same_source_beat_again() -> None:
    source = vector_major((3,), 1)
    sink = source.replayed(2, inner_beats=3)
    found = slots(source, sink)
    assert found.tolist() == [0, 1, 2, 0, 1, 2]
    assert live_window(found) == 3


def test_a_reorder_moves_whole_beats_only() -> None:
    refused("lanes-differ", vector_major((4, 6), 3), vector_major((4, 6), 2))
    refused("lanes-differ", vector_major((4, 6), 3), vector_major((4, 6), 1))


def test_an_element_the_source_never_presents_is_refused() -> None:
    # The sink reads all six, the source only its first row of three.
    source = Traversal((2, 3), (Loop(3, 1),), ())
    refused("element-never-presented", source, vector_major((2, 3), 1))
    # An element past the source's highest one is refused as well.
    low = Traversal((8,), (Loop(2, 1),), ())
    high = Traversal((8,), (Loop(2, 6),), ())
    refused("element-never-presented", low, high)
