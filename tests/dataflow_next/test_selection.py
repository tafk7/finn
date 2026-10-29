# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A selected pass (``Selected``): one of several sets a pass, chosen by a selector stream.

- Its positions over the stacked tensor ``(sets, *shape)`` are given a selection: pass
  ``j`` presents set ``selection[j]``, offset by the set's size times the index.
- A constant selection presents exactly what the pass repeated presents over that set:
  repetition is selection's constant case.
- ``steps`` joins two selected sides of the same sets through the same selector stream
  (the same name) by the steps inside one set, and refuses every other pairing with a
  selected side, ``selection-not-shared``: against a consumer that is not selected, the
  same sets under another selector stream, or other sets.

The cases: four sets of (6, 8), each pass ``vector_major((6, 8), 2)`` (24 beats of two
lanes), five passes.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
from dataflow_next.harness.checks import undelivered

from finn.dataflow_next.buffers import frame_slots
from finn.dataflow_next.steps import Step, Unrealizable, Unshared, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    Reorder,
    Selected,
    Traversal,
    offsets,
    vector_major,
)

SETS, SHAPE, PASSES = 4, (6, 8), 5
PASS = vector_major(SHAPE, 2)
SIZE = 48
SELECTION = (2, 0, 3, 3, 1)
# The same (6, 8) by column pairs: each pair of columns down the six rows, two lanes.
BY_COLUMNS = Traversal.over(SHAPE, ((1, 4, 2), (0, 6, 1)), ((1, 2, 1),))


def selected(form: Traversal, selector: str = "sets", sets: int = SETS) -> Selected:
    return Selected(BeatSequence(form), sets, selector)


def test_a_selection_offsets_each_pass_by_its_set() -> None:
    side = selected(PASS)
    assert side.shape == (4, 6, 8) and side.set_size == SIZE
    found = side.offsets(SELECTION)
    assert len(found) == PASSES * SIZE
    assert [int(found[j * SIZE]) for j in range(PASSES)] == [96, 0, 144, 144, 48]
    assert np.array_equal(found, np.concatenate([s * SIZE + offsets(PASS) for s in SELECTION]))
    beats = list(side.positions(SELECTION))
    assert len(beats) == PASSES * PASS.beats
    # Pass 2 reads set 3: its first beat is (3, 0, 0) and (3, 0, 1).
    assert beats[2 * PASS.beats] == ((3, 0, 0), (3, 0, 1))


@pytest.mark.parametrize("index", range(SETS))
def test_a_constant_selection_is_the_pass_repeated(index: int) -> None:
    side, repeated = selected(PASS), PASS.repeated(PASSES)
    assert np.array_equal(side.offsets((index,) * PASSES), index * SIZE + offsets(repeated))
    assert [
        tuple(position[1:] for position in beat) for beat in side.positions((index,) * PASSES)
    ] == list(repeated.positions())
    assert {position[0] for beat in side.positions((index,) * PASSES) for position in beat} == {
        index
    }


def test_a_selected_pass_is_no_traversal_and_checks_its_fields() -> None:
    side = selected(PASS)
    others: list[object] = [PASS, BeatSequence(PASS)]
    assert all(side != other for other in others)
    assert side == selected(PASS) and side != selected(PASS, "other")
    with pytest.raises(ValueError, match="one of 4 sets, not 4"):
        side.offsets((0, 4))
    with pytest.raises(ValueError, match="positive"):
        selected(PASS, sets=0)
    with pytest.raises(ValueError, match="nonempty name"):
        selected(PASS, selector="")
    with pytest.raises(TypeError, match="BeatSequence"):
        Selected(PASS, 4, "sets")  # type: ignore[arg-type]


def _delivered(chain_reorder: Reorder, source: Selected, sink: Selected) -> bool:
    """By enumeration over the stacked tensor: the reorder, frame by frame within each
    selected pass, turns what ``source`` presents over ``SELECTION`` into what ``sink``
    reads."""
    lanes = source.sequence.form.lanes
    held: npt.NDArray[np.int64] = source.offsets(SELECTION).reshape(-1, lanes)
    frames = len(held) // chain_reorder.frame_beats
    read = (
        np.arange(frames, dtype=np.int64)[:, None] * chain_reorder.frame_beats
        + frame_slots(chain_reorder)
    ).ravel()
    return bool(np.array_equal(held[read], sink.offsets(SELECTION).reshape(-1, lanes)))


def test_the_same_pass_under_the_same_selector_connects_directly() -> None:
    assert not steps(selected(PASS), selected(PASS))


def test_another_order_inside_a_set_is_the_reorder_between_the_two_passes() -> None:
    """Row by row against column pair by column pair, inside each selected set: one
    reorder whose frame is one set's pass, so it never reads across two selected sets."""
    source, sink = selected(PASS), selected(BY_COLUMNS)
    chain = steps(source, sink)
    inner = steps(BeatSequence(PASS), BeatSequence(BY_COLUMNS))
    assert chain == inner
    assert chain.kinds == (Step.REORDER,)
    reorder = chain.hops[0].reorder
    assert reorder == Reorder(24, (4, 6), (1, 4))
    assert reorder.frame_beats == PASS.beats
    assert undelivered(chain, BeatSequence(PASS), BeatSequence(BY_COLUMNS)) == ""
    assert _delivered(reorder, source, sink)


def _refused(source: BeatSequence | Selected, sink: BeatSequence | Selected, match: str) -> None:
    with pytest.raises(Unshared, match=match) as error:
        steps(source, sink)
    assert isinstance(error.value, Unrealizable)
    assert [found.code for found in error.value.findings] == ["selection-not-shared"]


def test_a_consumer_that_is_not_selected_is_refused() -> None:
    """Even one reading the repeated set: which set a pass holds is the selector stream's, at run
    time, so a constant selection is not known to be one."""
    _refused(selected(PASS), BeatSequence(PASS), "selected source")
    _refused(selected(PASS), BeatSequence(PASS.repeated(PASSES)), "selected source")
    _refused(BeatSequence(PASS), selected(PASS), "selected consumer")


def test_the_same_sets_under_another_selector_are_refused() -> None:
    _refused(selected(PASS, "rows"), selected(PASS, "frames"), "'rows' .* 'frames' share no")
    _refused(selected(PASS, "rows"), selected(BY_COLUMNS, "frames"), "share no selection")


def test_other_sets_under_the_same_selector_are_refused() -> None:
    _refused(selected(PASS), selected(PASS, sets=3), r"\(4, 6, 8\) .* \(3, 6, 8\)")
    other = selected(vector_major((6, 4), 2))
    _refused(selected(PASS), other, "share no selection")
