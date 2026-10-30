# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent oracles, by enumeration, for what the package derives:

- ``undelivered``: why a chain of steps does not deliver, hop by hop, the positions a
  consumer reads, in its order, from what its source presents (``""`` when it does);
- ``last_iterations``: for each beat of a schedule, whether every reduced index is at its
  last index value, which ``Schedule.closing``'s marker asserts.

The checkers, pricers and comparators that registered these against the line's refusals
and differing answers (``CHECKERS``, ``PRICERS``, ``COMPARATORS`` and their functions) are
retired with the line; they remain at the immutable baseline 1ac1616.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from finn.dataflow.buffers import Unslotted, frame_slots, slots
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.steps import Step, Steps
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    Repetition,
    Traversal,
    offsets,
)


def _rows(form: Traversal | Grouped) -> npt.NDArray[np.int64]:
    """``form``'s flat offsets, one row a beat, lane zero first."""
    return offsets(form).reshape(form.beats, form.lanes)


def _one_wiring(first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> bool:
    """The same positions in every beat through one fixed lane permutation: wires, not a
    step. Each lane of ``second`` is a lane of ``first`` over all the beats, and each lane
    of ``first`` is used once, so the lanes of the two, as whole columns, are the same
    multiset. Beats that are each a permutation, but not of one wiring, are not."""
    if first.shape != second.shape:
        return False
    if not first.shape[0]:
        return True
    return bool(np.array_equal(first[:, np.lexsort(first)], second[:, np.lexsort(second)]))


def _repetitions(form: Traversal | Grouped, source: BeatSequence) -> str:
    """Why ``form`` is not what ``source`` presents (once, or whole repetitions of its
    pass where it is cyclic), or ``""``."""
    each, found = _rows(source.form), _rows(form)
    count, left = divmod(len(found), len(each))
    if left or not count or (count > 1 and source.repetition is not Repetition.CYCLIC):
        return f"{form.beats} beats are no whole presentation of the source's {len(each)}"
    if not _one_wiring(found, np.tile(each, (count, 1))):
        return "the chain starts from beats the source does not present"
    return ""


def _hop_delivers(step: Step, source: BeatSequence, sink: BeatSequence, reorder: Any) -> str:
    """Why one step does not turn ``source`` into ``sink``, by enumeration, or ``""``."""
    before, after = source.form, sink.form
    if step is Step.MARKERS:
        if before != after or not set(source.markers) <= set(sink.markers):
            return "a marker step changes the sequence or drops a marker"
        return ""
    if step is Step.WIDTH:
        if before.shape != after.shape or not np.array_equal(offsets(before), offsets(after)):
            return "a width conversion changes the element order"
        return ""
    held, wanted = _rows(before), _rows(after)
    if not isinstance(before, Traversal) or not isinstance(after, Traversal):
        return "a reorder from or to a lane group"
    if reorder is None or before.lane_loops != after.lane_loops:
        return "a reorder without its reorder, or across lanes"
    frames, left = divmod(before.beats, reorder.frame_beats)
    local = frame_slots(reorder)
    if left or frames * len(local) != after.beats:
        return f"{reorder} does not frame {before.beats} beats into {after.beats}"
    read = (np.arange(frames, dtype=np.int64)[:, None] * reorder.frame_beats + local).ravel()
    if not np.array_equal(held[read], wanted):
        return f"{reorder} does not emit the sink's beats"
    try:
        first = slots(before, after)
    except Unslotted as error:
        return str(error)
    if not np.array_equal(held[first], wanted):
        return "a sink beat is no whole source beat"
    return ""


def undelivered(chain: Steps, source: BeatSequence, sink: BeatSequence) -> str:
    """Why ``chain`` does not deliver, hop by hop, the positions ``sink`` reads in its order
    from what ``source`` presents, or ``""`` when it does.

    By enumeration: the first hop starts from whole presentations of the source's pass, up
    to one lane permutation the same in every beat (wires: ``_one_wiring``);
    each hop starts where the one before ends; a width conversion keeps the element
    order (``offsets``); a reorder emits, frame by frame, the beats its ``Reorder``
    names, and every sink beat is a whole source beat (``buffers.slots``); a marker step
    keeps the sequence; the last hop ends at the sink's beats, up to one such
    permutation, with every marker the sink requires.
    """
    hops = chain.hops
    start = hops[0].source.form if hops else sink.form
    if start.shape != source.form.shape or start.shape != sink.form.shape:
        return "the chain's ends are another tensor"
    reason = _repetitions(start, source)
    if reason:
        return reason
    current = hops[0].source if hops else BeatSequence(start, markers=source.markers)
    for index, hop in enumerate(hops):
        if hop.source != current:
            return f"hop {index} does not start where the chain stands"
        reason = _hop_delivers(hop.step, hop.source, hop.sink, hop.reorder)
        if reason:
            return f"hop {index}: {reason}"
        current = hop.sink
    if not _one_wiring(_rows(current.form), _rows(sink.form)):
        return "the chain ends at other beats than the sink reads"
    required = {rule for rule in sink.markers if not rule.constant}
    if not required <= set(current.markers):
        return "the chain ends without a marker the sink requires"
    return ""


def last_iterations(schedule: Schedule, reduces: tuple[Index, ...]) -> npt.NDArray[np.bool_]:
    """For each beat of ``schedule``, in its order, whether every index of ``reduces`` is
    at its last index value: by enumeration of the beats' indices."""
    beats = np.arange(schedule.iterations, dtype=np.int64)
    found = np.ones(len(beats), dtype=bool)
    inner = 1
    for index in reversed(schedule.order):
        count = schedule.fold(index)
        if index in reduces:
            found &= (beats // inner) % count == count - 1
        inner *= count
    return found
