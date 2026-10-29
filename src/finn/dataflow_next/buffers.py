# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The buffer a reorder needs between two traversals of one tensor, measured by enumeration.

``slots(source, sink)`` lists, for each beat the sink presents, the source beat that
holds it (the first beat that presents the sink beat's first element). A reorder moves
whole beats, so the two traversals' lane loops must agree.

``live_window(slots)`` is the depth, in source beats, of the smallest circular buffer
that serves the sink from an in-order source that can be stalled after any beat, and
frees a beat once no later sink beat reads it. After sink beat ``t`` is served the
source has delivered up to the highest beat read so far, and the lowest beat still
wanted is the lowest one any later sink beat reads, so the buffer holds

    1 + max over t of (max(slots[:t + 1]) - min(slots[t + 1:]))

beats at its fullest (never less than one). The measure moves whole beats only and
counts nothing of time: it preserves order, not necessarily rate. A timed reorder
may need a larger capacity to sustain its frame period.

``frame_slots(reorder)`` lists the same for a ``Reorder`` value, within one frame: the
frame beat each emitted beat reads. A reorder holds beats within a frame only, so its
live window is ``live_window(frame_slots(reorder))`` whatever the frames carry; ``slots``,
which reads each element at the first beat presenting it, measures the same where the
source presents each element once, and points back to the first presentation where the
source repeats its pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow_next.traversal import Reorder, Traversal, offsets

Slots = npt.NDArray[np.int64]


class Unslotted(Refusal):
    """A reorder whose source beats cannot be measured against its sink's: one rejection,
    ``code`` by ``owner``, its message the error's."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),), message)


def slots(source: Traversal, sink: Traversal) -> Slots:
    """The source beat each sink beat reads, by the beat's first (lane zero's) element: a
    source beat's number, in source beats (not elements).

    ``Unslotted`` where the lane loops differ (``lanes-differ``: the reorder would
    move elements between lanes, not whole beats), or where the source never presents
    an element the sink reads (``element-never-presented``).
    """
    owner = "slots"
    if source.lane_loops != sink.lane_loops:
        raise Unslotted(
            "lanes-differ",
            owner,
            f"a reorder moves whole beats: the lane loops {source.lane_loops} "
            f"and {sink.lane_loops} must agree",
        )
    held = offsets(source).reshape(source.beats, source.lanes)[:, 0]
    wanted = offsets(sink).reshape(sink.beats, sink.lanes)[:, 0]
    # The first beat to present each element: np.unique's first occurrence, in element order.
    elements, first = np.unique(held, return_index=True)
    found = np.searchsorted(elements, wanted)
    found[found == len(elements)] = 0
    missing = elements[found] != wanted
    if missing.any():
        raise Unslotted(
            "element-never-presented",
            owner,
            f"the source never presents element {int(wanted[missing][0])}",
        )
    return np.asarray(first[found], dtype=np.int64)


def frame_slots(reorder: Reorder) -> Slots:
    """The frame beat each beat ``reorder`` emits in one frame reads: ``sum(index[i] *
    coefs[i])`` over its nested ``dims``, outer first."""
    found = np.zeros(1, dtype=np.int64)
    for dim, coef in zip(reorder.dims, reorder.coefs):
        found = (found[:, None] + np.arange(dim, dtype=np.int64) * coef).ravel()
    return found


def live_window(slot: Slots) -> int:
    """The live window of a reorder whose sink beats read source beats ``slot``, in source
    beats (a count of source beats, not elements): 0 for no beat, 1 for one beat (or an
    order that never holds one back)."""
    if len(slot) == 0:
        return 0
    if len(slot) == 1:
        return 1
    high = np.maximum.accumulate(slot)[:-1]
    low = np.minimum.accumulate(slot[::-1])[::-1][1:]
    return 1 + max(0, int((high - low).max()))


__all__ = ["Slots", "Unslotted", "frame_slots", "live_window", "slots"]
