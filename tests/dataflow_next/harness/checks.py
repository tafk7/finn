# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The checks a difference passes through: a checker accepts an answer where the line
refused, a comparator prices two differing answers. ``replay`` registers them by target.

For ``steps.steps``:

- ``steps_deliver`` (a checker): the revision's chain delivers, hop by hop, the positions
  the consumer reads, in its order, checked by enumeration (``undelivered``);
- ``steps_built_cost`` (a pricer): the cost of the chain a checker accepted, which the
  line has no counterpart for, with the chain's steps;
- ``steps_cost`` (a comparator): both chains deliver, and their costs (``steps.cost``),
  weighed by ``steps.smaller``.

For ``steps.presented``:

- ``presented_delivers`` (a checker): where the line refused (the consumer's elements no
  whole number of the source's passes), the revision's answer is whole repetitions of the
  source's pass, and the chain the revision derives from it delivers (``undelivered``).
  The revision counts a pass for each of the consumer's passes, not elements, so it
  builds where the line refuses: a window that skips and reads twice;
- ``presented_cost`` (a comparator): both answers are whole repetitions of the source's
  pass, and the chains from each to the consumer, derived by the revision's ``steps``
  from a source presenting it, are priced as above.

For ``schedule.Schedule.closing``, ``closing_asserted`` (a checker): where the line
refuses a reduction that is not innermost, the revision's marker is asserted, by
enumeration of the schedule's beats, exactly on the beats where every reduced index is
at its last index value.

Each check is given the line's values. It translates them (``translate.forward``) and
works on the revision's: the costs are the revision's ``Cost`` values, which the line
does not have, so they never cross back. An answer the line has no class for (a
``LevelDone``) reaches a checker as the revision's own value, which ``forward`` keeps.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from dataclasses import dataclass
from dataflow_next.harness import translate
from dataflow_next.harness.record import Call, Raised
from typing import Any, Literal

from finn.dataflow_next.buffers import Unslotted, frame_slots, slots
from finn.dataflow_next.schedule import Index, Refused, Schedule
from finn.dataflow_next.steps import (
    Cost,
    Step,
    Steps,
    Unrealizable,
    cost,
    smaller,
    steps,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Repetition,
    Traversal,
    offsets,
)


@dataclass(frozen=True)
class Comparison:
    """A comparator's reading of two differing answers: ``claim`` is ``smaller`` or
    ``larger_latency_counted``, or ``differs`` for a priced difference that is neither
    (``detail`` names it: a trade-off, a narrower chain); ``positions_equal`` says the two
    answers place the same positions; ``line_cost`` and ``next_cost`` are the costs
    compared; ``latency`` names what a ``larger_latency_counted`` claim costs."""

    claim: Literal["smaller", "larger_latency_counted", "differs"]
    positions_equal: bool
    line_cost: Any
    next_cost: Any
    latency: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Priced:
    """The revision's cost of an answer the line refused, and a word on it."""

    next_cost: Any
    detail: str = ""


Checker = Callable[[Call, Raised, Any], bool]
"""Accepts a call the line refused and the revision answered: given the call, the line's
exception and the revision's answer (as the line's values), True when the answer is
right (e.g. it is among an enumeration of positions)."""

Pricer = Callable[[Call, Any], Priced | None]
"""Prices an answer a checker accepted: given the call and the revision's answer (as the
line's values), a ``Priced`` with its cost, or None when it is not one this pricer reads."""

Comparator = Callable[[Call, Any, Any], Comparison | None]
"""Reads two differing value answers (the line's, the revision's as the line's values):
a ``Comparison`` when the difference is a cost, None when it is not one this comparator
explains."""


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
    from what ``source`` presents, or ``""`` when it does; the revision's values.

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


def _ends(call: Call) -> tuple[BeatSequence, BeatSequence] | None:
    """The call's source and sink as the revision's values, or None."""
    args, kwargs = list(call.args), dict(call.kwargs)
    ends = [args.pop(0) if args else kwargs.pop(name, None) for name in ("source", "sink")]
    found = translate.forward(tuple(ends))
    if all(isinstance(end, BeatSequence) for end in found):
        return found[0], found[1]
    return None


def steps_deliver(call: Call, refusal: Raised, answer: Any) -> bool:
    """A chain where the line refused is right when it delivers (``undelivered``)."""
    ends, chain = _ends(call), translate.forward(answer)
    if ends is None or not isinstance(chain, Steps):
        return False
    return not undelivered(chain, *ends)


def steps_built_cost(call: Call, answer: Any) -> Priced | None:
    """The cost of a chain built where the line refused, and its steps."""
    ends, chain = _ends(call), translate.forward(answer)
    if ends is None or not isinstance(chain, Steps):
        return None
    try:
        return Priced(cost(chain, ends[1]), chain.describe())
    except ValueError:
        return None


def weighed(line: Cost, revision: Cost) -> str:
    """How ``revision``'s cost stands to ``line``'s: ``smaller``, or what else it is."""
    if smaller(revision, line):
        return "smaller"
    if revision.narrowest_lanes < line.narrowest_lanes:
        return (
            f"narrower: {revision.narrowest_lanes} lanes against {line.narrowest_lanes}, "
            "so not smaller"
        )
    fewer = [
        name for name in ("live_window", "reads") if getattr(revision, name) < getattr(line, name)
    ]
    more = [
        name for name in ("live_window", "reads") if getattr(revision, name) > getattr(line, name)
    ]
    if fewer and more:
        return f"a trade-off: less {fewer[0]} for more {more[0]}, so not smaller"
    if more:
        return "larger"
    return "equal costs"


def _compared(line: Cost, revision: Cost, positions_equal: bool, reason: str) -> Comparison:
    relation = weighed(line, revision)
    if relation == "smaller":
        return Comparison("smaller", positions_equal, line, revision, detail=reason)
    detail = "; ".join(part for part in (relation, reason) if part)
    return Comparison("differs", positions_equal, line, revision, detail=detail)


def steps_cost(call: Call, line: Any, revision: Any) -> Comparison | None:
    """Two chains to one sink: their costs, and whether both deliver its positions."""
    ends = _ends(call)
    chains = translate.forward((line, revision))
    if ends is None or not all(isinstance(chain, Steps) for chain in chains):
        return None
    source, sink = ends
    try:
        costs = [cost(chain, sink) for chain in chains]
    except ValueError:
        return None
    reasons = [undelivered(chain, source, sink) for chain in chains]
    reason = "; ".join(
        f"{side}: {found}" for side, found in zip(("line", "next"), reasons) if found
    )
    return _compared(costs[0], costs[1], not reason, reason)


def presented_cost(call: Call, line: Any, revision: Any) -> Comparison | None:
    """Two presentations of a source over one frame of the consumer: whole repetitions of its
    pass each, priced by the chain the revision derives from each to the consumer."""
    ends = _ends(call)
    forms = translate.forward((line, revision))
    if ends is None or not all(isinstance(form, Traversal) for form in forms):
        return None
    source, sink = ends
    reasons = [_repetitions(form, source) for form in forms]
    try:
        chains = [
            steps(BeatSequence(form, sink.repetition, source.markers), sink) for form in forms
        ]
        costs = [cost(chain, sink) for chain in chains]
    except (Unrealizable, ValueError):
        return None
    reasons = [
        found or undelivered(chain, BeatSequence(form, sink.repetition), sink)
        for found, chain, form in zip(reasons, chains, forms)
    ]
    reason = "; ".join(
        f"{side}: {found}" for side, found in zip(("line", "next"), reasons) if found
    )
    return _compared(costs[0], costs[1], not reason, reason)


def presented_delivers(call: Call, refusal: Raised, answer: Any) -> bool:
    """A presentation where the line refused is right when it is whole repetitions of the
    source's pass and the chain the revision derives from it delivers."""
    ends, form = _ends(call), translate.forward(answer)
    if ends is None or not isinstance(form, (Traversal, Grouped)):
        return False
    source, sink = ends
    if _repetitions(form, source):
        return False
    shown = BeatSequence(form, sink.repetition, source.markers)
    try:
        chain = steps(shown, sink)
    except (Unrealizable, ValueError):
        return False
    return not undelivered(chain, BeatSequence(form, sink.repetition), sink)


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


def closing_asserted(call: Call, refusal: Raised, answer: Any) -> bool:
    """A marker where the line refused a reduction is right when it is asserted on exactly
    the beats where every reduced index is at its last index value (``last_iterations``)."""
    # A method's call is recorded with its schedule first.
    args, kwargs = translate.forward(call.args), translate.forward(call.kwargs)
    schedule = args[0] if args else None
    reduces = args[1] if len(args) > 1 else kwargs.get("reduces")
    marker = translate.forward(answer)
    if not isinstance(schedule, Schedule) or not isinstance(marker, (LevelEnd, LevelDone)):
        return False
    try:
        wanted = last_iterations(schedule, tuple(reduces))
    except (Refused, TypeError):
        return False
    asserted = np.array([marker.asserted(beat) for beat in range(len(wanted))], dtype=bool)
    return bool(np.array_equal(asserted, wanted))


CHECKERS: dict[str, Checker] = {
    "steps.steps": steps_deliver,
    "steps.presented": presented_delivers,
    "schedule.Schedule.closing": closing_asserted,
}
"""Checkers by target label."""

PRICERS: dict[str, Pricer] = {"steps.steps": steps_built_cost}
"""Pricers by target label."""

COMPARATORS: dict[str, Comparator] = {
    "steps.steps": steps_cost,
    "steps.presented": presented_cost,
}
"""Comparators by target label."""
