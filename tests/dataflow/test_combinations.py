# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The extensions of the package together in one channel: a lane group at an end (G), a
``LevelDone`` marker on the sink (L), a selected pair of sides (S) and the network's time
(N), through a position map or not (M). Each part has its own tests; here they meet, and a
case is built whole and by its parts.

**Four groups of cases**, A, M, D and T.

- **A**: one channel between two forms of one tensor, no map. Tensors of rank 1 to 3 with
  extents in {2, 3, 4} and at most 24 elements (28 shapes). Each end walks the tensor by an
  axis permutation, one element a beat; the sink also with its innermost axis's run replayed
  twice (the activation pattern of the tiled core). Lanes at each end in {1, 2, 3, 4}, the
  form ``grouped(order, lanes)`` (a count that divides no pass of the order is skipped), so
  counts that close whole loops and counts that cut a loop both occur. Markers on the sink:
  none, and one pair (beats, inner) of each class among those with ``inner | level beats | form
  beats``
  (an inner level of one beat, inner levels between the edges, the whole level, and one the sink's
  form
  refuses), drawn by a generator seeded with 7. Selection: none, or ``Selected(seq, 2,
  "sel")`` at both ends. Repetition: ``ONCE``, or ``CYCLIC`` against the sink's form
  repeated twice. The whole space has 287,936 cases, in 16 cells of (G, a ``LevelDone``
  between the edges, S, repetition), the smallest 8,706 cases and the largest 29,761.
- **M**: through a position map. Seven maps (two transposes of rank 2 and 3, a transpose of
  (6, 4, 2), two windows, two pads stated at the producer's side), a producer
  ``vector_major(tensor, lanes)`` at 1 to 4 lanes where they divide the innermost extent, a
  reader in each axis permutation at lanes in {1, 2, 3, 4, 6} (``grouped``), and every
  (beats, inner) pair of its beats (markers aligned or not) or none: 16,680 cases. S
  cannot join M, below.
- **D**: a set port selected a frame, a row and a beat (the selector stream varying with 0,
  1 or every beat index of the kernel's schedule), its sets stored in a memory presenting
  them cyclically. Three geometries (rows, C, PE, T) = (3, 6, 2, 2), (2, 4, 2, 3),
  (4, 6, 3, 2): 9 cases, all of them run.
- **T**: all four in one channel, the tiled MatMul's weights in its three geometries
  (6, 12, 6, 3, 4, 3), (8, 16, 8, 4, 4, 2) and (6, 12, 4, 2, 6, 3) of (N, MW, MH, PE, SIMD,
  TH), selected a frame among two matrices, with the core's marker for k on the weight
  port, in the network of the activations' source, the memory, the chain's stages and the
  core. 3 cases, all of them run.

**Four judges**, every case built through one pipeline and every part of it alone.

(a) *It delivers*: ``harness.checks.undelivered`` finds nothing; for S also over the stacked
tensor, over two selected passes choosing the sets (1, 0), by ``Selected.offsets``; for M the
composed form reads through the map what the reader reads.
(b) *Its cost is its parts' composed*: ``cost`` equals the cost composed by hand from an
enumeration of each step (a reorder's live window, a width step's least common multiple of
its lanes, a marker's none; the reads ``presented`` gives; the fewest lanes of any step's
ends).
(c) *Its refusals are its parts'*: the case is refused exactly when one part alone is
refused (G's ``grouped``, L's ``BeatSequence`` of the sink's form with its markers, the steps
without markers or selection, M's ``composed`` without the marker, then the marker on the
composed form), and by that part's refusal: class, finding code and message. A refusal no
part gives, or a part's refusal the case does not give, would be a finding.
(d) *Network time*: the stages of the chain, each reorder built unbounded (its capacity its
words a frame times ``FRAMES``), between a source and a sink; judged by ``bottleneck(network,
rules) == period`` under ``FINNLIB`` and ``ONE_CYCLE``, where the period is the most words a
frame on any channel (and, in D and T, the kernel's iterations), and by the stages'
delivering what the steps deliver. ``Run.on_time`` is not the judge: it reads only the sources,
and a slow stage behind an unbounded channel never pushes back to them. The least capacity of
a reorder keeping the period is measured apart, by a scan, and asserted on named cases only.

**What runs by default.** Group A: a seeded sample (seed 7) of 4,000 cases, 250 from each of
the 16 cells, judged (a) to (c); judge (d) on 200 of its built chains. Group M: a seeded
sample (seed 7) of 2,000 cases, and judge (d) on 100 of its built ones. Groups D and T whole.
**Marked slow**: group A whole (287,936 cases), judge (d) on 2,000 of its built chains, and
group M whole (16,680 cases).

**Representation boundaries.** S with M is refused: a reader is a ``PositionMap``
and a ``BeatSequence``, not a ``Selected``. A selected side reads sets from a memory;
movement before that memory is folded into its stored order. A selector stream that changes
position across lanes is refused by ``Schedule.present`` (``lanes-move-position``).
A selector stream varying with a beat index can be presented even when that index also moves
inside a whole set. No constructor associates that selector stream's count with the
``Selected`` passes. D's per-beat examples explicitly recut the sets to (PE, T); they do
not prove that the unassociated whole-set presentation is refused. L with N is no
conflict: a marker moves no word and is no stage.

**Selected passes.** The over-passes and timing checks include once sources whose
outer repetition equals an inner replay loop by value. Every selected row whose one-pass
chain builds is checked over the selected passes; no such rows are excluded.
"""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cache
from math import lcm, prod
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from dataflow.harness.checks import undelivered
from finn.core.reporting.findings import Refusal
from finn.dataflow import maps
from finn.dataflow.buffers import frame_slots, live_window
from finn.dataflow.gemm import ZERO, k, n
from finn.dataflow.reindex import (
    Reader,
    Reindex,
    Route,
    Unrouted,
    composed,
    route,
    route_cost,
    stage_lanes,
)
from finn.dataflow.schedule import Affine, Index, Pace, Refused, Schedule
from finn.dataflow.steps import (
    Cost,
    Step,
    Steps,
    Unrealizable,
    Unshared,
    cost,
    presented,
    steps,
)
from finn.dataflow.timing import (
    FINNLIB,
    FRAMES,
    ONE_CYCLE,
    Channel,
    Memory,
    Network,
    ReorderStage,
    Rules,
    Sink,
    Source,
    Stage,
    WidthStage,
    bottleneck,
    chained,
    kernel,
    route_stages,
    step_stages,
)
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Repetition,
    Selected,
    Traversal,
    grouped,
    level_done,
    offsets,
    stored,
    unreplayed,
    vector_major,
)

Array = npt.NDArray[np.int64]
Form = Traversal | Grouped
Pair = tuple[int, int]  # a marker's (beats, inner)
Outcome = tuple[str, str, str]  # a refusal's class, finding code ("" if none) and message
RULES = (("FINNLIB", FINNLIB), ("ONE_CYCLE", ONE_CYCLE))

SEED = 7
SAMPLE_PER_CELL = 250  # 16 cells: 4,000 cases
TIMED_SAMPLE = 200
TIMED_SLOW = 2_000
MAP_SAMPLE = 2_000
MAP_TIMED = 100
SETS, SELECTION = 2, (1, 0)  # the selected pair's sets, and the two passes' choice among them
PASSES = len(SELECTION)

BUILT = "built, delivers, cost composed"
PART = "refused as a part alone"


# --- what every group states ------------------------------------------------------------


def outcome(error: Exception) -> Outcome:
    """A refusal as it is compared: its class, its finding's code, its message."""
    code = error.findings[0].code if isinstance(error, Refusal) else ""
    return type(error).__name__, code, str(error)


def composed_by_hand(chain: Steps, source: BeatSequence, sink: BeatSequence) -> Cost:
    """The cost of a chain from an enumeration of each step: a reorder's live window of the
    reads its frames make over the step's source beats, in source beats times its lanes; a
    width step's least common multiple of its two lane counts; a marker step's none; summed.
    The reads are what ``source`` presents a frame of ``sink``; the lanes, the fewest of
    any step's two ends."""
    if not chain:
        return Cost(0, sink.form.beats, sink.form.lanes)
    window, lanes = 0, []
    for hop in chain.hops:
        lanes += [hop.source.form.lanes, hop.sink.form.lanes]
        if hop.step is Step.REORDER:
            assert hop.reorder is not None
            frame = hop.reorder.frame_beats
            frames = hop.source.form.beats // frame
            reads = (np.arange(frames)[:, None] * frame + frame_slots(hop.reorder)).ravel()
            window += live_window(reads) * hop.source.form.lanes
        elif hop.step is Step.WIDTH:
            window += lcm(hop.source.form.lanes, hop.sink.form.lanes)
    return Cost(window, presented(source, sink).beats, min(lanes))


def stacked_undelivered(
    chain: Steps, source: Selected, sink: Selected, selection: Sequence[int] = SELECTION
) -> str:
    """Why ``chain`` does not carry what ``source`` presents over ``selection`` to what
    ``sink`` reads over it, over the stacked tensor, or ``""``: a width step keeps the
    element order, a reorder reads its frames' beats in each selected pass, a marker step
    keeps the sequence. The chain starts at the beats it states for its first step (``undelivered``
    checks them against what the source presents, up to the wires of a lane permutation) and
    ends up to one lane permutation, the same in every beat."""
    size = source.set_size
    start = chain.hops[0].source.form if chain.hops else sink.sequence.form
    held: Array = np.concatenate([chosen * size + offsets(start) for chosen in selection])
    for hop in chain.hops:
        beats = held.reshape(-1, hop.source.form.lanes)
        if hop.step is Step.REORDER:
            assert hop.reorder is not None
            frame = hop.reorder.frame_beats
            frames = len(beats) // frame
            beats = beats[(np.arange(frames)[:, None] * frame + frame_slots(hop.reorder)).ravel()]
        held = beats.ravel()
    lanes = sink.sequence.form.lanes
    end, wanted = held.reshape(-1, lanes), sink.offsets(selection).reshape(-1, lanes)
    # Up to one lane permutation the same in every beat (wires): the columns are the same
    # multiset.
    if not np.array_equal(end[:, np.lexsort(end)], wanted[:, np.lexsort(wanted)]):
        return "the chain's elements over the selected passes are not the sink's"
    return ""


def unbounded(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """The stages with each reorder's capacity never binding over the frames run."""
    return tuple(
        replace(stage, capacity=stage.words * FRAMES) if isinstance(stage, ReorderStage) else stage
        for stage in stages
    )


def period_of(words: int, stages: Sequence[Stage]) -> int:
    """The most words a frame of any channel of ``stages`` between a source of ``words`` and
    a sink: each stage moves at most one word a cycle a port, so no period is shorter."""
    open_ = chained((Source("src", tuple(range(words)), words, words), *stages, Sink("snk")))
    return max(found for _, found in open_.words)


def timed(words: int, stages: Sequence[Stage], rules: Rules) -> tuple[int, int]:
    """The period and the bottleneck of a source of ``words`` words a frame through
    ``stages`` to a sink, every channel unbounded."""
    period = period_of(words, stages)
    network = chained((Source("src", tuple(range(words)), words, period), *stages, Sink("snk")))
    return period, bottleneck(network, rules)


def stages_deliver(chain: Steps, stages: Sequence[Stage]) -> str:
    """Why the stages of ``chain`` are not what its steps say, or ``""``: each width step
    at its two lane counts, each reorder reading its frames' slots over the step's source
    beats, each channel carrying its step's beats."""
    hops = [hop for hop in chain.hops if hop.step is not Step.MARKERS]
    if len(hops) != len(stages):
        return f"{len(hops)} steps with beats made {len(stages)} stages"
    for hop, stage in zip(hops, stages):
        if isinstance(stage, WidthStage):
            lanes = (hop.source.form.lanes, hop.sink.form.lanes)
            if hop.step is not Step.WIDTH or (stage.lanes_in, stage.lanes_out) != lanes:
                return f"{stage.name} is not the width step {lanes}"
        elif isinstance(stage, ReorderStage):
            assert hop.reorder is not None
            frame = hop.reorder.frame_beats
            frames = hop.source.form.beats // frame
            reads = (np.arange(frames)[:, None] * frame + frame_slots(hop.reorder)).ravel()
            if hop.step is not Step.REORDER or stage.reads != tuple(reads.tolist()):
                return f"{stage.name} does not read what its reorder reads"
        else:
            return f"{stage.name} is no stage of a step"
    first = hops[0].source.form.beats
    network = chained((Source("src", tuple(range(first)), first, first), *stages, Sink("snk")))
    carried = [found for _, found in network.words]
    if carried != [first, *(hop.sink.form.beats for hop in hops)]:
        return f"the channels carry {carried} words a frame"
    return ""


def findings_of_time(chain: Steps) -> list[str]:
    """What judge (d) finds wrong with the network of ``chain``'s stages: the stages'
    delivery, and the bottleneck against the period under each rule."""
    stages = unbounded(step_stages(chain, "s"))
    found: list[str] = []
    reason = stages_deliver(chain, stages)
    if reason:
        found.append(reason)
    first = next(hop for hop in chain.hops if hop.step is not Step.MARKERS).source.form.beats
    for label, rules in RULES:
        period, slowest = timed(first, stages, rules)
        if slowest != period:
            found.append(f"{label}: the bottleneck is {slowest} cycles, the period {period}")
    return found


# --- group A: one channel, no map --------------------------------------------------------

CORE_EXTENTS = (2, 3, 4)
CORE_BOUND = 24
CORE_LANES = (1, 2, 3, 4)

End = tuple[str, tuple[int, ...]]  # ("perm", axes), or ("replay", axes) for a sink


class Case(NamedTuple):
    shape: tuple[int, ...]
    source: End
    source_lanes: int
    sink: End
    sink_lanes: int
    marker: Pair | None
    selected: bool
    repetition: Repetition


@cache
def walk(shape: tuple[int, ...], end: End) -> Traversal:
    """Each axis walked in ``end``'s order, one element a beat; a replay has the innermost
    axis's run presented twice."""
    kind, perm = end
    base = Traversal.over(shape, [(axis, shape[axis], 1) for axis in perm], ())
    if kind == "replay":
        return base.replayed(2, inner_beats=shape[perm[-1]])
    return base


@cache
def at_lanes(shape: tuple[int, ...], end: End, lanes: int) -> Form | None:
    """The end's order at ``lanes`` a beat, or None where no pass of it divides."""
    try:
        return grouped(walk(shape, end), lanes)
    except ValueError:
        return None


def shapes() -> Iterator[tuple[int, ...]]:
    for rank in (1, 2, 3):
        for shape in itertools.product(CORE_EXTENTS, repeat=rank):
            if prod(shape) <= CORE_BOUND:
                yield shape


def ends(shape: tuple[int, ...], replays: bool) -> Iterator[End]:
    for perm in itertools.permutations(range(len(shape))):
        yield "perm", perm
        if replays and shape[perm[-1]] < prod(shape):
            yield "replay", perm


@cache
def marker_classes(form: Form) -> dict[str, tuple[Pair, ...]]:
    """Each (beats, inner) with ``inner | beats | beats`` and a level of two beats or
    more, by what the form does with it: an inner level of one beat, between the edges, the whole
    level, or refused (the marker closes no loop of the form)."""
    found: dict[str, list[Pair]] = {"one": [], "between": [], "whole": [], "refused": []}
    for beats in range(2, form.beats + 1):
        if form.beats % beats:
            continue
        for inner in range(1, beats + 1):
            if beats % inner:
                continue
            try:
                BeatSequence(form, markers=(level_done(beats, inner),))
            except ValueError:
                found["refused"].append((beats, inner))
                continue
            found["one" if inner == 1 else "whole" if inner == beats else "between"].append(
                (beats, inner)
            )
    return {kind: tuple(pairs) for kind, pairs in found.items()}


def marker_choices(form: Form, rng: random.Random) -> list[Pair | None]:
    """No marker, and one pair of each class the form has, drawn by ``rng``."""
    classes = marker_classes(form)
    return [None, *(rng.choice(classes[kind]) for kind in classes if classes[kind])]


def space(rng: random.Random) -> Iterator[Case]:
    """The core space of group A, in a fixed order."""
    for shape in shapes():
        sources, sinks = list(ends(shape, False)), list(ends(shape, True))
        for source, sink in itertools.product(sources, sinks):
            for source_lanes, sink_lanes in itertools.product(CORE_LANES, repeat=2):
                a, b = at_lanes(shape, source, source_lanes), at_lanes(shape, sink, sink_lanes)
                if a is None or b is None:
                    continue
                for marker in marker_choices(b, rng):
                    for selected in (False, True):
                        for repetition in Repetition:
                            yield Case(
                                shape,
                                source,
                                source_lanes,
                                sink,
                                sink_lanes,
                                marker,
                                selected,
                                repetition,
                            )


def forms_of(case: Case) -> tuple[Form, Form]:
    a = at_lanes(case.shape, case.source, case.source_lanes)
    b = at_lanes(case.shape, case.sink, case.sink_lanes)
    assert a is not None and b is not None
    return a, b


def sink_form(case: Case) -> Form:
    """The sink's form: its order, repeated twice against a cyclic source."""
    form = forms_of(case)[1]
    return form.repeated(2) if case.repetition is Repetition.CYCLIC else form


Cell = tuple[bool, bool, bool, str]  # (a lane group, a LevelDone between the edges, S, repetition)


def cell_of(case: Case) -> Cell:
    a, b = forms_of(case)
    group = isinstance(a, Grouped) or isinstance(b, Grouped)
    between = case.marker is not None and case.marker[1] not in (1, case.marker[0])
    return group, between, case.selected, case.repetition.name


class Built(NamedTuple):
    chain: Steps
    source: BeatSequence
    sink: BeatSequence
    selected: tuple[Selected, Selected] | None


def parts_alone(case: Case) -> dict[str, Outcome]:
    """What each part of the pipeline refuses when run alone on the case's own values: L's
    marker on the sink's form, and the steps between the forms without markers or
    selection (G's ``grouped`` built both ends: the space holds the forms it builds)."""
    found: dict[str, Outcome] = {}
    form = sink_form(case)
    markers = () if case.marker is None else (level_done(*case.marker),)
    try:
        BeatSequence(form, markers=markers)
    except ValueError as error:
        found["L"] = outcome(error)
    try:
        steps(BeatSequence(forms_of(case)[0], case.repetition), BeatSequence(form))
    except ValueError as error:
        found["steps"] = outcome(error)
    return found


def attempt(case: Case) -> Built | Outcome:
    """The case built whole: its ends (the sink's markers are checked as the end is
    built), then the steps between them; or the refusal."""
    form = sink_form(case)
    markers = () if case.marker is None else (level_done(*case.marker),)
    try:
        source = BeatSequence(forms_of(case)[0], case.repetition)
        sink = BeatSequence(form, markers=markers)
        if case.selected:
            ends = (Selected(source, SETS, "sel"), Selected(sink, SETS, "sel"))
            return Built(steps(*ends), source, sink, ends)
        return Built(steps(source, sink), source, sink, None)
    except ValueError as error:
        return outcome(error)


def over_passes(source: BeatSequence, sink: BeatSequence, passes: int) -> Steps:
    """The steps over ``passes`` selected passes, each a pass of ``sink``'s form: a once
    source repeated as often, a cyclic one presented as often as the sink needs."""
    form = source.form.repeated(passes) if source.repetition is Repetition.ONCE else source.form
    return steps(BeatSequence(form, source.repetition), BeatSequence(sink.form.repeated(passes)))


def loops_of(form: Form) -> tuple[Loop, ...]:
    if isinstance(form, Grouped):
        return form.order
    return (*form.beat_loops, *form.lane_loops)


def test_a_once_source_over_two_passes_against_a_sink_replaying_twice_builds() -> None:
    shape = (2, 2, 2)
    source = BeatSequence(grouped(walk(shape, ("perm", (0, 2, 1))), 1))
    sink = BeatSequence(grouped(walk(shape, ("replay", (1, 2, 0))), 1))
    one = steps(source, sink)
    assert one.kinds == (Step.REORDER,)
    for passes in (2, 3):
        many = over_passes(source, sink, passes)
        first = BeatSequence(source.form.repeated(passes))
        last = BeatSequence(sink.form.repeated(passes))
        assert undelivered(many, first, last) == ""
        assert [hop.reorder for hop in many.hops] == [hop.reorder for hop in one.hops]
        priced = cost(one, sink)
        assert cost(many, last) == Cost(
            priced.live_window, priced.reads * passes, priced.narrowest_lanes
        )


def judged(case: Case) -> tuple[str, Built | None]:
    """The case's verdict by judges (a) to (c): ``BUILT``, ``PART`` (refused, and by a
    part alone, with that part's class, code and message), or what is wrong."""
    parts = parts_alone(case)
    found = attempt(case)
    if not isinstance(found, Built):
        if not parts:
            return f"REFUSED ONLY IN COMBINATION: {found}", None
        # The sink is built before the steps: L's refusal comes first.
        expected = parts["L"] if "L" in parts else parts["steps"]
        if found != expected:
            return f"REFUSED BY NO PART: {found}, the parts' {parts}", None
        return PART, None
    if parts:
        return f"BUILT WHERE A PART ALONE REFUSES: {parts}", None
    if built_selected := found.selected:
        if found.chain != steps(found.source, found.sink):
            return "A SELECTED CHAIN IS NOT ITS SEQUENCES'", None
        reason = stacked_undelivered(found.chain, *built_selected)
        if reason:
            return f"UNDELIVERED over the selected passes: {reason}", None
    reason = undelivered(found.chain, found.source, found.sink)
    if reason:
        return f"UNDELIVERED: {reason}", None
    if cost(found.chain, found.sink) != composed_by_hand(found.chain, found.source, found.sink):
        return "COST NOT THE PARTS' COMPOSED", None
    return BUILT, found


@dataclass
class Judgement:
    """What judging a run of cases found: the verdicts by count, the first case of each,
    and the cases whose chains have a step with beats, to be timed."""

    verdicts: Counter[str]
    first: dict[str, Case]
    cells: Counter[Cell]
    timed: list[Case]


def judge_all(cases: Iterator[Case], reservoir: int, rng: random.Random) -> Judgement:
    """Judge every case; keep ``reservoir`` of the built ones with a step that moves beats,
    drawn uniformly (reservoir sampling) by ``rng``."""
    done = Judgement(Counter(), {}, Counter(), [])
    seen = 0
    for case in cases:
        verdict, built = judged(case)
        done.verdicts[verdict] += 1
        done.first.setdefault(verdict, case)
        done.cells[cell_of(case)] += 1
        if built is not None and any(hop.step is not Step.MARKERS for hop in built.chain.hops):
            seen += 1
            if len(done.timed) < reservoir:
                done.timed.append(case)
            elif (slot := rng.randrange(seen)) < reservoir:
                done.timed[slot] = case
    return done


def sampled(rng: random.Random, space_rng: random.Random) -> tuple[int, Counter[Cell], list[Case]]:
    """The space's size, its cells' sizes, and ``SAMPLE_PER_CELL`` cases drawn uniformly from
    each cell (reservoir sampling), in the space's order."""
    cells: Counter[Cell] = Counter()
    chosen: dict[Cell, list[Case]] = {}
    for case in space(space_rng):
        cell = cell_of(case)
        cells[cell] += 1
        picked = chosen.setdefault(cell, [])
        if len(picked) < SAMPLE_PER_CELL:
            picked.append(case)
        elif (slot := rng.randrange(cells[cell])) < SAMPLE_PER_CELL:
            picked[slot] = case
    return sum(cells.values()), cells, [case for cell in sorted(chosen) for case in chosen[cell]]


@cache
def sample() -> tuple[int, Counter[Cell], list[Case]]:
    return sampled(random.Random(SEED), random.Random(SEED))


@cache
def sample_judged() -> Judgement:
    return judge_all(iter(sample()[2]), TIMED_SAMPLE, random.Random(SEED))


def test_the_core_space_has_the_stated_cells() -> None:
    size, cells, chosen = sample()
    assert size == 287_936
    assert len(cells) == 16 and all(8_706 <= count <= 29_761 for count in cells.values())
    assert min(cells.values()) == 8_706 and max(cells.values()) == 29_761
    assert sorted(cells.values()) == [
        8706, 8706, 8706, 8706, 12162, 12162, 12162, 12162,
        21355, 21355, 21355, 21355, 29761, 29761, 29761, 29761,
    ]  # fmt: skip
    # Every cell is sampled alike: the 16 cells' 250 cases each.
    assert Counter(cell_of(case) for case in chosen) == {cell: SAMPLE_PER_CELL for cell in cells}
    assert len(chosen) == 16 * SAMPLE_PER_CELL


def test_the_sampled_cases_build_deliver_and_cost_their_parts_composed() -> None:
    """Judges (a) to (c) on the 4,000 sampled cases: each is built and delivers at its
    parts' composed cost, or is refused by a part alone."""
    done = sample_judged()
    assert set(done.verdicts) <= {BUILT, PART}, {
        verdict: done.first[verdict] for verdict in done.verdicts if verdict not in (BUILT, PART)
    }
    assert sum(done.verdicts.values()) == 4_000
    assert done.verdicts[PART] > 0 and done.verdicts[BUILT] > 0
    sampled_cases = sample()[2]
    for flag in range(4):  # every extension is on, and off, among the built cases
        assert any(cell_of(case)[flag] for case in sampled_cases)


def test_a_marker_refusal_precedes_the_request_for_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A marker-invalid end is refused before any step derivation is attempted."""
    case = next(case for case in sample()[2] if "L" in parts_alone(case))
    expected = parts_alone(case)["L"]

    def unexpected(*args: object, **kwargs: object) -> Steps:
        raise AssertionError("steps requested before refusing the invalid marker")

    monkeypatch.setattr(__name__ + ".steps", unexpected)
    assert attempt(case) == expected


def test_a_selected_chain_over_passes_is_the_chain_of_one_pass() -> None:
    """Over ``PASSES`` selected passes the chain is the one pass's: its steps and each
    reorder the same (a reorder frame lies inside one pass), its live window and narrowest
    lanes the same, its reads ``PASSES`` times, including equal repetition/replay loops."""
    checked = 0
    for case in sample()[2]:
        if not case.selected:
            continue
        form = sink_form(case)
        source = BeatSequence(forms_of(case)[0], case.repetition)
        try:
            one = steps(source, BeatSequence(form))
        except ValueError:
            continue
        many = over_passes(source, BeatSequence(form), PASSES)
        first = (
            source.form.repeated(PASSES) if source.repetition is Repetition.ONCE else source.form
        )
        assert (
            undelivered(
                many, BeatSequence(first, source.repetition), BeatSequence(form.repeated(PASSES))
            )
            == ""
        ), case
        assert many.kinds == one.kinds
        assert [hop.reorder for hop in many.hops] == [hop.reorder for hop in one.hops]
        by_one, by_many = (
            cost(one, BeatSequence(form)),
            cost(many, BeatSequence(form.repeated(PASSES))),
        )
        assert by_many == Cost(by_one.live_window, by_one.reads * PASSES, by_one.narrowest_lanes)
        checked += 1
    assert checked > 500


def test_the_sampled_built_chains_keep_the_period_under_both_rules() -> None:
    """Judge (d) on 200 built chains of the sample. A selected pair is timed over
    ``PASSES`` passes."""
    timed_cases = sample_judged().timed
    assert len(timed_cases) == TIMED_SAMPLE
    assert sum(case.selected for case in timed_cases) > 0
    timed = 0
    for case in timed_cases:
        chain = timing_chain(case)
        assert findings_of_time(chain) == [], case
        timed += 1
    assert timed == TIMED_SAMPLE


def timing_chain(case: Case) -> Steps:
    """The case's chain, or for S its chain over ``PASSES`` selected passes."""
    found = attempt(case)
    assert isinstance(found, Built)
    if case.selected:
        return over_passes(found.source, found.sink, PASSES)
    return found.chain


def test_a_marker_moves_no_word_and_makes_no_stage() -> None:
    """L with N: the stages of a chain with a marker step are those of the chain without."""
    checked = 0
    for case in sample()[2]:
        found = attempt(case)
        if not isinstance(found, Built) or case.selected:
            continue
        marked = [hop for hop in found.chain.hops if hop.step is Step.MARKERS]
        if not marked:
            continue
        rest = Steps(tuple(hop for hop in found.chain.hops if hop.step is not Step.MARKERS))
        assert step_stages(found.chain, "s") == step_stages(rest, "s")
        checked += 1
    assert checked > 100


@pytest.mark.slow
def test_the_whole_core_space_builds_delivers_and_costs_its_parts_composed() -> None:
    done = whole_core()
    assert sum(done.verdicts.values()) == 287_936
    assert set(done.verdicts) <= {BUILT, PART}, {
        verdict: done.first[verdict] for verdict in done.verdicts if verdict not in (BUILT, PART)
    }
    assert done.verdicts[BUILT] == 249_872 and done.verdicts[PART] == 38_064
    assert sorted(done.cells.values()) == sorted(sample()[1].values())


@pytest.mark.slow
def test_the_built_chains_of_the_whole_space_keep_the_period_under_both_rules() -> None:
    """Judge (d) on 2,000 of the built chains of the whole core space."""
    timed_cases = whole_core().timed
    assert len(timed_cases) == TIMED_SLOW
    timed = 0
    for case in timed_cases:
        chain = timing_chain(case)
        assert findings_of_time(chain) == [], case
        timed += 1
    assert timed == TIMED_SLOW


@cache
def whole_core() -> Judgement:
    return judge_all(space(random.Random(SEED)), TIMED_SLOW, random.Random(SEED))


# --- group M: through a position map ----------------------------------------------------

MAPS: dict[str, tuple[maps.PositionMap, maps.PositionMap | None]] = {
    "transpose (4, 6)": (maps.transpose((4, 6)), None),
    "transpose (2, 3, 4) by (1, 2, 0)": (maps.transpose((2, 3, 4), (1, 2, 0)), None),
    "transpose (6, 4, 2) by (2, 0, 1)": (maps.transpose((6, 4, 2), (2, 0, 1)), None),
    "window (1, 4, 4, 2) of 2": (maps.im2col((1, 4, 4, 2), (2, 2)), None),
    "window (1, 5, 1, 3) of (2, 1)": (maps.im2col((1, 5, 1, 3), (2, 1)), None),
    "pad (4, 4, 2) by 1 on H and W": (
        maps.identity((6, 6, 2)),
        maps.pad((4, 4, 2), (1, 1, 0, 1, 1, 0)),
    ),
    "pad (3, 6) by 2 on W": (maps.identity((3, 10)), maps.pad((3, 6), (0, 2, 0, 2))),
}  # fmt: skip
MAP_LANES = (1, 2, 3, 4, 6)

# Where a case ends.
READER_MARKER = "reader's marker refused (L alone)"
COMPOSED_MARKER = "marker unaligned on the composed form (L on M's form)"
MAP_ALONE = "composed refused as without the marker (M alone)"


BUILT_PLAIN = "built (no stage; sink Traversal)"
BUILT_PLAIN_GROUPED = "built (no stage; sink Grouped)"
BUILT_STAGED = "built (stage; sink Traversal)"
BUILT_STAGED_GROUPED = "built (stage; sink Grouped)"


def built_as(stage: bool, sink: Form) -> str:
    grouped_sink = isinstance(sink, Grouped)
    if stage:
        return BUILT_STAGED_GROUPED if grouped_sink else BUILT_STAGED
    return BUILT_PLAIN_GROUPED if grouped_sink else BUILT_PLAIN


class MapCase(NamedTuple):
    name: str
    producer_lanes: int
    perm: tuple[int, ...]
    reader_lanes: int
    marker: Pair | None


def producer_tensor(name: str) -> tuple[int, ...]:
    pmap, at_producer = MAPS[name]
    return at_producer.sources[0] if at_producer is not None else pmap.sources[0]


@cache
def reader_form(name: str, perm: tuple[int, ...], lanes: int) -> Form | None:
    shape = MAPS[name][0].shape
    order = Traversal.over(shape, [(axis, shape[axis], 1) for axis in perm], ())
    try:
        return grouped(order, lanes)
    except ValueError:
        return None


def map_cases() -> Iterator[MapCase]:
    for name, (pmap, _) in MAPS.items():
        for producer_lanes in (1, 2, 3, 4):
            try:
                vector_major(producer_tensor(name), producer_lanes)
            except ValueError:
                continue
            for perm in itertools.permutations(range(len(pmap.shape))):
                for lanes in MAP_LANES:
                    form = reader_form(name, perm, lanes)
                    if form is None:
                        continue
                    yield MapCase(name, producer_lanes, perm, lanes, None)
                    for beats in range(2, form.beats + 1):
                        if form.beats % beats == 0:
                            for inner in range(1, beats + 1):
                                if beats % inner == 0:
                                    yield MapCase(name, producer_lanes, perm, lanes, (beats, inner))


def reindex_of(case: MapCase, markers: bool = True) -> Reindex:
    """The case's one reader over its producer; ``markers`` False leaves the marker off."""
    pmap, at_producer = MAPS[case.name]
    form = reader_form(case.name, case.perm, case.reader_lanes)
    assert form is not None
    rules = (level_done(*case.marker),) if markers and case.marker is not None else ()
    producer = BeatSequence(vector_major(producer_tensor(case.name), case.producer_lanes))
    return Reindex(producer, (Reader(pmap, BeatSequence(form, markers=rules), at_producer),))


def stage_streams(reindex: Reindex, found: Route) -> tuple[BeatSequence, BeatSequence]:
    """The actual stage's row-major input and output at its whole-beat lane count."""
    stage = found.stage
    assert stage is not None
    lanes = stage_lanes(stage, reindex.producer.form.lanes)
    shapes = (
        (stage.shape, stage.kept.shape)
        if isinstance(stage, maps.Drop)
        else (stage.fills.sources[0], stage.shape)
    )
    first, last = (
        BeatSequence(
            Traversal(shape, (Loop(prod(shape) // lanes, lanes),), (Loop(lanes, 1),)),
            reindex.producer.repetition,
        )
        for shape in shapes
    )
    return first, last


def route_undelivered(reindex: Reindex, found: Route, sink: BeatSequence) -> str:
    """Check each step chain and enumerate the stage's movement of producer positions.

    The reference reads the reader's map and its producer-side map in succession.
    The actual stage inserts fills into, or drops positions from, its row-major input;
    its output is indexed by the composed sink. A fill is distinguished from every
    producer position by -1, and the fill value is checked separately.
    """
    (reader,) = reindex.readers
    label, read = maps.reads(reader.map)
    at = offsets(reader.stream.form)
    wanted = np.where(label < 0, -1, read)[at]
    if found.stage is None:
        if not np.array_equal(offsets(sink.form), wanted):
            return "composed positions differ from the map's reads"
        return undelivered(found.after, reindex.producer, sink)
    first, last = stage_streams(reindex, found)
    for name, chain, source, target in (
        ("before", found.before, reindex.producer, first),
        ("after", found.after, last, sink),
    ):
        reason = undelivered(chain, source, target)
        if reason:
            return f"{name}: {reason}"
    assert reader.at_producer is not None
    edge_label, edge_read = maps.reads(reader.at_producer)
    wanted = np.where(edge_label < 0, -1, edge_read)[wanted]
    produced = np.arange(prod(first.form.shape), dtype=np.int64)
    if isinstance(found.stage, maps.Insertion):
        inserted = found.stage.inserted()
        delivered = np.full(len(inserted), -1, dtype=np.int64)
        delivered[~inserted] = produced
        if found.stage.fills.fill != reader.at_producer.fill:
            return "stage inserts a different fill value"
    else:
        delivered = produced[~found.stage.dropped()]
    if not np.array_equal(delivered[offsets(sink.form)], wanted):
        return "stage and composed sink deliver different producer positions"
    return ""


def routed_cost_by_hand(reindex: Reindex, found: Route, sink: BeatSequence) -> Cost:
    """A route's cost composed from its parts as ``route_cost`` states them: without an
    insertion or a drop, its steps'; with one, the live windows of its steps summed, the
    reads of its first part (a ``before`` step's, times the passes its stage's output makes
    of the reader's; else the stage's own), the fewest lanes anywhere on it."""
    producer = reindex.producer
    if found.stage is None:
        return composed_by_hand(found.after, producer, sink)
    lanes = stage_lanes(found.stage, producer.form.lanes)
    shape_in = (
        found.stage.shape if isinstance(found.stage, maps.Drop) else found.stage.fills.sources[0]
    )
    shape_out = found.stage.kept.shape if isinstance(found.stage, maps.Drop) else found.stage.shape
    size_in, size_out = prod(shape_in), prod(shape_out)
    stage_in = BeatSequence(
        Traversal(shape_in, (Loop(size_in // lanes, lanes),), (Loop(lanes, 1),)),
        producer.repetition,
    )
    stage_out = BeatSequence(
        Traversal(shape_out, (Loop(size_out // lanes, lanes),), (Loop(lanes, 1),)),
        producer.repetition,
    )
    passes = presented(stage_out, sink).beats // stage_out.form.beats
    parts = [Cost(0, stage_in.form.beats * passes, lanes)]
    parts.append(composed_by_hand(found.after, stage_out, sink))
    if found.before:
        first = composed_by_hand(found.before, producer, stage_in)
        parts.insert(0, Cost(first.live_window, first.reads * passes, first.narrowest_lanes))
    return Cost(
        sum(part.live_window for part in parts),
        parts[0].reads,
        min(part.narrowest_lanes for part in parts),
    )


def map_verdict(case: MapCase) -> str:
    """The case's end: one of the kinds above, or what is wrong. L alone is the marker on
    the reader's own form; M alone is ``composed`` without the marker; then the marker on
    the composed form, which ``composed`` raises under M's code."""
    form = reader_form(case.name, case.perm, case.reader_lanes)
    assert form is not None
    markers = () if case.marker is None else (level_done(*case.marker),)
    try:
        BeatSequence(form, markers=markers)
    except ValueError:
        return READER_MARKER
    reindex = reindex_of(case)
    try:
        sink = composed(reindex)
    except Exception as error:  # a refusal is compared, whatever its class
        try:
            plain = composed(reindex_of(case, markers=False))
        except Exception as alone:
            return MAP_ALONE if outcome(alone) == outcome(error) else f"REFUSED BY NO PART: {error}"
        try:
            BeatSequence(plain.form, markers=markers)
        except ValueError as part:
            same = outcome(error) == ("Unrouted", "marker-unaligned", str(part))
            return COMPOSED_MARKER if same else f"REFUSED BY NO PART: {error}"
        return f"REFUSED ONLY IN COMBINATION: {outcome(error)}"
    try:
        found = route(reindex)
        priced = route_cost(reindex)
        route_stages(reindex, "r")
    except Exception as error:
        return f"ROUTE REFUSED: {outcome(error)}"
    assert isinstance(found, Route)
    reason = route_undelivered(reindex, found, sink)
    if reason:
        return f"UNDELIVERED: {reason}"
    if priced != routed_cost_by_hand(reindex, found, sink):
        return "COST NOT THE PARTS' COMPOSED"
    if found.stage is None and priced != cost(found.after, sink):
        return "ROUTE COST NOT THE STEPS' COST"
    return built_as(found.stage is not None, sink.form)


@pytest.mark.parametrize("part", ("before", "stage", "after"))
def test_a_wrong_staged_route_fails_the_map_delivery_judge(
    part: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = MapCase("pad (3, 6) by 2 on W", 3, (0, 1), 3, (10, 5))
    reindex = reindex_of(case)
    found = route(reindex)
    assert isinstance(found, Route) and found.before and found.after
    assert map_verdict(case) == BUILT_STAGED
    wrong = {
        "before": replace(found, before=Steps()),
        "stage": replace(found, stage=maps.Insertion(maps.pad((3, 6), (0, 1, 0, 3)))),
        "after": replace(found, after=Steps()),
    }[part]
    monkeypatch.setattr(__name__ + ".route", lambda _: wrong)
    assert map_verdict(case).startswith("UNDELIVERED:"), part


@pytest.mark.parametrize("grouped_reader", (False, True))
@pytest.mark.parametrize("marked", (False, True))
def test_staged_pads_with_grouped_or_marked_readers_are_checked(
    grouped_reader: bool, marked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = next(
        case
        for case in map_cases()
        if case.name.startswith("pad")
        and (case.marker is not None) == marked
        and isinstance(reader_form(case.name, case.perm, case.reader_lanes), Grouped)
        == grouped_reader
        and map_verdict(case).startswith("built")
    )
    checks = []
    original = undelivered

    def checked(chain: Steps, source: BeatSequence, sink: BeatSequence) -> str:
        checks.append((source.form.shape, sink.form.shape))
        return original(chain, source, sink)

    monkeypatch.setattr(__name__ + ".undelivered", checked)
    assert map_verdict(case).startswith("built")
    assert len(checks) == 2  # the stage's input chain and output chain


@pytest.mark.parametrize("lanes", (2, 3))
@pytest.mark.parametrize("marked", (False, True))
def test_a_drop_route_checks_both_chains_and_the_positions_kept(lanes: int, marked: bool) -> None:
    crop = maps.sliced((8, 6), (1, 0), (7, 6), None, (2, 1))
    order = Traversal.over((3, 6), [(1, 6, 1), (0, 3, 1)], ())
    form = grouped(order, lanes)
    markers = (LevelEnd(form.beats),) if marked else ()
    reader = Reader(maps.identity((3, 6)), BeatSequence(form, markers=markers), crop)
    reindex = Reindex(BeatSequence(vector_major((8, 6), 6)), (reader,))
    found = route(reindex)
    assert isinstance(found, Route) and isinstance(found.stage, maps.Drop)
    assert isinstance(form, Grouped) == (lanes == 2)
    sink = composed(reindex)
    assert route_undelivered(reindex, found, sink) == ""
    wrong_drop = maps.Drop(maps.sliced((8, 6), (0, 0), (6, 6), None, (2, 1)))
    assert route_undelivered(reindex, replace(found, stage=wrong_drop), sink)
    assert found.after
    assert route_undelivered(reindex, replace(found, after=Steps()), sink)


def map_findings_of_time(case: MapCase) -> list[str]:
    """Judge (d) on a built map case: the stages of its route (an insertion or a drop
    included), each reorder unbounded, between a source of the producer's beats and a sink."""
    reindex = reindex_of(case)
    stages = unbounded(route_stages(reindex, "r"))
    words = reindex.producer.form.beats
    found: list[str] = []
    for label, rules in RULES:
        period, slowest = timed(words, stages, rules)
        if slowest != period:
            found.append(f"{label}: the bottleneck is {slowest} cycles, the period {period}")
    return found


@cache
def map_sample() -> list[MapCase]:
    return random.Random(SEED).sample(list(map_cases()), MAP_SAMPLE)


def map_judged(cases: Sequence[MapCase]) -> tuple[Counter[str], dict[str, MapCase]]:
    verdicts: Counter[str] = Counter()
    first: dict[str, MapCase] = {}
    for case in cases:
        verdict = map_verdict(case)
        verdicts[verdict] += 1
        first.setdefault(verdict, case)
    return verdicts, first


MAP_KINDS = {
    READER_MARKER,
    COMPOSED_MARKER,
    MAP_ALONE,
    BUILT_PLAIN,
    BUILT_PLAIN_GROUPED,
    BUILT_STAGED,
    BUILT_STAGED_GROUPED,
}


def test_the_map_cases_are_the_stated_number() -> None:
    assert sum(1 for _ in map_cases()) == 16_680


def test_the_sampled_map_cases_deliver_and_cost_their_parts_composed() -> None:
    """Judges (a) to (c) on 2,000 sampled cases: a case is refused by a part alone (the
    reader's marker, the map without the marker, the marker on the composed form), or is
    built, delivers the reader's positions through the map and costs what its route's
    parts cost."""
    verdicts, first = map_judged(map_sample())
    assert set(verdicts) <= MAP_KINDS, {v: first[v] for v in verdicts if v not in MAP_KINDS}
    assert sum(verdicts.values()) == MAP_SAMPLE
    assert all(verdicts[kind] > 0 for kind in MAP_KINDS)


def test_the_sampled_map_chains_keep_the_period_under_both_rules() -> None:
    """Judge (d) on 100 built map cases: the route's stages, an insertion or a drop among
    them."""
    cases = [case for case in map_sample() if map_verdict(case).startswith("built")][:MAP_TIMED]
    assert len(cases) == MAP_TIMED
    for case in cases:
        assert map_findings_of_time(case) == [], case


@pytest.mark.slow
def test_every_map_case_is_judged() -> None:
    verdicts, first = map_judged(list(map_cases()))
    assert set(verdicts) <= MAP_KINDS, {v: first[v] for v in verdicts if v not in MAP_KINDS}
    assert dict(verdicts) == {
        BUILT_PLAIN: 7_013,
        READER_MARKER: 4_184,
        BUILT_PLAIN_GROUPED: 1_784,
        BUILT_STAGED: 1_625,
        COMPOSED_MARKER: 1_223,
        MAP_ALONE: 702,
        BUILT_STAGED_GROUPED: 149,
    }


# --- group D: a selector stream varying with 0, 1 or every beat index -------------------

a, cb, cl, t = Index("a"), Index("cb"), Index("cl"), Index("t")
D_SETS = 3
D_SELECTION = (2, 0, 1, 1)
D_GEOMETRIES = [(3, 6, 2, 2), (2, 4, 2, 3), (4, 6, 3, 2)]  # (rows, C, PE, T)
VARIATIONS = ("frame", "row", "beat")


@dataclass(frozen=True)
class Selector:
    """A kernel of schedule (a, cb, cl, t), ``c = cb * PE + cl``, lanes (cl, t) at the set
    port: ``rows * C / PE`` iterations, the data port (rows, C) at ``(a, cb * PE + cl)``.
    The selector stream varies with no index (``frame``), with ``a`` (``row``) or with
    ``a`` and ``cb`` (``beat``): the set port's pass is the set over the schedule's indices
    the selector tensor does not vary with, so the sets are (C, T) at ``(cb * PE + cl, t)`` for the
    first two and (PE, T) at ``(cl, t)`` for the last."""

    rows: int
    channels: int
    pe: int
    thresholds: int
    variation: str

    @property
    def full(self) -> Schedule:
        return Schedule(
            {a: self.rows, cb: self.channels // self.pe, cl: self.pe, t: self.thresholds},
            {cl: self.pe, t: self.thresholds},
            (a, cb, cl, t),
        )

    @property
    def scope(self) -> Schedule:
        """The indices inside the set port's pass."""
        if self.variation == "frame":
            return self.full
        if self.variation == "row":
            return Schedule(
                {cb: self.channels // self.pe, cl: self.pe, t: self.thresholds},
                {cl: self.pe, t: self.thresholds},
                (cb, cl, t),
            )
        return Schedule(
            {cl: self.pe, t: self.thresholds}, {cl: self.pe, t: self.thresholds}, (cl, t)
        )

    @property
    def set_shape(self) -> tuple[int, int]:
        if self.variation == "beat":
            return self.pe, self.thresholds
        return self.channels, self.thresholds

    @property
    def data(self) -> Traversal:
        return self.full.present((self.rows, self.channels), (a, cb * self.pe + cl), lanes=(cl,))

    @property
    def set_port(self) -> Traversal:
        """One pass of the set port: one set over the scope's indices."""
        index = (cl, t) if self.variation == "beat" else (cb * self.pe + cl, t)
        return self.scope.present(self.set_shape, index, lanes=(cl, t))

    @property
    def passes(self) -> int:
        """The selected side's passes a frame: the kernel's iterations over the set
        port's pass."""
        return self.full.iterations // self.scope.iterations

    @property
    def selector_shape(self) -> tuple[int, ...]:
        return {"frame": (1,), "row": (self.rows,), "beat": (self.rows, self.channels // self.pe)}[
            self.variation
        ]

    @property
    def selector_port(self) -> Traversal:
        choices: dict[str, tuple[Index | Affine, ...]] = {
            "frame": (ZERO,),
            "row": (a,),
            "beat": (a, cb),
        }
        return self.full.present(self.selector_shape, choices[self.variation])

    @property
    def stored(self) -> Form:
        """One set in the memory, threshold-major (``t`` outer, the channel inner), at the
        least lane count that cuts that order where a lane count does (a ``Grouped``), else
        at one lane."""
        order = Traversal.over(
            self.set_shape, [(1, self.set_shape[1], 1), (0, self.set_shape[0], 1)], ()
        )
        forms = (
            grouped(order, lanes) for lanes in range(2, order.beats + 1) if order.beats % lanes == 0
        )
        return next((form for form in forms if isinstance(form, Grouped)), grouped(order, 1))

    @property
    def sequences(self) -> tuple[BeatSequence, BeatSequence]:
        return BeatSequence(self.stored, Repetition.CYCLIC), BeatSequence(self.set_port)

    @property
    def selected(self) -> tuple[Selected, Selected]:
        source, sink = self.sequences
        return Selected(source, D_SETS, "S"), Selected(sink, D_SETS, "S")

    @property
    def frame_chain(self) -> Steps:
        """The set port's chain over one frame: its pass presented ``passes`` times."""
        source, sink = self.sequences
        return steps(source, BeatSequence(sink.form.repeated(self.passes)))

    @property
    def selector_chain(self) -> Steps:
        return steps(
            BeatSequence(vector_major(self.selector_shape, 1)), BeatSequence(self.selector_port)
        )

    def network(self, period: int) -> Network:
        """The kernel's data port fed by a source, its set port by the memory and the
        stages of ``frame_chain``, its selector port by a source and the stages of
        ``selector_chain``; the sources released every ``period`` cycles."""
        pace = Pace(self.full)
        core = kernel("core", (pace, pace, pace), ())
        set_stages = unbounded(step_stages(self.frame_chain, "w"))
        selector_stages = unbounded(step_stages(self.selector_chain, "s"))
        selector_words = prod(self.selector_shape)
        data = self.data.beats
        links = [Channel("x", "x", "core")]
        stages: list[Stage] = [Source("x", tuple(range(data)), data, period), Memory("mem")]
        stages += [*set_stages]
        names = ["mem", *(stage.name for stage in set_stages)]
        links += [Channel(f"{p}>{q}", p, q) for p, q in zip(names, names[1:])]
        links.append(Channel("set", names[-1], "core", input=1))
        stages.append(Source("sel", tuple(range(selector_words)), selector_words, period))
        stages += [*selector_stages, core]
        names = ["sel", *(stage.name for stage in selector_stages)]
        links += [Channel(f"{p}>{q}", p, q) for p, q in zip(names, names[1:])]
        links.append(Channel("selp", names[-1], "core", input=2))
        return Network(tuple(stages), tuple(links))

    @property
    def period(self) -> int:
        """The most words a frame on any channel, and the kernel's iterations."""
        words = [found for _, found in self.network(10**6).words]
        return max(*words, self.full.iterations)


D_CASES = [Selector(*geometry, variation) for geometry in D_GEOMETRIES for variation in VARIATIONS]
D_IDS = [f"{c.rows}-{c.channels}-{c.pe}-{c.thresholds}-{c.variation}" for c in D_CASES]
# The selected side's passes a frame, from the table of the variations: 1, rows, rows * C / PE.
D_PASSES: dict[str, Callable[[Selector], int]] = {
    "frame": lambda c: 1,
    "row": lambda c: c.rows,
    "beat": lambda c: c.rows * c.channels // c.pe,
}


@pytest.mark.parametrize("case", D_CASES, ids=D_IDS)
def test_a_selected_set_port_delivers_at_its_parts_cost(case: Selector) -> None:
    """The set port's selected sides: the chain between them is the chain between their
    sequences (inside one set), it delivers by enumeration and over the stacked sets, and
    over a frame's passes it is the one pass's with the reads ``passes`` times."""
    source, sink = case.selected
    chain = steps(source, sink)
    assert chain == steps(source.sequence, sink.sequence)
    assert undelivered(chain, source.sequence, sink.sequence) == ""
    assert stacked_undelivered(chain, source, sink, D_SELECTION) == ""
    assert cost(chain, sink.sequence) == composed_by_hand(chain, source.sequence, sink.sequence)
    one = cost(chain, sink.sequence)
    frame = case.frame_chain
    assert frame.kinds == chain.kinds
    assert [hop.reorder for hop in frame.hops] == [hop.reorder for hop in chain.hops]
    wide = BeatSequence(sink.sequence.form.repeated(case.passes))
    assert cost(frame, wide) == Cost(one.live_window, one.reads * case.passes, one.narrowest_lanes)


@pytest.mark.parametrize("case", D_CASES, ids=D_IDS)
def test_the_selector_stream_presents_one_index_a_selected_pass(case: Selector) -> None:
    """Nothing but this ties the selector stream's channel to the selected side: the indices a
    frame the selector tensor holds are the passes the set port makes a frame."""
    assert case.passes == D_PASSES[case.variation](case)
    assert prod(case.selector_shape) == case.passes
    # The set port over a frame presents its pass that many times, one beat each iteration.
    assert case.set_port.repeated(case.passes).beats == case.full.iterations


def test_the_stored_sets_are_a_lane_group_where_a_lane_count_cuts_their_order() -> None:
    grouped_stores = [isinstance(case.stored, Grouped) for case in D_CASES]
    assert grouped_stores == [True, True, False, True, True, True, True, True, True]


@pytest.mark.parametrize("case", D_CASES, ids=D_IDS)
def test_a_selected_set_port_keeps_the_period_in_its_kernel_s_network(case: Selector) -> None:
    """The kernel with all three ports in one network: the data port's source, the memory
    and the set port's stages, the selector stream's source and its stages. The period is the most
    words a frame on any channel or the kernel's iterations, and the bottleneck is it under
    both rules, each reorder unbounded."""
    period = case.period
    assert period >= case.full.iterations
    network = case.network(period)
    for _, rules in RULES:
        assert bottleneck(network, rules) == period


@pytest.mark.parametrize("geometry", D_GEOMETRIES)
def test_a_selector_varying_across_lanes_is_refused(geometry: tuple[int, int, int, int]) -> None:
    """One selector index per lane cannot be presented by a one-lane selector port.
    This checks lane variation only, not an association with whole-set passes."""
    rows, channels, pe, _ = geometry
    full = Selector(*geometry, "frame").full
    with pytest.raises(Refused, match="lanes move the position") as error:
        full.present((rows, channels), (a, cb * pe + cl))
    assert [found.code for found in error.value.findings] == ["lanes-move-position"]


def test_a_beat_index_selector_builds_without_a_whole_set_pass_association() -> None:
    """The beat index advances within the original whole set, yet its selector stream builds.

    There is no constructor tying the nine selector stream beats to the one selected pass.
    This is a representation boundary, not a refusal of either independent value.
    """
    whole = Selector(3, 6, 2, 2, "frame")
    selector = whole.full.present((3, 3), (a, cb))
    selected = whole.selected[1]
    assert selector.lanes == 1 and selector.beats == 9
    assert offsets(selector).tolist() == list(range(9))
    assert whole.set_shape == (6, 2) and selected.set_size == 12
    assert selected.sequence.form == whole.set_port
    assert whole.passes == 1 and selector.beats != whole.passes
    # The existing beat-scoped construction explicitly changes the set's extent.
    beat = Selector(3, 6, 2, 2, "beat")
    assert beat.set_shape == (2, 2) and beat.passes == selector.beats


# --- group T: all four in one channel, the tiled MatMul's weights -----------------------

G, TH = Index("g"), Index("th")
T_GEOMETRIES = [(6, 12, 6, 3, 4, 3), (8, 16, 8, 4, 4, 2), (6, 12, 4, 2, 6, 3)]
T_IDS = ["6-12-6-3-4-3", "8-16-8-4-4-2", "6-12-4-2-6-3"]
T_PERIODS = (36, 64, 24)  # the core's iterations a frame
T_PORT_GROUPED = (False, False, True)
T_SETS = 2


@dataclass(frozen=True)
class Tiled:
    vectors: int
    depth: int
    outputs: int
    pe: int
    simd: int
    tile: int

    @property
    def core(self) -> Schedule:
        return Schedule(
            {G: self.vectors // self.tile, n: self.outputs, k: self.depth, TH: self.tile},
            {n: self.pe, k: self.simd},
            (G, n, k, TH),
        )

    @property
    def weights(self) -> Traversal:
        return self.core.present((self.depth, self.outputs), (k, n), lanes=(n, k))

    @property
    def port(self) -> Form:
        return grouped(unreplayed(self.weights), self.pe * self.simd // self.tile)

    @property
    def marker(self) -> LevelEnd | LevelDone:
        return self.core.closing((k,))

    @property
    def selected(self) -> tuple[Selected, Selected]:
        sink = Selected(BeatSequence(self.weights, markers=(self.marker,)), T_SETS, "which")
        source = Selected(BeatSequence(stored(self.port), Repetition.CYCLIC), T_SETS, "which")
        return source, sink

    def network(self, period: int) -> Network:
        source, sink = self.selected
        stages = unbounded(step_stages(steps(source, sink), "w"))
        activations = self.core.present(
            (self.vectors, self.depth), (G * self.tile + TH, k), lanes=(k,)
        )
        pace = Pace(self.core)
        names = ["mem", *(stage.name for stage in stages)]
        return Network(
            (
                Source("x", tuple(range(activations.beats)), activations.beats, period),
                Memory("mem"),
                *stages,
                kernel("core", (pace, pace), ()),
            ),
            (
                Channel("x", "x", "core"),
                *(Channel(f"{p}>{q}", p, q) for p, q in zip(names, names[1:])),
                Channel("w", names[-1], "core", input=1),
            ),
        )


T_CASES = [Tiled(*geometry) for geometry in T_GEOMETRIES]


@pytest.mark.parametrize("case", T_CASES, ids=T_IDS)
def test_the_tiled_weights_selected_with_the_core_s_marker_build_and_deliver(case: Tiled) -> None:
    """G, L and S in one channel: the port a lane group in the third geometry, the marker k's
    last iteration (a ``LevelDone``), the weights selected among two matrices. The chain is
    the width step, the reorder replaying each beat, and the marker step; each part alone
    builds."""
    assert isinstance(case.port, Grouped) is T_PORT_GROUPED[T_CASES.index(case)]
    assert isinstance(case.marker, LevelDone) and case.marker.aligned(case.weights)
    source, sink = case.selected
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH, Step.REORDER, Step.MARKERS)
    assert chain == steps(source.sequence, sink.sequence)
    # Each part alone: the sink's marker on its form, the steps without it.
    BeatSequence(case.weights, markers=(case.marker,))
    without = steps(source.sequence, BeatSequence(case.weights))
    assert chain.kinds[:-1] == without.kinds
    assert undelivered(chain, source.sequence, sink.sequence) == ""
    assert stacked_undelivered(chain, source, sink, (1, 0, 0, 1)) == ""
    assert cost(chain, sink.sequence) == composed_by_hand(chain, source.sequence, sink.sequence)
    assert cost(chain, sink.sequence) == cost(without, BeatSequence(case.weights))


@pytest.mark.parametrize("case", T_CASES, ids=T_IDS)
def test_the_tiled_weights_keep_the_core_s_period_in_the_network(case: Tiled) -> None:
    period = T_PERIODS[T_CASES.index(case)]
    assert case.core.iterations == period
    words = [found for _, found in case.network(10**6).words]
    assert max(words) <= period
    network = case.network(period)
    for _, rules in RULES:
        assert bottleneck(network, rules) == period


# --- the pairings that cannot occur together, and the marker the map moves --------------


def test_a_selected_reader_is_refused_because_a_selected_side_reads_its_sets_from_a_memory() -> (
    None
):
    """S with M cannot occur together: a reader is a ``PositionMap`` and a ``BeatSequence``,
    and a ``Selected`` is neither. A selected side reads its sets from a memory with random
    access to them; a movement node before that memory is folded into the order the memory
    stores, so no producer's stream passes a map into a selected port."""
    side = Selected(BeatSequence(vector_major((4, 6), 2)), 2, "sel")
    with pytest.raises(TypeError, match="^a reader is a PositionMap and a BeatSequence$"):
        Reader(maps.identity((4, 6)), side)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="^a producer presents a BeatSequence$"):
        Reindex(side, ())  # type: ignore[arg-type]


PAIRS = [
    ((4, 3), ("perm", (0, 1)), 1, ("perm", (1, 0)), 1),
    ((3, 4), ("perm", (0, 1)), 2, ("perm", (1, 0)), 2),  # the sink a lane group
    ((2, 3, 4), ("perm", (0, 1, 2)), 4, ("replay", (2, 1, 0)), 3),
]


@pytest.mark.parametrize("pair", PAIRS, ids=["4x3", "3x4-lane-group", "2x3x4-replay"])
def test_a_selected_side_joins_only_the_same_sets_through_the_same_selector_stream(
    pair: tuple[tuple[int, ...], End, int, End, int],
) -> None:
    """S's own refusals on a pair of forms of one tensor that join without selection: a side
    not selected against one that is, the same sets under another selector stream, other
    sets; each ``Unshared`` (``selection-not-shared``)."""
    shape, source_end, source_lanes, sink_end, sink_lanes = pair
    a = at_lanes(shape, source_end, source_lanes)
    b = at_lanes(shape, sink_end, sink_lanes)
    assert a is not None and b is not None
    source, sink = BeatSequence(a), BeatSequence(b)
    assert steps(source, sink)  # without selection the pair is joined by steps
    assert steps(Selected(source, 2, "sel"), Selected(sink, 2, "sel")) == steps(source, sink)
    refused: list[tuple[BeatSequence | Selected, BeatSequence | Selected, str]] = [
        (Selected(source, 2, "sel"), sink, "selected source joins only a selected pass"),
        (source, Selected(sink, 2, "sel"), "selected consumer joins only a selected pass"),
        (
            Selected(source, 2, "sel"),
            Selected(sink, 2, "other"),
            re.escape(f"{(2, *shape)} through 'sel' and one of {(2, *shape)} through 'other'"),
        ),
        (
            Selected(source, 2, "sel"),
            Selected(sink, 3, "sel"),
            re.escape(f"{(2, *shape)} through 'sel' and one of {(3, *shape)} through 'sel'"),
        ),
    ]
    for ends, other, message in refused:
        with pytest.raises(Unshared, match=message) as error:
            steps(ends, other)
        assert isinstance(error.value, Unrealizable)
        assert [found.code for found in error.value.findings] == ["selection-not-shared"]


def test_a_marker_the_map_moves_off_a_loop_is_refused_where_the_reader_s_form_takes_it() -> None:
    """A marker aligned on a reader's form need not be on its form composed through the
    map, though the beats are the same: the reader's marker that closes its merged loop of
    24 beats cuts a loop of the composed form (loops (6, 1), (4, 6) over the source). It is
    L's alignment on the composed form, raised under M's code, ``marker-unaligned``: a
    part's refusal, not a combination's. Both kinds of marker, the smallest of each."""
    pmap = maps.transpose((4, 6))
    for rule, message in (
        (LevelEnd(3), "a marker every 3 beats closes no loop level"),
        (level_done(6, 2), "a marker on the last 2 beats of every 6 beats closes no loop level"),
    ):
        reader = vector_major((6, 4), 1)
        BeatSequence(reader, markers=(rule,))  # the reader's own form takes it
        producer = BeatSequence(vector_major((4, 6), 1))
        plain = composed(Reindex(producer, (Reader(pmap, BeatSequence(reader)),)))
        with pytest.raises(ValueError, match=message) as part:  # the part alone
            BeatSequence(plain.form, markers=(rule,))
        with pytest.raises(Unrouted) as error:
            composed(Reindex(producer, (Reader(pmap, BeatSequence(reader, markers=(rule,))),)))
        assert outcome(error.value) == ("Unrouted", "marker-unaligned", str(part.value))


# --- the least reorder capacity, measured apart -----------------------------------------


def least_reorder_capacity(chain: Steps, rules: Rules) -> tuple[int, int]:
    """For a chain of one reorder (markers aside): the reorder's live window in words, and
    the least capacity, by a scan from one word up, keeping the period with every channel
    unbounded and the source released every ``max(words a frame)`` cycles."""
    (hop,) = (found for found in chain.hops if found.step is not Step.MARKERS)
    (stage,) = step_stages(Steps((hop,)), "r")
    assert isinstance(stage, ReorderStage)
    words = hop.source.form.beats
    period = max(words, hop.sink.form.beats)
    for capacity in range(1, words * FRAMES + 1):
        network = chained(
            (
                Source("src", tuple(range(words)), words, period),
                replace(stage, capacity=capacity),
                Sink("snk"),
            )
        )
        try:
            if bottleneck(network, rules) == period:
                return stage.capacity, capacity
        except ValueError:  # a capacity too small deadlocks
            continue
    raise AssertionError("no capacity keeps the period, though the unbounded reorder does")


def vector_chain(source: Traversal, sink: Traversal) -> Steps:
    return steps(BeatSequence(source), BeatSequence(sink))


# (live window in words, least capacity keeping the period) under FINNLIB and ONE_CYCLE: a
# reorder at its live window keeps the order, not the rate, and by more than one word.
LEAST = [
    (
        "a transpose of (2, 3) at one lane",
        vector_chain(vector_major((2, 3), 1), Traversal.over((2, 3), [(1, 3, 1), (0, 2, 1)], [])),
        ((3, 7), (3, 6)),
    ),
    (
        "a transpose of (3, 4) at one lane",
        vector_chain(vector_major((3, 4), 1), Traversal.over((3, 4), [(1, 4, 1), (0, 3, 1)], [])),
        ((8, 15), (8, 14)),
    ),
    (
        "a transpose of (4, 4) at one lane",
        vector_chain(vector_major((4, 4), 1), Traversal.over((4, 4), [(1, 4, 1), (0, 4, 1)], [])),
        ((12, 21), (12, 20)),
    ),
    (
        "a row of (4, 4) read twice at two lanes",
        vector_chain(vector_major((4, 4), 2), vector_major((4, 4), 2).replayed(2, inner_beats=2)),
        ((2, 3), (2, 2)),
    ),
    (
        "a row of (4, 6) read three times at two lanes",
        vector_chain(vector_major((4, 6), 2), vector_major((4, 6), 2).replayed(3, inner_beats=3)),
        ((3, 3), (3, 3)),
    ),
    (
        "a window of three taps over eight at one lane",
        vector_chain(vector_major((8,), 1), Traversal.over((8,), [(0, 6, 1), (0, 3, 1)], [])),
        ((2, 3), (2, 2)),
    ),
]


@pytest.mark.parametrize(
    ("chain", "least"),
    [(chain, least) for _, chain, least in LEAST],
    ids=[name for name, _, _ in LEAST],
)
def test_a_reorder_needs_more_than_its_live_window_to_keep_the_rate(
    chain: Steps, least: tuple[tuple[int, int], tuple[int, int]]
) -> None:
    """The least capacity keeping the period against the live window, by rule: a transpose
    needs the next frame's words while it presents the last of the current one, which one
    frame's buffer cannot hold. Measured by a scan; asserted on these named cases only."""
    assert tuple(least_reorder_capacity(chain, rules) for _, rules in RULES) == least


def test_the_least_reorder_capacity_of_sampled_reorders_is_measured_as_data(
    record_property: Callable[[str, object], None],
) -> None:
    """The least capacity, by rule, of 12 sampled chains of one reorder: recorded beside
    the reorder's live window, and never failed but for finding no capacity at all."""
    measured = 0
    for case in sample()[2]:
        found = attempt(case)
        if not isinstance(found, Built):
            continue
        moving = [hop for hop in found.chain.hops if hop.step is not Step.MARKERS]
        if case.selected or [hop.step for hop in moving] != [Step.REORDER]:
            continue
        for label, rules in RULES:
            window, least = least_reorder_capacity(found.chain, rules)
            assert 1 <= least <= moving[0].source.form.beats * FRAMES
            record_property(f"{case} {label}", f"live window {window}, least capacity {least}")
        measured += 1
        if measured == 12:
            break
    assert measured == 12
