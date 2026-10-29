# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A reindex: one value from its producer's port to its readers' ports, inside a channel.

A ``Reindex`` holds facts only: the beat sequence the value's producer presents
(``producer``), and each of its ``readers``. A ``Reader`` is a position map from the
tensor the reader reads to the producer's (``map``; the movement nodes between them,
composed with ``maps.chain``, or ``maps.identity``) and the beat sequence the reader
requires over its own tensor (``stream``). What must happen between them is derived, not
stored:

- ``composed``: the reader's stream composed onto the tensor its steps reach
  (``maps.through``): the same elements in the same beats and lanes;
- ``route``: the reader's chain (``Route``): an optional insertion or drop at the
  producer's side, and the ``Steps`` around it;
- ``route_cost``: its cost, in ``steps.Cost``'s units.

**One reader.** With a strided map the route is ``steps(producer, composed)``, no more.
Through ``maps.identity`` that is exactly ``steps(producer, reader.stream)``: a reindex
asks nothing of the steps a channel derives today.

**A lane group.** A reader whose stream is a ``traversal.Grouped`` is composed by its
order: the order, one element a beat, through the map, then grouped again at its lanes
(``traversal.grouped``). Through ``maps.identity`` that is the stream itself, so the route
is ``steps(producer, reader.stream)``; through another strided map the composed form is a
``Traversal`` where the composed order closes whole loops at those lanes, a ``Grouped``
otherwise. A merge into a lane group is refused (``merge-grouped-reader``,
``MERGE_GROUPED_READER``) until a case needs it.

**Two or more readers are a fork.** Each reader is routed as it is alone (``branches``:
one reindex a reader), with its own steps and its own stage, and the fork sends each word
of the producer's stream to every branch at once, before any step (``Forked``, the
route of a fork). A fork holds no buffer; what one branch's stall does to another's is
the network's timing (``finn.dataflow.timing``: a ``Fork`` stage, each branch a channel),
where a fork that rejoins is sized. A derivation of one reader's (``composed``,
``stage_cost``) asked of a fork is refused (``fork-per-branch``, ``FORK_PER_BRANCH``): it
is each branch's. Branches sharing their first steps (one reorder, then the fork) are
not derived.

**Pad and Slice.** A map with fills composed with a window cannot be split back into the
two without a search, so the reader states the movement node at the producer's side
apart (``Reader.at_producer``): a Pad's map (fills) or a Slice's (a drop), from the
tensor ``map`` reads to the producer's. It is realized by its factorization
(``maps.factor_fills``, ``maps.factor_drop``): an in-order ``Insertion`` or ``Drop``,
which holds no buffer, then ``map``, which must be strided. The stage runs at the widest
lane count dividing the producer's at which it inserts or drops whole beats, on the
tensor in row-major order; the route's ``before`` steps bring the producer there (a width
step where the stage runs narrower, a reorder where the producer is not row-major), and
its ``after`` steps take the stage's output to the composed reader. Any other map that
is not strided is refused by ``maps.through`` with its code.

**Concat's merge.** A reader whose map has several sources (``maps.concat``) reads several
values: one reindex a source, each naming its source (``Reader.source``), joined at the
reader by a ``Merge``. Each reindex may state its own stage (``Reader.at_producer``, a
Pad's or a Slice's map from that source's tensor to its producer's), routed before the
merge as for one reader: a Pad of one source before a Concat is stated so. It is not
stated as a fill in the merge's own map (``chain(pad, concat, 0)`` read by one reader):
a merge's reader states each source's stage, it does not reach one through a fill, and
that map is refused (``merge-reads-a-fill``). The merge takes each beat whole from one
source, at its lanes (``merge_lanes``): the reader's own lane count where every one of its
beats reads one source, otherwise the widest count dividing it at which every beat does.
That narrower merge is exact: the merged stream is the reader's element order, so the
width step after it (``merged``) restores the reader's lanes, and the rate it costs shows
in the narrowest lane count. Each source's reindex is routed to the reader's beats from
that source, in order (``composed``), and the reader's markers are made after the merge.
``pattern`` names the source of each merged beat, as the shortest run that repeats through
the pass: along the channels at 8 + 4 channels and 4 lanes, two beats of the first source,
then one of the second, pixel by pixel.

**Insertion, drop and merge are no ``steps.Step``.** A step's kind says how it is
realized, and these are realized as none of them; they are facts of a route or a merge
beside its ``Steps``, each priced in ``Cost``'s three units. An insertion or a drop holds
no buffer, reads the beats its input presents (the producer's, at its lanes) and moves
its lanes a beat; a merge holds no buffer, reads the merged beats and moves its lanes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from math import prod
from typing import NamedTuple

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow_next.maps import (
    Drop,
    Insertion,
    PositionMap,
    factor_drop,
    factor_fills,
    reads,
    strided_modes,
    through,
)
from finn.dataflow_next.steps import Cost, Steps, cost, presented, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    Loop,
    Traversal,
    grouped,
    offsets,
    regrouped,
)

FORK_PER_BRANCH = "fork-per-branch"
"""The finding code for a derivation of one reader's (its composed stream, its stage's
cost) asked of a fork: each branch has its own (``branches``)."""

MERGE_GROUPED_READER = "merge-grouped-reader"
"""The finding code for a merge whose reader's stream is a lane group (``Grouped``):
refused until a case needs it."""


class Unrouted(Refusal):
    """A reindex, a merge or a route that cannot be stated or derived: one rejection,
    ``code`` by ``owner``, its message the error's."""

    def __init__(self, code: str, owner: str, message: str) -> None:
        super().__init__((Finding(FindingKind.REJECTION, code, owner, message),), message)


@dataclass(frozen=True)
class Reader:
    """One reader of a value: the position map from the tensor it reads to the producer's,
    and the beat sequence it requires over its own tensor.

    ``map`` reads ``stream``'s tensor (``map.shape``). ``source`` is which of ``map``'s
    sources the value is (a merge's reader reads several). ``at_producer``, where given,
    is a Pad's or a Slice's map at the producer's side: from the tensor ``map`` reads of
    this source (``map.sources[source]``) to the producer's.
    """

    map: PositionMap
    stream: BeatSequence
    at_producer: PositionMap | None = None
    source: int = 0

    def __post_init__(self) -> None:
        owner = "Reader"
        if not isinstance(self.map, PositionMap) or not isinstance(self.stream, BeatSequence):
            raise TypeError("a reader is a PositionMap and a BeatSequence")
        if self.at_producer is not None and not isinstance(self.at_producer, PositionMap):
            raise TypeError("a reader's map at the producer's side is a PositionMap")
        if self.stream.form.shape != self.map.shape:
            raise Unrouted(
                "shape-mismatch",
                owner,
                f"the stream walks {self.stream.form.shape}, the map reads {self.map.shape}",
            )
        if type(self.source) is not int or not 0 <= self.source < len(self.map.sources):
            raise Unrouted(
                "source-unknown", owner, f"source {self.source} of {len(self.map.sources)}"
            )
        if self.at_producer is not None:
            if len(self.at_producer.sources) != 1:
                raise Unrouted(
                    "stage-sources",
                    owner,
                    "an insertion or a drop at the producer's side reads one value",
                )
            if self.at_producer.shape != self.map.sources[self.source]:
                raise Unrouted(
                    "shape-mismatch",
                    owner,
                    f"the map reads {self.map.sources[self.source]} of source {self.source}, "
                    f"the map at the producer's side gives {self.at_producer.shape}",
                )

    @property
    def tensor(self) -> tuple[int, ...]:
        """The shape of the producer's tensor this reader reads."""
        if self.at_producer is not None:
            return self.at_producer.sources[0]
        return self.map.sources[self.source]


@dataclass(frozen=True)
class Reindex:
    """One value: the beat sequence its producer presents, and its readers (one, or a
    fork). Each reader reads the producer's tensor."""

    producer: BeatSequence
    readers: tuple[Reader, ...]

    def __post_init__(self) -> None:
        owner = "Reindex"
        readers = tuple(self.readers)
        if not isinstance(self.producer, BeatSequence):
            raise TypeError("a producer presents a BeatSequence")
        if not all(isinstance(reader, Reader) for reader in readers):
            raise TypeError("a value's readers are Reader values")
        if not readers:
            raise Unrouted("no-reader", owner, "a value is read by at least one reader")
        for reader in readers:
            if reader.tensor != self.producer.form.shape:
                raise Unrouted(
                    "shape-mismatch",
                    owner,
                    f"the producer presents {self.producer.form.shape}, "
                    f"a reader reads {reader.tensor}",
                )
        object.__setattr__(self, "readers", readers)


@dataclass(frozen=True)
class Merge:
    """Several values joined at one reader whose map reads them all: ``reindexes[i]`` is
    the value of the map's source ``i``, and each has that reader (the same map and
    stream, ``source`` ``i``), which may state its own stage (``Reader.at_producer``)."""

    reindexes: tuple[Reindex, ...]

    def __post_init__(self) -> None:
        reindexes = tuple(self.reindexes)
        if not all(isinstance(reindex, Reindex) for reindex in reindexes):
            raise TypeError("a merge joins Reindex values")
        object.__setattr__(self, "reindexes", reindexes)
        _joined(reindexes)


@dataclass(frozen=True)
class Route:
    """One reader's derived chain, producer first: ``before``, the steps that bring the
    producer's stream to the stage's input; ``stage``, the in-order insertion or drop at
    the producer's side, if any; ``after``, the steps from the stage's output (the
    producer's stream, without a stage) to the composed reader.

    The stage moves its input's lanes a beat: its input is ``before``'s last sink, or
    the producer's stream where ``before`` is empty; its output is that tensor's
    row-major order at the same lanes, the fills inserted or the positions dropped.
    """

    before: Steps
    stage: Insertion | Drop | None
    after: Steps


@dataclass(frozen=True)
class Forked:
    """The route of a value read by two or more readers: a fork at the producer's stream,
    before any step, sending each word to every branch at once, and each reader's own
    route behind it (``routes``, in the readers' order; ``branches``). The fork holds no
    buffer: each branch's capacity is a channel of the network's timing."""

    routes: tuple[Route, ...]

    def __post_init__(self) -> None:
        routes = tuple(self.routes)
        if len(routes) < 2 or not all(isinstance(found, Route) for found in routes):
            raise TypeError("a fork routes two readers or more, each a Route")
        object.__setattr__(self, "routes", routes)


# --- one reader ------------------------------------------------------------------------


def branches(reindex: Reindex) -> tuple[Reindex, ...]:
    """Each reader of the value as a reindex of its own: one reader is its own branch;
    a fork's readers are routed, composed and priced each in its branch."""
    return tuple(Reindex(reindex.producer, (reader,)) for reader in reindex.readers)


def _one(reindex: Reindex, owner: str) -> Reader:
    """The reindex's one reader; ``fork-per-branch`` where it has two or more."""
    if len(reindex.readers) > 1:
        raise Unrouted(
            FORK_PER_BRANCH,
            owner,
            f"a value read by {len(reindex.readers)} readers is a fork: {owner} is each "
            "branch's (branches)",
        )
    return reindex.readers[0]


def _factored(reader: Reader) -> tuple[Insertion | Drop, PositionMap]:
    """The reader's map at the producer's side as its in-order stage, and its strided map."""
    edge = reader.at_producer
    assert edge is not None
    if edge.fill is not None:
        return factor_fills(edge, reader.map)
    return factor_drop(edge, reader.map)


def _stage(reader: Reader) -> Insertion | Drop:
    """The reader's in-order stage. A merge's reader has no strided map to factor: its
    stage is its source's own, and the beats it reads of that source's tensor are checked
    in ``composed``."""
    if len(reader.map.sources) == 1:
        return _factored(reader)[0]
    edge = reader.at_producer
    assert edge is not None
    return Insertion(edge) if edge.fill is not None else Drop(edge)


def composed(reindex: Reindex) -> BeatSequence:
    """The one reader's stream composed onto the tensor its steps reach: the producer's,
    or, with an insertion or a drop, the stage's output. The same elements in the same
    beats and lanes, its repetition and markers kept. For a merge's reader, the reader's
    beats from this reindex's source, in order, at the merge's lanes (``merge_lanes``),
    with no markers: those are made after the merge. A fork's are its branches'
    (``fork-per-branch``)."""
    reader = _one(reindex, "composed")
    if len(reader.map.sources) > 1:
        lanes = merge_lanes(reader)
        return BeatSequence(_from_source(reader, lanes), reader.stream.repetition)
    window = reader.map if reader.at_producer is None else _factored(reader)[1]
    form = _through(window, reader.stream.form)
    try:
        return BeatSequence(form, reader.stream.repetition, reader.stream.markers)
    except ValueError as error:
        raise Unrouted("marker-unaligned", "composed", str(error)) from error


def _through(window: PositionMap, form: Traversal | Grouped) -> Traversal | Grouped:
    """``form`` composed through the strided ``window`` (``maps.through``). A lane group's
    order is composed at one lane a beat, then grouped again at its lanes (``grouped``):
    the same elements in the same beats and lanes, a ``Traversal`` where the composed
    order closes whole loops at them."""
    if isinstance(form, Traversal):
        return through(window, form)
    return grouped(through(window, Traversal(form.shape, form.order, ())), form.lanes)


def _row_major(shape: tuple[int, ...], lanes: int) -> Traversal:
    """``shape``'s positions once each in row-major order, ``lanes`` a beat."""
    size = prod(shape)
    return Traversal(shape, (Loop(size // lanes, lanes),), (Loop(lanes, 1),))


def _marks(stage: Insertion | Drop) -> npt.NDArray[np.bool_]:
    """For each position the stage passes over, in row-major order: whether it moves no
    producer element (a fill inserted, or a position dropped)."""
    return stage.inserted() if isinstance(stage, Insertion) else stage.dropped()


def stage_lanes(stage: Insertion | Drop, lanes: int) -> int:
    """The widest lane count dividing ``lanes`` (the producer's) at which ``stage``
    inserts or drops whole beats: every beat of the tensor it passes over, in row-major
    order, all moved or all not. One lane always does."""
    marks = _marks(stage)
    for count in range(lanes, 0, -1):
        if lanes % count or len(marks) % count:
            continue
        beats = marks.reshape(-1, count)
        if (beats == beats[:, :1]).all():
            return count
    raise AssertionError("one lane a beat moves whole beats")  # pragma: no cover


def stage_beats(stage: Insertion | Drop, lanes: int) -> tuple[bool, ...]:
    """For each beat ``stage`` passes over at ``lanes`` lanes (a count at which it moves
    whole beats, ``stage_lanes``), in row-major order: whether it moves a word of its
    input (a beat of the producer's, not a fill inserted or a position dropped)."""
    marks = _marks(stage)
    if len(marks) % lanes:
        raise ValueError(f"{len(marks)} positions are no whole beats of {lanes} lanes")
    beats = marks.reshape(-1, lanes)
    if not (beats == beats[:, :1]).all():
        raise ValueError(f"the stage does not insert or drop whole beats of {lanes} lanes")
    return tuple(not bool(mark) for mark in beats[:, 0])


def _stage_ends(stage: Insertion | Drop) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The shapes of the tensor the stage reads (the producer's) and the one it gives."""
    if isinstance(stage, Insertion):
        return stage.fills.sources[0], stage.shape
    return stage.shape, stage.kept.shape


class _Routed(NamedTuple):
    route: Route
    stage_input: BeatSequence | None
    stage_output: BeatSequence | None
    sink: BeatSequence


def _routed(reindex: Reindex, owner: str = "route") -> _Routed:
    reader = _one(reindex, owner)
    sink = composed(reindex)
    producer = reindex.producer
    if reader.at_producer is None:
        return _Routed(Route(Steps(), None, steps(producer, sink)), None, None, sink)
    stage = _stage(reader)
    lanes = stage_lanes(stage, producer.form.lanes)
    reads_shape, gives_shape = _stage_ends(stage)
    stage_input = BeatSequence(_row_major(reads_shape, lanes), producer.repetition)
    stage_output = BeatSequence(_row_major(gives_shape, lanes), producer.repetition)
    route = Route(steps(producer, stage_input), stage, steps(stage_output, sink))
    return _Routed(route, stage_input, stage_output, sink)


def route(reindex: Reindex) -> Route | Forked:
    """The derived chain: one reader's ``Route`` (without an insertion or a drop,
    ``steps(producer, composed(reindex))`` alone); for a fork, each reader's as it is
    alone (``branches``), in a ``Forked``. A fork whose branches read different numbers
    of beats of the producer is routed all the same: ``route_cost`` refuses it
    (``fork-reads-differ``), as the one place where its branches' reads are compared."""
    if len(reindex.readers) > 1:
        return Forked(tuple(_routed(branch).route for branch in branches(reindex)))
    return _routed(reindex).route


def stage_cost(reindex: Reindex) -> Cost | None:
    """The cost of the route's insertion or drop, or None without one: no buffer; the
    beats its input presents a frame of the reader, in source beats (the producer's, at its
    lanes, as many passes as the reader takes of its output); its lanes. A fork's are
    its branches' (``fork-per-branch``)."""
    return _stage_cost(_routed(reindex, "stage_cost"))


def _stage_cost(found: _Routed) -> Cost | None:
    if found.stage_input is None or found.stage_output is None:
        return None
    passes = presented(found.stage_output, found.sink).beats // found.stage_output.form.beats
    form = found.stage_input.form
    return Cost(0, form.beats * passes, form.lanes)


def route_cost(reindex: Reindex) -> Cost:
    """The cost of the route, in ``Cost``'s units.

    One reader's: the live windows of its steps and stage summed (in elements); the reads
    of its first part (in source beats: ``before``'s, or the stage's, or ``after``'s); the
    fewest lanes anywhere on it. Without an insertion or a drop it is ``cost(steps,
    composed)``.

    A fork's: its branches' live windows summed (each branch holds its own buffers, and
    the fork none); its reads the producer's beats once, since the fork reads each word
    once for every branch (``fork-reads-differ`` where two branches read different
    numbers of beats, as two counts of passes of a cyclic producer: one stream cannot
    serve both); the fewest lanes of any branch."""
    if len(reindex.readers) > 1:
        each = [route_cost(branch) for branch in branches(reindex)]
        if len({found.reads for found in each}) != 1:
            raise Unrouted(
                "fork-reads-differ",
                "route_cost",
                f"a fork sends each word to every branch, and its branches read "
                f"{[found.reads for found in each]} beats of the producer",
            )
        return Cost(
            sum(found.live_window for found in each),
            each[0].reads,
            min(found.narrowest_lanes for found in each),
        )
    found = _routed(reindex)
    after = cost(found.route.after, found.sink)
    if found.stage_input is None:
        return after
    staged = _stage_cost(found)
    assert staged is not None
    passes = staged.reads // found.stage_input.form.beats
    parts = [staged, after]
    if found.route.before:
        before = cost(found.route.before, found.stage_input)
        parts.insert(0, Cost(before.live_window, before.reads * passes, before.narrowest_lanes))
    return Cost(
        sum(part.live_window for part in parts),
        parts[0].reads,
        min(part.narrowest_lanes for part in parts),
    )


# --- the merge -------------------------------------------------------------------------


def _joined(reindexes: tuple[Reindex, ...]) -> Reader:
    owner = "Merge"
    count = len(reindexes)
    if count < 2:
        raise Unrouted("merge-too-few", owner, "a merge joins two values or more")
    found = {
        (reader.map, reader.stream)
        for reader in reindexes[0].readers
        if reader.source == 0 and len(reader.map.sources) == count
    }
    joined = [
        (pmap, stream)
        for pmap, stream in found
        if all(
            any(
                (reader.map, reader.stream, reader.source) == (pmap, stream, index)
                for reader in reindex.readers
            )
            for index, reindex in enumerate(reindexes)
        )
    ]
    if len(joined) != 1:
        raise Unrouted(
            "merge-reader",
            owner,
            f"the {count} values share no reader whose map reads them all, "
            "each as the source of its place",
        )
    return Reader(*joined[0])


def _into(reindex: Reindex, reader: Reader, source: int) -> Reindex:
    """The branch of ``reindex`` to the merge's ``reader``, as the value of its ``source``."""
    for branch in branches(reindex):
        mine = branch.readers[0]
        if (mine.map, mine.stream, mine.source) == (reader.map, reader.stream, source):
            return branch
    raise AssertionError("a merge's value has its reader")  # pragma: no cover


def joined(merge: Merge) -> Reader:
    """The reader the merge joins its values at (as its first source's reader)."""
    return _joined(merge.reindexes)


def _merge_form(reader: Reader) -> Traversal:
    """The merge's reader's form; a lane group is refused (``merge-grouped-reader``)."""
    form = reader.stream.form
    if isinstance(form, Grouped):
        raise Unrouted(
            MERGE_GROUPED_READER,
            "merge",
            "a merge into a lane group (Grouped) is not derived until a case needs it",
        )
    return form


def _beat_sources(reader: Reader, lanes: int) -> npt.NDArray[np.int64]:
    """The source each element of the reader's stream at ``lanes`` lanes reads, one row a
    beat (-1 for a fill)."""
    label = reads(reader.map)[0]
    return label[offsets(regrouped(_merge_form(reader), lanes))].reshape(-1, lanes)


def merge_lanes(reader: Reader) -> int:
    """The lanes a merge into ``reader`` runs at: the widest count dividing the reader's
    at which every beat of its stream reads one source. One lane always does; a fill read
    (no source) is refused (``merge-reads-a-fill``), and so is a lane group
    (``merge-grouped-reader``)."""
    form = _merge_form(reader)
    if (reads(reader.map)[0][offsets(form)] < 0).any():
        raise Unrouted(
            "merge-reads-a-fill",
            "merge_lanes",
            "a merge takes every beat from one of its values; this reader reads a fill",
        )
    for count in range(form.lanes, 0, -1):
        if form.lanes % count:
            continue
        try:
            beats = _beat_sources(reader, count)
        except ValueError:
            continue
        if (beats == beats[:, :1]).all():
            return count
    raise AssertionError("one lane a beat reads one source")  # pragma: no cover


def _from_source(reader: Reader, lanes: int) -> Traversal:
    """The reader's beats at ``lanes`` lanes that read its source ``reader.source``, in
    order, as a traversal of that source."""
    owner = "composed"
    form = regrouped(_merge_form(reader), lanes)
    label, offset = reads(reader.map)
    at = offsets(form).reshape(-1, lanes)
    mine = label[at][:, 0] == reader.source
    rows = offset[at][mine]
    shape = reader.map.sources[reader.source]
    if not len(rows):
        raise Unrouted("source-unread", owner, f"the reader reads nothing of source {shape}")
    lane = rows[0] - rows[0, 0]
    beats = strided_modes(rows[:, 0])
    lane_modes = strided_modes(lane)
    if (
        beats is None
        or lane_modes is None
        or beats[0] != 0
        or not (rows - rows[:, :1] == lane).all()
    ):
        raise Unrouted(
            "merge-order-no-loop-nest",
            owner,
            f"the reader's beats from source {shape} are no loop nest over it",
        )
    found = Traversal(shape, beats[1], lane_modes[1])
    if not np.array_equal(offsets(found), rows.ravel()):
        raise Unrouted(  # pragma: no cover - the factoring is exact
            "merge-order-no-loop-nest", owner, "the factored order differs from the reads"
        )
    return found


def pattern(merge: Merge) -> tuple[int, ...]:
    """The source each merged beat comes from, as the shortest run that repeats through
    the reader's pass (at ``merge_lanes``)."""
    reader = joined(merge)
    beats = _beat_sources(reader, merge_lanes(reader))[:, 0]
    total = len(beats)
    for length in range(1, total + 1):
        if total % length == 0 and np.array_equal(beats, np.tile(beats[:length], total // length)):
            return tuple(int(source) for source in beats[:length])
    raise AssertionError("the whole pass repeats itself")  # pragma: no cover


def _merged_stream(reader: Reader) -> BeatSequence:
    """What the merge gives: the reader's elements in its order, at the merge's lanes."""
    form = regrouped(_merge_form(reader), merge_lanes(reader))
    return BeatSequence(form, reader.stream.repetition)


def merged(merge: Merge) -> Steps:
    """The steps after the merge, to the reader: a width step where the merge runs
    narrower than the reader, then the reader's markers. Empty where the merge gives the
    reader's stream."""
    reader = joined(merge)
    return steps(_merged_stream(reader), reader.stream)


def merge_stage_cost(merge: Merge) -> Cost:
    """The merge's own cost: no buffer; the merged beats a frame of the reader, in source beats;
    the merge's lanes."""
    stream = _merged_stream(joined(merge))
    return Cost(0, stream.form.beats, stream.form.lanes)


def merge_cost(merge: Merge) -> Cost:
    """The cost of the whole merge: each value's route to the merge (``route_cost``), the
    merge (``merge_stage_cost``) and the steps after it (``merged``). Live windows
    summed, in elements; the fewest lanes anywhere.

    Its reads are every value's route's reads summed, in source beats: a count of
    positions read, beats of each source at that source's width, not a time. Sources of
    different widths sum beats of different sizes, and a merge's sources are read in turn,
    not at once: the merge's time is its stage's beat pattern in the network
    (``finn.dataflow.timing.merge_stage``), each source read on the beats ``pattern``
    names. A value also read elsewhere (a fork) is priced by its branch to the merge's
    reader; its other branches are their readers'."""
    reader = joined(merge)
    parts = [
        route_cost(_into(reindex, reader, source)) for source, reindex in enumerate(merge.reindexes)
    ]
    reads_total = sum(part.reads for part in parts)
    parts.append(merge_stage_cost(merge))
    after = merged(merge)
    if after:
        parts.append(cost(after, reader.stream))
    return Cost(
        sum(part.live_window for part in parts),
        reads_total,
        min(part.narrowest_lanes for part in parts),
    )


__all__ = [
    "FORK_PER_BRANCH",
    "MERGE_GROUPED_READER",
    "Forked",
    "Merge",
    "Reindex",
    "Reader",
    "Route",
    "Unrouted",
    "branches",
    "composed",
    "joined",
    "merge_cost",
    "merge_lanes",
    "merge_stage_cost",
    "merged",
    "pattern",
    "route",
    "route_cost",
    "stage_beats",
    "stage_cost",
    "stage_lanes",
]
