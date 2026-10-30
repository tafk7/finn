# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What must happen between two beat sequences of one tensor: the steps between a
channel's two streams.

``steps(source, sink)`` compares what a producer presents with what a consumer
requires and returns the canonical chain of steps that turns one into the
other, each step between two beat sequences:

- ``REORDER``: a buffered loop-nest reorder, replay and sliding windows
  included (``classify``'s ``Reorder``: the frame, loop extents and strides of
  the nest);
- ``WIDTH``: the same element order, another number of lanes a beat;
- ``MARKERS``: the consumer's marker rules synthesized on an unchanged
  sequence.

A compound mismatch of lanes and beat order is one ``WIDTH`` step, or none, where at
one end's lanes or at a lane count dividing both sides' the two orders differ by a lane
permutation: wires at an end of the chain. Else it is ``WIDTH`` then ``REORDER``, or
``REORDER`` then ``WIDTH``, at one end's lanes; where neither holds, a lane regroup
decomposes as ``WIDTH``, ``REORDER``, ``WIDTH`` through the largest lane count dividing
both sides' at which the two orders differ by a reorder, tried from their greatest
common divisor down. At one lane a beat each beat is one element, so a regroup of the
same positions is refused only for an order no loop-nest reorder reaches; the narrower
the lane count, the slower the chain (its ``cost``). A lane permutation is taken before
a reorder at any count: it is wider than a chain through fewer lanes, and so faster,
though its one width step may hold more (the least common multiple of the two lane
counts) than the narrower chain's steps together. Every sequence step invalidates the
markers before it, so a marker the consumer requires is synthesized after the last one.
What stays the same sequence (a lane permutation, carrier padding) is a property of the
connection, realized as wires, and not a step; so is a marker closing every beat
(``LevelEnd.constant``), which every sequence carries. A chain's first step may start at
a lane permutation of what the source presents, and its last end at one of what the
consumer reads: those wires are the connection's. With no steps, the channel's two sides
connect directly.

The steps split at the channel's transport (direct wires or a FIFO): their
``output`` side, the width conversions before the first reorder or marker step,
is carried out on the producer's side of the transport; their ``input`` side, the
rest (a reorder, replay included, the markers, and a width conversion after a
reorder), on the consumer's. Markers are made after the transport, which
carries data words only.

The steps say what must happen, not which hardware does it: a channel's two
adapter Decisions, one a side, choose a realization, and each candidate refuses
steps it cannot carry out. ``Unrealizable`` names what no chain can repair:
positions the source does not present, an order no reorder reaches, or a single
pass feeding a cyclic consumer (``Unrepeatable``, its own kind: no step repeats a
pass).

A lane group (``traversal.Grouped``, a port whose lanes cut its order where no loop
ends) joins its neighbour by a ``WIDTH`` step only, to or from a ``Traversal`` of its
own order: reorders stay between traversals, which move whole beats of a loop nest.
That traversal's lanes are the other end's, where both orders are traversals there and
differ by a reorder, else the most lanes down to one at which they are.

``cost`` prices a chain (``Cost``: its live window, its reads and its narrowest lane
count), so that two chains of the same positions can be compared (``smaller``). Two
cyclic sources of the same positions are compared over four fields, the memory each
stores (``traversal.stored_depth``) beside their chains' three (``source_verdict``), and
``store`` chooses by them the form a cyclic source stores for a consumer.

A marker is a level closing (``LevelEnd``) or a level on its last index value (``LevelDone``);
either kind the consumer requires and the chain does not carry is synthesized after the
last sequence step, as one ``MARKERS`` step.

A selected side (``traversal.Selected``: one of several sets a pass, chosen by a selector
stream) joins only a selected pass of the same sets through the same selector stream, by
the steps inside one set; any other pairing is refused (``Unshared``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd, lcm
from typing import Literal

from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.dataflow.buffers import frame_slots, live_window
from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    Grouped,
    Reorder,
    Repetition,
    Selected,
    Traversal,
    classify,
    grouped,
    one_pass,
    passes,
    regrouped,
    require_positive,
    stored,
)


class Step(Enum):
    REORDER = "reorder"
    WIDTH = "width_conversion"
    MARKERS = "markers"


@dataclass(frozen=True)
class Hop:
    """One step between two beat sequences: ``source`` in, ``sink`` out."""

    step: Step
    source: BeatSequence
    sink: BeatSequence
    reorder: Reorder | None = None


@dataclass(frozen=True)
class Steps:
    """The steps between two beat sequences; empty when they connect directly."""

    hops: tuple[Hop, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hops", tuple(self.hops))

    @property
    def kinds(self) -> tuple[Step, ...]:
        """Each step's kind, in order."""
        return tuple(hop.step for hop in self.hops)

    def __bool__(self) -> bool:
        return bool(self.hops)

    def describe(self) -> str:
        return " -> ".join(step.value for step in self.kinds) or "direct"

    @property
    def output(self) -> Steps:
        """The producer's side of the transport: the width conversions before the first
        reorder or marker step."""
        return Steps(self.hops[: self._transport])

    @property
    def input(self) -> Steps:
        """The consumer's side of the transport: every step from the first reorder or
        marker step on."""
        return Steps(self.hops[self._transport :])

    @property
    def _transport(self) -> int:
        """Where the transport sits: before the first step that is no width conversion."""
        return next(
            (index for index, hop in enumerate(self.hops) if hop.step is not Step.WIDTH),
            len(self.hops),
        )


class Unrealizable(ValueError):
    """No chain of steps turns the source's beat sequence into the sink's."""


class Unrepeatable(Unrealizable):
    """A single pass feeding a cyclic consumer: no step repeats a pass."""


class Unshared(Unrealizable, Refusal):
    """A selected side (``traversal.Selected``) joined to anything but a selected pass of
    the same sets through the same selector stream: no step selects a set. One rejection,
    ``selection-not-shared`` by ``steps``."""

    code = "selection-not-shared"

    def __init__(self, message: str) -> None:
        Refusal.__init__(
            self, (Finding(FindingKind.REJECTION, self.code, "steps", message),), message
        )


def presented(source: BeatSequence, sink: BeatSequence) -> Traversal | Grouped:
    """What ``source`` presents over one frame of ``sink``: a cyclic source repeats.

    **A cyclic source presents its pass once for each pass the consumer makes over the
    tensor:** as many times as ``sink``'s form repeats its whole pass (``passes``: its
    outermost stride-0 loops), over the times the source's own form repeats its pass.
    What the consumer reads inside one of its passes is the chain's, not more passes of
    the source:

    - a position it reads again (its replay, a window's overlap) the chain's reorder
      holds from the one pass and reads again;
    - a position it never reads the reorder takes and drops.

    So consumer pass ``j`` reads only source pass ``j``, whatever its reads count. This
    is the boundary rule (``traversal.unreplayed``): the receiver realizes its replay and
    its window, and whole-pass repetition is part of the stream. It is why the source
    ``traversal.stored`` or ``traversal.one_pass`` derives for a consumer, one pass of what
    the consumer reads, serves it.

    The rule counts passes, not elements, and so differs from counting how many of the
    source's passes the consumer's reads add up to:

    - a window whose reads overlap (``(6,)`` in windows of 3 at stride 1: 12 reads of 6
      elements) reads one pass, not two;
    - a window that skips and reads twice (18 reads of a 24-element frame) reads one
      pass, where its elements are no whole number of passes;
    - a source presenting one pass of a consumer with replay (``one_pass``), for a
      consumer that also repeats its pass, presents it as often as the consumer repeats
      it, not as often as the consumer's reads without the replay fill it.

    Where the source presents positions a frame of the consumer never reads, and the consumer
    repeats its pass, counting elements may pack several consumer passes into one of the
    source's and re-read it from a larger buffer: a trade-off (fewer reads, more held)
    this rule does not take, since holding the consumer's repetition is the stream's.

    Where the source's own repetition divides no whole number of the consumer's passes,
    it is refused (``Unrealizable``).
    """
    if sink.repetition is Repetition.CYCLIC and source.repetition is not Repetition.CYCLIC:
        raise Unrepeatable("a single pass cannot feed a cyclic consumer")
    form = source.form
    if source.repetition is Repetition.ONCE or form.shape != sink.form.shape:
        return form
    wanted, given = passes(sink.form), passes(form)
    if wanted % given:
        raise Unrealizable("the consumer's pass is not whole repetitions of the cyclic source")
    count = wanted // given
    return form if count == 1 else form.repeated(count)


_Chain = list[tuple[Step, Traversal | Grouped, Reorder | None]]
_Joined = tuple[Traversal | Grouped, _Chain]
"""Where a chain's first step starts (what the source presents, or a lane permutation of
it), and its steps as (step, the form after it, its reorder)."""


def _through(
    produced: Traversal | Grouped,
    wanted: Traversal | Grouped,
    low: Traversal,
    target: Traversal,
    *,
    permuted: bool,
) -> _Joined | None:
    """The chain from ``produced`` to ``wanted`` through ``low`` (``produced``'s order at
    some lanes) and ``target`` (``wanted``'s at the same), where those two differ by a
    reorder, or, if ``permuted``, by a lane permutation; else None.

    A lane permutation is wires, no step, so it goes to an end of the chain. At ``L``
    lanes it permutes each block of ``L`` elements alike, so at any multiple of ``L``
    lanes the two orders differ by a lane permutation too, the same one in every beat:

    - where ``L`` divides the consumer's lanes, the chain is the width step to
      ``produced``'s order at the consumer's lanes (none where those are its own), and
      ends at a lane permutation of what the consumer reads;
    - else, where ``L`` divides the producer's lanes, the chain starts at ``wanted``'s
      order at the producer's lanes, a lane permutation of what the source presents, with
      the width step from it.
    """
    between = classify(low, target)
    if between.adaptation is Adaptation.REORDER:
        chain: _Chain = []
        if low != produced:
            chain.append((Step.WIDTH, low, None))
        chain.append((Step.REORDER, target, between.reorder))
        if target != wanted:
            chain.append((Step.WIDTH, wanted, None))
        return produced, chain
    if not permuted or between.adaptation is not Adaptation.LANE_PERMUTATION:
        return None
    if not wanted.lanes % low.lanes:
        end = grouped(produced, wanted.lanes)
        return produced, [] if end == produced else [(Step.WIDTH, end, None)]
    if not produced.lanes % low.lanes:
        start = grouped(wanted, produced.lanes)
        return start, [] if start == wanted else [(Step.WIDTH, wanted, None)]
    return None


def _sequence(produced: Traversal | Grouped, wanted: Traversal | Grouped) -> _Joined:
    """Canonical sequence steps, and where the first starts."""
    if isinstance(produced, Grouped) or isinstance(wanted, Grouped):
        return _grouped_sequence(produced, wanted)
    verdict = classify(produced, wanted)
    if verdict.adaptation in (Adaptation.IDENTITY, Adaptation.LANE_PERMUTATION):
        return produced, []
    if verdict.adaptation is Adaptation.REORDER:
        return produced, [(Step.REORDER, wanted, verdict.reorder)]
    if verdict.adaptation is Adaptation.WIDTH_CONVERSION:
        return produced, [(Step.WIDTH, wanted, None)]
    if produced.shape != wanted.shape:
        raise Unrealizable(f"{verdict.detail}: {produced.shape} and {wanted.shape}")
    # Lanes and order both differ. At one end's lanes: width first over the producer's
    # element order, then the consumer's lanes permuted or a reorder; or the producer's
    # lanes permuted or a reorder at its lanes, then width. Else a regroup: through the
    # largest common lane count at which the two orders differ by a reorder. At one lane
    # a beat each beat is one element, and the reorder moves elements. A lane
    # permutation at any of these counts (wires and one width step) is taken before a
    # reorder at any (a buffer besides).
    pairs: list[tuple[Traversal, Traversal]] = []
    for lanes, flip in ((wanted.lanes, False), (produced.lanes, True)):
        try:
            middle = regrouped(wanted if flip else produced, lanes)
        except ValueError:
            continue
        pairs.append((produced, middle) if flip else (middle, wanted))
    common = gcd(produced.lanes, wanted.lanes)
    pairs.extend(_common(produced, wanted, common))
    found = _first(produced, wanted, pairs)
    if found is None:
        raise Unrealizable(verdict.detail)
    return found


def _common(
    produced: Traversal | Grouped, wanted: Traversal | Grouped, widest: int
) -> list[tuple[Traversal, Traversal]]:
    """Both orders at each divisor of ``widest`` lanes, the largest first, where both are
    traversals there."""
    pairs = []
    for lanes in (count for count in range(widest, 0, -1) if widest % count == 0):
        try:
            pairs.append((regrouped(produced, lanes), regrouped(wanted, lanes)))
        except ValueError:
            continue
    return pairs


def _first(
    produced: Traversal | Grouped,
    wanted: Traversal | Grouped,
    pairs: list[tuple[Traversal, Traversal]],
) -> _Joined | None:
    """The chain through the first of ``pairs`` whose two orders differ by a lane
    permutation, else through the first that differ by a reorder (``_through``)."""
    for permuted in (True, False):
        for low, target in pairs:
            found = _through(produced, wanted, low, target, permuted=permuted)
            if found is not None:
                return found
    return None


def _grouped_sequence(produced: Traversal | Grouped, wanted: Traversal | Grouped) -> _Joined:
    """The steps where one end, or both, is a lane group: it joins only by a width step,
    to or from a ``Traversal`` of its own order. Reorders stay between traversals.

    Over one element order the chain is one width step. Otherwise it goes through the
    lane count at which both orders are traversals and differ by a reorder or a lane
    permutation (wires at an end, ``_through``): the other end's own lane count, where it
    is a traversal, else the largest of its divisors, down to one (between two lane
    groups, the divisors of their common divisor, as a regroup of two traversals). At one
    lane a beat each beat is one element, so the same positions are refused only for an
    order no loop-nest reorder reaches.
    """
    if produced.shape != wanted.shape:
        raise Unrealizable(f"different operand shapes: {produced.shape} and {wanted.shape}")
    if produced == wanted:
        return produced, []
    if regrouped(produced, 1) == regrouped(wanted, 1):
        return produced, [(Step.WIDTH, wanted, None)]
    if isinstance(wanted, Traversal):
        widest = wanted.lanes
    elif isinstance(produced, Traversal):
        widest = produced.lanes
    else:
        widest = gcd(produced.lanes, wanted.lanes)
    found = _first(produced, wanted, _common(produced, wanted, widest))
    if found is None:
        raise Unrealizable("the sequences present different positions")
    return found


def steps(source: BeatSequence | Selected, sink: BeatSequence | Selected) -> Steps:
    """The canonical steps from ``source``'s beat sequence to ``sink``'s.

    Two selected sides (``Selected``) of the same sets through the same selector stream
    (``Selected.shares``) join by the steps between their sequences, inside one set: the
    selection loop is outermost and shared, so each step runs once a selected pass. Any
    other pairing with a selected side is refused (``Unshared``): a selected source
    against a consumer that is not selected, or one of other sets, or the same sets under
    another selector stream.
    """
    if isinstance(source, Selected) or isinstance(sink, Selected):
        return _selected_steps(source, sink)
    produced = presented(source, sink)
    start, chain = _sequence(produced, sink.form)
    return _hops(start, chain, source, sink)


def _selected_steps(source: BeatSequence | Selected, sink: BeatSequence | Selected) -> Steps:
    if not isinstance(source, Selected) or not isinstance(sink, Selected):
        side = "source" if isinstance(source, Selected) else "consumer"
        raise Unshared(f"a selected {side} joins only a selected pass through its selector stream")
    if not source.shares(sink):
        raise Unshared(
            f"a selected pass of {source.shape} through {source.selector!r} and one of "
            f"{sink.shape} through {sink.selector!r} share no selection"
        )
    return steps(source.sequence, sink.sequence)


def _hops(
    start: Traversal | Grouped, chain: _Chain, source: BeatSequence, sink: BeatSequence
) -> Steps:
    """The chain's hops from ``start`` (what the source presents, or a lane permutation of
    it, which keeps its beats and so its markers), and the markers the consumer requires
    synthesized after the last."""
    hops: list[Hop] = []
    current = BeatSequence(start, markers=source.markers)
    for step, form, reorder in chain:
        after = BeatSequence(form)
        hops.append(Hop(step, current, after, reorder))
        current = after
    missing = tuple(
        rule for rule in sink.markers if rule not in current.markers and not rule.constant
    )
    if missing:
        marked = BeatSequence(current.form, markers=(*current.markers, *missing))
        hops.append(Hop(Step.MARKERS, current, marked))
    return Steps(tuple(hops))


@dataclass(frozen=True)
class Cost:
    """What a chain of steps costs, each field in its own unit:

    - ``live_window``, in **elements**: the sum of each reorder's live window in source
      beats times its lanes, and a width step's least common multiple of its two lane
      counts (a cost convention, not the timed fullest hold); markers cost nothing;
    - ``reads``, in **source beats**: the beats its source presents a frame of the consumer (a
      merge's, every source's summed: a count of positions read, beats of different
      widths, not a time; the merge's time is its stage's beat pattern in the network,
      ``reindex.merge_cost``);
    - ``narrowest_lanes``, in **lanes**: the fewest lanes a beat it carries anywhere, its
      two ends included (a chain moves at most that many elements a beat).

    A reorder's live window preserves order; it need not preserve the rate. Time and
    capacity at a required period are judged by the network under its stated rules.
    """

    live_window: int
    reads: int
    narrowest_lanes: int

    def __post_init__(self) -> None:
        if type(self.live_window) is not int or self.live_window < 0:
            raise ValueError("a live window is a nonnegative number of elements")
        require_positive(self.reads, "reads")
        require_positive(self.narrowest_lanes, "narrowest_lanes")


def hop_cost(hop: Hop) -> Cost:
    """One step's cost (units as in ``Cost``: ``live_window`` in elements, ``reads`` in
    source beats, ``narrowest_lanes`` in lanes).

    - Its live window, in elements: for a reorder, ``live_window`` of its frame's slots
      (``frame_slots``), in its source's beats, times their lanes; for a width
      conversion, the least common multiple of its two lane counts, the elements of
      the whole beats it gathers on either side; for markers, nothing.
    - Its reads, in source beats: the beats its source presents.
    - Its narrowest lane count, in lanes: the fewer of its source's and sink's lanes.
    """
    source, sink = hop.source.form, hop.sink.form
    window = 0
    if hop.step is Step.REORDER:
        if not isinstance(source, Traversal) or not isinstance(sink, Traversal):
            raise ValueError("a reorder step is between traversals: a lane group joins by width")
        reorder = hop.reorder or classify(source, sink).reorder
        if reorder is None:
            raise ValueError("a reorder step without a reorder between its two sides")
        window = live_window(frame_slots(reorder)) * source.lanes
    elif hop.step is Step.WIDTH:
        window = lcm(source.lanes, sink.lanes)
    return Cost(window, source.beats, min(source.lanes, sink.lanes))


def cost(chain: Steps, sink: BeatSequence) -> Cost:
    """The cost of ``chain``, the steps to ``sink``, from its steps' (``hop_cost``); the
    units are ``Cost``'s.

    - Its live window, in elements, is the sum of its steps': each step contributes its own
      cost.
    - Its reads, in source beats, are its first step's: the beats the channel's source
      presents a frame of the consumer.
    - Its narrowest lane count, in lanes, is the least of its steps'.

    With no steps the source presents what ``sink`` reads: no buffer, ``sink``'s beats
    and lanes.
    """
    if not chain:
        return Cost(0, sink.form.beats, sink.form.lanes)
    each = [hop_cost(hop) for hop in chain.hops]
    return Cost(
        sum(found.live_window for found in each),
        each[0].reads,
        min(found.narrowest_lanes for found in each),
    )


def smaller(first: Cost, second: Cost) -> bool:
    """Whether ``first`` is smaller than ``second``: no field worse (its live window and
    reads no larger, its narrowest lane count not lower) and one strictly better.

    This prices buffer (live window, in elements), reads (in source beats) and rate (the
    narrowest lane count), not area: a chain that is wider only (the same live window and
    reads, more lanes) is smaller, counted no worse, though a wider chain may take more
    area. It is faster for the same buffers. A trade-off (fewer reads for a larger window, or
    the reverse) is not, and neither is a narrower chain, however small its buffers: it
    is slower. Comparing two costs says nothing of positions; ``smaller`` is meant for two
    chains of the same.
    """
    return (
        first.live_window <= second.live_window
        and first.reads <= second.reads
        and first.narrowest_lanes >= second.narrowest_lanes
        and first != second
    )


SourceVerdict = Literal["same", "smaller", "larger", "trade-off"]
"""How one source and its chain compare with another's (``source_verdict``)."""


def source_verdict(depth: int, chain: Cost, other_depth: int, other_chain: Cost) -> SourceVerdict:
    """How a cyclic source storing ``depth`` elements (``traversal.stored_depth``) and its
    chain's cost ``chain`` compare with another source of the same positions, over four
    fields: the stored depth and ``Cost``'s three, never summed (they are in different
    units). ``"smaller"``: no field worse (depth, live window and reads no larger, the
    narrowest lane count not lower) and one strictly better; ``"larger"``, the reverse;
    ``"same"``, every field equal; otherwise ``"trade-off"``, as a smaller memory with a
    larger reorder buffer is."""
    for value in (depth, other_depth):
        if type(value) is not int or value < 1:
            raise ValueError("a stored depth is a positive number of elements")
    first = (depth, chain.live_window, chain.reads, -chain.narrowest_lanes)
    second = (other_depth, other_chain.live_window, other_chain.reads, -other_chain.narrowest_lanes)
    if first == second:
        return "same"
    if all(a <= b for a, b in zip(first, second)):
        return "smaller"
    if all(a >= b for a, b in zip(first, second)):
        return "larger"
    return "trade-off"


def store(sink: BeatSequence) -> Traversal | Grouped:
    """The form a cyclic source stores for a consumer reading ``sink``: ``stored(sink.form)``
    or ``one_pass(sink.form)``, chosen by ``source_verdict`` over the two.

    Each candidate is weighed as a cyclic source presenting it: its depth, its beats times
    its lanes, and the cost of its steps to ``sink``.

    - ``one_pass`` always builds, directly: presented once for each of the consumer's
      passes (``presented``) it is ``sink``'s form itself. So ``store`` refuses nothing.
    - A ``stored`` candidate whose steps are refused (``Unrealizable``), or that is
      refused itself (``ValueError``: a lane group whose lanes group no pass of it),
      loses.
    - Where both build, ``stored`` wins unless it is ``"larger"``: on a trade-off too,
      since a memory holds its depth for as long as it runs, where a chain's buffer is
      one channel's to size.

    ``stored`` keeps the consumer's window but not its replay, ``one_pass`` both. A window
    that both passes positions and reads some twice stores the frame, every position
    once, read through a reorder: more than ``one_pass`` holds where the window reads few
    of the frame's positions, so ``one_pass`` wins there.
    """
    whole = one_pass(sink.form)
    other_chain = cost(steps(BeatSequence(whole, Repetition.CYCLIC), sink), sink)
    try:
        small = stored(sink.form)
        chain = cost(steps(BeatSequence(small, Repetition.CYCLIC), sink), sink)
    except ValueError:
        return whole
    depth, other_depth = small.beats * small.lanes, whole.beats * whole.lanes
    if source_verdict(depth, chain, other_depth, other_chain) == "larger":
        return whole
    return small


__all__ = [
    "Cost",
    "SourceVerdict",
    "Hop",
    "Step",
    "Steps",
    "Unrealizable",
    "Unrepeatable",
    "Unshared",
    "cost",
    "hop_cost",
    "presented",
    "smaller",
    "source_verdict",
    "steps",
    "store",
]
