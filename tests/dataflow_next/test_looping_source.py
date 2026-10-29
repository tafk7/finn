# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A looping source's count (``steps.presented``): a cyclic source is presented once for
each pass its consumer makes over the tensor, and the chain's reorder holds what the
consumer reads again inside a pass and drops what it never reads.

The rule, stated in ``presented``: consumer pass ``j`` (one presentation of the
consumer's ``one_pass``) reads only source pass ``j``. Each chain here is checked by
enumeration twice:

- ``undelivered``: the chain delivers, hop by hop, the positions the consumer reads in
  its order;
- ``source_passes``: every element of the consumer's pass ``j``, traced back through the
  chain's width steps and reorders, comes from the ``j``-th presentation of the source's
  pass.

The rule counts passes, not elements. Counting elements (how many of the source's passes
the consumer's reads add up to) is what ``finn.dataflow.steps.presented`` does; the last
test enumerates small consumers against it: where it builds, the rule builds, at the same
cost or a smaller one, except where the source presents positions a consumer pass never
reads and the consumer repeats its pass. There counting elements packs several consumer
passes into one of the source's and holds them in a larger buffer: a trade-off (fewer
reads, more held) the rule does not take.
"""

from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
from collections import Counter
from collections.abc import Sequence
from dataflow_next.harness.checks import undelivered

from finn.dataflow_next.buffers import frame_slots
from finn.dataflow_next.gemm import k, n
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import (
    Cost,
    Step,
    Steps,
    Unrealizable,
    cost,
    presented,
    smaller,
    steps,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    offsets,
    one_pass,
    passes,
    regrouped,
    stored,
    unreplayed,
    vector_major,
)

Array = npt.NDArray[np.int64]


def traced(chain: Steps, produced: Traversal | Grouped) -> Array:
    """For each element the chain's last step presents, in order, the index of the element
    of ``produced`` (what the source presents over the consumer's pass) it carries: a
    width step regroups the elements, a reorder emits its frames' beats."""
    beats = np.arange(produced.beats * produced.lanes, dtype=np.int64).reshape(
        produced.beats, produced.lanes
    )
    for hop in chain.hops:
        if hop.step is Step.WIDTH:
            beats = beats.reshape(-1, hop.sink.form.lanes)
        elif hop.step is Step.REORDER:
            assert hop.reorder is not None
            frame = hop.reorder.frame_beats
            frames = np.arange(len(beats) // frame, dtype=np.int64)[:, None] * frame
            beats = beats[(frames + frame_slots(hop.reorder)).ravel()]
    return beats.ravel()


def source_passes(chain: Steps, source: BeatSequence, sink: BeatSequence) -> Array:
    """For each of the consumer's passes (a row), the source pass each of its elements
    comes from, by enumeration through the chain."""
    form = source.form
    each = form.beats * form.lanes
    taken = traced(chain, presented(source, sink)) // each
    return taken.reshape(passes(sink.form), -1)


def one_pass_each(chain: Steps, source: BeatSequence, sink: BeatSequence) -> bool:
    """Whether consumer pass ``j`` reads only source pass ``j``, for every ``j``."""
    found = source_passes(chain, source, sink)
    return bool((found == np.arange(len(found))[:, None]).all())


def checked(given: Traversal | Grouped, form: Traversal | Grouped) -> tuple[Steps, Cost]:
    """The steps from ``given``, presented cyclically, to a consumer of ``form``, checked
    by enumeration both ways; and their cost."""
    source, sink = BeatSequence(given, Repetition.CYCLIC), BeatSequence(form)
    chain = steps(source, sink)
    assert undelivered(chain, source, sink) == ""
    assert one_pass_each(chain, source, sink)
    return chain, cost(chain, sink)


# -- the cases the rule was written for ----------------------------------------------------


def test_an_overlapping_window_reads_one_pass_of_its_frame() -> None:
    """(6,) at one lane through 4 windows of 3 at stride 1: 12 reads, two passes' worth of
    the 6-element frame, read from one pass. One reorder (frame 6, dims (4, 3), coefs
    (1, 1)) holds what the windows read again: cost (2, 6, 1), where counting the 12
    reads as two passes read 12 source beats for the same live window. Repeated three
    times, the consumer reads three passes, each its own."""
    form = Traversal((6,), (Loop(4, 1), Loop(3, 1)), ())
    frame = stored(form)
    assert frame == vector_major((6,), 1)
    assert form.beats == 2 * frame.beats
    source = BeatSequence(frame, Repetition.CYCLIC)
    assert presented(source, BeatSequence(form)) == frame
    chain, priced = checked(frame, form)
    assert [hop.reorder for hop in chain.hops] == [Reorder(6, (4, 3), (1, 1))]
    assert priced == Cost(2, 6, 1)
    thrice, priced = checked(frame, form.repeated(3))
    assert presented(source, BeatSequence(form.repeated(3))) == frame.repeated(3)
    assert [hop.reorder for hop in thrice.hops] == [Reorder(6, (4, 3), (1, 1))]
    assert priced == Cost(2, 18, 1)
    assert source_passes(thrice, source, BeatSequence(form.repeated(3))).tolist() == [
        [0] * 12,
        [1] * 12,
        [2] * 12,
    ]


def test_a_window_that_skips_and_reads_twice_reads_one_pass_of_its_frame() -> None:
    """(4, 6) at one lane: rows 0 and 2, each through 3 windows of 3 at stride 1 over its
    first 5 columns. 18 reads of the 24-element frame (``stored``), no whole number of
    its passes: one pass, which one reorder over two rows at a time (frame 12, dims
    (3, 3), coefs (1, 1)) reads the first row's windows out of, dropping the second row;
    cost (2, 24, 1). Elements 0 to 4 and 12 to 16 of the frame are read, columns 1 to 3
    of each row twice."""
    form = Traversal((4, 6), (Loop(2, 12), Loop(3, 1), Loop(3, 1)), ())
    frame = stored(form)
    assert frame == vector_major((4, 6), 1)
    chain, priced = checked(frame, form)
    assert [hop.reorder for hop in chain.hops] == [Reorder(12, (3, 3), (1, 1))]
    assert priced == Cost(2, 24, 1)
    source = BeatSequence(frame, Repetition.CYCLIC)
    assert sorted(set(traced(chain, presented(source, BeatSequence(form))).tolist())) == [
        *range(5),
        *range(12, 17),
    ]


def test_a_one_pass_with_its_replay_is_presented_once_for_each_consumer_pass() -> None:
    """(4, 6) as above, each window's start read twice (a stride-0 loop of 2 between the
    window and its 3 reads), the whole pass twice: 72 reads. ``one_pass`` is one pass, 36
    reads with the replay; the consumer's 36 reads without it once filled one pass, so
    the pass was counted once for two passes and refused. Presented twice it is the
    consumer's form: direct, cost (0, 72, 1)."""
    form = Traversal((4, 6), (Loop(2, 0), Loop(2, 12), Loop(3, 1), Loop(2, 0), Loop(3, 1)), ())
    once = one_pass(form)
    assert (passes(form), once.beats) == (2, 36)
    source = BeatSequence(once, Repetition.CYCLIC)
    assert presented(source, BeatSequence(form)) == once.repeated(2) == form
    chain, priced = checked(once, form)
    assert not chain and priced == Cost(0, 72, 1)


def test_the_tiled_weights_are_presented_once_for_each_group_of_vectors() -> None:
    """The tiled dot product (N, MW, MH, PE, SIMD, TH) = (6, 12, 6, 3, 4, 3): the core's
    schedule (g, n, k, th), g = N / TH = 2, reads its weights W (12, 6) at lanes (n, k),
    each beat TH times, the matrix once a group of TH vectors. From the matrix stored at
    the port's WSIMD = 4 lanes: two passes, one each group, through a width step 4 -> 12
    and one reorder replaying each beat 3 times; cost (24, 36, 4)."""
    g, th = Index("g"), Index("th")
    core = Schedule({g: 2, n: 6, k: 12, th: 3}, {n: 3, k: 4}, (g, n, k, th))
    weights = core.present((12, 6), (k, n), lanes=(n, k))
    matrix = one_pass(regrouped(unreplayed(weights), 4))
    assert (matrix.beats, matrix.lanes, passes(weights)) == (18, 4, 2)
    source = BeatSequence(matrix, Repetition.CYCLIC)
    assert presented(source, BeatSequence(weights)) == matrix.repeated(2)
    chain, priced = checked(matrix, weights)
    assert chain.kinds == (Step.WIDTH, Step.REORDER)
    assert chain.hops[1].reorder == Reorder(1, (3,), (0,))
    assert priced == Cost(24, 36, 4)


def held_most(read: Sequence[int]) -> int:
    """The most elements a reorder holds reading ``read`` in order, by enumeration, not by
    ``live_window``'s formula: elements of the source arrive in order up to the one a beat
    reads, and one leaves once it and every element before it has been read for the last
    time (an element never read leaves as it is passed). Counted after each beat."""
    last = {element: beat for beat, element in enumerate(read)}
    arrived = gone = most = 0
    for beat, element in enumerate(read):
        arrived = max(arrived, element + 1)
        while gone < arrived and last.get(gone, -1) <= beat:
            gone += 1
        most = max(most, arrived - gone)
    return most


def test_the_windows_costs_live_window_is_what_a_reorder_holds_by_enumeration() -> None:
    """The cost's live window of the (4, 6) window from ``stored`` (2), the overlapping
    window of 6 (2, once and three times) and the (4, 6) window with each start read twice
    (3): the most elements a reorder holds reading the elements the consumer reads, in order,
    from its source's pass (``traced``), enumerated; and the source beats a pass costs, the
    beats presented, the source's own."""
    skipping = Traversal((4, 6), (Loop(2, 12), Loop(3, 1), Loop(3, 1)), ())
    doubled = Traversal((4, 6), (Loop(2, 0), Loop(2, 12), Loop(3, 1), Loop(2, 0), Loop(3, 1)), ())
    overlapping = Traversal((6,), (Loop(4, 1), Loop(3, 1)), ())
    for form, given, window, reads in (
        (skipping, stored(skipping), 2, 24),
        (doubled, stored(doubled), 3, 48),
        (overlapping, stored(overlapping), 2, 6),
        (overlapping.repeated(3), stored(overlapping), 2, 18),
    ):
        chain, priced = checked(given, form)
        source, sink = BeatSequence(given, Repetition.CYCLIC), BeatSequence(form)
        read = traced(chain, presented(source, sink)).tolist()
        assert held_most(read) == window == priced.live_window, form
        assert presented(source, sink).beats == priced.reads == reads, form
    # The first: 18 reads, the elements 0 to 4 and 12 to 16 of the frame, the window's own
    # three reads of each of its columns 1 to 3 twice.
    chain, _ = checked(stored(skipping), skipping)
    source, sink = BeatSequence(stored(skipping), Repetition.CYCLIC), BeatSequence(skipping)
    assert traced(chain, presented(source, sink)).tolist() == [
        *(0, 1, 2, 1, 2, 3, 2, 3, 4),
        *(12, 13, 14, 13, 14, 15, 14, 15, 16),
    ]


# -- enumerated consumers, against counting elements ------------------------------------------


def _consumers() -> list[Traversal]:
    """Every consumer at one lane of (4,), (6,) and (2, 4) of one or two loops of extent 2
    or 3 at strides 0 to 4 (at least one moving), the whole pass once or twice: windows
    that overlap, skip or both, replays, and repeated passes."""
    found: dict[Traversal, None] = {}
    for shape in ((4,), (6,), (2, 4)):
        for depth in (1, 2):
            for extents in itertools.product((2, 3), repeat=depth):
                for strides in itertools.product(range(5), repeat=depth):
                    for whole in (1, 2):
                        loops = [Loop(e, s) for e, s in zip(extents, strides)]
                        try:
                            form = Traversal(shape, [Loop(whole, 0), *loops], ())
                        except ValueError:
                            continue
                        if any(loop.stride for loop in form.beat_loops):
                            found[form] = None
    return list(found)


def _by_elements(given: Traversal, form: Traversal) -> tuple[Steps, Traversal] | None:
    """The steps from ``given`` repeated as often as the consumer's elements fill its
    passes (counting elements), and that presentation; None where they fill no whole
    number, or the steps are refused."""
    count, left = divmod(form.beats * form.lanes, given.beats * given.lanes)
    if left:
        return None
    shown = given.repeated(count)
    try:
        return steps(BeatSequence(shown), BeatSequence(form)), shown
    except Unrealizable:
        return None


def test_the_count_over_enumerated_consumers_against_counting_elements() -> None:
    """For each of the 265 consumers of ``_consumers``, three cyclic sources: ``stored``,
    the ``one_pass`` and the row-major frame. The rule's count is ``passes`` of the
    consumer; its steps build from every source of every consumer, and they deliver and
    read each consumer pass from its own source pass, by enumeration. Against counting
    elements, per source (of 265):

    | source | built where counting elements is refused | same | smaller | trade-off |
    |---|---|---|---|---|
    | ``stored`` | 28 | 163 | 71 | 3 |
    | ``one_pass`` | 0 | 265 | 0 | 0 |
    | frame | 171 | 38 | 9 | 47 |

    None is refused where counting elements builds, and none is larger. Each trade-off is
    a source presenting positions a consumer pass never reads, for a consumer repeating
    its pass: ``stored``'s three are (2, 4) read through two windows of 2 at stride 1, 2
    or 3 (the frame of 8, each pass reading 3 of them), twice, cost (1, 16, 1) against
    (3, 8, 1), (5, 8, 1) and (7, 8, 1).
    """
    tally: Counter[tuple[str, str]] = Counter()
    for form in _consumers():
        sink = BeatSequence(form)
        read = set(offsets(one_pass(form)).tolist())
        for name, given in (
            ("stored", stored(form)),
            ("one_pass", one_pass(form)),
            ("frame", vector_major(form.shape, 1)),
        ):
            assert isinstance(given, Traversal)
            source = BeatSequence(given, Repetition.CYCLIC)
            assert presented(source, sink).beats == given.beats * passes(form)
            chain = steps(source, sink)
            assert undelivered(chain, source, sink) == ""
            assert one_pass_each(chain, source, sink)
            other = _by_elements(given, form)
            if other is None:
                tally[name, "built"] += 1
                continue
            assert undelivered(other[0], BeatSequence(other[1]), sink) == ""
            mine, theirs = cost(chain, sink), cost(other[0], sink)
            if mine == theirs:
                relation = "same"
            elif smaller(mine, theirs):
                relation = "smaller"
            else:
                assert not smaller(theirs, mine), (name, form)
                relation = "trade-off"
                presents = set(offsets(given).tolist())
                assert read < presents and passes(form) > 1, (name, form)
            tally[name, relation] += 1
    assert len(_consumers()) == 265
    assert dict(tally) == {
        ("stored", "built"): 28,
        ("stored", "same"): 163,
        ("stored", "smaller"): 71,
        ("stored", "trade-off"): 3,
        ("one_pass", "same"): 265,
        ("frame", "built"): 171,
        ("frame", "same"): 38,
        ("frame", "smaller"): 9,
        ("frame", "trade-off"): 47,
    }
