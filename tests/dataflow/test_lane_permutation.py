# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A lane permutation at a lane count the chain goes through is wires at an end of the
chain, not a reason to go through fewer lanes.

The geometry: (4, 4) in 2 x 2 blocks, two block rows of two blocks. The block's four
elements are a beat's four lanes in one of two orders: row-major ((r, c), (r, c + 1),
(r + 1, c), (r + 1, c + 1)) or column-major ((r, c), (r + 1, c), (r, c + 1), (r + 1,
c + 1)). Two orders that differ only so are one lane permutation, (0, 2, 1, 3), of every
four lanes: wires.

Before, a chain whose ends were at other lane counts took a reorder only, and passed a
lane permutation over for one lane a beat: correct, but every element on its own beat.
The network model shows the cost in time: where that chain is the bottleneck, its period
is longer. Each chain here is checked by enumeration (``undelivered``).
"""

from __future__ import annotations

import itertools
from collections import Counter
from math import gcd, prod

from dataflow.harness.checks import undelivered
from finn.dataflow.steps import Cost, Hop, Step, Steps, cost, steps
from finn.dataflow.timing import (
    FINNLIB,
    Sink,
    Source,
    bottleneck,
    chained,
    run,
    step_stages,
)
from finn.dataflow.traversal import (
    Adaptation,
    BeatSequence,
    Grouped,
    Loop,
    Reorder,
    Traversal,
    axis_strides,
    classify,
    grouped,
    regrouped,
)

TWO_BLOCKS_COLUMN_MAJOR = Traversal((4, 4), (Loop(2, 8),), (Loop(2, 2), Loop(2, 1), Loop(2, 4)))
"""8 lanes, a beat a block row: its two blocks, each column-major."""
BLOCKS_ROW_MAJOR = Traversal((4, 4), (Loop(2, 8), Loop(2, 2)), (Loop(2, 4), Loop(2, 1)))
"""4 lanes, a beat a block, row-major."""
BLOCKS_COLUMN_MAJOR = Traversal((4, 4), (Loop(2, 8), Loop(2, 2)), (Loop(2, 1), Loop(2, 4)))
"""4 lanes, a beat a block, column-major."""
HALF_BLOCKS = Traversal((4, 4), (Loop(2, 8), Loop(2, 2), Loop(2, 4)), (Loop(2, 1),))
"""2 lanes, a beat a block's row: its two columns."""


def _lanes(chain: Steps) -> list[tuple[int, int]]:
    return [(hop.source.form.lanes, hop.sink.form.lanes) for hop in chain.hops]


def _per_beat(form: Traversal | Grouped) -> list[set[tuple[int, ...]]]:
    return [set(beat) for beat in form.positions()]


def test_a_lane_permutation_after_a_width_step_is_wires() -> None:
    """8 lanes of two column-major blocks into 4 lanes of a row-major block: one width
    step 8 -> 4, ending at the column-major blocks, which the wires permute (0, 2, 1, 3).
    Cost (8, 2, 4); the reverse, 4 -> 8, (8, 4, 4). Before, both went through one lane:
    a width step to 1, a reorder (frame 4, dims (2, 2), coefs (1, 2)), a width step from
    1, cost (14, 2, 1) and (14, 4, 1)."""
    source, sink = BeatSequence(TWO_BLOCKS_COLUMN_MAJOR), BeatSequence(BLOCKS_ROW_MAJOR)
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH,) and _lanes(chain) == [(8, 4)]
    end = chain.hops[-1].sink.form
    assert end == BLOCKS_COLUMN_MAJOR
    assert classify(end, BLOCKS_ROW_MAJOR).lane_permutation == (0, 2, 1, 3)
    assert cost(chain, sink) == Cost(8, 2, 4)
    assert undelivered(chain, source, sink) == ""
    back = steps(sink, source)
    assert back.kinds == (Step.WIDTH,) and _lanes(back) == [(4, 8)]
    assert cost(back, source) == Cost(8, 4, 4)
    assert undelivered(back, sink, source) == ""


def test_a_lane_permutation_before_a_width_step_is_wires() -> None:
    """4 lanes of a column-major block into 2 lanes of a block's row: the producer's
    lanes, permuted (0, 2, 1, 3) by the wires, are the row-major block, and one width step
    4 -> 2 splits it into its rows. The chain starts at that permutation of what the
    source presents. Cost (4, 4, 2); the reverse, one width step 2 -> 4 ending at the
    permutation, (4, 8, 2). Before: through one lane, (8, 4, 1) and (8, 8, 1)."""
    source, sink = BeatSequence(BLOCKS_COLUMN_MAJOR), BeatSequence(HALF_BLOCKS)
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH,) and _lanes(chain) == [(4, 2)]
    start = chain.hops[0].source.form
    assert start == BLOCKS_ROW_MAJOR == regrouped(HALF_BLOCKS, 4)
    assert classify(BLOCKS_COLUMN_MAJOR, start).adaptation is Adaptation.LANE_PERMUTATION
    assert cost(chain, sink) == Cost(4, 4, 2)
    assert undelivered(chain, source, sink) == ""
    back = steps(sink, source)
    assert back.kinds == (Step.WIDTH,) and _lanes(back) == [(2, 4)]
    assert cost(back, source) == Cost(4, 8, 2)
    assert undelivered(back, sink, source) == ""


def test_two_lane_groups_that_differ_by_a_lane_permutation_connect_directly() -> None:
    """(4, 6) at 8 lanes, two lane groups of one order but for the two middle elements of
    every four (a block of rows 2r, 2r + 1 and columns c, c + 3): at 4 lanes the two are a
    lane permutation, so at 8 every beat holds the same positions, permuted alike. Direct,
    cost (0, 3, 8). Before: through one lane, cost (18, 3, 1)."""
    first = Grouped((4, 6), (Loop(2, 12), Loop(3, 1), Loop(2, 3), Loop(2, 6)), 8)
    second = Grouped((4, 6), (Loop(2, 12), Loop(3, 1), Loop(4, 3)), 8)
    low, target = regrouped(first, 4), regrouped(second, 4)
    assert classify(low, target).lane_permutation == (0, 2, 1, 3)
    assert _per_beat(first) == _per_beat(second)
    source, sink = BeatSequence(first), BeatSequence(second)
    chain = steps(source, sink)
    assert not chain and cost(chain, sink) == Cost(0, 3, 8)
    assert undelivered(chain, source, sink) == ""


def test_through_one_lane_the_join_is_correct_but_slower() -> None:
    """The chain the lane permutation was passed over for, from 8 lanes of two
    column-major blocks to 4 lanes of a row-major block: a width step 8 -> 1, a reorder
    (frame 4, dims (2, 2), coefs (1, 2)), a width step 1 -> 4. It delivers, at cost
    (14, 2, 1) against the one width step's (8, 2, 4).

    In time, under FinnLib's rules, a source presenting its 2 words a frame on cycles 0
    and 1, a sink taking a word a cycle, every channel unbounded: released every 4 cycles,
    the one width step keeps 4 cycles a frame, and through one lane the sink completes a
    frame every 32 (16 one-element words, through a reorder at its live window of 2
    beats); the bottleneck is that chain's. Released every 40 cycles, neither chain is the
    bottleneck, and both keep 40."""
    source, sink = BeatSequence(TWO_BLOCKS_COLUMN_MAJOR), BeatSequence(BLOCKS_ROW_MAJOR)
    one, target = regrouped(TWO_BLOCKS_COLUMN_MAJOR, 1), regrouped(BLOCKS_ROW_MAJOR, 1)
    reorder = classify(one, target).reorder
    assert reorder == Reorder(4, (2, 2), (1, 2))
    narrow = Steps(
        (
            Hop(Step.WIDTH, source, BeatSequence(one)),
            Hop(Step.REORDER, BeatSequence(one), BeatSequence(target), reorder),
            Hop(Step.WIDTH, BeatSequence(target), sink),
        )
    )
    assert undelivered(narrow, source, sink) == ""
    wide = steps(source, sink)
    assert (cost(narrow, sink), cost(wide, sink)) == (Cost(14, 2, 1), Cost(8, 2, 4))
    periods = {}
    for released in (4, 40):
        for name, chain in (("narrow", narrow), ("wide", wide)):
            network = chained(
                (Source("s", (0, 1), 2, released), *step_stages(chain, "c"), Sink("t"))
            )
            found = run(network.unbounded(), FINNLIB)
            assert not found.deadlocked
            periods[released, name] = (found.period("t"), bottleneck(network, FINNLIB))
    assert periods == {
        (4, "narrow"): (32, 32),
        (4, "wide"): (4, 4),
        (40, "narrow"): (40, 40),
        (40, "wide"): (40, 40),
    }


def _forms(shape: tuple[int, ...]) -> list[Traversal | Grouped]:
    """Every order of ``shape`` whose loops split each axis of 4 into 2 x 2 and of 6
    into 2 x 3 (the coarser part outer), in every sequence, at every lane count from 2
    dividing its elements and below them: a ``Traversal`` or a ``Grouped``."""
    strides = axis_strides(shape)
    parts: list[Loop] = []
    for axis, extent in enumerate(shape):
        inner = {4: 2, 6: 3}.get(extent, extent)
        if inner < extent:
            parts.append(Loop(extent // inner, inner * strides[axis]))
        parts.append(Loop(inner, strides[axis]))
    size = prod(shape)
    found: dict[Traversal | Grouped, None] = {}
    for order in itertools.permutations(parts):
        for lanes in range(2, size):
            if not size % lanes:
                found[grouped(Traversal(shape, order, ()), lanes)] = None
    return list(found)


def test_no_lane_permutation_is_passed_over_for_fewer_lanes() -> None:
    """Every ordered pair of distinct forms of (4, 4) and of (2, 6) from ``_forms`` (5,664
    pairs: 5,322 of two traversals, 135 from a traversal to a lane group, 135 back, 72 of
    two lane groups) builds a chain that delivers, by enumeration, and at no lane count
    dividing both ends' above the chain's narrowest are the two orders a lane
    permutation."""
    tally: Counter[str] = Counter()
    for shape in ((4, 4), (2, 6)):
        forms = _forms(shape)
        for first, second in itertools.permutations(forms, 2):
            source, sink = BeatSequence(first), BeatSequence(second)
            chain = steps(source, sink)
            assert undelivered(chain, source, sink) == ""
            narrowest = cost(chain, sink).narrowest_lanes
            common = gcd(first.lanes, second.lanes)
            for lanes in range(narrowest + 1, common + 1):
                if common % lanes:
                    continue
                try:
                    low, target = regrouped(first, lanes), regrouped(second, lanes)
                except ValueError:
                    continue
                assert classify(low, target).adaptation is not Adaptation.LANE_PERMUTATION
            kinds = "".join("G" if isinstance(f, Grouped) else "T" for f in (first, second))
            tally[kinds] += 1
    assert dict(tally) == {"TT": 5322, "TG": 135, "GT": 135, "GG": 72}
