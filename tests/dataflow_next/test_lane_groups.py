# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lane groups and the stored form: each test states its geometry and the numbers it
asserts, and checks what it derives by enumeration.

- **A lane group** (``Grouped``): a (6, 2) tensor read column by column (axis 1 outer,
  axis 0 inner) is a ``Traversal`` at 6 lanes and a ``Grouped`` at 4, whose second beat
  straddles the two columns; a row-major order is a ``Traversal`` at every lane count
  dividing it. The constructor refuses an aligned group and a group crossing a pass of a
  repetition; ``grouped`` refuses what ``regrouped`` refuses, by the same class and
  message, except where the lanes divide a pass's elements and cut a loop.
- **Steps with a lane group**: one width step over one order (the (6, 2) case, both
  ways); otherwise a width step to a ``Traversal`` of the group's order, a reorder, and a
  width step, through the most lanes at which both orders are traversals.
- **The tiled dot product's weights at (N, MW, MH, PE, SIMD, TH) = (6, 12, 4, 2, 6, 3)**:
  the port, 4 lanes against tile rows of 6, is a ``Grouped``; from a memory storing one
  pass of it the port connects directly, the matrix presented twice; to the core's
  weights the chain is a width step 4 -> 12 and a reorder replaying each beat 3 times.
- **The stored form**: what a cyclic source holds, ``stored`` and ``stored_depth``, and
  the four-field verdict between two sources (``source_verdict``).
- **A reindex with a lane-group reader**, through the identity and through a transpose;
  a merge into a lane group is refused.
- **The calls the harness records** (slow): every ``traversal.period`` call FINN makes on
  the chain of eight kernels, TFC and the geometries, against ``stored``; every
  ``traversal.regrouped`` call against ``grouped``.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
from collections.abc import Iterator
from dataflow_next.harness import translate
from dataflow_next.harness.checks import undelivered
from dataflow_next.harness.record import Call, Raised, Recorder
from dataflow_next.harness.sources import PREPARED
from typing import Any

from finn.core.reporting.findings import Refusal
from finn.dataflow_next.buffers import frame_slots, live_window
from finn.dataflow_next.gemm import k, n
from finn.dataflow_next.maps import concat, identity, transpose
from finn.dataflow_next.reindex import (
    MERGE_GROUPED_READER,
    Merge,
    Reader,
    Reindex,
    Route,
    Unrouted,
    composed,
    joined,
    merge_lanes,
    route,
    route_cost,
)
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import (
    Cost,
    Hop,
    Step,
    Steps,
    Unrealizable,
    cost,
    hop_cost,
    presented,
    source_verdict,
    steps,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelEnd,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    classify,
    each_once,
    grouped,
    offsets,
    one_pass,
    pack,
    passes,
    regrouped,
    stored,
    stored_depth,
    unpack,
    unreplayed,
    vector_major,
)

Array = npt.NDArray[np.int64]


def flat_positions(form: Traversal | Grouped) -> Array:
    """``form``'s positions, enumerated one by one (``position``), as flat row-major
    offsets, a row a beat."""
    rows = [
        [
            int(np.ravel_multi_index(form.position(beat, lane), form.shape))
            for lane in range(form.lanes)
        ]
        for beat in range(form.beats)
    ]
    return np.array(rows, dtype=np.int64).reshape(form.beats, form.lanes)


def element_order(form: Traversal | Grouped) -> Array:
    """``form``'s element order at one lane a beat (``regrouped``), cut into its beats."""
    return offsets(regrouped(form, 1)).reshape(form.beats, form.lanes)


# --- the lane group ----------------------------------------------------------------------

# A (6, 2) tensor column by column: axis 1 outer (stride 1), axis 0 inner (stride 2).
COLUMNS = Traversal.over((6, 2), ((1, 2, 1),), ((0, 6, 1),))


def test_an_order_cut_where_no_loop_ends_is_a_lane_group() -> None:
    """(6, 2) column by column: at 6 lanes each beat is a column, a ``Traversal``; at 4
    lanes the second beat holds the last two of column 0 and the first two of column 1, so
    the lanes are a group of the order: beats (0..3), (4, 5, 6, 7), (8..11) of it."""
    assert COLUMNS.lanes == 6 and COLUMNS.beats == 2
    found = grouped(COLUMNS, 4)
    assert found == Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 4)
    assert isinstance(found, Grouped)
    assert (found.beats, found.lanes) == (3, 4)
    assert list(found.positions()) == [
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((4, 0), (5, 0), (0, 1), (1, 1)),
        ((2, 1), (3, 1), (4, 1), (5, 1)),
    ]
    assert np.array_equal(offsets(found), offsets(COLUMNS))
    assert np.array_equal(flat_positions(found), element_order(found))
    assert grouped(found, 6) == COLUMNS == regrouped(found, 6)
    assert grouped(COLUMNS, 6) == COLUMNS
    assert not found.row_major


@pytest.mark.parametrize("lanes", [1, 2, 3, 4, 6, 12])
def test_a_row_major_order_is_a_traversal_at_every_dividing_lane_count(lanes: int) -> None:
    """(6, 2) row-major is one loop of 12, which any dividing lane count splits."""
    found = grouped(vector_major((6, 2), 2), lanes)
    assert isinstance(found, Traversal)
    assert found == regrouped(vector_major((6, 2), 2), lanes)
    assert found.row_major


def test_the_constructor_refuses_an_aligned_group_and_a_group_across_passes() -> None:
    # (6, 2) column by column at 6 lanes closes a column a beat: a Traversal's.
    with pytest.raises(ValueError, match="aligned group is a Traversal"):
        Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 6)
    with pytest.raises(ValueError, match="aligned group is a Traversal"):
        Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 1)
    # (6,) presented twice (a repetition of 2 outside a pass of 6) at 4 lanes: 4 divides
    # the 12 elements of both passes but not the 6 of one, so a beat would hold elements
    # of two passes.
    with pytest.raises(ValueError, match="4 lanes do not divide the 6 elements of a pass"):
        Grouped((6,), (Loop(2, 0), Loop(6, 1)), 4)
    with pytest.raises(ValueError, match="lanes must be a positive integer"):
        Grouped((6, 2), (Loop(2, 1), Loop(6, 2)), 0)
    with pytest.raises(ValueError, match="leaves an operand of 12 elements"):
        Grouped((6, 2), (Loop(2, 1), Loop(7, 2)), 4)


def test_a_lane_group_may_hold_two_presentations_of_a_replay() -> None:
    """(6,) with each element replayed twice inside the pass (a stride-0 loop of 2 inside
    the moving loop of 6), 12 presentations a pass. At 4 lanes a beat closes two whole
    replays, a Traversal; at 3 lanes beats cut the replays of elements 1 and 4."""
    order = (Loop(6, 1), Loop(2, 0))
    # 4 lanes close two whole replays: aligned, a Traversal.
    assert isinstance(grouped(Traversal((6,), order, ()), 4), Traversal)
    # 3 lanes cut a replay: element 1's two presentations are in beats 0 and 1.
    found = grouped(Traversal((6,), order, ()), 3)
    assert isinstance(found, Grouped)
    assert [tuple(position[0] for position in beat) for beat in found.positions()] == [
        (0, 0, 1),
        (1, 2, 2),
        (3, 3, 4),
        (4, 5, 5),
    ]


@pytest.mark.parametrize(
    ("form", "lanes", "message"),
    [
        (vector_major((4, 4), 4), 0, "inner_beats must be a positive integer"),
        (vector_major((4, 4), 4), -4, "inner_beats must be a positive integer"),
        (vector_major((2, 2), 2), 8, "inner_beats exceeds the pass"),
        # (6, 2) column by column at 5 lanes: 5 does not divide the 12 elements.
        (COLUMNS, 5, "inner_beats does not align with the loop nest"),
        # (6,) presented twice at 4 lanes: 4 divides both passes' 12, not one pass's 6.
        (
            Traversal((6,), (Loop(2, 0), Loop(6, 1)), ()),
            4,
            "inner_beats does not align with the loop nest",
        ),
    ],
    ids=["zero", "negative", "exceeds-the-pass", "no-divisor", "across-two-passes"],
)
def test_every_other_refusal_of_regrouped_is_grouped_s_by_class_and_message(
    form: Traversal, lanes: int, message: str
) -> None:
    with pytest.raises(ValueError) as line:
        regrouped(form, lanes)
    with pytest.raises(ValueError) as found:
        grouped(form, lanes)
    assert type(found.value) is type(line.value)
    assert str(found.value) == str(line.value) == message


def test_a_lane_group_presents_its_whole_pass_again_and_strips_it() -> None:
    found = grouped(COLUMNS, 4)
    assert isinstance(found, Grouped)
    again = found.repeated(3)
    assert again == Grouped((6, 2), (Loop(3, 0), Loop(2, 1), Loop(6, 2)), 4)
    assert (again.beats, passes(again), one_pass(again)) == (9, 3, found)
    assert np.array_equal(offsets(again), np.tile(offsets(found), 3))


def test_a_marker_on_a_lane_group_closes_whole_loops_of_its_order() -> None:
    """(6, 4) column by column (4 columns of 6) at 4 lanes: 6 beats. Every 3 beats closes
    two columns (12 elements), aligned; every 2 beats (8 elements) cuts a column; every
    6 beats closes the pass. A marker on every beat is a constant, which every sequence
    carries."""
    form = Grouped((6, 4), (Loop(4, 1), Loop(6, 4)), 4)
    assert form.beats == 6
    assert [LevelEnd(level).aligned(form) for level in (1, 2, 3, 4, 6)] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert BeatSequence(form, markers=(LevelEnd(3),)).markers == (LevelEnd(3),)
    with pytest.raises(ValueError, match="a marker every 2 beats closes no loop level"):
        BeatSequence(form, markers=(LevelEnd(2),))


def test_a_lane_group_packs_and_unpacks_at_its_positions() -> None:
    """(6, 2) column by column at 4 lanes, 8-bit lanes: word b holds elements 4b..4b+3
    of the column order, lane zero lowest; beyond 64 bits the exact path packs the same."""
    found = grouped(COLUMNS, 4)
    tensor = np.arange(12, dtype=np.int64).reshape(6, 2) - 6
    words = pack(found, tensor.ravel(), 8)
    expected = tuple(
        sum((int(tensor[position]) & 0xFF) << (8 * lane) for lane, position in enumerate(beat))
        for beat in found.positions()
    )
    assert words == expected
    assert np.array_equal(unpack(found, words, 8, signed=True), tensor)
    wide = [int(value) + (1 << 70) for value in tensor.ravel()]
    assert pack(found, wide, 8) == words


# --- steps with a lane group ------------------------------------------------------------


@pytest.mark.parametrize("forward", [True, False], ids=["6-to-4", "4-to-6"])
def test_an_unaligned_width_alone_is_one_width_step(forward: bool) -> None:
    """(6, 2) column by column at 6 lanes (a Traversal, 2 beats) against the same order
    at 4 lanes (a Grouped, 3 beats): one width step each way, holding lcm(6, 4) = 12
    elements; its reads are the source's beats, its narrowest lanes 4."""
    group = grouped(COLUMNS, 4)
    source, sink = (COLUMNS, group) if forward else (group, COLUMNS)
    chain = steps(BeatSequence(source), BeatSequence(sink))
    assert chain.kinds == (Step.WIDTH,)
    assert chain.describe() == "width_conversion"
    assert undelivered(chain, BeatSequence(source), BeatSequence(sink)) == ""
    assert cost(chain, BeatSequence(sink)) == Cost(12, source.beats, 4)
    assert chain.output == chain and not chain.input


def test_a_lane_group_of_another_order_joins_through_a_traversal_of_its_own() -> None:
    """(6, 2) row-major at 2 lanes into the column order at 4 lanes: at 2 lanes both are
    traversals but differ in their lane axis, so the chain goes through one lane: a width
    step 2 -> 1, a reorder of the 12 elements, a width step 1 -> 4 into the group."""
    source, sink = BeatSequence(vector_major((6, 2), 2)), BeatSequence(grouped(COLUMNS, 4))
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert [(hop.source.form.lanes, hop.sink.form.lanes) for hop in chain.hops] == [
        (2, 1),
        (1, 1),
        (1, 4),
    ]
    assert chain.hops[1].reorder == Reorder(12, (2, 6), (1, 2))
    assert undelivered(chain, source, sink) == ""
    # And back: the reverse chain, through one lane again.
    back = steps(sink, source)
    assert back.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert undelivered(back, sink, source) == ""


def test_two_lane_groups_of_one_order_are_one_width_step() -> None:
    """(15, 2) column by column: 30 elements, a Grouped at 6 lanes (5 beats) and at 10
    lanes (3 beats), whose beats both cut the columns of 15: one width step, lcm 30."""
    order = (Loop(2, 1), Loop(15, 2))
    six, ten = Grouped((15, 2), order, 6), Grouped((15, 2), order, 10)
    chain = steps(BeatSequence(six), BeatSequence(ten))
    assert chain.kinds == (Step.WIDTH,)
    assert cost(chain, BeatSequence(ten)) == Cost(30, 5, 6)
    assert undelivered(chain, BeatSequence(six), BeatSequence(ten)) == ""


def test_a_lane_group_sink_takes_its_markers_after_the_width_step() -> None:
    form = Grouped((6, 4), (Loop(4, 1), Loop(6, 4)), 4)
    source = BeatSequence(regrouped(form, 2))
    sink = BeatSequence(form, markers=(LevelEnd(3),))
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH, Step.MARKERS)
    assert chain.hops[-1].sink == sink
    assert undelivered(chain, source, sink) == ""


def test_a_lane_group_reading_positions_never_presented_is_refused() -> None:
    """(6, 2) column by column, rows 0..4 only, at 2 lanes (2 cuts the columns of 5): it
    reads part of the column order at 4 lanes, through a reorder at one lane that skips
    row 5; the column order reads row 5, which it never presents."""
    part = Grouped((6, 2), (Loop(2, 1), Loop(5, 2)), 2)
    whole = grouped(COLUMNS, 4)
    skipping = steps(BeatSequence(whole), BeatSequence(part))
    assert skipping.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert skipping.hops[1].reorder == Reorder(6, (5,), (1,))
    assert undelivered(skipping, BeatSequence(whole), BeatSequence(part)) == ""
    with pytest.raises(Unrealizable, match="different positions"):
        steps(BeatSequence(part), BeatSequence(whole))
    with pytest.raises(Unrealizable, match="different operand shapes"):
        steps(BeatSequence(grouped(COLUMNS, 4)), BeatSequence(vector_major((12,), 4)))


def test_classify_refuses_a_lane_group() -> None:
    group: Any = grouped(COLUMNS, 4)
    with pytest.raises(TypeError, match="a lane group joins by a width step"):
        classify(group, COLUMNS)
    with pytest.raises(TypeError, match="a lane group joins by a width step"):
        classify(COLUMNS, group)


def test_a_reorder_step_with_a_lane_group_end_has_no_cost() -> None:
    group = BeatSequence(grouped(COLUMNS, 4))
    hop = Hop(Step.REORDER, group, BeatSequence(vector_major((6, 2), 2)), Reorder(3, (6,), (1,)))
    with pytest.raises(ValueError, match="a lane group joins by width"):
        hop_cost(hop)


# --- the tiled dot product's weights at (6, 12, 4, 2, 6, 3) -------------------------------

G, TH = Index("g"), Index("th")


def tiled_weights(
    vectors: int, depth: int, outputs: int, pe: int, simd: int, tile: int
) -> tuple[Traversal, Traversal | Grouped]:
    """The core's weights W[k, n] under the schedule (g, n, k, th), lanes (n, k): a PE x
    SIMD tile, pe-major, replayed TH times; and the module's port: the same matrix
    without the replay at WSIMD = PE * SIMD / TH lanes."""
    core = Schedule(
        {G: vectors // tile, n: outputs, k: depth, TH: tile}, {n: pe, k: simd}, (G, n, k, TH)
    )
    weights = core.present((depth, outputs), (k, n), lanes=(n, k))
    return weights, grouped(unreplayed(weights), pe * simd // tile)


def port_contract(
    vectors: int, depth: int, outputs: int, pe: int, simd: int, tile: int
) -> Iterator[list[int]]:
    """The port's beats as the tiled module reads them, a flat offset of W (MW, MH) per
    lane: for each group of TH vectors, each tile (nf, sf), each of its TH beats, lane l
    holds element f = b * WSIMD + l of the tile, (pe, simd) = (f // SIMD, f % SIMD), at
    W[sf * SIMD + simd, nf * PE + pe]."""
    wsimd = pe * simd // tile
    for _ in range(vectors // tile):
        for nf in range(outputs // pe):
            for sf in range(depth // simd):
                for beat in range(tile):
                    row = []
                    for lane in range(wsimd):
                        f = beat * wsimd + lane
                        row.append((sf * simd + f % simd) * outputs + nf * pe + f // simd)
                    yield row


THIRD = (6, 12, 4, 2, 6, 3)
"""(N, MW, MH, PE, SIMD, TH): NF = 2, SF = 2, WSIMD = 4, tiles of 2 x 6."""


def test_the_tiled_weight_port_whose_lanes_cross_a_row_is_a_lane_group() -> None:
    """At (6, 12, 4, 2, 6, 3) WSIMD = 4 lanes against tile rows of SIMD = 6: the line's
    ``regrouped`` refuses; ``grouped`` gives a Grouped of 2 groups x 4 tiles x 3 beats =
    24 beats, whose positions are the module's contract beat by beat."""
    weights, port = tiled_weights(*THIRD)
    with pytest.raises(ValueError, match="does not align"):
        regrouped(unreplayed(weights), 4)
    assert isinstance(port, Grouped)
    assert (port.beats, port.lanes) == (24, 4)
    assert port.order == (Loop(2, 0), Loop(2, 2), Loop(2, 24), Loop(2, 1), Loop(6, 4))
    assert np.array_equal(flat_positions(port), np.array(list(port_contract(*THIRD))))
    assert np.array_equal(element_order(port), flat_positions(port))


def test_the_tiled_weights_from_a_memory_of_one_port_one_pass() -> None:
    """The memory stores ``stored(port)``: 12 beats of 4 lanes, 48 elements, the matrix
    once. To the port: direct, the matrix presented N / TH = 2 times. To the core's
    weights: a width step 4 -> 12, then a reorder replaying each beat TH = 3 times
    (frame 1 beat, dims (3,), coefs (0,)); cost (24, 24, 4): lcm(4, 12) = 12 for the width
    step and one 12-lane beat for the replay, 24 source beats read, 4 lanes narrowest."""
    weights, port = tiled_weights(*THIRD)
    memory = stored(port)
    assert isinstance(memory, Grouped)
    assert (memory.beats, memory.lanes, stored_depth(port)) == (12, 4, 48)
    source = BeatSequence(memory, Repetition.CYCLIC)

    to_port = steps(source, BeatSequence(port))
    assert not to_port and to_port.describe() == "direct"
    shown = presented(source, BeatSequence(port))
    assert shown == port
    assert shown.beats * shown.lanes // (12 * 4) == 2
    assert cost(to_port, BeatSequence(port)) == Cost(0, 24, 4)

    to_core = steps(source, BeatSequence(weights))
    assert to_core.kinds == (Step.WIDTH, Step.REORDER)
    assert [(hop.source.form.lanes, hop.sink.form.lanes) for hop in to_core.hops] == [
        (4, 12),
        (12, 12),
    ]
    reorder = to_core.hops[1].reorder
    assert reorder == Reorder(1, (3,), (0,))
    assert live_window(frame_slots(reorder)) == 1
    assert cost(to_core, BeatSequence(weights)) == Cost(24, 24, 4)
    assert undelivered(to_core, source, BeatSequence(weights)) == ""
    # By enumeration: the port's 24 beats, widened to 12 lanes (8 beats), each replayed 3
    # times, are the core's 24 beats.
    widened = offsets(port).reshape(-1, 12)
    assert np.array_equal(np.repeat(widened, 3, axis=0), offsets(weights).reshape(-1, 12))


def test_the_tiled_weights_stored_at_the_core_s_lanes_against_the_line_s_one_pass() -> None:
    """``stored(w)``, 4 beats x 12 = 48 elements, against ``one_pass(w)``, the line's
    stored form, 12 beats x 12 = 144 (it keeps the TH-fold replay). From ``stored(w)``:
    one reorder replaying each beat 3 times, cost (12, 8, 12); from ``one_pass(w)``:
    direct, cost (0, 24, 12). Over the four fields that is a trade-off: a memory a third
    the size and a third of the reads, for one beat of 12 lanes held."""
    weights, _ = tiled_weights(*THIRD)
    small, line = stored(weights), one_pass(weights)
    assert (small.beats, small.lanes, stored_depth(weights)) == (4, 12, 48)
    assert (line.beats, line.lanes) == (12, 12)
    chains = {}
    for name, form in (("stored", small), ("one_pass", line)):
        source = BeatSequence(form, Repetition.CYCLIC)
        chain = steps(source, BeatSequence(weights))
        assert undelivered(chain, source, BeatSequence(weights)) == ""
        chains[name] = cost(chain, BeatSequence(weights))
    assert steps(BeatSequence(small, Repetition.CYCLIC), BeatSequence(weights)).hops[
        0
    ].reorder == Reorder(1, (3,), (0,))
    assert chains == {"stored": Cost(12, 8, 12), "one_pass": Cost(0, 24, 12)}
    assert source_verdict(48, chains["stored"], 144, chains["one_pass"]) == "trade-off"


# --- the stored form ----------------------------------------------------------------------


def test_a_form_without_replay_or_window_stores_its_one_pass() -> None:
    form = vector_major((4, 6), 3)
    assert stored(form) == one_pass(form) == form
    assert stored(form.repeated(3)) == form
    assert stored_depth(form.repeated(3)) == 24


def test_a_consumer_s_inner_replay_is_not_stored() -> None:
    """(4, 16) at one lane, each row of 16 read 16 times (a stride-0 loop of 16 between
    the row loop and the column loop): the line's form of one pass stores 4 x 16 x 16 = 1024
    elements, ``stored`` the 64 of the tensor. The steps from it replay the rows: one
    reorder, its frame a row of 16 beats, dims (16, 16), coefs (0, 1)."""
    form = Traversal((4, 16), (Loop(4, 16), Loop(16, 0), Loop(16, 1)), ())
    assert stored(form) == vector_major((4, 16), 1)
    assert (stored_depth(form), one_pass(form).beats) == (64, 1024)
    source, sink = BeatSequence(stored(form), Repetition.CYCLIC), BeatSequence(form)
    chain = steps(source, sink)
    assert chain.kinds == (Step.REORDER,)
    assert [hop.reorder for hop in chain.hops] == [Reorder(16, (16, 16), (0, 1))]
    assert undelivered(chain, source, sink) == ""


def test_a_sliding_window_stores_its_frame_once() -> None:
    """(6,) at one lane through a window of 3 at stride 1 (4 windows): 12 reads, each
    position up to three times. ``stored`` is the frame, the 6 elements once, where the
    line's form of one pass holds the 12 reads. The steps from it: the frame presented once for the
    consumer's one pass, and one reorder (frame 6 beats, dims (4, 3), coefs (1, 1)) reads
    the windows out of it, holding what they read again; cost (2, 6, 1)."""
    form = Traversal((6,), (Loop(4, 1), Loop(3, 1)), ())
    assert not each_once(form)
    assert stored(form) == vector_major((6,), 1)
    assert (stored_depth(form), one_pass(form).beats) == (6, 12)
    source, sink = BeatSequence(stored(form), Repetition.CYCLIC), BeatSequence(form)
    chain = steps(source, sink)
    assert [hop.reorder for hop in chain.hops] == [Reorder(6, (4, 3), (1, 1))]
    assert cost(chain, sink) == Cost(2, 6, 1)
    assert undelivered(chain, source, sink) == ""
    windows = offsets(stored(form))[np.add.outer(np.arange(4), np.arange(3)).ravel()]
    assert np.array_equal(windows, offsets(form))


def test_a_window_that_only_passes_positions_stores_what_it_reads() -> None:
    """(6,) at one lane read at stride 2 (elements 0, 2, 4): no position is read twice, so
    ``stored`` holds the 3 read, each once, not the 6-element frame. From it the consumer
    connects directly, the 3 elements once a pass."""
    form = Traversal((6,), (Loop(3, 2),), ())
    assert stored(form) == form == one_pass(form)
    assert stored_depth(form) == 3
    source, sink = BeatSequence(stored(form), Repetition.CYCLIC), BeatSequence(form)
    chain = steps(source, sink)
    assert not chain
    assert undelivered(chain, source, sink) == ""
    assert offsets(presented(source, sink)).tolist() == [0, 2, 4]


def test_a_window_that_passes_positions_and_reads_some_twice() -> None:
    """(4, 6) at one lane, rows 0 and 2 only, each through 3 windows of 3 at stride 1 over
    its first 5 columns: 18 reads, columns 1..3 read twice, rows 1 and 3 and column 5
    never. ``stored`` is the frame, all 24 elements, presented once for the consumer's one
    pass: one reorder over two rows at a time (frame 12, dims (3, 3), coefs (1, 1)) reads
    the first row's windows and drops the second row; cost (2, 24, 1)."""
    form = Traversal((4, 6), (Loop(2, 12), Loop(3, 1), Loop(3, 1)), ())
    assert stored(form) == vector_major((4, 6), 1)
    assert (stored_depth(form), one_pass(form).beats) == (24, 18)
    source, sink = BeatSequence(stored(form), Repetition.CYCLIC), BeatSequence(form)
    chain = steps(source, sink)
    assert [hop.reorder for hop in chain.hops] == [Reorder(12, (3, 3), (1, 1))]
    assert cost(chain, sink) == Cost(2, 24, 1)
    assert undelivered(chain, source, sink) == ""


def test_a_lane_group_stores_its_order_s_one_pass_at_its_lanes() -> None:
    group = grouped(COLUMNS, 4)
    assert stored(group) == group == stored(group.repeated(2))
    assert stored_depth(group) == 12
    # (6,) with each element replayed twice inside the pass, at 3 lanes: the stored order
    # is the 6 elements once, which 3 lanes split as whole loops.
    replayed = grouped(Traversal((6,), (Loop(6, 1), Loop(2, 0)), ()), 3)
    assert isinstance(replayed, Grouped)
    assert stored(replayed) == vector_major((6,), 3)
    # (3,) with each element replayed 4 times, at 6 lanes: 6 groups the 12 presentations
    # (and cuts the replays of element 1), but no pass of the 3 elements.
    replays = Grouped((3,), (Loop(3, 1), Loop(4, 0)), 6)
    assert [tuple(p[0] for p in beat) for beat in replays.positions()] == [
        (0, 0, 0, 0, 1, 1),
        (1, 1, 2, 2, 2, 2),
    ]
    with pytest.raises(ValueError, match="6 lanes group no pass of the order without its replay"):
        stored(replays)


@pytest.mark.parametrize(
    ("first", "second", "verdict"),
    [
        ((48, Cost(12, 8, 12)), (48, Cost(12, 8, 12)), "same"),
        ((48, Cost(12, 8, 12)), (144, Cost(12, 24, 12)), "smaller"),
        ((48, Cost(12, 8, 12)), (48, Cost(12, 8, 4)), "smaller"),  # wider
        ((144, Cost(12, 24, 12)), (48, Cost(12, 8, 12)), "larger"),
        ((48, Cost(12, 8, 12)), (144, Cost(0, 24, 12)), "trade-off"),  # memory for buffer
        ((48, Cost(0, 8, 4)), (144, Cost(0, 24, 12)), "trade-off"),  # memory for rate
    ],
    ids=["same", "smaller", "wider", "larger", "memory-for-buffer", "memory-for-rate"],
)
def test_a_source_is_weighed_over_four_fields_never_summed(
    first: tuple[int, Cost], second: tuple[int, Cost], verdict: str
) -> None:
    assert source_verdict(*first, *second) == verdict


def test_a_stored_depth_is_a_positive_number_of_elements() -> None:
    with pytest.raises(ValueError, match="stored depth"):
        source_verdict(0, Cost(0, 1, 1), 1, Cost(0, 1, 1))


# --- a reindex with a lane-group reader -------------------------------------------------


def test_a_lane_group_reader_through_the_identity_is_the_steps() -> None:
    """(6, 2) row-major at 2 lanes, read column by column at 4 lanes through the
    identity: the route is ``steps(producer, reader)``, its cost theirs."""
    producer = BeatSequence(vector_major((6, 2), 2))
    reader = BeatSequence(grouped(COLUMNS, 4))
    reindex = Reindex(producer, (Reader(identity((6, 2)), reader),))
    assert composed(reindex) == reader
    assert route(reindex) == Route(Steps(), None, steps(producer, reader))
    assert route_cost(reindex) == cost(steps(producer, reader), reader)


def test_a_lane_group_reader_through_a_transpose_is_composed_by_its_order() -> None:
    """X (3, 4) row-major at 4 lanes; the reader reads Y = Xᵀ (4, 3) in the order (i // 2,
    j, i % 2) at 4 lanes, which cuts the loop of j (3) in two: a Grouped over Y. Composed
    onto X by its order it is the order (2 of stride 2, 3 of stride 4, 2 of stride 1) at 4
    lanes, again a Grouped. The route: a width step 4 -> 2, a reorder at 2 lanes (frame 6,
    dims (2, 3), coefs (1, 2)), a width step 2 -> 4; cost (16, 3, 2)."""
    y_order = Grouped((4, 3), (Loop(2, 6), Loop(3, 1), Loop(2, 3)), 4)
    producer = BeatSequence(vector_major((3, 4), 4))
    reindex = Reindex(producer, (Reader(transpose((3, 4), (1, 0)), BeatSequence(y_order)),))
    found = composed(reindex)
    assert found.form == Grouped((3, 4), (Loop(2, 2), Loop(3, 4), Loop(2, 1)), 4)
    # The same elements in the same beats and lanes: Y[i, j] is X[j, i].
    transposed = [tuple((j, i) for i, j in beat) for beat in y_order.positions()]
    assert list(found.form.positions()) == transposed
    routed = route(reindex)
    assert isinstance(routed, Route)
    chain = routed.after
    assert chain.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert chain.hops[1].reorder == Reorder(6, (2, 3), (1, 2))
    assert undelivered(chain, producer, found) == ""
    assert route_cost(reindex) == Cost(16, 3, 2)


def test_a_lane_group_composed_onto_a_loop_nest_is_a_traversal() -> None:
    """Y = Xᵀ (4, 3) read column by column at 6 lanes cuts the columns of 4: a Grouped. On
    X (3, 4) that order is row-major, which 6 lanes split: the composed form is a
    Traversal, and from X at 4 lanes the route is one width step."""
    y_order = grouped(Traversal.over((4, 3), ((1, 3, 1),), ((0, 4, 1),)), 6)
    assert isinstance(y_order, Grouped)
    producer = BeatSequence(vector_major((3, 4), 4))
    reindex = Reindex(producer, (Reader(transpose((3, 4), (1, 0)), BeatSequence(y_order)),))
    assert composed(reindex).form == regrouped(vector_major((3, 4), 4), 6)
    found = route(reindex)
    assert isinstance(found, Route) and found.after.kinds == (Step.WIDTH,)


def test_a_merge_into_a_lane_group_is_refused_by_name() -> None:
    """Two (3, 2) values joined along axis 1 into (3, 4), read column by column at 2
    lanes (a Grouped: 2 cuts the columns of 3)."""
    form = Grouped((3, 4), (Loop(4, 1), Loop(3, 4)), 2)
    join = concat(((3, 2), (3, 2)), 1)
    halves = tuple(
        Reindex(BeatSequence(vector_major((3, 2), 2)), (Reader(join, BeatSequence(form), None, i),))
        for i in (0, 1)
    )
    merge = Merge(halves)
    for derive in (lambda: merge_lanes(joined(merge)), lambda: composed(halves[0])):
        with pytest.raises(Unrouted) as refused:
            derive()
        assert [finding.code for finding in refused.value.findings] == [MERGE_GROUPED_READER]
        assert isinstance(refused.value, Refusal)


# --- the calls the harness records ---------------------------------------------------------

SOURCES = ("chain-8", "tfc", "geometries")


@pytest.fixture(scope="module")
def recorded() -> dict[str, list[Call]]:
    """Every ``traversal.period`` and ``traversal.regrouped`` call FINN makes while it
    derives the chain of eight kernels, TFC and the geometries, by source. Runs the three
    sources in this process (about three and a half minutes, TFC's most)."""
    labels = ("traversal.period", "traversal.regrouped")
    found: dict[str, list[Call]] = {}
    for source in SOURCES:
        run = PREPARED[source]()
        with Recorder() as recorder:
            recorder.source = source
            run()
        found[source] = [call for call in recorder.recorded() if call.target.label in labels]
    return found


def _outcome(derive: Any) -> Any:
    try:
        return derive()
    except Exception as error:  # an exception is an answer, compared by type and message
        return Raised.of(error)


@pytest.mark.slow
def test_stored_is_the_line_s_one_pass_on_every_recorded_call(
    recorded: dict[str, list[Call]],
) -> None:
    """Where the form has no replay inside a pass and no window (``unreplayed`` leaves it
    as it is), ``stored`` equals the line's ``period``. Every other call is listed with
    both depths, in elements: each a smaller stored depth whose steps deliver the
    consumer's pass, checked by enumeration. On this tree: chain-8 16 calls, 14 equal, 2
    listed, both (4, 16) at one lane with a row read 16 or 8 times: 64 against 1024 and
    512; TFC 208 calls, all equal; the geometries 2, both equal."""
    counts: dict[str, tuple[int, int]] = {}
    listed: list[tuple[str, int, int]] = []
    for source, calls in recorded.items():
        compared = equal = 0
        for call in calls:
            if call.target.label != "traversal.period":
                continue
            assert not isinstance(call.outcome, Raised)
            form = translate.forward(call.args[0])
            assert one_pass(form) == translate.forward(call.outcome)
            compared += 1
            if unreplayed(form) == form:
                assert stored(form) == one_pass(form)
                equal += 1
                continue
            depth, line = stored_depth(form), one_pass(form).beats * one_pass(form).lanes
            listed.append((source, depth, line))
            assert depth < line
            supply = BeatSequence(stored(form), Repetition.CYCLIC)
            reader = BeatSequence(form)
            assert undelivered(steps(supply, reader), supply, reader) == ""
        counts[source] = (compared, equal)
    assert counts == {"chain-8": (16, 14), "tfc": (208, 208), "geometries": (2, 2)}
    assert sorted(listed) == [("chain-8", 64, 512), ("chain-8", 64, 1024)]


@pytest.mark.slow
def test_grouped_is_the_line_s_regrouped_on_every_recorded_call(
    recorded: dict[str, list[Call]],
) -> None:
    """Where the line returns, ``grouped`` returns the same ``Traversal``. Where it raises
    "does not align" and the lanes divide the elements of a pass, ``grouped`` gives a
    ``Grouped`` whose positions, beat by beat, are the order's enumeration; every other
    refusal is ``grouped``'s, by class and message. On this tree the three calls are the
    geometries' tiled weights: two returned (4 and 8 lanes), and one Grouped (4 lanes
    against tile rows of 6)."""
    returned = groups = refused = 0
    for calls in recorded.values():
        for call in calls:
            if call.target.label != "traversal.regrouped":
                continue
            form, lanes = translate.forward(call.args)
            found = _outcome(lambda: grouped(form, lanes))
            if not isinstance(call.outcome, Raised):
                assert found == translate.forward(call.outcome)
                returned += 1
            elif isinstance(found, Grouped):
                assert call.outcome.message == "inner_beats does not align with the loop nest"
                order = offsets(regrouped(form, 1)).reshape(found.beats, found.lanes)
                assert np.array_equal(flat_positions(found), order)
                groups += 1
            else:
                assert found == call.outcome
                refused += 1
    assert (returned, groups, refused) == (2, 1, 0)
