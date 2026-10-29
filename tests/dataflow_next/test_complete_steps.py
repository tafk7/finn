# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Complete steps and their cost: the regroup through the largest common lane count at
which the orders differ by a reorder, a looping source counted against its consumer's
order without the replay, and what a chain costs.

Each chain is checked by enumeration (``_carried``: the source's beats carried through
every step) besides its numbers, and against the line (``finn.dataflow``), which refuses
the regroups and counts the looping source more often.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
from dataflow_next.harness import translate
from typing import Any

import finn.dataflow.steps as line
from finn.dataflow_next.buffers import frame_slots
from finn.dataflow_next.gemm import k, m, n
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
    smaller,
    steps,
)
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelEnd,
    Repetition,
    Traversal,
    offsets,
    one_pass,
    regrouped,
    unreplayed,
    vector_major,
)


def _rows(form: Traversal | Grouped) -> npt.NDArray[np.int64]:
    return offsets(form).reshape(form.beats, form.lanes)


def _carried(chain: Steps, source: Traversal | Grouped) -> npt.NDArray[np.int64]:
    """``source``'s beats (flat offsets, a row a beat) carried through ``chain``: a width
    conversion regroups the element sequence, a reorder emits its frames' beats."""
    beats = _rows(source)
    for hop in chain.hops:
        if hop.step is Step.WIDTH:
            beats = beats.reshape(-1, hop.sink.form.lanes)
        elif hop.step is Step.REORDER:
            assert hop.reorder is not None
            frame = hop.reorder.frame_beats
            local = frame_slots(hop.reorder)
            frames = np.arange(len(beats) // frame, dtype=np.int64)[:, None] * frame
            beats = beats[(frames + local).ravel()]
    return beats


def _line_steps(source: BeatSequence, sink: BeatSequence) -> Any:
    """The line's answer to the same question, as the revision's values."""
    return translate.forward(line.steps(translate.back(source), translate.back(sink)))


def _lanes(chain: Steps) -> list[tuple[int, int]]:
    return [(hop.source.form.lanes, hop.sink.form.lanes) for hop in chain.hops]


# -- the regroup ---------------------------------------------------------------------------

ALONG_ROWS = vector_major((4, 4), 4)
ALONG_COLUMNS = Traversal.over((4, 4), ((1, 4, 1),), ((0, 4, 1),))
TILES = Traversal.over((4, 8), ((0, 2, 2), (1, 4, 2)), ((0, 2, 1), (1, 2, 1)))
"""(4, 8) in 2 x 2 tiles: two rows of two columns a beat."""


@pytest.mark.parametrize(
    ("source", "sink", "reorder", "window"),
    [
        (ALONG_ROWS, ALONG_COLUMNS, (16, (4, 4), (1, 4)), 12),
        (ALONG_COLUMNS, ALONG_ROWS, (16, (4, 4), (1, 4)), 12),
    ],
    ids=["rows-to-columns", "columns-to-rows"],
)
def test_four_lanes_on_one_axis_regroup_through_one_lane_onto_the_other(
    source: Traversal, sink: Traversal, reorder: tuple[object, ...], window: int
) -> None:
    """4 x 4: at gcd(4, 4) and at 2 lanes the lanes still lie on different axes; at one
    lane a beat the join is a transpose, one reorder."""
    found = steps(BeatSequence(source), BeatSequence(sink))
    assert found.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert _lanes(found) == [(4, 1), (1, 1), (1, 4)]
    (moved,) = [hop.reorder for hop in found.hops if hop.reorder]
    assert (moved.frame_beats, moved.dims, moved.coefs) == reorder
    assert np.array_equal(_carried(found, source), _rows(sink))
    # 4 + 12 + 4 elements held; the source's 4 beats; one lane at the narrowest.
    assert hop_cost(found.hops[1]).live_window == window
    assert cost(found, BeatSequence(sink)) == Cost(4 + window + 4, 4, 1)
    with pytest.raises(line.Unrealizable, match="another lane axis"):
        _line_steps(BeatSequence(source), BeatSequence(sink))


@pytest.mark.parametrize(
    ("source", "sink", "reorder", "window"),
    [
        (TILES, vector_major((4, 8), 4), (8, (2, 4), (1, 2)), 12),
        (vector_major((4, 8), 4), TILES, (8, (4, 2), (1, 4)), 8),
    ],
    ids=["tiles-to-rows", "rows-to-tiles"],
)
def test_a_regroup_goes_through_the_largest_lane_count_that_reorders(
    source: Traversal, sink: Traversal, reorder: tuple[object, ...], window: int
) -> None:
    """2 x 2 tiles against four lanes along rows: refused at gcd(4, 4), a reorder at 2
    lanes (a pair of columns a beat), so 1 lane is not needed."""
    found = steps(BeatSequence(source), BeatSequence(sink))
    assert found.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH)
    assert _lanes(found) == [(4, 2), (2, 2), (2, 4)]
    (moved,) = [hop.reorder for hop in found.hops if hop.reorder]
    assert (moved.frame_beats, moved.dims, moved.coefs) == reorder
    assert np.array_equal(_carried(found, source), _rows(sink))
    # Held elements: 4 a width conversion, the reorder's beats times their 2 lanes.
    assert hop_cost(found.hops[1]).live_window == window
    assert cost(found, BeatSequence(sink)) == Cost(4 + window + 4, 8, 2)
    with pytest.raises(line.Unrealizable, match="another lane axis"):
        _line_steps(BeatSequence(source), BeatSequence(sink))


def test_a_regroup_of_other_positions_or_no_loop_nest_stays_refused() -> None:
    # A sink reading positions the source never presents, at any lane count.
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    with pytest.raises(Unrealizable, match="different positions"):
        steps(BeatSequence(half), BeatSequence(vector_major((4,), 2)))
    # Another tensor.
    with pytest.raises(Unrealizable, match=r"\(4, 4\) and \(2, 8\)"):
        steps(BeatSequence(ALONG_ROWS), BeatSequence(vector_major((2, 8), 4)))


# -- the transpose before a dense dot product ------------------------------------------------

ROWS, COLUMNS, OUTPUTS = 32, 16, 16
"""X (32, 16) row-major; Y = Xᵀ read by a dot product with M 16, K 32, N 16."""


def _transpose(lanes: int, pe: int, simd: int) -> tuple[BeatSequence, BeatSequence]:
    """The producer's X at ``lanes``, and the dot product's port over Y = Xᵀ stated over X."""
    schedule = Schedule({m: COLUMNS, n: OUTPUTS, k: ROWS}, {n: pe, k: simd}, (m, n, k))
    port = Traversal.over(
        (ROWS, COLUMNS),
        ((1, COLUMNS, 1), (None, OUTPUTS // pe, 0), (0, ROWS // simd, simd)),
        ((0, simd, 1),),
    )
    return (
        BeatSequence(vector_major((ROWS, COLUMNS), lanes)),
        BeatSequence(port, markers=(schedule.closing((k,)),)),
    )


@pytest.mark.parametrize(
    ("arrangement", "lanes", "reads", "window"),
    [
        ((4, 4, 4), [(4, 1), (1, 1), (1, 4), (4, 4)], 128, 4 + 497 + 4),
        ((2, 4, 2), [(2, 1), (1, 1), (1, 2), (2, 2)], 256, 2 + 497 + 2),
    ],
    ids=["lanes-4-pe-4-simd-4", "lanes-2-pe-4-simd-2"],
)
def test_the_transposes_the_line_refuses_build_through_one_lane(
    arrangement: tuple[int, int, int],
    lanes: list[tuple[int, int]],
    reads: int,
    window: int,
) -> None:
    source, sink = _transpose(*arrangement)
    with pytest.raises(line.Unrealizable, match="different positions"):
        _line_steps(source, sink)
    found = steps(source, sink)
    assert found.kinds == (Step.WIDTH, Step.REORDER, Step.WIDTH, Step.MARKERS)
    assert _lanes(found) == lanes
    (moved,) = [hop.reorder for hop in found.hops if hop.reorder]
    assert (moved.frame_beats, moved.dims, moved.coefs) == (512, (16, 4, 32), (1, 0, 16))
    assert np.array_equal(_carried(found, source.form), _rows(sink.form))
    assert found.hops[-1].sink.markers == sink.markers
    # The reorder at one lane a beat holds 497 elements, the composed transpose and replay.
    assert hop_cost(found.hops[1]).live_window == 497
    assert cost(found, sink) == Cost(window, reads, 1)


@pytest.mark.parametrize("arrangement", [(1, 4, 1), (4, 4, 1), (1, 1, 4)])
def test_the_transposes_the_line_builds_keep_their_chain(
    arrangement: tuple[int, int, int],
) -> None:
    source, sink = _transpose(*arrangement)
    found = steps(source, sink)
    assert found == _line_steps(source, sink)
    assert np.array_equal(_carried(found, source.form), _rows(sink.form))


# -- the looping source --------------------------------------------------------------------

G, TH = Index("g"), Index("th")
TILED = [(6, 12, 6, 3, 4, 3), (8, 16, 8, 4, 4, 2), (6, 12, 4, 2, 6, 3)]
"""A tiled dot product (N, MW, MH, PE, SIMD, TH): TH vectors share each weight tile."""


def _weights(
    vectors: int, depth: int, outputs: int, pe: int, simd: int, tile: int
) -> tuple[Traversal, Traversal]:
    """The core's weight port, and the memory's port: the matrix without its replay, in
    tiles of PE x SIMD cut into TH beats (no loop nest where those lanes cross a row)."""
    core = Schedule(
        {G: vectors // tile, n: outputs, k: depth, TH: tile}, {n: pe, k: simd}, (G, n, k, TH)
    )
    weights = core.present((depth, outputs), (k, n), lanes=(n, k))
    return weights, regrouped(unreplayed(weights), pe * simd // tile)


@pytest.mark.parametrize(
    ("geometry", "line_cost", "next_cost"),
    [
        (TILED[0], Cost(12 + 6 * 12, 108, 4), Cost(12 + 1 * 12, 36, 4)),
        (TILED[1], Cost(16 + 8 * 16, 128, 8), Cost(16 + 1 * 16, 64, 8)),
    ],
    ids=["6-12-6-3-4-3", "8-16-8-4-4-2"],
)
def test_a_looping_source_presents_the_matrix_once_a_group_of_tiled_vectors(
    geometry: tuple[int, ...], line_cost: Cost, next_cost: Cost
) -> None:
    """The weights from a memory looping over one matrix: the matrix N/TH times, then
    each beat replayed TH times, where the line streams it N times and keeps one in TH."""
    vectors, depth, outputs, pe, simd, tile = geometry
    weights, port = _weights(*geometry)
    source, sink = BeatSequence(one_pass(port), Repetition.CYCLIC), BeatSequence(weights)
    elements = depth * outputs
    found = presented(source, sink)
    assert found.beats * found.lanes // elements == vectors // tile
    lined = translate.forward(line.presented(translate.back(source), translate.back(sink)))
    assert lined.beats * lined.lanes // elements == vectors
    chain, before = steps(source, sink), _line_steps(source, sink)
    assert chain.kinds == before.kinds == (Step.WIDTH, Step.REORDER)
    assert chain.hops[1].reorder is not None
    reorder = chain.hops[1].reorder
    assert (reorder.frame_beats, reorder.dims, reorder.coefs) == (1, (tile,), (0,))
    assert np.array_equal(_carried(chain, found), _rows(weights))
    assert np.array_equal(_carried(before, lined), _rows(weights))
    assert (cost(before, sink), cost(chain, sink)) == (line_cost, next_cost)
    assert smaller(next_cost, line_cost)


def test_a_tiled_weight_port_whose_lanes_cross_a_row_stays_refused() -> None:
    """PE 2, SIMD 6, TH 3: four lanes a beat against tile rows of six is no loop nest, so
    no source to count; a later change gives such lanes a stream."""
    with pytest.raises(ValueError, match="does not align"):
        _weights(*TILED[2])


def test_a_consumer_reading_part_of_its_pass_twice_counts_by_what_it_reads() -> None:
    """Half of a cycled (4, 4) tensor, each beat twice: without its replay the consumer
    reads half a pass, no whole one, so its elements with the replay count, as the line's."""
    whole = vector_major((4, 4), 4)
    half_twice = Traversal.over((4, 4), ((0, 2, 1), (None, 2, 0)), ((1, 4, 1),))
    source, sink = BeatSequence(whole, Repetition.CYCLIC), BeatSequence(half_twice)
    assert presented(source, sink) == whole
    assert steps(source, sink) == _line_steps(source, sink)


# -- costs -------------------------------------------------------------------------------


def test_a_pure_reorder_costs_its_live_window_in_its_source_beats() -> None:
    """(2, 3) one element a beat, read column by column: slots 0 3 1 4 2 5, so after the
    second beat read the source has given 4 beats and still owes beat 1: 3 held."""
    sink = BeatSequence(Traversal.over((2, 3), ((1, 3, 1), (0, 2, 1)), ()))
    found = steps(BeatSequence(vector_major((2, 3), 1)), sink)
    assert found.kinds == (Step.REORDER,)
    reorder = found.hops[0].reorder
    assert reorder is not None
    assert list(frame_slots(reorder)) == [0, 3, 1, 4, 2, 5]
    assert cost(found, sink) == Cost(3, 6, 1)


def test_a_reorder_at_several_lanes_holds_its_beats_lanes() -> None:
    found = steps(BeatSequence(vector_major((4, 8), 4)), BeatSequence(TILES))
    reorder = found.hops[1]
    assert reorder.step is Step.REORDER and reorder.source.form.lanes == 2
    assert hop_cost(reorder) == Cost(4 * 2, 16, 2)


def test_a_width_conversion_holds_the_common_multiple_of_its_lanes() -> None:
    sink = BeatSequence(vector_major((2, 6), 3))
    found = steps(BeatSequence(vector_major((2, 6), 2)), sink)
    assert found.kinds == (Step.WIDTH,)
    assert cost(found, sink) == Cost(6, 6, 2)


def test_markers_and_a_direct_connection_hold_nothing() -> None:
    form = vector_major((2, 4), 2)
    sink = BeatSequence(form, markers=(LevelEnd(2),))
    found = steps(BeatSequence(form), sink)
    assert found.kinds == (Step.MARKERS,)
    assert cost(found, sink) == Cost(0, 4, 2)
    assert cost(Steps(), BeatSequence(form)) == Cost(0, 4, 2)


def test_a_reorder_step_states_or_finds_its_reorder() -> None:
    found = steps(BeatSequence(ALONG_ROWS), BeatSequence(ALONG_COLUMNS))
    bare = Hop(Step.REORDER, found.hops[1].source, found.hops[1].sink)
    assert hop_cost(bare) == hop_cost(found.hops[1])
    with pytest.raises(ValueError, match="without a reorder"):
        hop_cost(Hop(Step.REORDER, BeatSequence(ALONG_ROWS), BeatSequence(ALONG_COLUMNS)))


@pytest.mark.parametrize("fields", [(-1, 1, 1), (0, 0, 1), (0, 1, 0), (True, 1, 1)], ids=str)
def test_a_cost_refuses_what_no_chain_costs(fields: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        Cost(*fields)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Cost(10, 4, 2), Cost(12, 6, 2), True),  # both smaller
        (Cost(10, 6, 2), Cost(12, 6, 2), True),  # one smaller, one equal
        (Cost(10, 6, 4), Cost(12, 6, 2), True),  # and wider
        (Cost(12, 6, 4), Cost(12, 6, 2), True),  # wider only: the same window and reads
        (Cost(12, 6, 2), Cost(12, 6, 4), False),  # narrower only
        (Cost(12, 8, 4), Cost(12, 6, 2), False),  # wider, but more reads
        (Cost(12, 6, 2), Cost(12, 6, 2), False),  # equal
        (Cost(10, 8, 2), Cost(12, 6, 2), False),  # a trade-off: a smaller window, more reads
        (Cost(12, 6, 2), Cost(10, 8, 2), False),  # the same trade-off the other way
        (Cost(4, 2, 1), Cost(12, 6, 2), False),  # narrower: smaller buffers, but slower
        (Cost(14, 8, 2), Cost(12, 6, 2), False),  # larger
    ],
)
def test_smaller_is_no_field_worse_and_one_strictly_better(
    first: Cost, second: Cost, expected: bool
) -> None:
    assert smaller(first, second) is expected
