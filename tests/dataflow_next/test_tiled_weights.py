# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The tiled dot product's weights in three cases, each stating its geometry and
the numbers it asserts, and checking what it derives by enumeration.

The core's schedule is (g, n, k, th) with v = g * TH + th, n folded by PE and k by SIMD:
its weights ``w`` are W[k, n] (MW, MH) at lanes (n, k), a PE x SIMD tile, pe-major,
replayed TH times, the whole matrix N / TH times. The module's port is the same matrix
without the replay at WSIMD = PE * SIMD / TH lanes: each tile is cut into TH beats, and
the module buffers a tile for TH beats. The cases are (N, MW, MH, PE, SIMD, TH)
with NF = MH / PE, SF = MW / SIMD:

- (6, 12, 6, 3, 4, 3): NF = 2, SF = 3, WSIMD = 4; the port is a ``Traversal`` (4 lanes
  divide the tile rows of 4).
- (8, 16, 8, 4, 4, 2): NF = 2, SF = 4, WSIMD = 8; a ``Traversal`` (8 lanes are two tile
  rows).
- (6, 12, 4, 2, 6, 3): NF = 2, SF = 2, WSIMD = 4; a ``Grouped`` (4 lanes against tile rows
  of 6: a beat straddles two rows).

What each test asserts, per case: the port's positions beat by beat; the depths
of the memories that could feed it, in elements; the chain from a memory storing the port's
one_pass to the port (direct, the matrix N / TH times) and to the core's weights (a width
step, then a reorder replaying each beat TH times); and the four-field verdict between
storing ``w`` at the core's lanes and storing its one_pass with the replay.

The unaligned width alone, outside the matrix (a (6, 2) tensor column by column at 6 lanes
and at 4), is ``test_lane_groups.test_an_unaligned_width_alone_is_one_width_step``, both
directions, so it is not repeated here.
"""

from __future__ import annotations

import pytest

import numpy as np
import numpy.typing as npt
from collections.abc import Sequence
from dataclasses import dataclass
from dataflow_next.harness.checks import undelivered

from finn.dataflow_next.gemm import k, n
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import Cost, Step, cost, presented, source_verdict, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    Reorder,
    Repetition,
    Traversal,
    grouped,
    offsets,
    one_pass,
    regrouped,
    stored,
    stored_depth,
    unreplayed,
)

Array = npt.NDArray[np.int64]
Beats = Sequence[Sequence[tuple[int, ...]]]

G, TH = Index("g"), Index("th")


@dataclass(frozen=True)
class Tiled:
    """One case and every number a test of it states. Depths are in elements,
    costs are ``Cost(live_window, reads, narrowest_lanes)`` of a chain into ``w``."""

    vectors: int  # N
    depth: int  # MW
    outputs: int  # MH
    pe: int
    simd: int
    tile: int  # TH
    port_is_grouped: bool
    one_pass_depth: int  # beats x lanes of ``one_pass(w)``, which keeps the TH-fold replay
    stored_w_depth: int  # ``stored(w)``, at PE * SIMD lanes
    stored_port_beats: int  # ``stored(port)``, at WSIMD lanes
    presentations: int  # N / TH, the matrix presented a frame
    port_chain: Cost  # from ``stored(port)`` to ``w``: a width step, then the reorder
    stored_w_chain: Cost  # from ``stored(w)`` to ``w``
    one_pass_chain: Cost  # from ``one_pass(w)`` to ``w``
    one_pass_kinds: tuple[Step, ...]  # the steps from ``one_pass(w)`` to ``w``
    verdict: str  # ``stored(w)`` against ``one_pass(w)``

    @property
    def nf(self) -> int:
        return self.outputs // self.pe

    @property
    def sf(self) -> int:
        return self.depth // self.simd

    @property
    def wsimd(self) -> int:
        return self.pe * self.simd // self.tile

    @property
    def elements(self) -> int:
        return self.depth * self.outputs

    def weights(self) -> Traversal:
        core = Schedule(
            {G: self.vectors // self.tile, n: self.outputs, k: self.depth, TH: self.tile},
            {n: self.pe, k: self.simd},
            (G, n, k, TH),
        )
        return core.present((self.depth, self.outputs), (k, n), lanes=(n, k))

    def port(self) -> Traversal | Grouped:
        return grouped(unreplayed(self.weights()), self.wsimd)


CASES = [
    Tiled(
        6, 12, 6, 3, 4, 3, False, 216, 72, 18, 2,
        Cost(24, 36, 4), Cost(12, 12, 12), Cost(0, 36, 12), (), "trade-off",
    ),
    Tiled(
        8, 16, 8, 4, 4, 2, False, 256, 128, 16, 4,
        Cost(32, 64, 8), Cost(16, 32, 16), Cost(0, 64, 16), (), "trade-off",
    ),
    Tiled(
        6, 12, 4, 2, 6, 3, True, 144, 48, 12, 2,
        Cost(24, 24, 4), Cost(12, 8, 12), Cost(0, 24, 12), (), "trade-off",
    ),
]  # fmt: skip
IDS = ["6-12-6-3-4-3", "8-16-8-4-4-2", "6-12-4-2-6-3"]


def port_positions(case: Tiled) -> Beats:
    """The port's beats as the module reads them, a (row, column) of W (MW, MH) per lane:
    for each group of TH vectors, each tile (nf, sf) in that order, each of the tile's TH
    beats, lane l of beat b holds element f = b * WSIMD + l of the tile, that is
    (pe, simd) = (f // SIMD, f % SIMD), at W[sf * SIMD + simd, nf * PE + pe]."""
    beats = []
    for _group in range(case.vectors // case.tile):
        for nf in range(case.nf):
            for sf in range(case.sf):
                for beat in range(case.tile):
                    first = beat * case.wsimd
                    beats.append(
                        [
                            (
                                sf * case.simd + (first + lane) % case.simd,
                                nf * case.pe + (first + lane) // case.simd,
                            )
                            for lane in range(case.wsimd)
                        ]
                    )
    return beats


def weight_positions(case: Tiled) -> Beats:
    """The core's beats of ``w``: for each group of TH vectors, each tile (nf, sf), each
    of TH vectors, the whole tile at once, lane (pe, simd) = pe * SIMD + simd holding
    W[sf * SIMD + simd, nf * PE + pe]."""
    beats = []
    for _group in range(case.vectors // case.tile):
        for nf in range(case.nf):
            for sf in range(case.sf):
                for _vector in range(case.tile):
                    beats.append(
                        [
                            (sf * case.simd + simd, nf * case.pe + pe)
                            for pe in range(case.pe)
                            for simd in range(case.simd)
                        ]
                    )
    return beats


def positions_of(form: Traversal | Grouped) -> Beats:
    """``form``'s positions, enumerated one by one (``position``), a row a beat."""
    return [
        [tuple(int(i) for i in form.position(beat, lane)) for lane in range(form.lanes)]
        for beat in range(form.beats)
    ]


def flat(positions: Beats, columns: int) -> Array:
    """Beats of (row, column) positions as flat row-major offsets of a (rows, columns)
    matrix, a row a beat."""
    return np.array([[row * columns + col for row, col in beat] for beat in positions])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_weight_port_is_the_module_s_contract_beat_by_beat(case: Tiled) -> None:
    """The port, ``grouped(unreplayed(w), WSIMD)``: the first two cases a
    ``Traversal`` equal to ``regrouped``'s, the third a ``Grouped`` (``regrouped``
    refuses it). Its positions, beat by beat, are the contract computed by plain loops;
    the core's weights are the same tiles, each replayed TH times."""
    weights, port = case.weights(), case.port()
    assert (weights.beats, weights.lanes) == (
        case.vectors // case.tile * case.nf * case.sf * case.tile,
        case.pe * case.simd,
    )
    if case.port_is_grouped:
        assert isinstance(port, Grouped)
        with pytest.raises(ValueError, match="does not align"):
            regrouped(unreplayed(weights), case.wsimd)
    else:
        assert isinstance(port, Traversal)
        assert port == regrouped(unreplayed(weights), case.wsimd)
    assert (port.beats, port.lanes) == (
        case.vectors // case.tile * case.nf * case.sf * case.tile,
        case.wsimd,
    )
    contract = port_positions(case)
    assert positions_of(port) == contract
    assert np.array_equal(
        offsets(port).reshape(port.beats, port.lanes), flat(contract, case.outputs)
    )
    # The core's weights: positions by enumeration, and each tile of the port widened to
    # PE * SIMD lanes (TH beats into one) then replayed TH times is the core's beats.
    assert positions_of(weights) == weight_positions(case)
    widened = flat(contract, case.outputs).reshape(-1, case.pe * case.simd)
    replayed = np.repeat(widened, case.tile, axis=0)
    assert np.array_equal(replayed, flat(weight_positions(case), case.outputs))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_depths_of_the_memories_that_could_feed_the_port_in_elements(case: Tiled) -> None:
    """The line's stored form, ``one_pass(w)``, keeps the TH-fold replay: NF * SF * TH beats
    of PE * SIMD lanes. ``stored(w)`` holds the matrix once at PE * SIMD lanes; the port's
    ``stored`` holds it once at WSIMD lanes: the same elements at two widths. The matrix
    is presented N / TH times to the port."""
    weights, port = case.weights(), case.port()
    line = one_pass(weights)
    assert (line.beats, line.lanes) == (
        case.nf * case.sf * case.tile,
        case.pe * case.simd,
    )
    assert line.beats * line.lanes == case.one_pass_depth
    assert stored_depth(weights) == case.stored_w_depth == case.elements
    assert stored_depth(one_pass(weights)) == case.stored_w_depth
    form = stored(port)
    assert (form.beats, form.lanes) == (case.stored_port_beats, case.wsimd)
    assert form.beats * form.lanes == stored_depth(port) == case.elements
    shown = presented(BeatSequence(form, Repetition.CYCLIC), BeatSequence(port))
    assert shown == port
    assert shown.beats * shown.lanes // case.elements == case.presentations
    assert case.presentations == case.vectors // case.tile


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_from_the_stored_port_to_the_port_is_direct(case: Tiled) -> None:
    port = case.port()
    source = BeatSequence(stored(port), Repetition.CYCLIC)
    chain = steps(source, BeatSequence(port))
    assert not chain and chain.describe() == "direct"
    assert undelivered(chain, source, BeatSequence(port)) == ""
    assert cost(chain, BeatSequence(port)) == Cost(0, port.beats, case.wsimd)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_from_the_stored_port_to_the_core_s_weights_is_a_width_step_and_a_replay(
    case: Tiled,
) -> None:
    """A width step WSIMD -> PE * SIMD (the module's tile buffer filling), then a reorder
    of one beat, replayed TH times (frame 1 beat, dims (TH,), coefs (0,)). The first two
    cases are the chains the line gives from ``one_pass(port)``, which is
    ``stored(port)`` there."""
    weights, port = case.weights(), case.port()
    source = BeatSequence(stored(port), Repetition.CYCLIC)
    sink = BeatSequence(weights)
    chain = steps(source, sink)
    assert chain.kinds == (Step.WIDTH, Step.REORDER)
    assert chain.describe() == "width_conversion -> reorder"
    assert [(hop.source.form.lanes, hop.sink.form.lanes) for hop in chain.hops] == [
        (case.wsimd, case.pe * case.simd),
        (case.pe * case.simd, case.pe * case.simd),
    ]
    reorder = chain.hops[1].reorder
    assert reorder == Reorder(1, (case.tile,), (0,))
    assert reorder is not None
    assert (reorder.frame_beats, reorder.dims, reorder.coefs) == (1, (case.tile,), (0,))
    assert cost(chain, sink) == case.port_chain
    assert undelivered(chain, source, sink) == ""
    if not case.port_is_grouped:
        assert stored(port) == one_pass(port)
        assert chain == steps(BeatSequence(one_pass(port), Repetition.CYCLIC), sink)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_storing_the_core_s_weights_once_against_storing_their_one_pass(case: Tiled) -> None:
    """``stored(w)`` at PE * SIMD lanes, against ``one_pass(w)``, the line's stored form:
    from the first one reorder replaying each beat TH times, from the second the replay
    is already in the memory: direct, the pass presented once for each of the N / TH
    passes the core makes over the matrix. Every case is a trade-off: a memory a TH-th of
    the size and a TH-th of the reads, for one beat held for the replay. The four fields,
    never summed: depth in elements, then the chain's live window, reads and narrowest
    lanes."""
    weights = case.weights()
    sink = BeatSequence(weights)
    small, line = stored(weights), one_pass(weights)
    assert (small.beats, small.lanes) == (case.nf * case.sf, case.pe * case.simd)
    small_source = BeatSequence(small, Repetition.CYCLIC)
    line_source = BeatSequence(line, Repetition.CYCLIC)
    from_small, from_line = steps(small_source, sink), steps(line_source, sink)
    assert from_small.hops[0].reorder == Reorder(1, (case.tile,), (0,))
    assert from_small.kinds == (Step.REORDER,)
    assert from_line.kinds == case.one_pass_kinds
    assert undelivered(from_small, small_source, sink) == ""
    assert undelivered(from_line, line_source, sink) == ""
    assert cost(from_small, sink) == case.stored_w_chain
    assert cost(from_line, sink) == case.one_pass_chain
    assert (
        source_verdict(
            stored_depth(weights), case.stored_w_chain, case.one_pass_depth, case.one_pass_chain
        )
        == case.verdict
    )
    # By enumeration: each beat of the stored matrix, replayed TH times, is the core's
    # beats of one pass of tiles.
    matrix_pass = flat(weight_positions(case), case.outputs)[: case.nf * case.sf * case.tile]
    assert np.array_equal(
        np.repeat(offsets(small).reshape(-1, case.pe * case.simd), case.tile, axis=0),
        matrix_pass,
    )
