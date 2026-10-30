# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The tiled MatMul taken whole, in three geometries: the activations' replay into the
core, the output reorder out of it, and the network of all of them in time.

The core's schedule is (g, n, k, th) with v = g * TH + th, n folded by PE and k by SIMD,
so a frame is NF * SF * N steps (N / TH * NF * SF * TH). The geometries are
(N, MW, MH, PE, SIMD, TH) with NF = MH / PE, SF = MW / SIMD, WSIMD = PE * SIMD / TH:

- (6, 12, 6, 3, 4, 3): NF = 2, SF = 3, WSIMD = 4; 36 steps a frame.
- (8, 16, 8, 4, 4, 2): NF = 2, SF = 4, WSIMD = 8; 64 steps a frame.
- (6, 12, 4, 2, 6, 3): NF = 2, SF = 2, WSIMD = 4; 24 steps a frame. The weight port is a
  ``Grouped``: 4 lanes run against tile rows of 6.

**The activations.** From a row-major producer, (N, MW) at SIMD lanes, to the core's port,
the steps are one reorder of frame SF * TH beats, dims (NF, SF, TH), coefs (0, 1, SF):
(9, (2, 3, 3), (0, 1, 3)), (8, (2, 4, 2), (0, 1, 4)) and (6, (2, 2, 3), (0, 1, 2)). Each
delivers (``harness.checks.undelivered`` finds nothing).

**The output.** From the core's port, (N, MH) at PE lanes, to a row-major consumer, the
steps are one reorder of frame NF * TH beats, dims (TH, NF), coefs (1, TH):
(6, (3, 2), (1, 3)), (4, (2, 2), (1, 2)) and (6, (3, 2), (1, 3)). Each delivers.

**The weights and the marker** are asserted by other tests and not repeated:
``test_tiled_weights`` (the port beat by beat, the memories' depths, the chain from the
memory to the port and to the core's weights, the four-field verdict between the two
stored forms) and ``test_level_done.test_the_tiled_core_s_k_is_on_its_last_step_across_th``
(k's last index value, a ``LevelDone`` of beats SF * TH and inner TH). Here only: the weight chain
into the core is a width step then a replay of each beat TH times (frame one beat, dims
(TH,), coefs (0,)), and in the third geometry it builds from a ``Grouped``.

**The whole, in time.** One ``timing.Network`` a geometry: a source of the activations
(released every step count of the core), their reorder, a memory presenting the weight
port's matrix N / TH times through the width step and the TH-fold replay the derived
weight steps say, the core (it reads both ports on every step and writes its output when k
closes, its latency 0), the output reorder and a sink. The seven channels are, in the order
the search lowers them, ``act``, ``act_replayed``, ``w_mem``, ``w_wide``, ``w_replayed``,
``result`` and ``out``; 6 frames are run.

Each reorder is built unbounded: its capacity is its words over the 6 frames run
(``words * RUN_FRAMES``), which never binds. The live window is the buffer the order needs,
not the one the rate needs, so it is no capacity to size a reorder by: with each reorder at
its live window the sink's period, every channel unbounded, is 60, 128 and 40 cycles under
FinnLib's rules against the core's 36, 64 and 24 (48, 96 and 32 under the one-cycle
rules). With the reorders unbounded it is the core's, at the sink and at the core, under
both rules. Each reorder's least capacity keeping that period, the others unbounded, is
recorded next to its live window as data, exactly: ``timing.least`` sizes a channel, and a
reorder's capacity is its stage's, so each is the least of a scan from one word up that
keeps the period at the sink (and keeps it from there on).

``minimal`` sizes the channels. Its default criterion (``period_kept``) reads the sink as
well as the sources: the sources' lateness alone (``Run.on_time``) is hidden by a reorder
that holds the backlog, and under the one-cycle rules it returns a register a channel (a
word moving every second cycle) at twice the core's period at the sink, which is recorded
too, as the reason the default reads the ends. The default returns the sets below, each
channel above direct one word short of the period when lowered by one, and so does a hand
criterion that reads the sink's period (``keeps``), the same sets. Direct is a handshake
(0 words) under FinnLib's rules, a one-word register under the one-cycle rules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pytest

from dataflow.harness.checks import undelivered
from finn.dataflow.gemm import k, n
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.steps import Step, Steps, steps
from finn.dataflow.timing import (
    FINNLIB,
    ONE_CYCLE,
    Channel,
    Memory,
    Network,
    ReorderStage,
    Run,
    Sink,
    Source,
    WidthStage,
    kernel,
    minimal,
    period_kept,
    run,
    step_stages,
)
from finn.dataflow.traversal import (
    BeatSequence,
    Grouped,
    Reorder,
    Traversal,
    grouped,
    unreplayed,
    vector_major,
)

G, TH = Index("g"), Index("th")
RUN_FRAMES = 6
UNBOUNDED = 10**6  # more words than any reorder of these geometries holds
CHANNELS = ("act", "act_replayed", "w_mem", "w_wide", "w_replayed", "result", "out")
RULES = {"FinnLib": FINNLIB, "one cycle": ONE_CYCLE}


@dataclass(frozen=True)
class Tiled:
    """One geometry and every number a test of it states."""

    vectors: int  # N
    depth: int  # MW
    outputs: int  # MH
    pe: int
    simd: int
    tile: int  # TH
    port_is_grouped: bool
    iterations_a_frame: int  # the core's period
    activation_reorder: tuple[int, tuple[int, ...], tuple[int, ...]]
    output_reorder: tuple[int, tuple[int, ...], tuple[int, ...]]
    window: tuple[int, int, int]  # the live windows of the activation, weight, output reorders
    window_period: tuple[int, int]  # the sink's period at the windows: FinnLib, one cycle
    least: dict[str, tuple[int, int, int]]  # the least capacity of each of the three, by rules

    @property
    def nf(self) -> int:
        return self.outputs // self.pe

    @property
    def sf(self) -> int:
        return self.depth // self.simd

    @property
    def wsimd(self) -> int:
        return self.pe * self.simd // self.tile

    def core(self) -> Schedule:
        return Schedule(
            {G: self.vectors // self.tile, n: self.outputs, k: self.depth, TH: self.tile},
            {n: self.pe, k: self.simd},
            (G, n, k, TH),
        )

    def activations(self) -> Traversal:
        return self.core().present((self.vectors, self.depth), (G * self.tile + TH, k), lanes=(k,))

    def results(self) -> Traversal:
        return self.core().present(
            (self.vectors, self.outputs), (G * self.tile + TH, n), lanes=(n,), reduces=(k,)
        )

    def weights(self) -> Traversal:
        return self.core().present((self.depth, self.outputs), (k, n), lanes=(n, k))

    def weight_port(self) -> Traversal | Grouped:
        return grouped(unreplayed(self.weights()), self.wsimd)

    def activation_steps(self) -> Steps:
        row_major = BeatSequence(vector_major((self.vectors, self.depth), self.simd))
        return steps(row_major, BeatSequence(self.activations()))

    def output_steps(self) -> Steps:
        row_major = BeatSequence(vector_major((self.vectors, self.outputs), self.pe))
        return steps(BeatSequence(self.results()), row_major)

    def weight_steps(self) -> Steps:
        return steps(BeatSequence(self.weight_port()), BeatSequence(self.weights()))


# For each geometry: the numbers stated above, then the live windows (activation replay,
# weight replay, output reorder), as found, and what they cost in time (see the docstring).
CASES = [
    Tiled(
        6, 12, 6, 3, 4, 3, False, 36, (9, (2, 3, 3), (0, 1, 3)), (6, (3, 2), (1, 3)),
        (9, 1, 3), (60, 48), {"FinnLib": (11, 2, 3), "one cycle": (10, 2, 3)},
    ),
    Tiled(
        8, 16, 8, 4, 4, 2, False, 64, (8, (2, 4, 2), (0, 1, 4)), (4, (2, 2), (1, 2)),
        (8, 1, 2), (128, 96), {"FinnLib": (9, 2, 2), "one cycle": (8, 2, 2)},
    ),
    Tiled(
        6, 12, 4, 2, 6, 3, True, 24, (6, (2, 2, 3), (0, 1, 2)), (6, (3, 2), (1, 3)),
        (6, 1, 3), (40, 32), {"FinnLib": (7, 2, 3), "one cycle": (6, 2, 3)},
    ),
]  # fmt: skip
IDS = ["6-12-6-3-4-3", "8-16-8-4-4-2", "6-12-4-2-6-3"]


def delivers(chain: Steps, source: Traversal | Grouped, sink: Traversal | Grouped) -> None:
    assert undelivered(chain, BeatSequence(source), BeatSequence(sink)) == ""


def one_reorder(chain: Steps) -> Reorder:
    assert chain.kinds == (Step.REORDER,)
    found = chain.hops[0].reorder
    assert found is not None
    return found


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_activations_are_one_replay_of_a_frame_of_sf_times_th(case: Tiled) -> None:
    chain = case.activation_steps()
    reorder = one_reorder(chain)
    assert (reorder.frame_beats, reorder.dims, reorder.coefs) == case.activation_reorder
    # The frame is SF * TH beats, the dims (NF, SF, TH): NF times over, k fast, th slow.
    assert case.activation_reorder == (
        case.sf * case.tile,
        (case.nf, case.sf, case.tile),
        (0, 1, case.sf),
    )
    delivers(chain, vector_major((case.vectors, case.depth), case.simd), case.activations())


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_output_is_one_reorder_of_a_frame_of_nf_times_th(case: Tiled) -> None:
    chain = case.output_steps()
    reorder = one_reorder(chain)
    assert (reorder.frame_beats, reorder.dims, reorder.coefs) == case.output_reorder
    assert case.output_reorder == (case.nf * case.tile, (case.tile, case.nf), (1, case.tile))
    delivers(chain, case.results(), vector_major((case.vectors, case.outputs), case.pe))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_weights_into_the_core_are_a_width_step_then_a_replay_of_each_beat(
    case: Tiled,
) -> None:
    """The memory presents the port's matrix N / TH times, so the chain is the one the
    weights' own tests derive from the stored form; the third geometry's port is a
    ``Grouped`` and the chain builds from it all the same."""
    port = case.weight_port()
    assert isinstance(port, Grouped) == case.port_is_grouped
    chain = case.weight_steps()
    assert chain.kinds == (Step.WIDTH, Step.REORDER)
    assert chain.hops[0].source.form == port
    assert [(hop.source.form.lanes, hop.sink.form.lanes) for hop in chain.hops] == [
        (case.wsimd, case.pe * case.simd),
        (case.pe * case.simd, case.pe * case.simd),
    ]
    assert chain.hops[1].reorder == Reorder(1, (case.tile,), (0,))
    delivers(chain, port, case.weights())


def stages_of(case: Tiled) -> tuple[ReorderStage, WidthStage, ReorderStage, ReorderStage]:
    """The chains' stages: the activation replay, the weights' width step and replay, and
    the output reorder, each reorder at its live window (``step_stages``' capacity)."""
    (replay,) = step_stages(case.activation_steps(), "act")
    wide, w_replay = step_stages(case.weight_steps(), "w")
    (back,) = step_stages(case.output_steps(), "result")
    assert isinstance(replay, ReorderStage) and isinstance(w_replay, ReorderStage)
    assert isinstance(wide, WidthStage) and isinstance(back, ReorderStage)
    return replay, wide, w_replay, back


def reorders(case: Tiled) -> tuple[ReorderStage, ReorderStage, ReorderStage]:
    replay, _, w_replay, back = stages_of(case)
    return replay, w_replay, back


def whole(case: Tiled, capacities: Sequence[int] | None = None) -> Network:
    """The tiled MatMul as a network, every channel unbounded. Each reorder is unbounded
    (``words * RUN_FRAMES``) unless ``capacities`` states the activation replay's, the
    weight replay's and the output reorder's."""
    core = case.core()
    replay, wide, w_replay, back = stages_of(case)
    sized = [stage.words * RUN_FRAMES for stage in (replay, w_replay, back)]
    if capacities is not None:
        sized = list(capacities)
    replay, w_replay, back = (
        replace(stage, capacity=capacity)
        for stage, capacity in zip((replay, w_replay, back), sized)
    )
    words = case.vectors * case.depth // case.simd
    network = Network(
        (
            Source("source", tuple(range(words)), words, core.iterations),
            replay,
            Memory("memory"),
            wide,
            w_replay,
            kernel("core", (Pace(core), Pace(core)), (Pace(core, reduces=(k,)),)),
            back,
            Sink("sink"),
        ),
        (
            Channel("act", "source", replay.name),
            Channel("act_replayed", replay.name, "core"),
            Channel("w_mem", "memory", wide.name),
            Channel("w_wide", wide.name, w_replay.name),
            Channel("w_replayed", w_replay.name, "core", input=1),
            Channel("result", "core", back.name),
            Channel("out", back.name, "sink"),
        ),
    )
    return network.unbounded()


def sink_period(found: Run) -> int | None:
    return found.period("sink")


def keeps(case: Tiled) -> Callable[[Run], bool]:
    """The hand criterion that reads the sink's period as well as the sources' lateness."""

    def criterion(found: Run) -> bool:
        return found.on_time and sink_period(found) == case.iterations_a_frame

    return criterion


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_network_runs_the_core_s_iterations_a_frame_and_its_channels_carry_the_frames_words(
    case: Tiled,
) -> None:
    network = whole(case)
    assert network.period == case.iterations_a_frame == case.core().iterations
    assert tuple(channel.name for channel in network.channels) == CHANNELS
    assert tuple(stage.capacity for stage in reorders(case)) == case.window
    words = case.vectors * case.depth // case.simd
    assert dict(network.words) == {
        "act": words,
        "act_replayed": case.iterations_a_frame,
        "w_mem": case.iterations_a_frame,
        "w_wide": case.iterations_a_frame // case.tile,
        "w_replayed": case.iterations_a_frame,
        "result": case.vectors * case.nf,
        "out": case.vectors * case.nf,
    }


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_with_every_channel_and_reorder_unbounded_it_keeps_the_core_s_period(
    case: Tiled, name: str
) -> None:
    found = run(whole(case), RULES[name], RUN_FRAMES)
    assert not found.deadlocked and period_kept(whole(case), RULES[name], RUN_FRAMES)(found)
    assert sink_period(found) == found.period("core") == case.iterations_a_frame


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_with_each_reorder_at_its_live_window_the_sink_s_period_is_longer(case: Tiled) -> None:
    """As found (data): the windows are what the order needs, and the period at them."""
    network = whole(case, case.window)
    for (name, rules), period in zip(RULES.items(), case.window_period):
        found = run(network, rules, RUN_FRAMES)
        assert not found.deadlocked and sink_period(found) == period, name
        assert period > case.iterations_a_frame


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_least_capacity_of_each_reorder_keeping_the_period_is_this_data(
    case: Tiled, name: str
) -> None:
    """(activation replay, weight replay, output reorder) beside the live windows
    (``Tiled.window``): the others unbounded, the least capacity from one word up that
    keeps the period at the sink, and every capacity above it does."""
    rules, criterion = RULES[name], keeps(case)
    found = []
    for place in range(3):
        words = reorders(case)[place].words * RUN_FRAMES
        keeping = []
        for capacity in range(1, case.window[place] + 6):
            sized = [words, words, words]
            sized[place] = capacity
            keeping.append(criterion(run(whole(case, sized), rules, RUN_FRAMES)))
        assert all(keeping[keeping.index(True) :])
        found.append(keeping.index(True) + 1)
    assert tuple(found) == case.least[name]


# The minimal capacities, in CHANNELS' order, with every reorder unbounded: with the hand
# criterion that reads the sink's period, which the default (``period_kept``) gives too; and
# with the sources' lateness alone, under the one-cycle rules a register a channel at twice
# the core's period.
MINIMAL = {"FinnLib": (0,) * 7, "one cycle": (1, 2, 2, 1, 2, 2, 1)}
MINIMAL_BY_SOURCES = {"FinnLib": (0,) * 7, "one cycle": (1,) * 7}


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_minimal_channels_keep_the_core_s_period_at_the_sink(case: Tiled, name: str) -> None:
    rules, criterion, network = RULES[name], keeps(case), whole(case)
    found = minimal(network, rules, RUN_FRAMES, criterion)
    assert found.order == CHANNELS and found.period == case.iterations_a_frame
    assert found.frames == RUN_FRAMES
    assert found.capacities == MINIMAL[name]
    at = network.at(found.found)
    assert criterion(run(at, rules, RUN_FRAMES))
    # Minimal: one word less in any channel above direct loses the period.
    for channel in CHANNELS:
        if found.capacity(channel) > rules.direct_capacity:
            fewer = run(at.at({channel: found.capacity(channel) - 1}), rules, RUN_FRAMES)
            assert not criterion(fewer), channel


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_default_criterion_gives_the_hand_criterion_s_set_at_the_core_s_period(
    case: Tiled, name: str
) -> None:
    rules, network = RULES[name], whole(case)
    found = minimal(network, rules, RUN_FRAMES)
    assert found.capacities == MINIMAL[name]
    assert found.capacities == minimal(network, rules, RUN_FRAMES, keeps(case)).capacities
    assert found.criterion == "period_kept"
    at = run(network.at(found.found), rules, RUN_FRAMES)
    assert period_kept(network, rules, RUN_FRAMES)(at) and not at.deadlocked
    assert sink_period(at) == case.iterations_a_frame


def sources_only(found: Run) -> bool:
    return found.on_time


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_sources_lateness_alone_loses_the_period_at_the_sink_under_the_one_cycle_rules(
    case: Tiled, name: str
) -> None:
    """The old default's fault, the reason the default reads the ends: the sources are on
    time, a reorder holds the backlog, and a register a channel moves a word every second
    cycle, so the sink completes a frame every 2 * steps cycles."""
    rules, network = RULES[name], whole(case)
    found = minimal(network, rules, RUN_FRAMES, sources_only)
    assert found.capacities == MINIMAL_BY_SOURCES[name]
    assert found.criterion == "sources_only"
    at = run(network.at(found.found), rules, RUN_FRAMES)
    assert at.on_time and not at.deadlocked
    assert sink_period(at) == case.iterations_a_frame * (1 if name == "FinnLib" else 2)
    assert period_kept(network, rules, RUN_FRAMES)(at) == (name == "FinnLib")
