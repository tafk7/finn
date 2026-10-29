# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A residual's time (``timing``): a fork that rejoins, sized under both channel rules.

**The geometry.** x of shape (16, 8) at 2 lanes (64 words a frame) is forked to an input
generator and to a skip channel. The generator feeds a MatMul (M 16, K 8, N 8, PE 2,
SIMD 2: 256 steps a frame, each row of 4 words read 4 times), then a Thresholding (PE 2,
latency 2), then an Add (PE 2, latency 1) which also reads the skip channel. The
elementwise kernels take 64 steps a frame. The sources are released every 256 cycles (the
MatMul's steps), 6 frames are run, and the MatMul's latency is 0, 4, 16 and 40. The
channels, in the order the search lowers them: ``x`` (source to fork), ``to_gen`` (fork to
generator), ``skip`` (fork to Add), ``gen_out``, ``y`` (MatMul to Thresholding), ``t``
(Thresholding to Add) and ``out``.

**Under the one-cycle rules** (every word visible the cycle after it is written, a direct
connection a one-word register), exactly:

- the skip channel's unbounded peak is 50, 51, 54 and 60 words;
- the joint minimal skip / y is 5/1, 6/2, 9/5 and 15/11 words, and the sink's period at
  that set is 256;
- with every channel direct the network deadlocks (the fork needs both branches to take
  its word, and the Add needs the skip word before the MatMul's reaches it);
- one budget between the FIFO before the generator and the generator's own capacity, at
  MatMul latency 0 and 16: 2 words and a generator of 4 keep 256; 1 and 5 keep 256; 1 and
  4 give 272.

**Under FinnLib's rules** the shape is what is stated, and the numbers are as found:

1. At the depths the line's per-channel sizing (``fifo_sizing.least_depth``) gives each
   channel from its own two sides, the network deadlocks or fails the period (found: it
   deadlocks, the line giving every channel direct).
2. The skip channel's minimal capacity never falls as the MatMul's latency grows, and is
   larger at 40 than at 0 (5, 6, 9, 15 words; its unbounded peaks 52, 53, 56, 62).
"""

from __future__ import annotations

import pytest

from finn.dataflow_next.buffers import live_window, slots
from finn.dataflow_next.gemm import k, m, n
from finn.dataflow_next.schedule import Index, Pace, Schedule
from finn.dataflow_next.timing import (
    FINNLIB,
    ONE_CYCLE,
    Channel,
    Fork,
    Network,
    ReorderStage,
    Rules,
    Run,
    Sink,
    Source,
    frees,
    kernel,
    minimal,
    period_kept,
    reorder,
    run,
)
from finn.dataflow_next.traversal import vector_major
from finn.kernels.fifo_sizing import Buffered, Pattern, least_depth
from finn.kernels.input_generator import NestBuffer, nest_buffer

M, K, N, PE, SIMD, LANES = 16, 8, 8, 2, 2, 2
WORDS = M * K // LANES  # 64 words of x a frame
RUN_FRAMES = 6
LATENCIES = (0, 4, 16, 40)

A, C = Index("a"), Index("c")
MATMUL = Schedule({m: M, n: N, k: K}, {n: PE, k: SIMD}, (m, n, k))
ELEMENTWISE = Schedule({A: M, C: N}, {C: PE}, (A, C))
PERIOD = MATMUL.iterations  # 256
X_PORT = MATMUL.present((M, K), (m, k), lanes=(k,))
SLOTS = slots(vector_major((M, K), LANES), X_PORT)
READS = [int(s) for s in SLOTS]
WINDOW = live_window(SLOTS)
CHANNELS = ("x", "to_gen", "skip", "gen_out", "y", "t", "out")


def residual(latency: int, generator: ReorderStage | None = None) -> Network:
    """The residual, every channel unbounded; the generator is the package's (its live
    window of 4 words) unless one is given."""
    stages = (
        Source("source", tuple(range(WORDS)), WORDS, PERIOD),
        Fork("fork", 2),
        generator or reorder("input_gen", READS, WORDS),
        kernel("matmul", (Pace(MATMUL),), (Pace(MATMUL, reduces=(k,)),), latency),
        kernel("thresholding", (Pace(ELEMENTWISE),), (Pace(ELEMENTWISE),), 2),
        kernel("add", (Pace(ELEMENTWISE), Pace(ELEMENTWISE)), (Pace(ELEMENTWISE),), 1),
        Sink("sink"),
    )
    links = (
        Channel("x", "source", "fork"),
        Channel("to_gen", "fork", "input_gen"),
        Channel("skip", "fork", "add", output=1),
        Channel("gen_out", "input_gen", "matmul"),
        Channel("y", "matmul", "thresholding"),
        Channel("t", "thresholding", "add", input=1),
        Channel("out", "add", "sink"),
    )
    return Network(stages, links)


def direct(rules: Rules) -> dict[str, int | None]:
    return {name: rules.direct_capacity for name in CHANNELS}


def sink_period(found: Run) -> int | None:
    return found.period("sink")


# --- the one-cycle rules ---------------------------------------------------------------

# Per latency: the unbounded run's skip peak, and the joint minimal set in channel order
# (x, to_gen, skip, gen_out, y, t, out).
ONE_CYCLE_FOUND = {
    0: (50, (1, 2, 5, 2, 1, 1, 1)),
    4: (51, (1, 2, 6, 2, 2, 1, 1)),
    16: (54, (1, 2, 9, 2, 5, 1, 1)),
    40: (60, (1, 2, 15, 2, 11, 1, 1)),
}


def test_the_geometry_is_the_generators_live_window_of_four_words() -> None:
    assert (MATMUL.iterations, WORDS, WINDOW) == (256, 64, 4)
    # Each row of 4 words read 4 times, row after row.
    assert READS[:20] == [0, 1, 2, 3] * 4 + [4, 5, 6, 7]
    net = residual(0)
    assert dict(net.words) == {
        "x": 64, "to_gen": 64, "skip": 64, "gen_out": 256, "y": 64, "t": 64, "out": 64
    }  # fmt: skip
    assert net.period == 256


@pytest.mark.parametrize("latency", LATENCIES)
def test_one_cycle_skip_peak_and_the_joint_minimal_set(latency: int) -> None:
    peak, capacities = ONE_CYCLE_FOUND[latency]
    net = residual(latency)
    keeps = period_kept(net, ONE_CYCLE, RUN_FRAMES)
    free = run(net.unbounded(), ONE_CYCLE, RUN_FRAMES)
    assert free.peak("skip") == peak and sink_period(free) == 256 and keeps(free)
    found = minimal(net, ONE_CYCLE, RUN_FRAMES)
    assert found.order == CHANNELS and found.period == 256 and found.frames == RUN_FRAMES
    assert found.capacities == capacities
    skip, y = found.capacity("skip"), found.capacity("y")
    assert (skip, y) == {0: (5, 1), 4: (6, 2), 16: (9, 5), 40: (15, 11)}[latency]
    at = net.at(found.found)
    kept = run(at, ONE_CYCLE, RUN_FRAMES)
    assert keeps(kept) and sink_period(kept) == 256
    # Minimal: one word less in the skip channel, or in y, loses the period.
    for name in ("skip", "y"):
        if found.capacity(name) > ONE_CYCLE.direct_capacity:
            fewer = run(at.at({name: found.capacity(name) - 1}), ONE_CYCLE, RUN_FRAMES)
            assert not keeps(fewer), name
    # With every channel direct (a one-word register each): a deadlock.
    assert run(net.at(direct(ONE_CYCLE)), ONE_CYCLE, RUN_FRAMES).deadlocked


@pytest.mark.parametrize("latency", (0, 16))
def test_one_cycle_one_budget_between_the_fifo_and_the_generator(latency: int) -> None:
    """The FIFO before the generator and the generator's capacity are one budget: 2 and
    4 words keep 256 cycles a frame, so do 1 and 5, and 1 and 4 give 272 (a frame late)."""
    base = minimal(residual(latency), ONE_CYCLE, RUN_FRAMES).found
    for fifo, capacity, expected in ((2, 4, 256), (1, 5, 256), (1, 4, 272)):
        net = residual(latency, reorder("input_gen", READS, WORDS, capacity))
        found = run(net.at(base | {"to_gen": fifo}), ONE_CYCLE, RUN_FRAMES)
        assert not found.deadlocked
        assert sink_period(found) == expected, (fifo, capacity)
        assert period_kept(net, ONE_CYCLE, RUN_FRAMES)(found) == (expected == 256)


# --- FinnLib's rules -------------------------------------------------------------------

SOURCE = Pattern(tuple(range(WORDS)), WORDS)
ELEMENTWISE_SIDE = Pattern(ELEMENTWISE.beat_times(), ELEMENTWISE.iterations)
MATMUL_SIDE = Pattern(MATMUL.beat_times(), PERIOD)
MATMUL_OUT = Pattern(MATMUL.beat_times(reduces=(k,)), PERIOD)
TAKEN_EACH_CYCLE = Pattern(tuple(range(WORDS)), WORDS)

#: The line's buffer for the generator (its capacity and its row-end frees), and the
#: package's live window; the same reads.
LINE_BUFFER = nest_buffer(WORDS, (M, N // PE, K // SIMD), (K // SIMD, 0, 1))
WINDOW_BUFFER = NestBuffer(WINDOW, tuple(READS), frees(READS, WORDS))


def generator_of(buffer: NestBuffer) -> ReorderStage:
    return ReorderStage("input_gen", buffer.reads, buffer.freed, buffer.capacity)


def line_depths(buffer: NestBuffer) -> dict[str, int | None]:
    """Each channel's depth by the line's per-channel sizing on its own two sides: the
    fork's input and the generator's output are no channel the line sizes (direct)."""
    return {
        "x": 0,
        "gen_out": 0,
        "to_gen": least_depth(SOURCE, Buffered(buffer, MATMUL_SIDE), PERIOD, RUN_FRAMES),
        "skip": least_depth(SOURCE, ELEMENTWISE_SIDE, PERIOD, RUN_FRAMES),
        "y": least_depth(MATMUL_OUT, ELEMENTWISE_SIDE, PERIOD, RUN_FRAMES),
        "t": least_depth(ELEMENTWISE_SIDE, ELEMENTWISE_SIDE, PERIOD, RUN_FRAMES),
        "out": least_depth(ELEMENTWISE_SIDE, TAKEN_EACH_CYCLE, PERIOD, RUN_FRAMES),
    }


def test_the_lines_buffer_reads_the_packages_slots_and_frees_them_later() -> None:
    assert LINE_BUFFER.reads == tuple(READS) == WINDOW_BUFFER.reads
    assert (LINE_BUFFER.capacity, WINDOW_BUFFER.capacity) == (15, 4)
    assert LINE_BUFFER.freed != WINDOW_BUFFER.freed


@pytest.mark.parametrize("buffer", (WINDOW_BUFFER, LINE_BUFFER), ids=("window", "line's"))
@pytest.mark.parametrize("latency", LATENCIES)
def test_finnlib_at_the_lines_depths_the_network_deadlocks(
    buffer: NestBuffer, latency: int
) -> None:
    depths = line_depths(buffer)
    # As found: the line sizes every channel direct, one at a time.
    assert depths == dict.fromkeys(CHANNELS, 0)
    net = residual(latency, generator_of(buffer))
    at = run(net.at(depths), FINNLIB, RUN_FRAMES)
    assert not period_kept(net, FINNLIB, RUN_FRAMES)(at)
    assert at.deadlocked  # as found; the statement is "deadlocks or fails the period"


# The skip channel's unbounded peak, minimal capacity and the joint minimal set (the order
# of CHANNELS): every other channel direct, y too.
FINNLIB_FOUND = {
    0: (52, (0, 0, 5, 0, 0, 0, 0)),
    4: (53, (0, 0, 6, 0, 0, 0, 0)),
    16: (56, (0, 0, 9, 0, 0, 0, 0)),
    40: (62, (0, 0, 15, 0, 0, 0, 0)),
}


@pytest.mark.parametrize("buffer", (WINDOW_BUFFER, LINE_BUFFER), ids=("window", "line's"))
def test_finnlib_skip_minimal_capacity_grows_with_the_matmuls_latency(buffer: NestBuffer) -> None:
    skips = []
    for latency in LATENCIES:
        peak, capacities = FINNLIB_FOUND[latency]
        net = residual(latency, generator_of(buffer))
        free = run(net.unbounded(), FINNLIB, RUN_FRAMES)
        keeps = period_kept(net, FINNLIB, RUN_FRAMES)
        assert free.peak("skip") == peak and keeps(free) and sink_period(free) == 256
        found = minimal(net, FINNLIB, RUN_FRAMES)
        assert found.capacities == capacities
        at = run(net.at(found.found), FINNLIB, RUN_FRAMES)
        assert keeps(at) and sink_period(at) == 256
        # Minimal: one word less in the skip channel loses the period.
        skip = found.capacity("skip")
        fewer = run(net.at(found.found | {"skip": skip - 1}), FINNLIB, RUN_FRAMES)
        assert not keeps(fewer)
        skips.append(skip)
    assert skips == [5, 6, 9, 15]
    assert skips == sorted(skips) and skips[-1] > skips[0]
