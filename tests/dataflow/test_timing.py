# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A whole network's time (``timing``): each test states its geometry and its numbers.

- **The two rules' direct**, on a source of 4 words a cycle (span 4, period 8) and a
  kernel reading one a step: a handshake moves each word in its cycle (0, 1, 2, 3); a
  one-word register takes two cycles a word (written 0, 2, 4, 6, read 1, 3, 5, 7). A FIFO
  of one word under FinnLib's rules takes three (written 0, 3, 6, 9): late 6 against an
  idle time of 4, where direct keeps the period, so more words can lose it.
- **Chains of two stages against the line's per-channel model**
  (``finn.kernels.fifo_sizing``): every word's leaving and taking cycle and the verdict
  equal ``simulate`` and ``keeps`` at every depth, for paces and for an input generator's
  buffer, and ``least`` equals ``least_depth``.
- **The stages**: a width step's words are the line's ``converted``; a memory source, an
  insertion, a drop, a fork, a kernel's latency, each by hand; a Concat of two values of
  4 and 2 channels at 4 and 1 lanes merged at 2 lanes (pattern 0, 0, 1), each source's
  beats by hand; a Concat of two values of 4 channels at 2 and 4 lanes merged at 2 lanes
  (pattern 0, 0, 1, 1); a value read by two readers through different maps (one through a
  width step, one through a reorder), each branch staged on its own after a fork.
- **The input generator**: the frees after each beat and the live window as its default
  capacity; the line's buffer for the same reorder reads the same words, frees them at
  row ends and holds 7 words where the live window is 2, and under FinnLib's rules the
  live window holds a two-word row read twice to 20 cycles a frame where 16 are due.
- **Sizing** on a small residual (x (4, 4) at 2 lanes forked to an input generator and a
  skip channel, MatMul M 4, K 4, N 4 at PE 2, SIMD 2, Thresholding and Add at PE 2,
  latencies 2 and 1, MatMul's 0 or 4; period 16): the joint minimal set under both rules,
  minimal (one word less in any channel loses the period or deadlocks), and the period
  kept at every capacity of the skip channel above its minimal (monotone), all direct a
  deadlock.
- **Rules that hold a slot longer** (a slot free two cycles after its word is read): the
  peaks count the slot as held, a source, a kernel of latency 2 and a consumer by hand.
- **The criterion and the order** of a search are parameters: a stricter criterion
  sizes more, another order ends at another set, and a loop is refused a size. The
  default criterion (``period_kept``) is the subject of ``test_period_kept``.
"""

from __future__ import annotations

from math import prod

import pytest

from finn.dataflow.buffers import live_window, slots
from finn.dataflow.gemm import k, m, n
from finn.dataflow.maps import concat, identity, pad, transpose
from finn.dataflow.reindex import Merge, Reader, Reindex, branches
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.steps import steps
from finn.dataflow.timing import (
    FINNLIB,
    FRAMES,
    ONE_CYCLE,
    Channel,
    Direct,
    DropStage,
    Fork,
    InsertionStage,
    Kernel,
    Memory,
    MergeStage,
    Network,
    ReorderStage,
    Rules,
    Run,
    Sink,
    Source,
    WidthStage,
    bottleneck,
    chained,
    ends,
    frees,
    kernel,
    least,
    merge_stage,
    minimal,
    order,
    period_kept,
    reorder,
    route_stages,
    run,
    step_stages,
)
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.fifo_sizing import (
    Buffered,
    Pattern,
    converted,
    keeps,
    least_depth,
    simulate,
)
from finn.kernels.input_generator import NestBuffer, nest_buffer

STEADY = (0, 1, 2, 3)


def two(capacity: int | None, period: int = 8) -> Network:
    """A source of four words a cycle (span 4) and a kernel reading one a step (4 steps)."""
    return Network(
        (Source("s", STEADY, 4, period), Kernel("k", 4, (STEADY,), ())),
        (Channel("c", "s", "k", capacity),),
    )


# --- the rules -------------------------------------------------------------------------


def test_the_two_named_rules() -> None:
    assert FINNLIB == Rules(2, Direct.HANDSHAKE, 1, 2)
    assert ONE_CYCLE == Rules(1, Direct.REGISTER, 1, 1)
    assert (FINNLIB.direct_capacity, ONE_CYCLE.direct_capacity) == (0, 1)
    assert [FINNLIB.latency(c) for c in (0, 1, 5, None)] == [0, 2, 2, 2]
    assert [ONE_CYCLE.latency(c) for c in (1, 5, None)] == [1, 1, 1]
    with pytest.raises(ValueError, match="at least 1 words"):
        ONE_CYCLE.latency(0)
    with pytest.raises(ValueError, match="nonnegative"):
        Rules(-1, Direct.HANDSHAKE, 1, 2)


def test_direct_is_a_handshake_or_a_one_word_register_on_a_chain_of_two() -> None:
    handshake = run(two(0), FINNLIB, 2)
    assert handshake.written("c") == handshake.taken("c") == (0, 1, 2, 3, 8, 9, 10, 11)
    assert handshake.late == (("s", (0, 0)),) and handshake.on_time
    register = run(two(1), ONE_CYCLE, 2)
    # Written at 0, visible at 1 and taken; its slot free at 2: a word every two cycles.
    assert register.written("c") == (0, 2, 4, 6, 8, 10, 12, 14)
    assert register.taken("c") == (1, 3, 5, 7, 9, 11, 13, 15)
    assert register.late == (("s", (3, 3)),) and register.on_time  # 3 late, 4 idle
    assert not run(two(1, period=4), ONE_CYCLE, 2).on_time  # no idle time: 3 late is lost


def test_a_fifo_of_one_word_under_finnlibs_rules_loses_what_direct_keeps() -> None:
    # Written at 0, visible two cycles later (2), its slot free the cycle after (3).
    one = run(two(1), FINNLIB, 2)
    assert one.written("c") == (0, 3, 6, 9, 12, 15, 18, 21)
    assert one.taken("c") == (2, 5, 8, 11, 14, 17, 20, 23)
    assert one.late == (("s", (6, 10)),) and not one.on_time
    assert run(two(0), FINNLIB, 2).on_time and run(two(2), FINNLIB, 2).on_time
    # So the search tries direct apart, then bisects the FIFOs above it.
    assert least(two(None), "c", FINNLIB) == 0


# --- chains of two against the line ----------------------------------------------------

MM = Schedule({m: 4, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))  # 16 steps
A, C = Index("a"), Index("c")
EW = Schedule({A: 4, C: 4}, {C: 2}, (A, C))  # 8 steps

# (supply, acceptance, period): MatMul's output (a word at the end of each two-step
# reduction) into an elementwise kernel; a burst of 4 words in a span of 12 into a reader
# of one word every other step; two bursts into a reader of a word a step. The last two at
# a period of their producer's span: no idle time, so each word the reader holds back the
# FIFO must take (least 3 and 5 words).
CHAINS = [
    (Pattern(MM.beat_times(reduces=(k,)), 16), Pattern(EW.beat_times(), 8), 16),
    (Pattern((0, 1, 2, 3), 12), Pattern((0, 2, 4, 6), 8), 12),
    (Pattern((0, 1, 4, 5, 6, 9), 10), Pattern((0, 1, 2, 3, 4, 5), 10), 10),
]


def paced(supply: Pattern, accept: Pattern, period: int, depth: int) -> Network:
    return Network(
        (
            Source("s", supply.times, supply.span, period),
            Kernel("k", accept.span, (accept.times,), ()),
        ),
        (Channel("c", "s", "k", depth),),
    )


@pytest.mark.parametrize("supply, accept, period", CHAINS)
def test_a_chain_of_two_is_the_lines_simulate_word_for_word(
    supply: Pattern, accept: Pattern, period: int
) -> None:
    for depth in range(3 * len(supply.times)):
        flow = simulate(supply, accept, period, depth, FRAMES)
        found = run(paced(supply, accept, period, depth), FINNLIB)
        assert found.on_time == keeps(supply, accept, period, depth)
        assert (flow is None) == found.deadlocked
        if flow is not None:
            assert found.written("c") == flow.left and found.taken("c") == flow.taken
    upper = paced(supply, accept, period, 0)
    assert least(upper, "c", FINNLIB) == least_depth(supply, accept, period)


def test_the_least_depths_of_the_chains_are_the_lines() -> None:
    assert [least(paced(s, a, p, 0), "c", FINNLIB) for s, a, p in CHAINS] == [0, 3, 5]


def test_an_input_generator_on_a_chain_of_two_is_the_lines_buffered_acceptance() -> None:
    # x (4, 4) at 2 lanes read by MatMul (M 4, K 4, N 4, PE 2, SIMD 2): each row of two
    # words read twice; the line's buffer for that nest, and the source a word a cycle.
    buffer = nest_buffer(8, (4, 2, 2), (2, 0, 1))
    reads = Pattern(MM.beat_times(), 16)
    supply = Pattern(tuple(range(8)), 8)
    for depth in range(10):
        flow = simulate(supply, Buffered(buffer, reads), 16, depth, FRAMES)
        net = Network(
            (
                Source("s", supply.times, 8, 16),
                ReorderStage("g", buffer.reads, buffer.freed, buffer.capacity),
                Kernel("k", 16, (reads.times,), ()),
            ),
            (Channel("c", "s", "g", depth), Channel("o", "g", "k", 0)),
        )
        found = run(net, FINNLIB)
        assert flow is not None and not found.deadlocked
        assert found.written("c") == flow.left and found.taken("c") == flow.taken
        assert found.on_time == keeps(supply, Buffered(buffer, reads), 16, depth)


# --- the stages ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lanes_in, lanes_out, times",
    [
        (4, 2, (0, 1, 2, 5)),
        (1, 2, (0, 1, 2, 3, 6, 7)),
        (3, 2, (0, 2, 3, 4)),
        (2, 3, (0, 1, 2, 3, 4, 5)),
        (2, 6, (0, 1, 2)),
        (6, 2, (0, 1)),
    ],
)
def test_a_width_steps_words_are_the_lines_converted(
    lanes_in: int, lanes_out: int, times: tuple[int, ...]
) -> None:
    # Three frames, released far enough apart that the step's words never queue.
    period = 3 * len(times) * max(1, lanes_in // lanes_out) + 4
    net = Network(
        (
            Source("s", times, times[-1] + 1, period),
            WidthStage("w", lanes_in, lanes_out),
            Sink("t"),
        ),
        (Channel("a", "s", "w", 0), Channel("b", "w", "t", None)),
    )
    arrived = [frame * period + time for frame in range(3) for time in times]
    assert run(net, FINNLIB, 3).written("b") == converted(arrived, lanes_in, lanes_out)


def test_a_memory_source_presents_whenever_its_channel_has_room() -> None:
    # Into a FIFO of 2 words read every other step of 8: words 0 and 1 at once, then each
    # the cycle after a word is taken (taken 2, 4, ...: visible two cycles after writing).
    net = Network((Memory("w"), Kernel("k", 8, ((0, 2, 4, 6),), ())), (Channel("c", "w", "k", 2),))
    found = run(net, FINNLIB, 2)
    assert found.written("c") == (0, 1, 3, 5, 7, 9, 11, 13)
    assert found.taken("c") == (2, 4, 6, 8, 10, 12, 14, 16)
    assert found.late == () and found.on_time  # paced by its consumer, never late


def test_an_insertion_and_a_drop_move_a_beat_a_cycle() -> None:
    # Two words a frame (span 2, period 4), a fill before and after them: the source's
    # words leave with beats 1 and 2, the fills take beats 0 and 3.
    inserted = run(
        chained(
            (Source("s", (0, 1), 2, 4), InsertionStage("i", (False, True, True, False)), Sink("t")),
            0,
        ),
        FINNLIB,
        2,
    )
    assert inserted.written("s>i") == (1, 2, 5, 6)
    assert inserted.taken("i>t") == tuple(range(8))
    assert inserted.late == (("s", (1, 1)),) and inserted.on_time
    dropped = run(
        chained(
            (Source("s", STEADY, 4, 4), DropStage("d", (True, False, True, False)), Sink("t")), 0
        ),
        FINNLIB,
        2,
    )
    assert dropped.written("s>d") == tuple(range(8))
    assert dropped.taken("d>t") == (0, 2, 4, 6)


def test_a_fork_sends_each_word_to_every_branch_at_once() -> None:
    # Direct into a sink and into a kernel reading every other step: both take each word
    # when the slower can.
    net = Network(
        (Source("s", STEADY, 4, 8), Fork("f", 2), Sink("t"), Kernel("k", 8, ((0, 2, 4, 6),), ())),
        (
            Channel("x", "s", "f", 0),
            Channel("a", "f", "t", 0),
            Channel("b", "f", "k", 0, output=1),
        ),
    )
    found = run(net, FINNLIB, 1)
    assert found.taken("a") == found.taken("b") == found.written("x") == (0, 2, 4, 6)


def test_a_kernels_latency_on_a_fifo_and_on_a_handshake() -> None:
    def latent(capacity: int | None) -> Network:
        return chained(
            (Source("s", STEADY, 4, 8), Kernel("k", 4, (STEADY,), (STEADY,), 3), Sink("t")),
            capacity,
        ).at({"s>k": 0})

    # On a FIFO: visible its latency and the FIFO's after the step.
    fifo = run(latent(None), FINNLIB, 1)
    assert fifo.written("k>t") == (0, 1, 2, 3) and fifo.taken("k>t") == (5, 6, 7, 8)
    # On a handshake: the pipeline's three registers hold the words in flight.
    held = run(latent(0), FINNLIB, 1)
    assert held.written("k>t") == (0, 1, 2, 3) and held.taken("k>t") == (3, 4, 5, 6)
    assert kernel("mm", (Pace(MM),), (Pace(MM, reduces=(k,)),), 3) == Kernel(
        "mm", 16, (tuple(range(16)),), ((1, 3, 5, 7, 9, 11, 13, 15),), 3
    )


def test_a_fork_rejoined_through_handshakes_alone_deadlocks() -> None:
    # A source forked to two kernels joined by a third: one branch's kernel takes a word
    # only when the join can, and the join only when the other branch's word arrived.
    def diamond(latency: int, first: int) -> Network:
        steady = Kernel("k0", 4, (STEADY,), (STEADY,))
        late = Kernel("k1", 4, (STEADY,), (STEADY,), latency)
        join = Kernel("j", 4, (STEADY, STEADY), (STEADY,))
        return Network(
            (Source("s", STEADY, 4, 8), Fork("f", 2), steady, late, join, Sink("t")),
            (
                Channel("x", "s", "f", 0),
                Channel("a", "f", "k0", 0),
                Channel("b", "f", "k1", 0, output=1),
                Channel("a2", "k0", "j", first),
                Channel("b2", "k1", "j", 0, input=1),
                Channel("o", "j", "t", 0),
            ),
        )

    assert run(diamond(0, 0), FINNLIB, 2).taken("o") == (0, 1, 2, 3, 8, 9, 10, 11)
    stuck = run(diamond(1, 0), FINNLIB, 2)
    assert stuck.deadlocked and stuck.completed("t") == () and not stuck.on_time
    # A FIFO of two words on the other branch: the join waits two cycles for it.
    assert run(diamond(1, 2), FINNLIB, 1).taken("o") == (2, 4, 6, 8)


def test_a_concat_of_two_values_at_different_lanes_is_merged_on_its_pattern() -> None:
    # (2, 2, 4) at 4 lanes and (2, 2, 2) at 1 lane joined along the channels into (2, 2, 6)
    # read at 2 lanes: merged at 2 lanes, two beats of the first, then one of the second.
    pmap = concat(((2, 2, 4), (2, 2, 2)), 2)
    stream = BeatSequence(vector_major((2, 2, 6), 2))
    first = Reindex(BeatSequence(vector_major((2, 2, 4), 4)), (Reader(pmap, stream, None, 0),))
    second = Reindex(BeatSequence(vector_major((2, 2, 2), 1)), (Reader(pmap, stream, None, 1),))
    merge = Merge((first, second))
    assert merge_stage("merge", merge) == MergeStage("merge", (0, 0, 1), 12)
    assert route_stages(first, "a") == (WidthStage("a.after.0", 4, 2),)
    assert route_stages(second, "b") == (WidthStage("b.after.0", 1, 2),)
    net = Network(
        (
            Source("a", STEADY, 4, 16),
            WidthStage("wa", 4, 2),
            Source("b", tuple(range(8)), 8, 16),
            WidthStage("wb", 1, 2),
            merge_stage("merge", merge),
            Sink("sink"),
        ),
        (
            Channel("a>wa", "a", "wa", 0),
            Channel("wa>merge", "wa", "merge", 0),
            Channel("b>wb", "b", "wb", 0),
            Channel("wb>merge", "wb", "merge", 0, input=1),
            Channel("merge>sink", "merge", "sink", 0),
        ),
    )
    found = run(net, FINNLIB, 1)
    # Beat 0 waits for the first half of a's word 0 (taken at 0, out at 1); b's first word
    # at 2 lanes for its second element (taken at 1, out at 2), the merge's beat 2 at 3;
    # a's word 3 waits until the halves of word 1 left (at 5).
    assert found.taken("wa>merge") == (1, 2, 4, 5, 7, 8, 10, 11)
    assert found.taken("wb>merge") == (3, 6, 9, 12)
    assert found.written("a>wa") == (0, 1, 2, 5)
    assert found.written("b>wb") == (0, 1, 2, 3, 4, 6, 7, 9)
    assert found.taken("merge>sink") == tuple(range(1, 13))


def test_a_concat_of_values_at_two_and_four_lanes_merges_two_beats_of_each_by_pixel() -> None:
    # Two (2, 2, 4) values, the first at 2 lanes (8 beats), the second at 4 (4 beats), joined
    # along the channels into (2, 2, 8) read at 2 lanes: a pixel is two beats of the first,
    # then two of the second (which a width step narrows from 4 lanes to 2), 16 beats.
    pmap = concat(((2, 2, 4), (2, 2, 4)), 2)
    stream = BeatSequence(vector_major((2, 2, 8), 2))
    first = Reindex(BeatSequence(vector_major((2, 2, 4), 2)), (Reader(pmap, stream, None, 0),))
    second = Reindex(BeatSequence(vector_major((2, 2, 4), 4)), (Reader(pmap, stream, None, 1),))
    merge = Merge((first, second))
    assert merge_stage("merge", merge) == MergeStage("merge", (0, 0, 1, 1), 16)
    assert route_stages(first, "a") == ()
    assert route_stages(second, "b") == (WidthStage("b.after.0", 4, 2),)
    net = Network(
        (
            Source("a", tuple(range(8)), 8, 32),
            Source("b", (0, 1, 2, 3), 4, 32),
            WidthStage("wb", 4, 2),
            merge_stage("merge", merge),
            Sink("sink"),
        ),
        (
            Channel("a>merge", "a", "merge", 0),
            Channel("b>wb", "b", "wb", 0),
            Channel("wb>merge", "wb", "merge", 0, input=1),
            Channel("merge>sink", "merge", "sink", 0),
        ),
    )
    found = run(net, FINNLIB, 1)
    # The merge takes a beat a cycle: the first source's words at beats 0, 1, 4, 5, 8, 9, 12,
    # 13 (due at 0 to 7, none waits for them), the second's at 2, 3, 6, 7, 10, 11, 14, 15. A
    # width step's word leaves the cycle after its last element arrived: its first two (one
    # input word, taken at 0) are ready at 1, and wait for their beats, 2 and 3; its next two
    # need the second input word (taken at 1), ready at 2, beats 6 and 7. The third input
    # word waits for the first four output words to leave (taken at 3, due 2), the fourth for
    # the next four (taken at 7, due 3).
    assert found.taken("a>merge") == (0, 1, 4, 5, 8, 9, 12, 13)
    assert found.taken("wb>merge") == (2, 3, 6, 7, 10, 11, 14, 15)
    assert found.written("b>wb") == (0, 1, 3, 7)
    assert found.taken("merge>sink") == tuple(range(16))
    assert found.late == (("a", (6,)), ("b", (4,))) and found.on_time  # idle 24 and 28


def test_a_value_read_through_two_maps_is_staged_one_branch_at_a_time() -> None:
    # x (2, 2) at one lane, a word a cycle, forked to two readers: the identity read at 2 lanes
    # (a width step, 1 to 2) and its transpose read at one lane (a reorder of the words 0, 2, 1,
    # 3: word 0 is free after the first beat, 1 and 2 after the third, all after the last, so
    # it holds 2 words).
    producer = BeatSequence(vector_major((2, 2), 1))
    wide = Reader(identity((2, 2)), BeatSequence(vector_major((2, 2), 2)))
    flipped = Reader(transpose((2, 2), (1, 0)), BeatSequence(vector_major((2, 2), 1)))
    fork = Reindex(producer, (wide, flipped))
    with pytest.raises(ValueError, match="one branch at a time"):
        route_stages(fork, "x")
    first, second = (route_stages(branch, name) for branch, name in zip(branches(fork), "ab"))
    assert first == (WidthStage("a.after.0", 1, 2),)
    assert second == (ReorderStage("b.after.0", (0, 2, 1, 3), (1, 1, 3, 4), 2),)
    net = Network(
        (Source("x", STEADY, 4, 8), Fork("fork", 2), *first, *second, Sink("sa"), Sink("sb")),
        (
            Channel("x>fork", "x", "fork", 0),
            Channel("fork>a", "fork", "a.after.0", 0),
            Channel("fork>b", "fork", "b.after.0", 0, output=1),
            Channel("a>sa", "a.after.0", "sa", 0),
            Channel("b>sb", "b.after.0", "sb", 0),
        ),
    )
    found = run(net, FINNLIB, 1)
    # The reorder presents word 0 two cycles after taking it (beat at 2), word 2 (taken at 3,
    # when it has room: word 0's slot is free a cycle after its beat) at 5, word 1 (taken at 1,
    # but the beat before it left at 5) at 6, freeing 3 words; word 3 waits for that, taken at 7,
    # presented at 9. The fork moves a word when both branches take it, so the width step's
    # words are held too: its first, from words 0 and 1 (taken at 0, 1), leaves at 2, its
    # second, from words 2 and 3 (taken at 3, 7), at 8.
    assert found.taken("fork>a") == found.taken("fork>b") == (0, 1, 3, 7)
    assert found.written("a>sa") == (2, 8)
    assert found.written("b>sb") == (2, 5, 6, 9)
    assert found.late == (("x", (4,)),) and found.on_time  # the last word 4 late, 4 idle


def test_a_pad_before_its_reader_is_an_insertion_stage() -> None:
    # (4, 4, 2) at 2 lanes padded by one on H and W to (6, 6, 2), read at 2 lanes: 36
    # beats, a fill row, then each of four rows a fill, four words and a fill, a fill row.
    reindex = Reindex(
        BeatSequence(vector_major((4, 4, 2), 2)),
        (
            Reader(
                identity((6, 6, 2)),
                BeatSequence(vector_major((6, 6, 2), 2)),
                pad((4, 4, 2), (1, 1, 0, 1, 1, 0)),
            ),
        ),
    )
    row = (False, True, True, True, True, False)
    assert route_stages(reindex, "p") == (
        InsertionStage("p.stage", (False,) * 6 + row * 4 + (False,) * 6),
    )


# --- the input generator ---------------------------------------------------------------

X_PORT = MM.present((4, 4), (m, k), lanes=(k,))
X_STEPS = steps(BeatSequence(vector_major((4, 4), 2)), BeatSequence(X_PORT))
ROWS = (0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7)


def test_a_reorders_frees_and_its_live_window() -> None:
    assert frees(ROWS, 8) == (0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8)
    assert frees((1, 0), 3) == (0, 3)  # word 2, never read, is free once 0 and 1 are
    found = reorder("g", ROWS, 8)
    assert found == ReorderStage("g", ROWS, frees(ROWS, 8), 2)
    # The steps' reorder (frame 2, read twice) is unrolled over the pass of 8 words.
    assert step_stages(X_STEPS, "x") == (ReorderStage("x.0", ROWS, frees(ROWS, 8), 2),)
    assert live_window(slots(vector_major((4, 4), 2), X_PORT)) == 2


def test_the_lines_buffer_frees_at_row_ends_and_holds_more_than_the_live_window() -> None:
    line = nest_buffer(8, (4, 2, 2), (2, 0, 1))
    assert line == NestBuffer(7, ROWS, (0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8))
    assert reorder("g", ROWS, 8).capacity == 2


def test_under_finnlibs_rules_the_live_window_alone_does_not_keep_the_rate() -> None:
    # A row of two words read twice: the third word waits for the first row's first free
    # (after beat 2) plus one, and is presented two cycles after it is taken, a cycle late
    # every row: 20 cycles a frame. One more word keeps 16; one-cycle rules keep it at 2.
    assert bottleneck(residual(0, generator=2), FINNLIB) == 20
    assert bottleneck(residual(0, generator=3), FINNLIB) == 16
    assert bottleneck(residual(0, generator=2), ONE_CYCLE) == 16


# --- the network and sizing ------------------------------------------------------------

CHANNELS = ("x", "to_gen", "skip", "gen_out", "y", "t", "out")


def residual(latency: int, generator: int = 3, frames: int | None = None) -> Network:
    """x (4, 4) at 2 lanes, forked to an input generator (of ``generator`` words) and a
    skip channel; MatMul (latency ``latency``), Thresholding (2) and Add (1)."""
    stages = (
        Source("x", tuple(range(8)), 8, 16),
        Fork("fork", 2),
        ReorderStage("gen", ROWS, frees(ROWS, 8), generator),
        kernel("matmul", (Pace(MM),), (Pace(MM, reduces=(k,)),), latency),
        kernel("thresholding", (Pace(EW),), (Pace(EW),), 2),
        kernel("add", (Pace(EW), Pace(EW)), (Pace(EW),), 1),
        Sink("sink"),
    )
    links = (
        Channel("x", "x", "fork"),
        Channel("to_gen", "fork", "gen"),
        Channel("skip", "fork", "add", output=1),
        Channel("gen_out", "gen", "matmul"),
        Channel("y", "matmul", "thresholding"),
        Channel("t", "thresholding", "add", input=1),
        Channel("out", "add", "sink"),
    )
    return Network(stages, links)


def test_a_networks_words_ends_and_order() -> None:
    net = residual(0)
    assert dict(net.words) == {
        "x": 8, "to_gen": 8, "skip": 8, "gen_out": 16, "y": 8, "t": 8, "out": 8
    }  # fmt: skip
    assert ends(net) == ("sink",)
    assert order(net) == CHANNELS
    assert net.period == 16


def test_a_network_refuses_what_is_not_one() -> None:
    source, sink = Source("s", STEADY, 4, 8), Sink("t")
    with pytest.raises(ValueError, match="writes 4 words a frame, k reads 2"):
        Network((source, Kernel("k", 2, ((0, 1),), ())), (Channel("c", "s", "k"),))
    with pytest.raises(ValueError, match="every port of t"):
        Network((source, sink, Sink("u")), (Channel("c", "s", "u"),))
    with pytest.raises(ValueError, match="own name"):
        Network((source, Sink("s")), (Channel("c", "s", "s"),))
    with pytest.raises(ValueError, match="share one period"):
        Network(
            (source, Source("r", STEADY, 4, 9), Kernel("k", 4, (STEADY, STEADY), ())),
            (Channel("a", "s", "k"), Channel("b", "r", "k", input=1)),
        ).period
    with pytest.raises(ValueError, match="cannot keep a period"):
        Source("s", STEADY, 4, 3)
    with pytest.raises(ValueError, match="one word a cycle"):
        Kernel("k", 4, ((0, 0),), ())


def test_a_network_slower_than_its_sources_is_refused_a_size() -> None:
    slow = Network(
        (Source("s", STEADY, 4, 4), Kernel("k", 8, ((0, 2, 4, 6),), ())),
        (Channel("c", "s", "k"),),
    )
    assert bottleneck(slow, FINNLIB) == 8
    with pytest.raises(ValueError, match="bottleneck period is 8 .* sources' 4"):
        minimal(slow, FINNLIB)
    with pytest.raises(ValueError, match="bottleneck period is 8"):
        least(slow, "c", FINNLIB)


def test_at_the_unbounded_runs_peaks_the_run_is_the_unbounded_run() -> None:
    for rules in (FINNLIB, ONE_CYCLE, SLOT_TWO):
        free = run(residual(4).unbounded(), rules)
        at = run(residual(4).at(dict(free.peaks)), rules)
        assert (at.writes, at.reads, at.completions) == (free.writes, free.reads, free.completions)


# A slot free two cycles after its word is read, direct a handshake.
SLOT_TWO = Rules(fifo_latency=2, direct=Direct.HANDSHAKE, slot_free=2, reorder_latency=2)


def held_slots() -> Network:
    """A source of 2 words (cycles 1 and 2 of a frame of 12), a kernel of latency 2 reading
    at steps 0 and 11 and writing at 6 and 11, a consumer reading at steps 2 and 4 of 8;
    period 12."""
    return chained(
        [
            Source("s", (1, 2), 12, 12),
            Kernel("k", 12, ((0, 11),), ((6, 11),), 2),
            Kernel("c", 8, ((2, 4),), ()),
        ]
    )


def test_a_slot_is_held_until_its_free_time_after_the_take_and_the_peaks_count_it() -> None:
    """Under ``SLOT_TWO`` the kernel's words are written at cycles 9 and 14 and the
    consumer takes the first at 13. Its slot is free at 15, so at 14 one word is still
    held and the channel needs 2 words, where a peak that counted the slot free as it is
    taken read 1 (and a channel of that capacity loses the period)."""
    net = held_slots()
    free = run(net.unbounded(), SLOT_TWO)
    assert free.written("k>c")[:4] == (9, 14, 21, 26)
    assert free.taken("k>c")[:4] == (13, 18, 25, 30)
    assert free.peaks == (("s>k", 3), ("k>c", 2))
    # At them, the unbounded run; one word less in the second channel loses the period.
    at = run(net.at(dict(free.peaks)), SLOT_TWO)
    assert (at.writes, at.reads, at.completions) == (free.writes, free.reads, free.completions)
    criterion = period_kept(net, SLOT_TWO)
    assert criterion(at)
    assert not criterion(run(net.at({"s>k": 3, "k>c": 1}), SLOT_TWO))
    found = minimal(net, SLOT_TWO)
    assert (found.order, found.peaks, found.capacities) == (("s>k", "k>c"), (3, 2), (3, 0))
    assert least(net, "k>c", SLOT_TWO) == 0


# The joint minimal sets, in the order above (x, to_gen, skip, gen_out, y, t, out), and the
# unbounded run's peaks; all direct deadlocks under both rules.
FOUND = {
    (ONE_CYCLE, 0): ((2, 3, 8, 2, 1, 2, 2), (1, 1, 5, 2, 1, 2, 2)),
    (ONE_CYCLE, 4): ((2, 3, 10, 2, 3, 2, 2), (1, 1, 7, 2, 3, 2, 2)),
    (FINNLIB, 0): ((3, 5, 11, 3, 2, 3, 2), (0, 0, 4, 0, 0, 0, 0)),
    (FINNLIB, 4): ((3, 5, 13, 3, 4, 3, 2), (0, 0, 6, 0, 0, 0, 0)),
}


@pytest.mark.parametrize("rules, latency", list(FOUND), ids=lambda v: str(v))
def test_the_joint_search_ends_at_a_minimal_set(rules: Rules, latency: int) -> None:
    net = residual(latency)
    found = minimal(net, rules)
    assert found.order == CHANNELS and found.period == 16 and found.frames == FRAMES
    assert (found.peaks, found.capacities) == FOUND[(rules, latency)]
    at = net.at(found.found)
    criterion = period_kept(net, rules)
    assert criterion(run(at, rules))
    # Minimal: one word less in any one channel loses the period or deadlocks.
    for name, capacity in found.found.items():
        if capacity > rules.direct_capacity:
            assert not criterion(run(at.at({name: capacity - 1}), rules)), name
    direct = run(net.at({name: rules.direct_capacity for name in CHANNELS}), rules)
    assert direct.deadlocked


def kept_by_capacity(net: Network, name: str, rules: Rules, upper: int) -> list[Run]:
    return [run(net.at({name: c}), rules) for c in range(rules.direct_capacity + 1, upper + 1)]


@pytest.mark.parametrize("rules, latency", list(FOUND), ids=lambda v: str(v))
def test_more_words_in_one_channel_never_lose_the_period(rules: Rules, latency: int) -> None:
    """Monotonicity, on hand cases at every FIFO capacity up to the unbounded peak, the
    others fixed at the minimal set: once the period is kept it stays kept, and no word
    moves later with more room."""
    net = residual(latency)
    found = minimal(net, rules)
    at = net.at(found.found)
    criterion = period_kept(net, rules)
    for name, peak in zip(found.order, found.peaks):
        runs = kept_by_capacity(at, name, rules, peak)
        kept = [criterion(each) for each in runs]
        assert kept == sorted(kept), (name, kept)
        for fewer, more in zip(runs, runs[1:]):
            if fewer.deadlocked:
                continue
            assert all(
                b <= a for (_, x), (_, y) in zip(fewer.reads, more.reads) for a, b in zip(x, y)
            )


def test_the_chains_keep_the_period_monotonically_above_direct() -> None:
    for supply, accept, period in CHAINS:
        net = paced(supply, accept, period, 0)
        runs = kept_by_capacity(net, "c", FINNLIB, 12)
        kept = [period_kept(net, FINNLIB)(each) for each in runs]
        assert kept == sorted(kept)


def test_a_sizing_states_its_order_and_its_total() -> None:
    found = minimal(residual(0), ONE_CYCLE)
    assert found.capacity("skip") == 5 and found.total == 14
    assert found.passes == 2  # the second pass moved nothing
    assert least(residual(0).at(found.found), "skip", ONE_CYCLE) == 5
    assert prod(found.peaks) > 0


def test_a_sizing_whose_start_does_not_keep_the_criterion_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="channel s>k does not keep the period at 3 words"):
        minimal(held_slots(), SLOT_TWO, keeps=lambda found: False)


def never_late(found: Run) -> bool:
    """A criterion stricter than the period's: no source is late at all."""
    return not found.deadlocked and all(each == 0 for _, late in found.late for each in late)


def test_a_criterion_of_the_callers_sizes_by_that_criterion() -> None:
    """The one-word register keeps the period at 4 words of 8 cycles, its sources late
    within their idle time; a source never late needs a second word."""
    assert least(two(None), "c", ONE_CYCLE) == 1
    assert least(two(None), "c", ONE_CYCLE, keeps=never_late) == 2
    default, strict = (
        minimal(residual(0), ONE_CYCLE),
        minimal(residual(0), ONE_CYCLE, keeps=never_late),
    )
    assert default.total == 14 and strict.total == 19
    assert (default.criterion, strict.criterion) == ("period_kept", "never_late")
    assert all(strict.capacity(name) >= default.capacity(name) for name in CHANNELS)
    assert period_kept(residual(0), ONE_CYCLE)(run(residual(0).at(strict.found), ONE_CYCLE))


def test_an_order_of_the_callers_is_the_order_the_channels_are_lowered_in() -> None:
    """A source of 4 words (cycles 2, 4, 9, 12 of 15, period 16), a kernel of latency 1
    taking and giving a word a step over 4 steps, and a consumer reading at steps 1, 3, 5
    and 6 of 14: from the sources down the first channel is lowered first and the second
    keeps 2 words; lowered from the sink up the second goes to a handshake and the first
    keeps 1. Each set is minimal, and they differ."""
    net = chained(
        [
            Source("s", (2, 4, 9, 12), 15, 16),
            Kernel("k", 4, ((0, 1, 2, 3),), ((0, 1, 2, 3),), 1),
            Kernel("c", 14, ((1, 3, 5, 6),), ()),
        ]
    )
    down = minimal(net, FINNLIB)
    up = minimal(net, FINNLIB, channels=("k>c", "s>k"))
    assert down.order == ("s>k", "k>c") and down.found == {"s>k": 0, "k>c": 2}
    assert up.order == ("k>c", "s>k") and up.found == {"k>c": 0, "s>k": 1}
    for found in (down, up):
        at = net.at(found.found)
        criterion = period_kept(net, FINNLIB)
        assert criterion(run(at, FINNLIB))
        for name, capacity in found.found.items():
            if capacity > FINNLIB.direct_capacity:
                assert not criterion(run(at.at({name: capacity - 1}), FINNLIB)), name
    with pytest.raises(ValueError, match="names each of the network's channels once"):
        minimal(net, FINNLIB, channels=("s>k",))
    with pytest.raises(ValueError, match="names each of the network's channels once"):
        minimal(net, FINNLIB, channels=("s>k", "s>k"))


def test_a_loop_is_run_but_refused_a_size() -> None:
    """Two kernels each writing a word at step 0 and reading the other's at step 1: a run
    times it (no word waits on a word that cannot come), but a period is read down a
    network from its sources, so its bottleneck and its sizes are refused."""
    ring = Network(
        (Kernel("a", 2, ((1,),), ((0,),)), Kernel("b", 2, ((1,),), ((0,),))),
        (Channel("a>b", "a", "b"), Channel("b>a", "b", "a")),
    )
    assert not run(ring, FINNLIB).deadlocked
    for call in (
        lambda: bottleneck(ring, FINNLIB),
        lambda: least(ring, "a>b", FINNLIB),
        lambda: minimal(ring, FINNLIB),
    ):
        with pytest.raises(ValueError, match="channels form a loop"):
            call()


def test_an_unknown_stage_or_channel_is_refused_by_name() -> None:
    net = two(None)
    with pytest.raises(ValueError, match="no stage x"):
        net.stage("x")
    with pytest.raises(ValueError, match="no channel x"):
        net.channel("x")
    assert net.stage("k").name == "k" and net.channel("c").name == "c"
