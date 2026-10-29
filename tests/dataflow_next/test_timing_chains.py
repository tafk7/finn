# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Chains against the line's per-channel sizing (``finn.kernels.fifo_sizing``), under
FinnLib's rules, no fork: the package's ``least`` for one channel against the line's
``least_depth`` on the same two sides.

**How a chain's sides are translated.** The line's supply is a producer's beat times over
its span, released every period; its acceptance is a consumer's beat times, or an input
generator's buffer and the beat times of its reader. The line's consumer has no phase, so
an acceptance's first read is at step 0. In the network:

- a producer kernel is fed a word a step by a steady source through a direct channel (so
  the source, the one stage held to the period, is late as the kernel is) and writes at
  the supply's beat times; the consumer kernel reads at the acceptance's beat times;
- a supply already width-converted enters as a ``Source`` of the line's converted pattern
  (``fifo_sizing.converted``), not a source and a ``WidthStage``: the package's width step
  holds its producer back, which the line does not model (the same output times, a
  different lateness).

**What is tested**

- *Two kernels joined directly*: equal, on four named chains (least depths 0, 3, 5, 3)
  at every depth to 8, and on all 100 pairs of 3-word patterns of spans 3 to 5 (period
  the longer span).
- *A width step, then a reorder through an input generator*: a ``Buffered`` acceptance
  whose least depth is above 0, two cases by hand (a producer at 2 lanes into one lane, a
  row read twice from a buffer of one word more than the row; a producer at 1 lane into 2
  lanes, a buffer of exactly the row): equal at every depth, and the period shown by a
  run. The source and the width step as the model has them, no longer a ``Source`` of the
  converted pattern, need no FIFO at all (the step holds the producer back). A generator
  that cannot keep the rate is refused by the package, where the line gives a depth that
  grows with the frames it runs.
- *A stated latency on a chain*: unlike the line's, where a side's latency is a constant
  shift and changes no depth, the package's kernel holds the words in flight in a FIFO's
  slots, so a latency of one cycle needs one word more than the line's on some chains
  (12 of the 100 pairs above, never more than one), and 2 or more cycles none more (a
  handshake's pipeline registers hold them: a direct connection then keeps what a FIFO of
  the line's depth does not).
"""

from __future__ import annotations

import pytest

import itertools

from finn.dataflow_next.gemm import k, m, n
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.timing import (
    FINNLIB,
    Channel,
    Kernel,
    Network,
    ReorderStage,
    Source,
    WidthStage,
    bottleneck,
    due,
    least,
    minimal,
    period_kept,
    run,
)
from finn.kernels.fifo_sizing import Buffered, Pattern, converted, keeps, least_depth
from finn.kernels.input_generator import NestBuffer, nest_buffer

A, C = Index("a"), Index("c")
MATMUL = Schedule({m: 4, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))  # 16 steps
ELEMENTWISE = Schedule({A: 4, C: 4}, {C: 2}, (A, C))  # 8 steps

#: Depths compared, the line's ``keeps`` against the run's ``Run.on_time``.
DEPTHS = range(9)


def kernels(
    supply: Pattern, accept: Pattern, period: int, depth: int | None = None, latency: int = 0
) -> Network:
    """A producer kernel (a source feeds it a word a step; it writes at ``supply``'s beat
    times, visible ``latency`` cycles late) and a consumer kernel reading at ``accept``'s,
    joined by channel ``c`` of ``depth`` words."""
    steady = tuple(range(supply.beats))
    return Network(
        (
            Source("feed", steady, supply.beats, period),
            Kernel("producer", supply.beats, (steady,), (supply.times,), latency),
            Kernel("consumer", accept.beats, (accept.times,), ()),
        ),
        (Channel("in", "feed", "producer", 0), Channel("c", "producer", "consumer", depth)),
    )


# (the chain, supply, acceptance, period, the line's least depth): matmul's output (a word
# at the end of each two-step reduction) into an elementwise kernel; a burst of 4 words in
# a span of 12 into a reader of a word every other step; two bursts into a reader of a word
# a step, both at a period of the producer's span (no idle time: each word the reader
# holds back the FIFO must take); and 3 words of a span of 4 into a reader that skips a step.
TWO_KERNELS = [
    ("matmul into elementwise", Pattern(MATMUL.beat_times(reduces=(k,)), 16),
     Pattern(ELEMENTWISE.beat_times(), 8), 16, 0),
    ("a burst into a slow reader", Pattern((0, 1, 2, 3), 12), Pattern((0, 2, 4, 6), 8), 12, 3),
    ("two bursts, no idle time", Pattern((0, 1, 4, 5, 6, 9), 10),
     Pattern((0, 1, 2, 3, 4, 5), 10), 10, 5),
    ("three words in a span of four", Pattern((0, 1, 2), 4), Pattern((0, 2, 3), 4), 4, 3),
]  # fmt: skip


@pytest.mark.parametrize(
    "supply, accept, period, expected",
    [case[1:] for case in TWO_KERNELS],
    ids=[case[0] for case in TWO_KERNELS],
)
def test_two_kernels_joined_directly_have_the_lines_least_depth(
    supply: Pattern, accept: Pattern, period: int, expected: int
) -> None:
    assert least_depth(supply, accept, period) == expected
    net = kernels(supply, accept, period)
    assert least(net, "c", FINNLIB) == expected
    for depth in DEPTHS:
        found = run(kernels(supply, accept, period, depth), FINNLIB)
        assert found.on_time == keeps(supply, accept, period, depth), depth


def small_patterns() -> list[Pattern]:
    """Every 3-word pattern of span 3 to 5: the first word at 0, the others at distinct later
    beats within the span."""
    return [
        Pattern((0, *later), span)
        for span in range(3, 6)
        for later in itertools.combinations(range(1, span), 2)
    ]


def test_two_kernels_have_the_lines_least_depth_on_every_small_pair() -> None:
    patterns = small_patterns()
    assert len(patterns) == 10  # 1 + 3 + 6
    positive = 0
    for supply, accept in itertools.product(patterns, patterns):
        period = max(supply.beats, accept.beats)
        line = least_depth(supply, accept, period)
        assert least(kernels(supply, accept, period), "c", FINNLIB) == line, (supply, accept)
        positive += bool(line)
    assert positive > 20  # a FIFO is needed on a fair share of them, not only direct


# --- a width step, then a reorder through an input generator -------------------------------


def row_replay(
    rows: int, passes: int, width: int, capacity: int, spacing: int
) -> tuple[NestBuffer, Pattern]:
    """An input generator over ``rows`` rows of ``width`` words, each row read ``passes``
    times (reads 0..width-1 repeated, row after row), freeing a row after its last pass, of
    ``capacity`` words (``nest_buffer``'s own is the RTL's, larger); and its reader taking
    a beat every ``spacing`` cycles."""
    line = nest_buffer(rows * width, (rows, passes, width), (width, 0, 1))
    beats = rows * passes * width
    return (
        NestBuffer(capacity, line.reads, line.freed),
        Pattern(tuple(range(0, spacing * beats, spacing)), spacing * beats),
    )


def converted_supply(producer: Pattern, lanes_in: int, lanes_out: int) -> Pattern:
    """The line's supply behind an output width conversion: the converted word times over
    the producer's span, or the conversion's if that is longer."""
    times = converted(producer.times, lanes_in, lanes_out)
    return Pattern(times, max(producer.beats, times[-1] - times[0] + 1))


def generated(
    supply: Pattern, buffer: NestBuffer, reads: Pattern, period: int, depth: int | None = None
) -> Network:
    """A source of ``supply``, channel ``c``, the input generator, a kernel reading it at
    ``reads`` (the generator's output direct)."""
    return Network(
        (
            Source("s", supply.times, supply.beats, period),
            ReorderStage("g", buffer.reads, buffer.freed, buffer.capacity),
            Kernel("k", reads.beats, (reads.times,), ()),
        ),
        (Channel("c", "s", "g", depth), Channel("o", "g", "k", 0)),
    )


def widened(
    producer: Pattern,
    lanes: tuple[int, int],
    buffer: NestBuffer,
    reads: Pattern,
    period: int,
    depth: int | None = None,
) -> Network:
    """The same, the producer and a width step as the package has them, the channel after
    the step."""
    return Network(
        (
            Source("s", producer.times, producer.beats, period),
            WidthStage("w", *lanes),
            ReorderStage("g", buffer.reads, buffer.freed, buffer.capacity),
            Kernel("k", reads.beats, (reads.times,), ()),
        ),
        (
            Channel("a", "s", "w", 0),
            Channel("c", "w", "g", depth),
            Channel("o", "g", "k", 0),
        ),
    )


# (lanes in and out, the producer's beat times and span, the nest (rows, passes, width), the
# generator's capacity, the reader's spacing, the period, the supply after the width step, the
# line's least depth).
WIDTH_THEN_REORDER = [
    pytest.param(
        (2, 1), Pattern((0, 3), 6), (1, 2, 4), 5, 1, 9, Pattern((1, 2, 4, 5), 6), 3,
        id="2 lanes to 1, a row of 4 read twice",
    ),
    pytest.param(
        (1, 2), Pattern((0, 2, 4, 6), 8), (1, 2, 2), 2, 2, 9, Pattern((3, 7), 8), 2,
        id="1 lane to 2, a row of 2 read twice",
    ),
]  # fmt: skip


@pytest.mark.parametrize("lanes, producer, nest, capacity, spacing, period, supply, expected",
                         WIDTH_THEN_REORDER)  # fmt: skip
def test_a_width_step_then_a_reorder_has_the_lines_least_depth_above_direct(
    lanes: tuple[int, int],
    producer: Pattern,
    nest: tuple[int, int, int],
    capacity: int,
    spacing: int,
    period: int,
    supply: Pattern,
    expected: int,
) -> None:
    """The first: a producer at 2 lanes of 2 beats a frame (cycles 0 and 3) becomes 4 words
    at 1 lane (1, 2, 4, 5); the generator holds a row of 4 words read twice, in 5 words,
    freeing the row after its last read, read a word a cycle; each frame's 4 words (9
    cycles on) arrive while the last still holds the previous, one fits the spare slot, so
    3 wait in the FIFO. The second: 4 beats at 1 lane (0, 2, 4, 6) become 2 words at 2 lanes
    (3, 7), a row of 2 read twice in a buffer of 2 words, read every other cycle; the
    next frame's 2 words both wait: 2. Each at a period of 9 cycles."""
    buffer, reads = row_replay(nest[0], nest[1], nest[2], capacity, spacing)
    assert converted_supply(producer, *lanes) == supply
    line = Buffered(buffer, reads)
    assert least_depth(supply, line, period) == expected > 0
    net = generated(supply, buffer, reads, period)
    assert least(net, "c", FINNLIB) == expected
    for depth in DEPTHS:
        found = run(generated(supply, buffer, reads, period, depth), FINNLIB)
        assert found.on_time == keeps(supply, line, period, depth) == (depth >= expected), depth
    keeping = period_kept(net, FINNLIB)
    at = run(generated(supply, buffer, reads, period, expected), FINNLIB)
    assert keeping(at) and at.period("k") == period
    fewer = run(generated(supply, buffer, reads, period, expected - 1), FINNLIB)
    assert not keeping(fewer)
    # With the width step and its producer as the package has them the width step holds the
    # producer back, and no FIFO is needed: direct keeps the period too.
    raw = widened(producer, lanes, buffer, reads, period)
    assert least(raw, "c", FINNLIB) == 0
    held = run(raw.at({"c": 0}), FINNLIB)
    assert period_kept(raw, FINNLIB)(held) and held.period("k") == period


def test_a_generator_that_cannot_keep_the_rate_is_refused_where_the_line_gives_a_depth() -> None:
    """One word a frame (cycle 0) of a period of 2, into a generator of 1 word that reads
    it twice, a reader taking a beat each cycle: the generator takes 4 cycles a frame, not
    2 (it presents a word two cycles after taking it). The package says so: the network's
    bottleneck is 4 and it will not size it. The line gives a depth that only delays the
    loss: a word of backlog grows every two frames, so its least depth is half the frames
    it runs (8 over 16, 16 over 32, 32 over 64)."""
    buffer = NestBuffer(1, (0, 0), (0, 1))
    supply, reads = Pattern((0,), 1), Pattern((0, 1), 2)
    line = Buffered(buffer, reads)
    assert [least_depth(supply, line, 2, frames) for frames in (16, 32, 64)] == [8, 16, 32]
    net = generated(supply, buffer, reads, 2)
    assert bottleneck(net, FINNLIB) == 4
    with pytest.raises(ValueError, match="bottleneck period is 4 .* sources' 2"):
        least(net, "c", FINNLIB)
    # The period is lost whatever the FIFO: the reader completes a frame every 4 cycles.
    for depth in (None, 16):
        assert run(generated(supply, buffer, reads, 2, depth), FINNLIB).period("k") == 4


def test_a_reorder_whose_direct_output_is_slower_than_the_period_has_no_depth() -> None:
    """The line and the sources' half answer 4; the default answers None. A row of 2 words
    (at cycles 0 and 1 of a period of 8) read twice from a buffer of 2 words, its reader
    taking a beat every other cycle: 8 cycles a frame. With every channel unbounded the
    network's bottleneck is 8, since ``bottleneck`` runs the generator's output unbounded
    too. With that output direct, as the network states it, the generator and its reader
    take 9 cycles a frame: the sources are on time at 4 words (the FIFO holds the backlog
    for the 16 frames) while the reader falls a cycle further behind each frame, 25 cycles
    after its frame's due cycle at the last. No depth keeps the period, so ``least`` is None.

    This is the model's claim about a generator whose output is direct; whether the
    hardware's generator into a direct reader loses a cycle a frame is for a run of the
    hardware to settle."""
    buffer, reads = row_replay(1, 2, 2, 2, 2)
    supply, period = Pattern((0, 1), 2), 8
    line = Buffered(buffer, reads)
    net = generated(supply, buffer, reads, period)
    assert least_depth(supply, line, period) == 4
    assert bottleneck(net, FINNLIB) == 8
    assert run(net.at({"c": None}), FINNLIB).period("k") == 9
    # The sources' half alone, the old default: 4 words, the line's.
    assert least(net, "c", FINNLIB, keeps=lambda found: found.on_time) == 4
    assert least(net, "c", FINNLIB) is None
    # The due cycles are the unbounded run's worst frame (12 after k * 8) and the idle time.
    assert due(net, FINNLIB).by == (("k", 12 + 6),)
    at = run(generated(supply, buffer, reads, period, 4), FINNLIB)
    assert at.on_time and keeps(supply, line, period, 4)  # the line's ``within`` passes
    assert not period_kept(net, FINNLIB)(at)
    late = [done - frame * period for frame, done in enumerate(at.completed("k"))]
    assert late == list(range(10, 26))


def test_a_consumer_slower_than_the_period_is_refused_where_the_line_gives_a_depth() -> None:
    """One word a frame at a period of 1 cycle into a consumer taking a word every 2 cycles.
    The line gives 9 words, and ``within`` passes the run at 9: the FIFO holds the backlog
    of the 16 frames and the consumer falls a cycle further behind each frame. The sources
    are never late. The network's bottleneck is 2 against the sources' 1, so the default
    has no due cycles and refuses it."""
    supply, accept, period = Pattern((0,), 1), Pattern((0,), 2), 1

    def chain(depth: int | None) -> Network:
        return Network(
            (Source("s", (0,), 1, period), Kernel("k", 2, ((0,),), ())),
            (Channel("c", "s", "k", depth),),
        )

    assert least_depth(supply, accept, period) == 9
    assert keeps(supply, accept, period, 9) and not keeps(supply, accept, period, 8)
    at = run(chain(9), FINNLIB)
    assert at.on_time and set(dict(at.late)["s"]) == {0}
    assert [done - frame * period for frame, done in enumerate(at.completed("k"))] == list(
        range(2, 18)
    )
    assert bottleneck(chain(None), FINNLIB) == 2
    for refused in (
        lambda: least(chain(None), "c", FINNLIB),
        lambda: minimal(chain(None), FINNLIB),
        lambda: due(chain(None), FINNLIB),
        lambda: period_kept(chain(None), FINNLIB),
    ):
        with pytest.raises(ValueError, match="bottleneck period is 2 .* sources' 1"):
            refused()


# --- a stated latency on a chain -------------------------------------------------------------

LATENT = Pattern((0, 1, 2), 4), Pattern((0, 2, 3), 4), 4  # the supply, the acceptance, the period


def test_a_latency_of_one_cycle_needs_a_word_more_than_the_lines_depth() -> None:
    """Three words in a span of four, read at steps 0, 2, 3, no idle time: the line's least
    depth is 3. The producer kernel's latency of 1 delays each word's arrival at the
    reader by a cycle, the words in flight hold their slots in the FIFO, and the producer,
    with no idle time to absorb a cycle, needs a fourth: with 3 words its second word of
    each frame leaves a cycle later than its time, a cycle more every other frame (late 0,
    1, 1, 2, 2, 3 over the first frames); with 4 it is never late."""
    supply, accept, period = LATENT
    assert least_depth(supply, accept, period) == 3
    assert least(kernels(supply, accept, period, latency=0), "c", FINNLIB) == 3
    assert least(kernels(supply, accept, period, latency=1), "c", FINNLIB) == 4
    three = run(kernels(supply, accept, period, 3, latency=1), FINNLIB)
    assert not three.on_time and dict(three.late)["feed"][:6] == (0, 1, 1, 2, 2, 3)
    four = run(kernels(supply, accept, period, 4, latency=1), FINNLIB)
    assert four.on_time and set(dict(four.late)["feed"]) == {0}
    # The reader's words: visible a cycle later than with no latency.
    assert run(kernels(supply, accept, period, 4), FINNLIB).taken("c")[:3] == (2, 4, 5)
    assert four.taken("c")[:3] == (3, 5, 6)


def test_a_latency_of_two_cycles_is_held_by_the_pipelines_registers() -> None:
    """The same chain at latency 2: a handshake lets the kernel issue while fewer than 2 of
    its words are untaken, so direct keeps the period (least 0, below the line's 3); a
    FIFO of 1 to 4 words does not (its slots hold the words in flight), 5 does. The least
    capacity is not monotone from direct to the FIFOs: the search tries direct apart."""
    supply, accept, period = LATENT
    kept = [
        run(kernels(supply, accept, period, depth, latency=2), FINNLIB).on_time
        for depth in range(7)
    ]
    assert kept == [True, False, False, False, False, True, True]
    assert least(kernels(supply, accept, period, latency=2), "c", FINNLIB) == 0


def test_latency_changes_the_least_depth_by_one_word_at_most_on_every_small_pair() -> None:
    """Over the 100 pairs of 3-word patterns (the period the longer span): with a latency of
    1 the least depth is one word above the line's on 12 of them, below it on 30 (the
    pipeline's one register holds a word), and never more than one above; with a latency
    of 2 it is none above, 54 below (9 by two words, 45 by three)."""
    patterns = small_patterns()
    for latency, above, below in ((0, 0, 0), (1, 12, 30), (2, 0, 54)):
        differences = []
        for supply, accept in itertools.product(patterns, patterns):
            period = max(supply.beats, accept.beats)
            found = least(kernels(supply, accept, period, latency=latency), "c", FINNLIB)
            line = least_depth(supply, accept, period)
            assert found is not None and line is not None
            differences.append(found - line)
        assert sum(1 for each in differences if each > 0) == above, latency
        assert sum(1 for each in differences if each < 0) == below, latency
        assert max(differences) <= 1, latency
