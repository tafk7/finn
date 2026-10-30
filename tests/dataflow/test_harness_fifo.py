# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness's FIFO comparison: a call with sides only is the two-stage network and its
depth is the line's ``least_depth`` on hand cases (a ``Buffered`` one and a ``None`` too);
each class by hand; what is a fault (a broken tool: it fails the run) and what is only counted
(a difference, a finding, a stubbed call); a whole partition as one network; the table, its
coverage and its pass; and the recorder on direct calls, on a strategy's and on a stand-in's."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from support import chain
from support.helpers import FULL_DSP48E2, with_defaults

from dataflow.harness import fifo
from dataflow.harness.fifo_record import (
    DEPTH,
    NONE,
    NOT_MODELLED,
    RAISED,
    STUBBED,
    Acceptance,
    Beats,
    Buffer,
    Edge,
    FifoCall,
    FifoRecorder,
    Line,
    Partition,
    Port,
    Stored,
    Unpaced,
    Wire,
    acceptance_of,
    beats_of,
)
from finn.dataflow import timing
from finn.dataflow.gemm import k, m, n
from finn.dataflow.schedule import Index, Pace, Schedule
from finn.dataflow.timing import FINNLIB, Channel, Kernel, Network, Source, least
from finn.kernels import fifo_sizing
from finn.kernels.explore import Seam, SizeFifos, TargetCycles
from finn.kernels.fifo_sizing import Buffered, Flow, Pattern, converted, least_depth
from finn.kernels.input_generator import NestBuffer, nest_buffer

A, C = Index("a"), Index("c")
MATMUL = Schedule({m: 4, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))  # 16 steps
ELEMENTWISE = Schedule({A: 4, C: 4}, {C: 2}, (A, C))  # 8 steps
FRAMES = fifo_sizing.FRAMES

# The sides of four hand chains and their least depths: a burst of 4 words in a span of 12
# into a reader of a word every other step (3); two bursts into a reader of a word a step,
# no idle time (5); 3 words of a span of 4 into a reader that skips a step (3); matmul's
# output into an elementwise kernel (0, direct).
CASES = [
    ("a burst", Pattern((0, 1, 2, 3), 12), Pattern((0, 2, 4, 6), 8), 12, 3),
    (
        "two bursts",
        Pattern((0, 1, 4, 5, 6, 9), 10),
        Pattern((0, 1, 2, 3, 4, 5), 10),
        10,
        5,
    ),
    ("three words", Pattern((0, 1, 2), 4), Pattern((0, 2, 3), 4), 4, 3),
    (
        "direct",
        Pattern(MATMUL.beat_times(reduces=(k,)), 16),
        Pattern(ELEMENTWISE.beat_times(), 8),
        16,
        0,
    ),
]
SIDES = [pytest.param(*case[1:], id=case[0]) for case in CASES]


def a_call(
    supply: Pattern,
    accept: Pattern | Buffered,
    period: int,
    line: Line,
    function: str = "least_depth",
    channel: str = "c",
) -> FifoCall:
    """A call as the recorder keeps it, the line's answer ``line`` given by hand."""
    plain: Acceptance = beats_of(accept) if isinstance(accept, Pattern) else acceptance_of(accept)
    return FifoCall("here", channel, function, period, FRAMES, beats_of(supply), plain, line)


def row_replay(rows: int, passes: int, width: int, capacity: int, spacing: int = 1) -> Buffered:
    """An input generator over ``rows`` rows of ``width`` words, each row read ``passes``
    times, freeing a row after its last pass, in ``capacity`` words; its reader taking a
    beat every ``spacing`` cycles."""
    line = nest_buffer(rows * width, (rows, passes, width), (width, 0, 1))
    beats = rows * passes * width
    return Buffered(
        NestBuffer(capacity, line.reads, line.freed),
        Pattern(tuple(range(0, spacing * beats, spacing)), spacing * beats),
    )


# A supply of the line's converted pattern into a buffer of a row of 4 words read twice (5
# words), at a period of 9 (3 words wait in the FIFO); and of 2 words read twice (2 words), a
# beat every other cycle, at a period of 9 (2 words wait).
ROWS = [
    ("a row read twice", Pattern((1, 2, 4, 5), 6), row_replay(1, 2, 4, 5), 9, 3),
    ("a short row", Pattern((3, 7), 8), row_replay(1, 2, 2, 2, 2), 9, 2),
]
BUFFERED = [pytest.param(*case[1:], id=case[0]) for case in ROWS]

# -- a call with sides only is a two-stage network ------------------------------------------


@pytest.mark.parametrize("supply, accept, period, expected", SIDES)
def test_a_call_with_sides_only_has_the_lines_least_depth(
    supply: Pattern, accept: Pattern, period: int, expected: int
) -> None:
    assert least_depth(supply, accept, period) == expected
    network = fifo.sides_network(beats_of(supply), beats_of(accept), period)
    assert least(network, fifo.FIFO, FINNLIB, FRAMES) == expected
    judged = fifo.judge_sides(a_call(supply, accept, period, Line(DEPTH, expected)), 1)
    (found,) = judged.entries
    assert (found.verdict, found.package, found.detail) == ("same", f"{expected} words", "")
    assert judged.findings == [] and judged.faults == []


@pytest.mark.parametrize("supply, accept, period, expected", BUFFERED)
def test_a_buffered_call_has_the_lines_least_depth_above_direct(
    supply: Pattern, accept: Buffered, period: int, expected: int
) -> None:
    assert least_depth(supply, accept, period) == expected > 0
    judged = fifo.judge_sides(a_call(supply, accept, period, Line(DEPTH, expected)), 1)
    (found,) = judged.entries
    assert (found.verdict, found.package) == ("same", f"{expected} words")
    assert (found.depth, found.acceptance) == ("above 0", "Buffered")
    assert judged.findings == [] and judged.faults == []


def test_a_consumers_frame_that_holds_several_of_the_buffers_is_its_run_that_many_times() -> None:
    """A buffer of 4 beats over a word each (a frame of 1 word, read 4 times) in a reader of
    12 beats: the line counts words across frames, so 3 of the buffer's frames come in one of
    the consumer's, 3 words a frame."""
    buffer = NestBuffer(2, (0, 0, 0, 0), (0, 0, 0, 1))
    accept = Buffered(buffer, Pattern(tuple(range(12)), 12))
    supply = Pattern((0, 1, 2), 12)
    expected = least_depth(supply, accept, 12)
    assert expected is not None
    (found,) = fifo.judge_sides(a_call(supply, accept, 12, Line(DEPTH, expected)), 1).entries
    assert (found.verdict, found.package) == ("same", f"{expected} words")
    reads, freed = fifo.repeated((0, 0, 0, 0), (0, 0, 0, 1), 12)
    assert reads == (0,) * 4 + (1,) * 4 + (2,) * 4
    assert freed == (0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3)


def test_a_reader_that_is_no_whole_number_of_the_buffers_frames_is_untranslatable() -> None:
    buffer = NestBuffer(2, (0, 0, 0, 0), (0, 0, 0, 1))
    accept = Buffered(buffer, Pattern(tuple(range(6)), 6))
    judged = fifo.judge_sides(a_call(Pattern((0,), 6), accept, 6, Line(DEPTH, 0)), 1)
    (found,) = judged.entries
    assert found.verdict == "untranslatable" and "its reader takes 6" in found.detail
    assert [fault.key for fault in judged.faults] == ["here :: c"]


def test_a_supply_of_other_words_than_the_buffer_takes_a_frame_is_untranslatable() -> None:
    buffer = NestBuffer(1, (0, 1), (0, 2))
    accept = Buffered(buffer, Pattern((0, 1), 2))
    judged = fifo.judge_sides(a_call(Pattern((0,), 2), accept, 4, Line(NONE)), 1)
    (found,) = judged.entries
    assert found.verdict == "untranslatable"
    assert "supply writes 1 words a frame, generator reads 2" in found.detail
    assert len(judged.faults) == 1


# -- the classes, by hand -------------------------------------------------------------------


def judged(call: FifoCall) -> fifo.Judged:
    return fifo.judge_sides(call, 1)


def classed(call: FifoCall) -> tuple[str, str, str, str]:
    """A call's class as a partition's channel gets it (no comparison with the line's own run):
    the network of its sides at the line's depth, its channel against ``least``."""
    network = fifo.sides_network(call.supply, call.accept, call.period, call.line.depth)  # type: ignore[arg-type]
    return fifo.judge(network, fifo.FIFO, call.line, call.frames)


BURST = CASES[0][1:]  # supply, accept, period, least depth 3
RUN = "supply late by 0 (idle 0); accept every 12"


def test_a_line_depth_the_network_finds_smaller_is_smaller_and_shows_both_runs() -> None:
    supply, accept, period, least_ = BURST
    package, verdict, detail, broken = classed(a_call(supply, accept, period, Line(DEPTH, 4)))
    assert (package, verdict, broken) == (f"{least_} words", "smaller", "")
    assert detail == f"at the line's 4 words: {RUN}; at the package's {least_} words: {RUN}"


def test_a_line_depth_the_network_finds_larger_differs_and_shows_both_runs() -> None:
    supply, accept, period, least_ = BURST
    package, verdict, detail, broken = classed(a_call(supply, accept, period, Line(DEPTH, 2)))
    assert (package, verdict, broken) == (f"{least_} words", "differs", "")
    # The run at the line's depth shows which source fell late, and the period at both depths.
    assert detail == (
        f"larger than the line's 2 words; at the line's 2 words: supply late by 32 (idle 0); "
        f"accept every 14; at the package's {least_} words: {RUN}"
    )


def test_a_larger_depth_from_a_stated_latency_is_counted_and_named() -> None:
    """A kernel of latency 1 holds a word in flight in the FIFO's slots: one more than the
    line's 3 (the line's latency is a constant shift)."""
    supply, accept, period, _ = CASES[2][1:]
    steady = tuple(range(supply.span))
    network = Network(
        (
            Source("feed", steady, supply.span, period),
            Kernel("producer", supply.span, (steady,), (supply.times,), 1),
            Kernel("consumer", accept.span, (accept.times,), ()),
        ),
        (Channel("in", "feed", "producer", 0), Channel("c", "producer", "consumer", None)),
    )
    package, verdict, detail, broken = fifo.judge(network, "c", Line(DEPTH, 3), FRAMES)
    assert (package, verdict, broken) == ("4 words", "larger_latency_counted", "")
    assert "a latency of 1 cycles" in detail and "late by" in detail


def test_a_none_is_same_when_the_network_deadlocks_too() -> None:
    buffer = NestBuffer(1, (0, 1), (0, 2))
    accept = Buffered(buffer, Pattern((0, 1), 2))
    supply = Pattern((0, 1), 2)
    assert least_depth(supply, accept, 4) is None
    result = judged(a_call(supply, accept, 4, Line(NONE)))
    (entry,) = result.entries
    assert (entry.verdict, entry.package) == ("same", f"none up to {FRAMES * 2} words")
    assert entry.detail == "deadlocks" and result.findings == [] and result.faults == []
    assert (entry.depth, entry.acceptance) == ("none", "Buffered")


def test_a_none_differs_when_the_network_keeps_the_period_at_that_bound() -> None:
    supply, accept, period, _ = BURST
    call = a_call(supply, accept, period, Line(NONE))
    package, verdict, detail, _ = classed(call)
    assert verdict == "differs" and package == f"{FRAMES * 4} words keep the period"
    assert "the line finds no depth" in detail
    assert judged(call).faults  # on two stages, a least the line does not find is a fault


# -- a fault is a broken tool; a difference or a finding is counted ---------------------------


def test_a_two_stage_call_whose_least_differs_from_the_lines_is_a_fault() -> None:
    supply, accept, period, least_ = BURST
    for line in (least_ + 1, least_ - 1):  # smaller, then larger
        result = judged(a_call(supply, accept, period, Line(DEPTH, line)))
        assert len(result.faults) == 1
        assert "least differs from the line's with the bottleneck 12 not above" in (
            result.faults[0].reason
        )
        table = fifo.table([(a_call(supply, accept, period, Line(DEPTH, line)), 1)])
        assert not table.passed and table.report()["passed"] is False
        assert [fault.key for fault in table.faults] == ["here :: c"]


def test_a_larger_line_depth_loses_the_period_in_the_network_too_and_is_also_a_finding() -> None:
    supply, accept, period, least_ = BURST
    result = judged(a_call(supply, accept, period, Line(DEPTH, least_ - 1)))
    (finding,) = result.findings
    assert finding.kind == fifo.NOT_KEPT and "supply late by" in finding.run
    assert finding.restored_by == (fifo.FIFO,)


def test_a_run_whose_word_times_differ_from_the_lines_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supply, accept, period, least_ = BURST
    call = a_call(supply, accept, period, Line(DEPTH, least_))
    assert judged(call).faults == []
    real = fifo_sizing.simulate

    def later(*arguments: object) -> Flow | None:
        flow = real(*arguments)  # type: ignore[arg-type]
        assert flow is not None
        taken = (*flow.taken[:5], flow.taken[5] + 1, *flow.taken[6:])
        return Flow(flow.left, taken, flow.late)

    monkeypatch.setattr("dataflow.harness.fifo.simulate", later)
    (fault,) = judged(call).faults
    assert fault.key == "here :: c"
    assert fault.reason.startswith("at the line's depth: words taken: word 5 is at cycle")
    assert not fifo.table([(call, 1)]).passed


def test_a_run_whose_word_leaving_times_differ_from_the_lines_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supply, accept, period, least_ = BURST
    call = a_call(supply, accept, period, Line(DEPTH, least_))
    real = fifo_sizing.simulate

    def earlier(*arguments: object) -> Flow | None:
        flow = real(*arguments)  # type: ignore[arg-type]
        assert flow is not None
        return Flow((flow.left[0] - 1, *flow.left[1:]), flow.taken, flow.late)

    monkeypatch.setattr("dataflow.harness.fifo.simulate", earlier)
    (fault,) = judged(call).faults
    assert fault.reason.startswith("at the line's depth: words leaving: word 0 is at cycle")


def test_a_run_whose_kept_verdict_differs_from_the_lines_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supply, accept, period, least_ = BURST
    call = a_call(supply, accept, period, Line(DEPTH, least_))
    monkeypatch.setattr("dataflow.harness.fifo.within", lambda *arguments: False)
    (fault,) = judged(call).faults
    assert fault.reason == ("at the line's depth: the line loses the period, the package the other")


def test_a_line_that_deadlocks_where_the_package_runs_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supply, accept, period, least_ = BURST
    call = a_call(supply, accept, period, Line(DEPTH, least_))
    monkeypatch.setattr("dataflow.harness.fifo.simulate", lambda *arguments: None)
    reasons = [fault.reason for fault in judged(call).faults]
    assert "at the line's depth: the line deadlocks, the package runs" in reasons


def test_a_crash_in_the_comparison_is_a_fault_and_the_table_still_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supply, accept, period, least_ = BURST
    ordinary = a_call(supply, accept, period, Line(DEPTH, least_), channel="ok")
    crashing = a_call(supply, accept, period, Line(DEPTH, least_))

    def boom(call: FifoCall, count: int) -> fifo.Judged:
        if call is crashing:
            raise RuntimeError("the comparison broke")
        return real(call, count)

    real = fifo.judge_sides
    monkeypatch.setattr(fifo, "judge_sides", boom)
    table = fifo.table([(ordinary, 1), (crashing, 1)])
    assert table.counts["same"] == 1 and table.counts["crashed"] == 1
    assert [(fault.key, fault.reason) for fault in table.faults] == [
        ("here :: c", "crash: RuntimeError: the comparison broke")
    ]
    assert not table.passed

    def broken(group: object) -> fifo.Judged:
        raise RuntimeError("the partition broke")

    monkeypatch.setattr(fifo, "judge_partition", broken)
    partition = fifo.table(partition_calls(4))
    assert partition.counts["crashed"] == 3 and not partition.passed
    assert partition.faults == (fifo.Fault("st", "crash: RuntimeError: the partition broke"),)


def test_a_smaller_whose_own_run_does_not_keep_the_period_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel's least that does not keep the period at its own depth: the strategy's result
    failing its own criterion (here, a ``least`` that answers 0 where 3 are needed)."""
    supply, accept, period, least_ = BURST
    monkeypatch.setattr(timing, "least", lambda *arguments: 0)
    package, verdict, detail, broken = classed(a_call(supply, accept, period, Line(DEPTH, 4)))
    assert (package, verdict) == ("0 words", "differs")
    assert broken == "its least depth, 0 words, does not keep the period"
    assert "at the package's 0 words: supply late by" in detail
    partition = fifo.table(partition_calls(4))
    assert [fault.key for fault in partition.faults] == ["st :: y"]
    assert not partition.passed


def test_a_joint_minimal_set_that_loses_the_period_is_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = timing.minimal

    def zeroed(network: Network, *arguments: object) -> timing.Sizing:
        sizing = real(network, *arguments)  # type: ignore[arg-type]
        return replace(sizing, capacities=(0,) * len(sizing.capacities))

    monkeypatch.setattr(timing, "minimal", zeroed)
    table = fifo.table(partition_calls(4))
    ((fault),) = table.faults
    assert fault.key == "st :: joint" and fault.reason.startswith("the minimal set loses")
    assert table.joint[0].kept is False and not table.passed


def test_a_difference_with_its_run_listed_does_not_fail() -> None:
    """The line's depth 0 in a partition where the network needs 4: a ``differs``, with the run
    at the line's depth and at the package's, a finding that the network does not keep the
    period at the line's depths, and a table that passes."""
    table = fifo.table(partition_calls(0))
    assert table.passed and table.faults == () and table.counts["differs"] == 1
    (difference,) = table.differences
    assert (difference.key, difference.verdict, difference.package) == (
        "st :: y",
        "differs",
        "4 words",
    )
    assert "at the line's 0 words: x.source late by" in difference.detail
    assert "at the package's 4 words: x.source late by 0 (idle 0)" in difference.detail
    report = table.report()
    assert report["passed"] is True and report["differences"][0]["detail"] == difference.detail
    assert table.finding_counts[fifo.NOT_KEPT] == 1
    assert any("fifo difference (differs) st :: y" in line for line in table.lines())


def test_a_bottleneck_above_the_period_is_slower_than_the_period_counted_not_failing() -> None:
    """One word a frame of a period of 2 into a generator reading it twice: its frame takes
    4 cycles. The package refuses to size it (the bottleneck is 4); the line gives 8."""
    buffer = NestBuffer(1, (0, 0), (0, 1))
    accept = Buffered(buffer, Pattern((0, 1), 2))
    supply = Pattern((0,), 1)
    assert least_depth(supply, accept, 2) == 8
    result = judged(a_call(supply, accept, 2, Line(DEPTH, 8)))
    (entry,) = result.entries
    assert (entry.verdict, entry.package) == ("slower_than_period", "refused")
    (finding,) = result.findings
    assert (finding.kind, finding.period, finding.depth) == ("slower_than_period", 2, 8)
    assert (finding.bottleneck, finding.at_depth) == (4, 4)  # unbounded, and at the line's 8
    assert result.faults == []
    table = fifo.table([(a_call(supply, accept, 2, Line(DEPTH, 8)), 1)])
    assert table.passed and table.counts["slower_than_period"] == 1
    assert table.finding_counts == {
        "network_not_kept": 0,
        "period_other_than_p": 0,
        "slower_than_period": 1,
        "not_modelled": 0,
    }
    assert table.report()["findings"][0]["bottleneck"] == 4
    assert "findings: network_not_kept 0, period_other_than_p 0, slower_than_period 1" in (
        table.summary()
    )


# A row of 2 words at cycles 0 and 1 of a period of 8, into a generator of 2 words reading a
# row twice, its reader taking a beat every other cycle.
DIRECT_SLOW = (Pattern((0, 1), 2), row_replay(1, 2, 2, 2, 2), 8, 4)


def test_a_generator_whose_direct_output_is_slower_than_the_period_is_classed_slower() -> None:
    """The line gives 4 words and the sources are on time at 4, and the network's bottleneck is
    the period, 8, since it runs the generator's output unbounded too. With the FIFO unbounded
    and that output direct, as stated, the end completes a frame every 9 cycles: the call is
    classed slower than the period, a finding of its own, and gets no second finding that the
    network does not keep the period at the line's depth."""
    supply, accept, period, line = DIRECT_SLOW
    assert least_depth(supply, accept, period) == line
    result = judged(a_call(supply, accept, period, Line(DEPTH, line)))
    (entry,) = result.entries
    assert (entry.verdict, entry.package) == ("slower_than_period", "none")
    (finding,) = result.findings
    assert (finding.kind, finding.period, finding.depth) == ("slower_than_period", 8, 4)
    assert (finding.bottleneck, finding.at_depth) == (8, 9)
    assert (
        "loses the rate at its end" in finding.run and "loses the rate at its end" in entry.detail
    )
    assert result.faults == []
    table = fifo.table([(a_call(supply, accept, period, Line(DEPTH, line)), 1)])
    assert table.passed and table.counts["slower_than_period"] == 1
    assert table.finding_counts[fifo.NOT_KEPT] == 0
    assert table.finding_counts[fifo.SLOWER] == 1


def test_the_line_s_verdict_is_compared_with_the_sources_half_of_the_criterion() -> None:
    """At the line's 4 words the sources of that generator are on time, as ``within`` says,
    and the end is not (``period_kept``): comparing the line's verdict with the whole
    criterion would be a fault of the tool, comparing it with ``Run.on_time`` is none."""
    supply, accept, period, line = DIRECT_SLOW
    call = a_call(supply, accept, period, Line(DEPTH, line))
    network = fifo.sides_network(call.supply, call.accept, period, line)  # type: ignore[arg-type]
    found = timing.run(network, FINNLIB, FRAMES)
    assert found.on_time and not timing.period_kept(network, FINNLIB, FRAMES)(found)
    assert fifo_sizing.keeps(supply, accept, period, line)
    assert fifo.against_the_line(call, found) == []


def test_a_partition_whose_bottleneck_is_not_the_period_is_a_finding_not_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line sized the partition at 16; the network's bottleneck, as the package finds it, is
    20 (stated, since no hand partition of this size is slower than its sources), or it
    deadlocks with every channel unbounded."""
    for found, said in ((20, "is 20"), (None, "deadlocks with every channel unbounded")):
        monkeypatch.setattr(fifo, "_bottleneck", lambda network, frames, found=found: found)
        table = fifo.table(partition_calls(4))
        (other,) = [each for each in table.findings if each.kind == fifo.OTHER_PERIOD]
        assert (other.period, other.bottleneck) == (16, found) and said in other.run
        assert table.passed and table.finding_counts[fifo.OTHER_PERIOD] == 1


def test_a_refused_call_and_an_unmodelled_channel_are_counted_apart_by_reason() -> None:
    supply, accept, period, _ = BURST
    raised = a_call(supply, accept, period, Line(RAISED, reason="ValueError: too slow"))
    unmodelled = a_call(supply, accept, period, Line(NOT_MODELLED, reason="a boundary"), "size")
    stubbed = a_call(supply, accept, period, Line(STUBBED, reason="stand-in"), "size", "s")
    table = fifo.table([(raised, 2), (unmodelled, 1), (stubbed, 1)])
    counts = table.counts
    assert (counts["raised"], counts["not_modelled"], counts["stubbed"]) == (1, 1, 1)
    assert counts["same"] == counts["differs"] == 0
    assert table.apart["raised"] == {"ValueError: too slow": 1}
    assert table.apart["not_modelled"] == {"a boundary": 1}
    assert table.apart["stubbed"] == {"stand-in": 1}
    assert table.passed  # counted apart, never a pass and never a failure


def test_a_source_the_package_cannot_take_is_untranslatable_and_fails() -> None:
    twice = Pattern((0, 0), 2)  # two words in one cycle: no pattern the line holds
    bad = FifoCall("", "c", "least_depth", 4, FRAMES, Beats((0, 0), 2), Beats((0,), 1), Line(NONE))
    result = judged(bad)
    assert twice and result.entries[0].verdict == "untranslatable"
    assert not fifo.table([(bad, 1)]).passed


# -- a partition is one network -------------------------------------------------------------


def a_partition() -> tuple[Partition, Pattern]:
    """A matmul writing a word at the end of each reduction (4 words in 16 steps) into a
    kernel that reads 16 words through an input generator of capacity 10, the matmul fed and
    the kernel's output taken at the edge."""
    y = Pace(MATMUL, reduces=(k,))
    line = nest_buffer(8, (2, 8), (0, 1))
    buffer = Buffer(10, line.reads, line.freed)
    ending = Schedule({A: 16}, {}, (A,))
    wires = (
        Wire("x", Edge(Beats(tuple(range(16)), 16)), Port("matmul", "x", Pace(MATMUL))),
        Wire("y", Port("matmul", "y", y), Port("act", "x", Pace(ending)), buffer=buffer),
        Wire("z", Port("act", "z", Pace(ending)), Edge(Beats(tuple(range(16)), 16))),
    )
    return Partition(("matmul", "act"), wires, 16), Pattern(y.times, 16)


def partition_calls(depth: int) -> list[tuple[FifoCall, int]]:
    partition, supply = a_partition()
    boundary = Line(NOT_MODELLED, reason="a boundary: not modelled")
    return [
        (FifoCall("st", "x", "size", 16, FRAMES, None, None, boundary, partition), 1),
        (
            FifoCall(
                "st", "y", "size", 16, FRAMES, beats_of(supply), None, Line(DEPTH, depth), partition
            ),
            1,
        ),
        (FifoCall("st", "z", "size", 16, FRAMES, None, None, boundary, partition), 1),
    ]


def test_a_buffered_channel_of_a_partition_above_depth_zero_is_the_whole_networks_least() -> None:
    entries, findings, faults, joint = fifo.judge_partition(partition_calls(4))
    by_key = {entry.key: entry for entry in entries}
    assert by_key["st :: y"].verdict == "same" and by_key["st :: y"].package == "4 words"
    assert by_key["st :: x"].verdict == by_key["st :: z"].verdict == "not_modelled"
    assert findings == [] and faults == []
    assert joint is not None and (joint.total, dict(zip(joint.order, joint.capacities))) == (
        4,
        {"x": 0, "y": 4, "y.generated": 0, "z": 0},
    )
    assert joint.kept is True


def test_a_partition_at_a_line_depth_that_loses_the_period_is_a_finding_not_a_failure() -> None:
    entries, findings, faults, joint = fifo.judge_partition(partition_calls(0))
    (entry,) = [each for each in entries if each.key == "st :: y"]
    assert (entry.verdict, entry.package) == ("differs", "4 words")
    (finding,) = findings
    assert finding.key == "st" and "x.source late by" in finding.run and faults == []
    assert finding.restored_by == ("y",) and not finding.memory_only
    assert joint is not None and joint.total == 4


def test_a_part_of_a_partition_the_line_does_not_pace_has_no_whole_network() -> None:
    partition, _ = a_partition()
    kept = partition.wires[1]
    parts = {
        "a mover that states no pace": Wire("w", Unpaced("a mover that states no pace"),
                                            Port("act", "x", None)),
        "kernel act's port x states no pace": Wire("w", Edge(Beats((0,), 1)),
                                                   Port("act", "x", None)),
        "two input_gens": Wire("w", Edge(Beats((0,), 1)), Port("act", "x", Pace(MATMUL)),
                               refused="two input_gens"),
    }  # fmt: skip
    for reason, wire in parts.items():
        each = Partition(partition.kernels, (wire, kept), 16)
        boundary = Line(NOT_MODELLED, reason="a boundary: not modelled")
        calls = [
            (FifoCall("st", each_wire.channel, "size", 16, FRAMES, None, None, boundary, each), 1)
            for each_wire in each.wires
        ]
        entries, findings, faults, joint = fifo.judge_partition(calls)
        assert joint is None and findings == [] and faults == []
        (whole,) = [entry for entry in entries if entry.key == "st :: partition"]
        assert whole.verdict == "not_modelled" and reason in whole.detail


def test_a_partition_that_could_not_be_read_is_untranslatable_and_fails() -> None:
    partition, _ = a_partition()
    refused = Partition(partition.kernels, partition.wires, 16, "a network of no members")
    calls = [(replace(call, partition=refused), count) for call, count in partition_calls(4)]
    entries, _, faults, joint = fifo.judge_partition(
        [(c, count) for c, count in calls if c.line.depth]
    )
    assert joint is None
    assert [(entry.verdict, entry.detail) for entry in entries] == [
        ("untranslatable", "a network of no members")
    ]
    # One fault for the partition, which fails the run.
    assert faults == [fifo.Fault("st", "untranslatable: a network of no members")]
    table = fifo.table([(c, count) for c, count in calls if c.line.depth])
    assert not table.passed and table.counts["untranslatable"] == 1


def test_a_memory_source_is_a_stage_paced_by_its_consumer_not_a_cause_of_the_loss() -> None:
    partition, supply = a_partition()
    weights = Wire("w", Stored(), Port("matmul", "w", Pace(MATMUL)))
    stored = Partition(partition.kernels, (*partition.wires, weights), 16)
    memory = Line(NOT_MODELLED, reason="a memory source: paced by its consumer")
    calls = [(replace(call, partition=stored), count) for call, count in partition_calls(4)]
    calls.append((FifoCall("st", "w", "size", 16, FRAMES, None, None, memory, stored), 1))
    entries, findings, faults, joint = fifo.judge_partition(calls)
    assert {entry.key: entry.verdict for entry in entries} == {
        "st :: x": "not_modelled",
        "st :: y": "same",
        "st :: z": "not_modelled",
        "st :: w": "not_modelled",
    }
    assert findings == [] and faults == [] and joint is not None and joint.total == 4
    assert "w" in joint.order


def test_a_channel_the_strategy_did_not_size_is_untranslatable() -> None:
    entries, _, faults, joint = fifo.judge_partition(partition_calls(4)[1:2])
    assert joint is None and len(faults) == 1
    assert [(entry.key, entry.verdict) for entry in entries] == [("st :: y", "untranslatable")]
    assert "is not sized by these calls" in entries[0].detail


# -- the table ------------------------------------------------------------------------------


def a_table() -> fifo.FifoTable:
    burst, bursts, *_ = (a_call(*case[1:4], Line(DEPTH, case[4])) for case in CASES)
    return fifo.table([(burst, 3), (bursts, 1)])


def test_the_table_counts_distinct_calls_and_calls_made_and_covers_by_depth_and_acceptance() -> (
    None
):
    table = a_table()
    assert (table.calls, table.calls_made) == (2, 4)
    assert table.counts["same"] == 2 and table.counts["smaller"] == 0
    assert table.passed
    assert table.coverage == {
        "0 Pattern": 0,
        "0 Buffered": 0,
        "above 0 Pattern": 2,
        "above 0 Buffered": 0,
    }
    assert table.gap == "no compared row has a Buffered acceptance above depth 0"
    assert table.gap in table.summary()
    assert "same 2, smaller 0" in table.summary()


def test_coverage_counts_only_the_rows_compared() -> None:
    """A row the line refused, a stubbed one, a line answer of ``None`` and one the package finds
    slower than the period are not compared: none is in the coverage, and the gap is said."""
    supply, accept, period, expected = ROWS[0][1:]
    slow = Buffered(NestBuffer(1, (0, 0), (0, 1)), Pattern((0, 1), 2))
    deadlock = Buffered(NestBuffer(1, (0, 1), (0, 2)), Pattern((0, 1), 2))
    others = [
        a_call(Pattern((0,), 1), slow, 2, Line(DEPTH, 8)),  # slower than the period
        a_call(supply, accept, period, Line(STUBBED, reason="stand-in"), "size", "stubbed"),
        a_call(supply, accept, period, Line(RAISED, reason="too slow"), "size", "raised"),
        a_call(supply, accept, period, Line(NOT_MODELLED, reason="a boundary"), "size", "unread"),
        a_call(Pattern((0, 1), 2), deadlock, 4, Line(NONE)),  # a same, with no depth
    ]
    direct = a_call(*CASES[3][1:4], Line(DEPTH, 0))
    table = fifo.table([(call, 1) for call in others])
    assert table.counts["same"] == 1 and table.counts["slower_than_period"] == 1
    assert set(table.coverage.values()) == {0}
    assert table.gap in table.summary() and table.gap
    compared = fifo.table(
        [
            (call, 1)
            for call in (*others, direct, a_call(supply, accept, period, Line(DEPTH, expected)))
        ]
    )
    assert compared.coverage == {
        "0 Pattern": 1,
        "0 Buffered": 0,
        "above 0 Pattern": 0,
        "above 0 Buffered": 1,
    }
    assert compared.gap == "" and "no compared row" not in compared.summary()


def test_the_gap_is_not_said_when_a_buffered_call_above_depth_zero_is_compared() -> None:
    supply, accept, period, expected = ROWS[0][1:]
    table = fifo.table([(a_call(supply, accept, period, Line(DEPTH, expected)), 1)])
    assert table.coverage["above 0 Buffered"] == 1
    assert table.gap == "" and "no compared row" not in table.summary()


def test_a_fault_fails_the_table_and_its_report_says_so() -> None:
    supply, accept, period, least_ = BURST
    table = fifo.table([(a_call(supply, accept, period, Line(DEPTH, least_ - 1)), 1)])
    assert not table.passed and table.counts["differs"] == 1
    report = table.report()
    assert report["passed"] is False and report["frames"] == FRAMES
    assert report["findings"] and report["entries"][0]["verdict"] == "differs"
    assert report["faults"] and report["differences"][0]["verdict"] == "differs"
    assert json.loads(json.dumps(report))["counts"] == report["counts"]  # plain values


def test_the_joint_minimal_sets_are_in_the_report_and_not_classed() -> None:
    table = fifo.table(partition_calls(4))
    report = table.report()
    (joint,) = report["joint"]
    assert (joint["key"], joint["total"]) == ("st", 4)
    assert sum(report["counts"].values()) == len(report["entries"])
    assert json.loads(json.dumps(report))["joint"][0]["order"] == list(joint["order"])


def test_a_fault_fails_a_table_and_a_difference_or_a_finding_is_only_listed() -> None:
    """A table fails on a fault (a broken tool), and passes when it only has a difference and a
    finding, both listed in its lines and in its report."""
    supply, accept, period, least_ = BURST
    slow = Buffered(NestBuffer(1, (0, 0), (0, 1)), Pattern((0, 1), 2))
    faulty = fifo.table([(a_call(supply, accept, period, Line(DEPTH, least_ - 1)), 1)])
    counted = fifo.table(
        [*partition_calls(0), (a_call(Pattern((0,), 1), slow, 2, Line(DEPTH, 8)), 1)]
    )
    assert faulty.faults
    assert counted.passed and counted.counts["differs"] == 1
    assert counted.counts["slower_than_period"] == 1
    for name, found in (("fault", faulty), ("counted", counted)):
        report = json.loads(json.dumps(found.report()))
        assert found.passed is report["passed"] is (name == "counted")
        out = "\n".join(found.lines())
        assert ("fifo FAULT" in out) is (name == "fault")
        assert ("fifo difference (differs) st :: y" in out) is (name == "counted")
        assert ("fifo difference (slower_than_period)" in out) is (name == "counted")


# -- the recorder ---------------------------------------------------------------------------


def test_the_recorder_keeps_a_direct_call_and_its_answer_and_a_refusal() -> None:
    supply, accept, period, expected = BURST
    with FifoRecorder() as recorder:
        assert fifo_sizing.least_depth(supply, accept, period) == expected
        assert fifo_sizing.least_depth(supply, accept, period) == expected
        with pytest.raises(ValueError, match="cannot keep a period"):
            fifo_sizing.least_depth(supply, accept, 3)
    assert fifo_sizing.least_depth(supply, accept, period) == expected
    (found, repeated), (refused, once) = recorder.recorded()
    assert (repeated, once) == (2, 1)
    assert found.line == Line(DEPTH, expected) and found.supply == beats_of(supply)
    assert found.accept == beats_of(accept) and found.partition is None
    assert refused.line.kind == RAISED and "cannot keep a period" in refused.line.reason


def test_the_recorder_keeps_a_strategys_whole_partition_with_each_channels_answer() -> None:
    explorer = Seam(FULL_DSP48E2)
    explorer.held.update(chain.HELD)
    point = TargetCycles(12).explore(explorer, with_defaults(chain.network()))
    with FifoRecorder() as recorder:
        SizeFifos().explore(explorer, point)
    calls = {call.channel: call for call, _ in recorder.recorded()}
    assert set(calls) == {"x", "w1", "hidden", "levels", "w2", "y"}
    assert {call.line.kind for call in calls.values()} == {DEPTH, NOT_MODELLED}
    assert calls["hidden"].line == calls["levels"].line == Line(DEPTH, 0)
    assert calls["levels"].accept is not None and calls["hidden"].accept is not None
    partitions = {call.partition for call in calls.values()}
    assert len(partitions) == 1 and None not in partitions
    (partition,) = partitions
    assert partition is not None and partition.period == 12
    assert {wire.channel for wire in partition.wires} == set(calls)
    table = fifo.table(recorder.recorded())
    assert table.counts["same"] == 2 and table.passed
    assert table.counts["not_modelled"] == 5  # four channels and the partition's whole


def test_the_recorder_leaves_the_line_as_it_found_it() -> None:
    before = fifo_sizing.least_depth
    with FifoRecorder():
        assert fifo_sizing.least_depth is not before
    assert fifo_sizing.least_depth is before


def test_a_call_made_while_least_depth_is_a_test_stand_in_is_stubbed_and_never_compared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line's own style test answers 300 words for every channel through a stand-in for
    ``least_depth``. The real function gives those channels 0 (the strategy test above): the
    stand-in's answer is not the line's, so the calls are kept apart, marked, and the table
    has no ``same`` and no ``smaller`` from them."""
    explorer = Seam(FULL_DSP48E2)
    explorer.held.update(chain.HELD)
    point = TargetCycles(12).explore(explorer, with_defaults(chain.network()))
    with FifoRecorder() as recorder, monkeypatch.context() as patch:
        patch.setattr(fifo_sizing, "least_depth", lambda supply, accept, period: 300)
        SizeFifos().explore(explorer, point)
    calls = {call.channel: call for call, _ in recorder.recorded()}
    assert set(calls) == {"x", "w1", "hidden", "levels", "w2", "y"}
    assert calls["hidden"].line.kind == calls["levels"].line.kind == STUBBED
    assert "least_depth stubbed in the line's test" in calls["hidden"].line.reason
    assert not any(call.line == Line(DEPTH, 300) for call in calls.values())
    table = fifo.table(recorder.recorded())
    assert table.counts["stubbed"] == 2 and table.counts["not_modelled"] == 4
    assert table.counts["same"] == table.counts["smaller"] == table.counts["differs"] == 0
    assert table.apart["stubbed"] == {"least_depth stubbed in the line's test": 2}
    assert table.passed and set(table.coverage.values()) == {0}
    # With the stand-in gone, the same calls are the line's again.
    with FifoRecorder() as again:
        SizeFifos().explore(explorer, point)
    assert {call.channel: call.line for call, _ in again.recorded()}["hidden"] == Line(DEPTH, 0)


@pytest.mark.parametrize("name", ["least_depth", "sides", "keeps", "simulate", "within"])
def test_a_call_a_module_that_imported_the_name_makes_is_stubbed_while_the_line_is_replaced(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A module that imported ``least_depth`` by name reaches the real function through the
    recorder's wrapper (as this one does); while the line's module has a stand-in under the
    name of ``least_depth``, of ``sides``, or of what ``least_depth`` calls (``keeps``, which
    calls ``simulate`` and ``within``), the call is made in a test's world and is marked, not
    compared. The stand-in gives the real answer: it is another function, and that is all the
    recorder reads."""
    supply, accept, period, expected = BURST
    real = getattr(fifo_sizing, name)
    with FifoRecorder() as recorder, monkeypatch.context() as patch:
        patch.setattr(
            fifo_sizing, name, lambda *arguments, **keywords: real(*arguments, **keywords)
        )
        assert least_depth(supply, accept, period) == expected  # the name this module imported
    ((call, _),) = recorder.recorded()
    assert call.line.kind == STUBBED and f"{name} stubbed in the line's test" in call.line.reason
    assert fifo.table([(call, 1)]).counts["stubbed"] == 1


# -- a kernel producer with a width conversion ----------------------------------------------


def test_a_kernel_producer_with_a_width_conversion_in_a_partition_reads_smaller_than_the_line() -> (
    None
):
    """A kernel writing 2 words a frame (at cycles 3 and 7 of 8) through a conversion from 2
    lanes to 1 into a kernel that reads 4 beats. The line enters the converted words as one
    source (times 4, 5, 8, 9: the conversion's words follow their last element, one a cycle) and
    gives 2 words. A partition translates the producer's kernel and the conversion as a
    ``WidthStage``, which holds its producer back, so the network needs none: ``smaller`` (the
    line 2, the package 0), the same run (a frame every 8, no source late) at both depths. The
    same sides as a call of their own (the converted ``Source``) are ``same``."""
    first, second = Index("a"), Index("c")
    writer_schedule = Schedule({first: 2, second: 4}, {second: 1}, (first, second))
    reader_schedule = Schedule({first: 4, second: 2}, {second: 1}, (first, second))
    writes = Pace(writer_schedule, reduces=(second,))
    reads = Pace(reader_schedule, reduces=(second,))
    assert (writes.times, writes.span, reads.times) == ((3, 7), 8, (1, 3, 5, 7))
    times = converted(writes.times, 2, 1)
    supply = Pattern(times, max(writes.span, times[-1] - times[0] + 1))
    accept = Pattern(reads.times, reads.span)
    assert (times, least_depth(supply, accept, 8)) == ((4, 5, 8, 9), 2)
    edge = Edge(Beats(tuple(range(8)), 8))
    partition = Partition(
        ("writer", "reader"),
        (
            Wire("x", edge, Port("writer", "x", Pace(writer_schedule))),
            Wire("m", Port("writer", "m", writes), Port("reader", "m", reads), convert=(2, 1)),
            Wire("o", Port("reader", "o", Pace(reader_schedule)), edge),
        ),
        8,
    )
    boundary = Line(NOT_MODELLED, reason="a boundary: not modelled")
    sized = FifoCall(
        "st", "m", "size", 8, FRAMES, beats_of(supply), beats_of(accept), Line(DEPTH, 2), partition
    )
    calls = [
        (replace(sized, channel=name, supply=None, accept=None, line=boundary), 1)
        for name in ("x", "o")
    ]
    judged_partition = fifo.judge_partition([calls[0], (sized, 1), calls[1]])
    (entry,) = [each for each in judged_partition.entries if each.key == "st :: m"]
    assert (entry.verdict, entry.line, entry.package) == ("smaller", "2 words", "0 words")
    assert "at the line's 2 words: x.source late by 0 (idle 0); o.sink every 8" in entry.detail
    assert "at the package's 0 words: x.source late by 0 (idle 0); o.sink every 8" in entry.detail
    assert judged_partition.findings == [] and judged_partition.faults == []
    # Alone, with the converted supply as a source, the two agree.
    (alone,) = judged(replace(sized, partition=None)).entries
    assert (alone.verdict, alone.package) == ("same", "2 words")
