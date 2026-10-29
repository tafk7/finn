# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The default criterion of the timing model's sizing (``timing.period_kept``): the sources
on time (``Run.on_time``) and every end complete every frame by its due cycle (``timing.due``).

- *The due cycles* are read from the network's unbounded run: an end's worst frame after
  ``k * period``, plus a slack: each end's own, the most idle time of the sources it is
  reached from (0 where none), unless the caller gives one. A run is held to them
  (``Due.met``), and refused where it ran other frames.
- *A slack is an end's own*: two sources of different idle times feed two ends, and a run
  late at one end by more than its source's idle time and less than the other's is refused
  by the per-end rule, where taking the most idle time of any source in the network accepts
  it.
- *The criterion is part of the result* (``Sizing.criterion``), and a network slower than
  its sources is refused before any criterion runs.
- *Monotonicity*: among capacities of one word or more, a set the criterion accepts is
  accepted at every pointwise larger set, on seeded random chains and fork-joins under
  three rules. A larger sample is marked ``slow``.
"""

from __future__ import annotations

import pytest

import random
from collections.abc import Callable
from dataclasses import replace

from finn.dataflow_next.timing import (
    FINNLIB,
    ONE_CYCLE,
    Channel,
    Direct,
    Due,
    Fork,
    Kernel,
    Memory,
    MergeStage,
    Network,
    Rules,
    Run,
    Sink,
    Source,
    chained,
    due,
    least,
    minimal,
    period_kept,
    run,
)

# --- the due cycles --------------------------------------------------------------------


def two_ends(a_span: int, b_span: int, a_capacities: tuple[int | None, int | None]) -> Network:
    """Two chains of one shape apart, ``a`` and ``b``: a source of 2 words a frame at cycles 0
    and 4 of a period of 7 (its span the only difference between them: its idle time is 7
    less the span), a kernel (5 steps, a latency of 1) and a consumer (3 steps). Ends
    ``a.end`` and ``b.end``, each reached from one source. Chain ``a``'s two channels are at
    ``a_capacities``, ``b``'s unbounded."""
    stages: list[Source | Kernel] = []
    channels = []
    for name, span, capacities in (
        ("a", a_span, a_capacities),
        ("b", b_span, (None, None)),
    ):
        stages += [
            Source(f"{name}.source", (0, 4), span, 7),
            Kernel(f"{name}.kernel", 5, ((0, 2),), ((2, 4),), 1),
            Kernel(f"{name}.end", 3, ((0, 1),), ()),
        ]
        channels += [
            Channel(f"{name}.in", f"{name}.source", f"{name}.kernel", capacities[0]),
            Channel(f"{name}.out", f"{name}.kernel", f"{name}.end", capacities[1]),
        ]
    return Network(tuple(stages), tuple(channels))


def lateness(found: Run, end: str, period: int) -> list[int]:
    return [at - frame * period for frame, at in enumerate(found.completed(end))]


def test_a_slack_is_the_idle_time_of_the_sources_an_end_is_reached_from() -> None:
    """Source ``a`` has no idle time (span 7 of 7) and source ``b`` 2 (span 5). With the
    channels of chain ``a`` one word each, its source is on time (late by 0, within its idle
    time) and its end completes each frame 1 cycle later than in the unbounded run, steadily:
    the run keeps the period at its rate, with a latency of 1 more than its source's idle
    time (0) allows. The per-end rule gives the end of ``a`` the slack 0 and refuses the run;
    the rule that took the most idle time of any source in the network (2, as ``slack=2`` is
    for every end) would give it 2 and accept it. ``b``'s end, reached from the source of
    idle time 2, has 2 under both."""
    net = two_ends(7, 5, (1, 1))
    free = run(net.unbounded(), ONE_CYCLE)
    unbounded_a = max(lateness(free, "a.end", 7))
    unbounded_b = max(lateness(free, "b.end", 7))
    by = due(net, ONE_CYCLE).by
    assert by == (("a.end", unbounded_a + 0), ("b.end", unbounded_b + 2))

    judged = run(net.at({"a.in": 1, "a.out": 1}), ONE_CYCLE)
    assert judged.on_time and dict(judged.late)["a.source"] == (0,) * 16
    assert max(lateness(judged, "a.end", 7)) == unbounded_a + 1  # 1 past the per-end bound
    assert judged.period("a.end") == judged.period("b.end") == 7  # at the rate throughout
    per_end, any_source = period_kept(net, ONE_CYCLE), period_kept(net, ONE_CYCLE, slack=2)
    assert per_end(judged) is False
    assert any_source(judged) is True
    assert due(net, ONE_CYCLE, slack=2).by == (
        ("a.end", unbounded_a + 2),
        ("b.end", unbounded_b + 2),
    )


def test_a_due_cycle_is_the_unbounded_runs_worst_frame_plus_the_slack() -> None:
    """On one source, a kernel and its consumer (a latency of 1, steps as in ``two_ends``),
    each slack: the default, the source's idle time; 0; or the caller's. An end reached from
    no source (a memory's) has slack 0 by default."""
    net = two_ends(5, 5, (None, None))
    free = run(net.unbounded(), ONE_CYCLE)
    worst = {end: max(lateness(free, end, 7)) for end in ("a.end", "b.end")}
    for slack, expected in ((None, 2), (0, 0), (5, 5)):
        found = due(net, ONE_CYCLE, slack=slack)
        assert (found.period, found.frames) == (7, 16)
        assert found.by == tuple((end, worst[end] + expected) for end in ("a.end", "b.end"))

    memory = Network(
        (
            Source("s", (0, 4), 5, 7),
            Kernel("k", 5, ((0, 2),), ((2, 4),), 1),
            Kernel("end", 3, ((0, 1),), ()),
            Memory("m"),
            Kernel("t", 3, ((0, 1),), ()),
        ),
        (
            Channel("in", "s", "k"),
            Channel("out", "k", "end"),
            Channel("mem", "m", "t"),
        ),
    )
    free = run(memory.unbounded(), ONE_CYCLE)
    by = dict(due(memory, ONE_CYCLE).by)
    assert by["end"] == max(lateness(free, "end", 7)) + 2
    assert by["t"] == max(lateness(free, "t", 7)) + 0  # reached from no source


def test_the_due_cycles_judge_runs_of_their_frames_only() -> None:
    net = two_ends(5, 5, (None, None))
    cycles = due(net, ONE_CYCLE, frames=6)
    assert isinstance(cycles, Due) and cycles.frames == 6
    assert cycles.met(run(net.unbounded(), ONE_CYCLE, 6))
    with pytest.raises(ValueError, match="6 frames judge no run of 16 frames"):
        cycles.met(run(net.unbounded(), ONE_CYCLE))
    # A run that deadlocked, or finished fewer frames than due, meets nothing.
    found = run(net.unbounded(), ONE_CYCLE, 6)
    assert not cycles.met(replace(found, deadlocked=True))
    short = tuple((stage, done[:-1]) for stage, done in found.completions)
    assert not cycles.met(replace(found, completions=short))


def test_a_slack_is_a_nonnegative_number_of_cycles() -> None:
    net = two_ends(5, 5, (None, None))
    with pytest.raises(ValueError, match="nonnegative"):
        due(net, ONE_CYCLE, slack=-1)


# --- the criterion is part of the result, and the network's rate is checked first ------------


def never_late(found: Run) -> bool:
    return found.on_time and all(each == 0 for _, late in found.late for each in late)


def test_a_sizing_records_the_criterion_it_used() -> None:
    net = two_ends(5, 5, (None, None))
    assert minimal(net, ONE_CYCLE).criterion == "period_kept"
    assert minimal(net, ONE_CYCLE, keeps=period_kept(net, ONE_CYCLE)).criterion == "period_kept"
    assert minimal(net, ONE_CYCLE, keeps=never_late).criterion == "never_late"
    assert minimal(net, ONE_CYCLE, keeps=lambda found: found.on_time).criterion.endswith("<lambda>")


def slower_than_its_source() -> Network:
    """A source of one word a frame at a period of 1 into a consumer taking a word every 2
    cycles: the network's bottleneck is 2."""
    return Network(
        (Source("s", (0,), 1, 1), Kernel("k", 2, ((0,),), ())), (Channel("c", "s", "k"),)
    )


def test_a_looser_criterion_is_refused_before_it_runs_on_a_network_slower_than_its_sources() -> (
    None
):
    """The precondition is the network's, whatever ``keeps`` is: no criterion is asked."""
    asked: list[Run] = []

    def anything(found: Run) -> bool:
        asked.append(found)
        return True

    net = slower_than_its_source()
    with pytest.raises(ValueError, match="bottleneck period is 2 .* sources' 1"):
        least(net, "c", FINNLIB, keeps=anything)
    with pytest.raises(ValueError, match="bottleneck period is 2 .* sources' 1"):
        minimal(net, FINNLIB, keeps=anything)
    with pytest.raises(ValueError, match="bottleneck period is 2 .* sources' 1"):
        least(net, "c", FINNLIB, keeps=lambda found: found.on_time)
    assert asked == []


def test_least_holds_a_channel_the_caller_fixed_to_the_networks_own_due_cycles() -> None:
    """The reference is the run with every channel unbounded, not the run with only the
    channel asked about unbounded: with the other channel at one word, whose lateness at the
    end passes the network's bound, no capacity of this one keeps the period (None), where a
    reference read with the first channel unbounded would excuse the other's lateness."""
    net = two_ends(7, 5, (None, 1))
    assert least(net, "a.in", ONE_CYCLE) is None
    assert least(net, "a.out", ONE_CYCLE) is not None
    assert least(net.at({"a.out": None}), "a.in", ONE_CYCLE) is not None


# --- monotonicity in capacity ----------------------------------------------------------------

RULES: dict[str, Rules] = {
    "FINNLIB": FINNLIB,
    "ONE_CYCLE": ONE_CYCLE,
    "Rules(2, HANDSHAKE, 2, 2)": Rules(2, Direct.HANDSHAKE, 2, 2),
}


def pattern(rng: random.Random, words: int, span: int) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(span), words)))


def random_chain(rng: random.Random, period: int) -> Network:
    """A source, a kernel (a latency of 0, 0, 1 or 2) and a consumer, each of random beat
    times of the same number of words a frame."""
    words = rng.randint(2, 6)
    span = rng.randint(words, period)
    source = Source("s", pattern(rng, words, span), span, period)
    steps = rng.randint(words, period)
    kernel = Kernel(
        "k",
        steps,
        (pattern(rng, words, steps),),
        (pattern(rng, words, steps),),
        rng.choice([0, 0, 1, 2]),
    )
    steps = rng.randint(words, period)
    return chained([source, kernel, Kernel("c", steps, (pattern(rng, words, steps),), ())])


def random_fork_join(rng: random.Random, period: int) -> Network:
    """A source into a fork, two kernels of random beat times and a merge of their words
    into a sink."""
    words = rng.randint(1, 3) * 2
    span = rng.randint(words, period)
    kernels = []
    for name in "ab":
        steps = rng.randint(words, period)
        kernels.append(
            Kernel(
                name,
                steps,
                (pattern(rng, words, steps),),
                (pattern(rng, words // 2, steps),),
                rng.choice([0, 1, 3]),
            )
        )
    return Network(
        (
            Source("s", pattern(rng, words, span), span, period),
            Fork("f", 2),
            *kernels,
            MergeStage("m", (0, 1), words),
            Sink("t"),
        ),
        (
            Channel("s>f", "s", "f"),
            Channel("f>a", "f", "a", output=0),
            Channel("f>b", "f", "b", output=1),
            Channel("a>m", "a", "m", input=0),
            Channel("b>m", "b", "m", input=1),
            Channel("m>t", "m", "t"),
        ),
    )


def monotone(
    networks: int, sets: int, larger: int, seed: int
) -> tuple[int, int, int, int, list[tuple[str, dict[str, int | None], dict[str, int | None]]]]:
    """Draw ``sets`` capacities (each channel 1 up to its unbounded peak plus one, about one
    in five with a channel unbounded) for each random network and rules; for each that
    ``period_kept`` accepts, check ``larger`` random pointwise larger sets and each set one
    word more in one channel. Returns the networks judged, the sets accepted, the sets whose
    sources are on time and an end is not, the larger sets checked, and the violations (an
    accepted set and a larger one refused)."""
    rng = random.Random(seed)
    judged = accepted = ends_lose = checked = 0
    violations: list[tuple[str, dict[str, int | None], dict[str, int | None]]] = []
    for index in range(networks):
        period = rng.randint(6, 24)
        net = (random_chain if index % 2 == 0 else random_fork_join)(rng, period)
        for name, rules in RULES.items():
            try:
                keeps: Callable[[Run], bool] = period_kept(net, rules)
            except ValueError:
                continue  # its bottleneck is not the sources' period, or it deadlocks unbounded
            judged += 1
            peaks = dict(run(net.unbounded(), rules).peaks)
            for _ in range(sets):
                capacities: dict[str, int | None] = {
                    channel: rng.randint(1, max(1, peak) + 1) for channel, peak in peaks.items()
                }
                if rng.random() < 0.2:
                    capacities[rng.choice(list(peaks))] = None
                drawn = run(net.at(capacities), rules)
                if not keeps(drawn):
                    ends_lose += drawn.on_time
                    continue
                accepted += 1
                more: list[dict[str, int | None]] = [
                    {**capacities, channel: capacity + 1}
                    for channel, capacity in capacities.items()
                    if capacity is not None
                ]
                for _ in range(larger):
                    more.append(
                        {
                            channel: None
                            if capacity is None or rng.random() < 0.1
                            else capacity + rng.randint(0, 3)
                            for channel, capacity in capacities.items()
                        }
                    )
                for each in more:
                    checked += 1
                    if not keeps(run(net.at(each), rules)):
                        violations.append((name, capacities, each))
    return judged, accepted, ends_lose, checked, violations


def test_the_default_criterion_accepts_every_larger_set_of_a_set_it_accepts() -> None:
    judged, accepted, ends_lose, checked, violations = monotone(
        networks=16, sets=6, larger=2, seed=7
    )
    assert violations == []
    # The sample exercises the ends' half: sets whose sources are on time and an end is late.
    assert judged >= 30 and accepted >= 100 and ends_lose >= 1 and checked >= 500


@pytest.mark.slow
def test_the_default_criterion_is_monotone_on_a_larger_sample() -> None:
    judged, accepted, ends_lose, checked, violations = monotone(
        networks=300, sets=12, larger=4, seed=7
    )
    assert violations == []
    assert judged >= 600 and accepted >= 4000 and ends_lose >= 100 and checked >= 40000
