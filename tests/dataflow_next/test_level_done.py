# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A marker holding while a level is on its last index value (``LevelDone``), and the marker a
reduction's last index value is (``Schedule.closing``).

- **The marker** is asserted on the last ``inner`` beats of every ``beats`` beats. Its one
  form has two edges, both ``LevelEnd``: a inner of one beat (``LevelEnd(beats)``) and a
  inner of the whole beats (``LevelEnd(1)``). On a beat sequence the inner closes whole
  innermost loops and the beats whole loops around them.
- **``closing``** takes a reduction that is one contiguous block of the iteration order. The
  indices after the block are spanned; with none, the marker is today's ``LevelEnd``.
- **The tiled dot product's core**, order (g, n, k, th) with v = g * TH + th, reduces k
  with th inside it: k's last index value holds across the TH beats of th, the last TH of
  every SF * TH beats. The line refuses it.
"""

from __future__ import annotations

import pytest

import itertools
from dataflow_next.harness import translate

import finn.dataflow.schedule as line_schedule
from finn.dataflow_next.gemm import k, m, n
from finn.dataflow_next.schedule import Index, Refused, Schedule
from finn.dataflow_next.steps import Step, steps
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    LevelDone,
    LevelEnd,
    Loop,
    Traversal,
    level_done,
    vector_major,
)

G, TH = Index("g"), Index("th")

# (N, MW, MH, PE, SIMD, TH): the tiled dot product's three cases, and the marker of
# k's last index value, (SF * TH, TH) with SF = MW / SIMD.
TILED = [
    ((6, 12, 6, 3, 4, 3), (9, 3)),
    ((8, 16, 8, 4, 4, 2), (8, 2)),
    ((6, 12, 4, 2, 6, 3), (6, 3)),
]
TILED_IDS = ["6-12-6-3-4-3", "8-16-8-4-4-2", "6-12-4-2-6-3"]
# The core's beats and the beats the marker holds on: 12 of 36, 16 of 64, 12 of 24.
TILED_BEATS = [(36, 12), (64, 16), (24, 12)]


def tiled_core(
    vectors: int, depth: int, outputs: int, pe: int, simd: int, tile: int
) -> tuple[Schedule, Traversal]:
    """The core's schedule and its activation port, which reads every iteration."""
    core = Schedule(
        {G: vectors // tile, n: outputs, k: depth, TH: tile}, {n: pe, k: simd}, (G, n, k, TH)
    )
    return core, core.present((vectors, depth), (G * tile + TH, k), lanes=(k,))


def last_iterations(schedule: Schedule, reduces: tuple[Index, ...]) -> list[bool]:
    """Each iteration, in order, by its indices: whether every reduced index is at
    its last index value."""
    ranges = [range(schedule.fold(index)) for index in schedule.order]
    return [
        all(
            digit == schedule.fold(index) - 1
            for index, digit in zip(schedule.order, digits)
            if index in reduces
        )
        for digits in itertools.product(*ranges)
    ]


# -- the marker ----------------------------------------------------------------------------


def test_the_marker_holds_on_the_last_inner_beats_of_every_level() -> None:
    rule = LevelDone(6, 2)
    assert [beat for beat in range(18) if rule.asserted(beat)] == [4, 5, 10, 11, 16, 17]
    assert not rule.constant


@pytest.mark.parametrize("beats", [1, 2, 6, 12])
def test_an_inner_of_one_beat_is_the_level_closing(beats: int) -> None:
    rule = level_done(beats, 1)
    assert type(rule) is LevelEnd and rule == LevelEnd(beats)
    assert [rule.asserted(beat) for beat in range(24)] == [
        beat % beats == beats - 1 for beat in range(24)
    ]


@pytest.mark.parametrize("beats", [2, 6, 12])
def test_inner_levels_of_the_whole_level_is_the_constant(beats: int) -> None:
    rule = level_done(beats, beats)
    assert type(rule) is LevelEnd and rule == LevelEnd(1) and rule.constant
    assert all(rule.asserted(beat) for beat in range(24))


def test_between_the_edges_it_is_the_new_kind_and_the_constructor_refuses_the_edges() -> None:
    assert level_done(12, 4) == LevelDone(12, 4)
    with pytest.raises(ValueError, match=r"is LevelEnd\(6\)"):
        LevelDone(6, 1)
    with pytest.raises(ValueError, match=r"LevelEnd\(1\)"):
        LevelDone(6, 6)
    for beats, inner in ((6, 4), (4, 6)):
        with pytest.raises(ValueError, match="divide no level"):
            LevelDone(beats, inner)
        with pytest.raises(ValueError, match="divide no level"):
            level_done(beats, inner)
    for beats, inner in ((0, 1), (2, 0)):
        with pytest.raises(ValueError, match="positive"):
            level_done(beats, inner)


def test_it_is_aligned_where_the_level_and_its_inner_levels_close_whole_loops() -> None:
    """The first tiled case's activation port, loops (g 2, n 2, k 3, th 3): the inner
    closes th (3) or k and th (9), the beats n, k and th (18) or k and th (9)."""
    _, port = tiled_core(6, 12, 6, 3, 4, 3)
    assert [loop.extent for loop in port.beat_loops] == [2, 2, 3, 3]
    for rule in (LevelDone(9, 3), LevelDone(18, 9), LevelDone(18, 3), LevelDone(36, 18)):
        assert rule.aligned(port)
        assert BeatSequence(port, markers=(rule,)).markers == (rule,)
    # 6 beats end inside k's 3 index values, 2 beats inside th's 3; 72 beats are two frames.
    for rule in (LevelDone(6, 3), LevelDone(4, 2), LevelDone(72, 36)):
        assert not rule.aligned(port)
    with pytest.raises(ValueError, match="last 2 beats of every 4 beats closes no loop level"):
        BeatSequence(port, markers=(LevelDone(4, 2),))
    with pytest.raises(TypeError, match="LevelEnd or LevelDone"):
        BeatSequence(port, markers=(9,))  # type: ignore[arg-type]


def test_on_a_lane_group_it_closes_whole_loops_of_the_order() -> None:
    """(2, 4, 3) in the order (2 at 12, 3 at 1, 4 at 3), three elements a beat: 8 beats.
    The two inner loops are 12 elements, 4 beats: inner 4, beats 8. A inner of 2 beats,
    6 elements, ends inside the loop of 4."""
    group = Grouped((2, 4, 3), (Loop(2, 12), Loop(3, 1), Loop(4, 3)), 3)
    assert group.beats == 8
    assert LevelDone(8, 4).aligned(group)
    assert not LevelDone(8, 2).aligned(group)
    assert not LevelDone(16, 8).aligned(group)


def test_steps_synthesize_it_in_one_markers_step() -> None:
    """The first tiled case's activations, row-major at SIMD 4, to the core's port with
    k's last index value: the reorder, then the marker, after the last sequence step."""
    _, port = tiled_core(6, 12, 6, 3, 4, 3)
    sink = BeatSequence(port, markers=(LevelDone(9, 3),))
    chain = steps(BeatSequence(vector_major((6, 12), 4)), sink)
    assert chain.kinds == (Step.REORDER, Step.MARKERS)
    assert set(Step) == {Step.REORDER, Step.WIDTH, Step.MARKERS}
    marked = chain.hops[-1]
    assert marked.source.form == marked.sink.form == port
    assert marked.sink.markers == (LevelDone(9, 3),)
    # A source offering it already, in the same order: nothing to do.
    assert not steps(sink, sink)
    # Offering the level closing instead is another marker: synthesized.
    other = BeatSequence(port, markers=(LevelEnd(9),))
    assert steps(other, sink).kinds == (Step.MARKERS,)


# -- closing -------------------------------------------------------------------------------


def _schedules() -> list[tuple[Schedule, tuple[Index, ...]]]:
    oh, kh, kw, c = (Index(name) for name in ("oh", "kh", "kw", "c"))
    return [
        (Schedule({m: 4, n: 6, k: 8}, {n: 2, k: 2}, (m, n, k)), (k,)),
        (Schedule({m: 4, n: 6, k: 8}, {n: 2, k: 2}, (m, k, n)), (n, k)),
        (Schedule({oh: 5, n: 4, kh: 3, kw: 3, c: 8}, {c: 4}, (oh, n, kh, kw, c)), (kw, kh, c)),
        (Schedule({m: 3, k: 4}, {k: 4}, (m, k)), (k,)),
    ]


@pytest.mark.parametrize("case", range(4))
def test_a_reduction_with_nothing_inside_is_today_s_level_end(case: int) -> None:
    schedule, reduces = _schedules()[case]
    line = translate.back(schedule)
    assert isinstance(line, line_schedule.Schedule)
    found = schedule.closing(reduces)
    assert type(found) is LevelEnd
    assert found == translate.forward(line.closing(translate.back(reduces)))
    assert [found.asserted(beat) for beat in range(schedule.iterations)] == last_iterations(
        schedule, reduces
    )


@pytest.mark.parametrize(
    ("case", "marker", "beats"), [(*t, b) for t, b in zip(TILED, TILED_BEATS)], ids=TILED_IDS
)
def test_the_tiled_core_s_k_is_on_its_last_step_across_th(
    case: tuple[int, ...], marker: tuple[int, int], beats: tuple[int, int]
) -> None:
    vectors, depth, outputs, pe, simd, tile = case
    core, port = tiled_core(*case)
    found = core.closing((k,))
    assert found == LevelDone(*marker) == LevelDone((depth // simd) * tile, tile)
    assert found.aligned(port)
    # By enumeration of the core's beats: exactly where k is at its last index value.
    wanted = last_iterations(core, (k,))
    assert [found.asserted(beat) for beat in range(core.iterations)] == wanted
    assert sum(wanted) == core.iterations // (depth // simd)
    assert (core.iterations, sum(wanted)) == beats
    assert sum(found.asserted(beat) for beat in range(port.beats)) == beats[1]
    # The line refuses it: k is not the innermost.
    line = translate.back(core)
    with pytest.raises(line_schedule.Refused, match="are not the innermost"):
        line.closing(translate.back((k,)))


def test_a_block_of_several_indices_spans_every_index_after_it() -> None:
    """(oh 5, kh 3, kw 3, c 2 index values, t 4): the block (kh, kw, c) spans t's 4 beats, its
    beats 3 * 3 * 2 * 4 = 72; in any order of naming."""
    oh, kh, kw, c, t = (Index(name) for name in ("oh", "kh", "kw", "c", "t"))
    schedule = Schedule({oh: 5, kh: 3, kw: 3, c: 8, t: 4}, {c: 4}, (oh, kh, kw, c, t))
    for reduces in ((kh, kw, c), (c, kh, kw)):
        found = schedule.closing(reduces)
        assert found == LevelDone(72, 4)
        assert [found.asserted(beat) for beat in range(schedule.iterations)] == last_iterations(
            schedule, reduces
        )


def test_edges_of_closing_are_level_ends() -> None:
    """Indices of one step inside the block inner one beat: ``LevelEnd(beats)``. A block
    of one step holds on every beat: ``LevelEnd(1)``."""
    t = Index("t")
    inner_of_one = Schedule({m: 2, k: 8, t: 3}, {k: 2, t: 3}, (m, k, t))
    assert inner_of_one.closing((k,)) == LevelEnd(4)
    block_of_one = Schedule({m: 2, k: 4, t: 3}, {k: 4}, (m, k, t))
    found = block_of_one.closing((k,))
    assert found == LevelEnd(1)
    assert [found.asserted(b) for b in range(block_of_one.iterations)] == last_iterations(
        block_of_one, (k,)
    )


def test_an_index_between_two_reduced_ones_is_refused() -> None:
    schedule = Schedule({m: 2, n: 4, k: 4}, {k: 2}, (k, n, m))
    with pytest.raises(Refused, match=r"not one block .*\[n\] lie between them") as error:
        schedule.closing((k, m))
    assert [found.code for found in error.value.findings] == ["reduction-not-innermost"]


@pytest.mark.parametrize("reduces", [(), (k, k), (Index("q"),)])
def test_nothing_reduced_twice_or_unknown_is_refused_as_today(
    reduces: tuple[Index, ...],
) -> None:
    schedule = Schedule({m: 2, n: 4, k: 4}, {k: 2}, (m, n, k))
    line = translate.back(schedule)
    with pytest.raises(line_schedule.Refused) as today:
        line.closing(translate.back(reduces))
    with pytest.raises(Refused) as error:
        schedule.closing(reduces)
    assert str(error.value) == str(today.value)
    assert [found.code for found in error.value.findings] == ["reduction-not-innermost"]
