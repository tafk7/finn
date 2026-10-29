# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The harness's checks for the steps: a chain where the line refused is accepted only
when it delivers the consumer's positions by enumeration, and two differing chains pass
as ``smaller`` only when both deliver them and the revision's cost is smaller; a
trade-off or a narrower chain fails, named."""

from __future__ import annotations

import pytest

import dataclasses
import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from dataflow_next.harness import checks, geometries, replay, translate
from dataflow_next.harness.checks import (
    closing_asserted,
    steps_built_cost,
    steps_cost,
    steps_deliver,
    undelivered,
    weighed,
)
from dataflow_next.harness.record import Call, Raised, Recorder
from dataflow_next.harness.replay import FAILING, judge, table
from typing import Any

import finn.dataflow.steps as line_steps
import finn.dataflow_next.steps as next_steps
from finn.dataflow.gemm import k, m, n
from finn.dataflow.schedule import Index, Refused, Schedule
from finn.dataflow.traversal import BeatSequence, Repetition, Traversal, period
from finn.dataflow.traversal import regrouped as line_regrouped
from finn.dataflow.traversal import unreplayed, vector_major
from finn.dataflow_next.schedule import Schedule as NextSchedule
from finn.dataflow_next.steps import Cost, Hop, Steps, cost
from finn.dataflow_next.traversal import BeatSequence as NextSequence
from finn.dataflow_next.traversal import LevelDone, LevelEnd
from finn.dataflow_next.traversal import Loop as NextLoop
from finn.dataflow_next.traversal import Reorder
from finn.dataflow_next.traversal import Traversal as NextTraversal
from finn.dataflow_next.traversal import classify, regrouped
from finn.dataflow_next.traversal import vector_major as next_vector_major


def _recorded(ask: Callable[[], None], label: str = "steps.steps") -> list[Call]:
    """The distinct calls of ``label`` that ``ask`` makes of the line."""
    with Recorder() as recorder:
        ask()
    return [call for call in recorder.recorded() if call.target.label == label]


def _ask(source: BeatSequence, sink: BeatSequence) -> Callable[[], None]:
    def ask() -> None:
        try:
            line_steps.steps(source, sink)
        except line_steps.Unrealizable:
            pass

    return ask


def _transpose(lanes: int, pe: int, simd: int) -> tuple[BeatSequence, BeatSequence]:
    """X (32, 16) at ``lanes``; a dot product (M 16, K 32, N 16) reading Y = Xᵀ, over X."""
    schedule = Schedule({m: 16, n: 16, k: 32}, {n: pe, k: simd}, (m, n, k))
    port = Traversal.over(
        (32, 16), ((1, 16, 1), (None, 16 // pe, 0), (0, 32 // simd, simd)), ((0, simd, 1),)
    )
    return BeatSequence(vector_major((32, 16), lanes)), BeatSequence(
        port, markers=(schedule.closing((k,)),)
    )


ALONG_ROWS = vector_major((4, 4), 4)
ALONG_COLUMNS = Traversal.over((4, 4), ((1, 4, 1),), ((0, 4, 1),))


# -- real pairs ----------------------------------------------------------------------------


def test_the_geometries_build_where_the_line_refuses_and_price_the_looping_source() -> None:
    tables = {}
    for name, case in geometries.CASES.items():
        with Recorder() as recorder:
            case()
        tables[name] = table(name, list(recorder.recorded()))
    assert all(found.passed for found in tables.values())
    assert tables["windows"].totals["same"] == tables["windows"].calls
    assert tables["regroup"].totals["builds_where_today_refuses"] == 2
    assert tables["transposes"].totals["builds_where_today_refuses"] == 5
    # Each build carries the revision's cost and no line cost; a one-lane regroup shows.
    for name, count in (("regroup", 2), ("transposes", 5)):
        built = tables[name].priced
        assert len(built) == count
        assert all(v.verdict == "builds_where_today_refuses" and v.line_cost is None for v in built)
        assert all(isinstance(v.next_cost, Cost) and v.note for v in built)
    assert [v.next_cost for v in tables["regroup"].priced] == [Cost(20, 4, 1)] * 2
    tiled = tables["tiled"]
    assert tiled.totals["smaller"] == 2
    # The three cases' marker of k, which the line refuses (k is not the innermost).
    assert tiled.totals["builds_where_today_refuses"] == 3
    assert ("schedule.Schedule.closing", "builds_where_today_refuses", 3) in tiled.counts
    assert [(priced.line_cost, priced.next_cost) for priced in tiled.priced] == [
        (Cost(84, 108, 4), Cost(24, 36, 4)),
        (Cost(144, 128, 8), Cost(32, 64, 8)),
    ]


@pytest.mark.parametrize("arrangement", [(4, 4, 4), (2, 4, 2)], ids=str)
def test_the_transposes_the_line_refuses_are_checked_and_pass(
    arrangement: tuple[int, int, int],
) -> None:
    (call,) = _recorded(_ask(*_transpose(*arrangement)))
    assert isinstance(call.outcome, Raised)
    verdict = judge(call)
    assert verdict.verdict == "builds_where_today_refuses"


def test_the_harness_s_own_looping_source_is_smaller() -> None:
    """A source cycling (2, 4) at 4 lanes into a port replaying each row twice (N 4, PE 2):
    the line presents the matrix twice, the revision once, then replays."""
    schedule = Schedule({m: 2, n: 4, k: 4}, {n: 2, k: 2}, (m, n, k))
    port = schedule.present((2, 4), (m, k), lanes=(k,))
    source = BeatSequence(vector_major((2, 4), 4), Repetition.CYCLIC)
    (call,) = _recorded(_ask(source, BeatSequence(port)))
    verdict = judge(call)
    assert verdict.verdict == "smaller"
    assert isinstance(verdict.line_cost, Cost) and isinstance(verdict.next_cost, Cost)
    assert verdict.next_cost.reads < verdict.line_cost.reads


def test_the_presented_comparator_prices_the_looping_source() -> None:
    source, sink = _tiled_weights()
    with Recorder() as recorder:
        line_steps.presented(source, sink)
    (call,) = [c for c in recorder.recorded() if c.target.label == "steps.presented"]
    verdict = judge(call)
    assert verdict.verdict == "smaller"
    assert (verdict.line_cost, verdict.next_cost) == (Cost(84, 108, 4), Cost(24, 36, 4))


def test_a_presentation_where_the_line_refuses_is_checked_and_passes() -> None:
    """(4, 6) at one lane, rows 0 and 2 through 3 windows of 3 at stride 1 each, from the
    24-element frame cycled: the line counts 18 reads, no whole number of 24, and refuses;
    the revision presents the frame once, whose chain delivers. A presentation that is no
    whole number of the source's passes, or whose chain does not deliver, is rejected."""
    form = Traversal.over((4, 6), ((0, 2, 2), (1, 3, 1), (1, 3, 1)), ())
    source = BeatSequence(vector_major((4, 6), 1), Repetition.CYCLIC)

    def ask() -> None:
        try:
            line_steps.presented(source, BeatSequence(form))
        except line_steps.Unrealizable:
            pass

    (call,) = _recorded(ask, "steps.presented")
    refusal = call.outcome
    assert isinstance(refusal, Raised)
    assert judge(call).verdict == "builds_where_today_refuses"
    frame = translate.forward(vector_major((4, 6), 1))
    assert checks.presented_delivers(call, refusal, translate.back(frame))
    half = NextTraversal(frame.shape, (NextLoop(12, 1),), ())
    assert not checks.presented_delivers(call, refusal, translate.back(half))
    assert not checks.presented_delivers(call, refusal, (1, 2))


def _tiled_weights() -> tuple[BeatSequence, BeatSequence]:
    """(N, MW, MH, PE, SIMD, TH) = (6, 12, 6, 3, 4, 3): the core's weight port and a
    memory looping over one matrix, as the line states them."""
    g, th = Index("g"), Index("th")
    core = Schedule({g: 2, n: 6, k: 12, th: 3}, {n: 3, k: 4}, (g, n, k, th))
    weights = core.present((12, 6), (k, n), lanes=(n, k))
    port = line_regrouped(unreplayed(weights), 4)
    return BeatSequence(period(port), Repetition.CYCLIC), BeatSequence(weights)


# -- the checker ---------------------------------------------------------------------------


def _regroup_call() -> tuple[Call, Steps]:
    (call,) = _recorded(_ask(BeatSequence(ALONG_ROWS), BeatSequence(ALONG_COLUMNS)))
    chain = next_steps.steps(*translate.forward((call.args[0], call.args[1])))
    return call, chain


def test_the_checker_accepts_the_regroup_and_rejects_what_does_not_deliver() -> None:
    call, chain = _regroup_call()
    refusal = call.outcome
    assert isinstance(refusal, Raised)
    assert steps_deliver(call, refusal, translate.back(chain))
    priced = steps_built_cost(call, translate.back(chain))
    assert priced is not None
    assert (priced.next_cost, priced.detail) == (
        Cost(20, 4, 1),
        "width_conversion -> reorder -> width_conversion",
    )
    assert steps_built_cost(call, (1, 2)) is None
    width, reorder, widen = chain.hops
    assert reorder.reorder is not None
    wrong = dataclasses.replace(reorder, reorder=Reorder(16, (4, 4), (4, 1)))
    cases = {
        "a wrong reorder": Steps((width, wrong, widen)),
        "a missing step": Steps((width, reorder)),
        "no step": Steps(),
        "a step that does not follow": Steps((width, widen, reorder)),
    }
    for name, broken in cases.items():
        assert not steps_deliver(call, refusal, translate.back(broken)), name
    assert not steps_deliver(call, refusal, (1, 2))


def test_the_checker_rejects_a_chain_from_beats_the_source_does_not_present() -> None:
    call, chain = _regroup_call()
    source, sink = translate.forward((call.args[0], call.args[1]))
    # Start from the source presented twice, though it presents its pass once.
    twice = source.form.repeated(2)
    start = Hop(chain.hops[0].step, NextSequence(twice), chain.hops[0].sink)
    assert "no whole presentation" in undelivered(Steps((start, *chain.hops[1:])), source, sink)
    other = NextSequence(NextTraversal((4, 4), (), ()))
    assert undelivered(chain, source, other)


# -- one wiring at an end ------------------------------------------------------------------

SAME = np.array([[0, 1], [2, 3]], dtype=np.int64)
SWAPPED = np.array([[1, 0], [3, 2]], dtype=np.int64)  # one permutation, (1, 0), in every beat
EACH_BEAT = np.array([[1, 0], [2, 3]], dtype=np.int64)  # a permutation in each beat, not one
OTHER = np.array([[0, 1], [2, 4]], dtype=np.int64)


@pytest.mark.parametrize(
    ("first", "second", "wired"),
    [
        (SAME, SAME, True),
        (SAME, SWAPPED, True),
        (SWAPPED, SAME, True),
        (SAME, EACH_BEAT, False),
        (EACH_BEAT, SAME, False),
        (SAME, OTHER, False),
        (SAME, SAME[:1], False),
        (SAME[:, :1], SAME[:, 1:], False),
    ],
    ids=["equal", "one swap", "swap back", "no single", "no single back", "other", "beats", "lane"],
)
def test_one_wiring_is_one_lane_permutation_over_every_beat(
    first: npt.NDArray[np.int64], second: npt.NDArray[np.int64], wired: bool
) -> None:
    assert checks._one_wiring(first, second) is wired


def test_one_wiring_finds_the_permutation_among_repeated_lanes() -> None:
    """Two lanes of one position all through the frame are interchangeable."""
    first = np.array([[7, 7, 1], [7, 7, 2], [7, 7, 3]], dtype=np.int64)
    second = np.array([[1, 7, 7], [2, 7, 7], [3, 7, 7]], dtype=np.int64)
    assert checks._one_wiring(first, second)
    mixed = np.array([[1, 7, 7], [7, 2, 7], [3, 7, 7]], dtype=np.int64)
    assert not checks._one_wiring(first, mixed)


def _ends_with_beats(
    monkeypatch: pytest.MonkeyPatch, source: npt.NDArray[np.int64], sink: npt.NDArray[np.int64]
) -> tuple[Steps, NextSequence, NextSequence, NextSequence]:
    """A chain of one marker step over the (2, 2) tensor's row-major form, with a source
    and a sink that are other forms of that tensor, and ``checks._rows`` reading the beats
    given for each: the row-major form's are ``SAME``, the source's ``source``, the sink's
    ``sink``. (One ``Traversal`` differs from another in its lanes by one fixed order, so a
    permutation that changes from beat to beat is put in by hand; the forms stand for
    them, and the arrays are what the check reads.) Returns the chain, the
    source, the form the chain runs on and the sink."""
    here = NextSequence(next_vector_major((2, 2), 2))
    start = NextSequence(NextTraversal.over((2, 2), ((1, 2, 1),), ((0, 2, 1),)))
    end = NextSequence(next_vector_major((2, 2), 1))
    beats = {here.form: SAME, start.form: source, end.form: sink}
    monkeypatch.setattr(checks, "_rows", lambda form: beats[form])
    return Steps((Hop(next_steps.Step.MARKERS, here, here),)), start, here, end


@pytest.mark.parametrize(
    ("sink", "wired"), [(SAME, True), (SWAPPED, True), (EACH_BEAT, False), (OTHER, False)]
)
def test_a_chain_ends_at_the_sink_through_one_wiring_only(
    monkeypatch: pytest.MonkeyPatch, sink: npt.NDArray[np.int64], wired: bool
) -> None:
    chain, _, here, end = _ends_with_beats(monkeypatch, SAME, sink)
    found = undelivered(chain, here, end)
    assert (found == "") is wired
    if not wired:
        assert found == "the chain ends at other beats than the sink reads"


@pytest.mark.parametrize(
    ("source", "wired"), [(SAME, True), (SWAPPED, True), (EACH_BEAT, False), (OTHER, False)]
)
def test_a_chain_starts_from_the_source_through_one_wiring_only(
    monkeypatch: pytest.MonkeyPatch, source: npt.NDArray[np.int64], wired: bool
) -> None:
    chain, start, here, _ = _ends_with_beats(monkeypatch, source, SAME)
    found = undelivered(chain, start, here)
    assert (found == "") is wired
    if not wired:
        assert found == "the chain starts from beats the source does not present"


# -- the comparator ------------------------------------------------------------------------


def _through(lanes: int) -> Steps:
    """2 x 2 tiles of (4, 8) into four lanes along rows, regrouped through ``lanes``."""
    tiles = NextSequence(NextTraversal.over((4, 8), ((0, 2, 2), (1, 4, 2)), ((0, 2, 1), (1, 2, 1))))
    rows = translate.forward(vector_major((4, 8), 4))
    low, target = regrouped(tiles.form, lanes), regrouped(rows, lanes)
    between = classify(low, target)
    assert between.reorder is not None
    first, middle, last = (NextSequence(form) for form in (low, target, rows))
    return Steps(
        (
            Hop(next_steps.Step.WIDTH, tiles, first),
            Hop(next_steps.Step.REORDER, first, middle, between.reorder),
            Hop(next_steps.Step.WIDTH, middle, last),
        )
    )


def _tiles_call(line: Steps, revision: Steps, monkeypatch: pytest.MonkeyPatch) -> Call:
    """A ``steps`` call on the tiles whose line answered ``line`` and whose revision
    answers ``revision``."""
    tiles = Traversal.over((4, 8), ((0, 2, 2), (1, 4, 2)), ((0, 2, 1), (1, 2, 1)))
    (call,) = _recorded(_ask(BeatSequence(tiles), BeatSequence(vector_major((4, 8), 4))))
    monkeypatch.setattr(next_steps, "steps", lambda source, sink: revision)
    return dataclasses.replace(call, outcome=translate.back(line))


def test_a_narrower_chain_is_priced_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through one lane rather than two: the same positions, buffers and reads, but
    slower, so not smaller."""
    two, one = _through(2), _through(1)
    sink = translate.forward(BeatSequence(vector_major((4, 8), 4)))
    assert (cost(two, sink), cost(one, sink)) == (Cost(20, 8, 2), Cost(20, 8, 1))
    verdict = judge(_tiles_call(two, one, monkeypatch))
    assert verdict.verdict == "differs" and verdict.verdict in FAILING
    assert verdict.detail.startswith("narrower")
    assert (verdict.line_cost, verdict.next_cost) == (Cost(20, 8, 2), Cost(20, 8, 1))
    # The other way round it is wider for the same buffers and reads: smaller.
    wider = judge(_tiles_call(one, two, monkeypatch))
    assert wider.verdict == "smaller" and wider.verdict not in FAILING
    assert (wider.line_cost, wider.next_cost) == (Cost(20, 8, 1), Cost(20, 8, 2))


def test_a_trade_off_is_priced_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    two, one = _through(2), _through(1)
    priced = {two: Cost(20, 8, 2), one: Cost(10, 16, 2)}
    monkeypatch.setattr(checks, "cost", lambda chain, sink: priced[chain])
    verdict = judge(_tiles_call(two, one, monkeypatch))
    assert verdict.verdict == "differs"
    assert verdict.detail.startswith("a trade-off: less live_window for more reads")


def test_a_cheaper_chain_of_other_positions_changes_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two = _through(2)
    width, reorder, widen = two.hops
    assert reorder.reorder is not None
    skipping = dataclasses.replace(
        reorder, reorder=dataclasses.replace(reorder.reorder, coefs=(0, 2))
    )
    broken = Steps((width, skipping, widen))
    priced = {two: Cost(20, 8, 2), broken: Cost(10, 8, 2)}
    monkeypatch.setattr(checks, "cost", lambda chain, sink: priced[chain])
    assert judge(_tiles_call(two, broken, monkeypatch)).verdict == "smaller_changes_positions"


def test_weighed_names_each_relation() -> None:
    line = Cost(20, 8, 2)
    assert weighed(line, Cost(10, 8, 2)) == "smaller"
    assert weighed(line, Cost(20, 8, 2)) == "equal costs"
    assert weighed(line, Cost(30, 8, 2)) == "larger"
    assert weighed(line, Cost(20, 8, 4)) == "smaller"  # wider only
    assert weighed(line, Cost(30, 8, 4)) == "larger"  # wider, but a larger window
    assert weighed(line, Cost(30, 4, 2)).startswith("a trade-off: less reads for more live_window")
    assert weighed(line, Cost(1, 1, 1)).startswith("narrower")


def test_a_comparator_leaves_answers_that_are_no_chains_unexplained() -> None:
    (call,) = _recorded(_ask(BeatSequence(ALONG_ROWS), BeatSequence(ALONG_ROWS)))
    assert steps_cost(call, (1, 2, 3), (1, 2)) is None
    weird: Any = dataclasses.replace(call, args=(1, 2))
    assert steps_cost(weird, translate.back(Steps()), translate.back(Steps())) is None


# -- the marker of a reduction's last index value -------------------------------------------------


def _tiled_closing() -> list[Call]:
    """The line's ``closing`` of the tiled core's k, (N, MW, MH, PE, SIMD, TH) =
    (6, 12, 6, 3, 4, 3), order (g, n, k, th): refused, k is not the innermost."""
    g, th = Index("g"), Index("th")
    core = Schedule({g: 2, n: 6, k: 12, th: 3}, {n: 3, k: 4}, (g, n, k, th))

    def ask() -> None:
        try:
            core.closing((k,))
        except Refused:
            pass

    return _recorded(ask, "schedule.Schedule.closing")


def test_a_reduction_the_line_refuses_builds_and_is_checked_by_enumeration() -> None:
    (call,) = _tiled_closing()
    assert isinstance(call.outcome, Raised) and "not the innermost" in call.outcome.message
    verdict = judge(call)
    assert verdict.verdict == "builds_where_today_refuses"
    assert "LevelDone(beats=9, inner=3)" in verdict.detail
    assert closing_asserted(call, call.outcome, LevelDone(9, 3))


@pytest.mark.parametrize("wrong", [LevelEnd(9), LevelEnd(3), LevelDone(18, 9), LevelDone(36, 3), 9])
def test_the_closing_checker_rejects_another_marker(wrong: Any) -> None:
    (call,) = _tiled_closing()
    assert not closing_asserted(call, call.outcome, wrong)


def test_a_wrong_marker_where_the_line_refuses_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    (call,) = _tiled_closing()
    monkeypatch.setattr(NextSchedule, "closing", lambda self, reduces: LevelDone(18, 9))
    verdict = judge(call)
    assert verdict.verdict == "differs" and verdict.detail.startswith("checker rejects")


def test_a_value_the_line_has_no_class_for_is_untranslatable_without_a_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (call,) = _tiled_closing()
    monkeypatch.delitem(replay.CHECKERS, "schedule.Schedule.closing")
    verdict = judge(call)
    assert verdict.verdict == "untranslatable" and "LevelDone" in verdict.detail
