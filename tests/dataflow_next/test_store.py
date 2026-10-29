# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The form a cyclic source stores for a consumer (``steps.store``): ``stored`` or
``one_pass`` of the consumer's form, chosen over the four fields of ``source_verdict``.

``one_pass``, presented once for each of the consumer's passes, is always the consumer's
form, so it always builds, directly. A refused ``stored`` loses; where both build,
``stored`` wins unless it is larger (a trade-off included).
"""

from __future__ import annotations

import pytest

from dataflow_next.harness.checks import undelivered

from finn.dataflow_next.gemm import k, n
from finn.dataflow_next.schedule import Index, Schedule
from finn.dataflow_next.steps import Cost, cost, presented, source_verdict, steps, store
from finn.dataflow_next.traversal import (
    BeatSequence,
    Grouped,
    Loop,
    Reorder,
    Repetition,
    Traversal,
    one_pass,
    stored,
    vector_major,
)

G, TH = Index("g"), Index("th")


def from_cyclic(form: Traversal | Grouped, sink: Traversal) -> tuple[BeatSequence, BeatSequence]:
    return BeatSequence(form, Repetition.CYCLIC), BeatSequence(sink)


def test_a_window_that_skips_and_reads_twice_is_stored_as_its_one_pass() -> None:
    """(4, 6) at one lane: rows 0 and 2, each through 3 windows of 3 at stride 1 over its
    first 5 columns. 18 reads; columns 1 to 3 twice, rows 1 and 3 and column 5 never.

    - ``stored`` is the 24-element frame, presented once for the consumer's one pass: a
      reorder over two rows at a time (frame 12, dims (3, 3), coefs (1, 1)) reads the
      first row's windows and drops the second row; cost (2, 24, 1).
    - ``one_pass`` is the 18 reads: direct, cost (0, 18, 1).

    Over the four fields ``stored`` is larger in every one but the lanes (24 against 18
    elements, a live window of 2 against none, 24 reads against 18), so ``one_pass`` wins."""
    form = Traversal((4, 6), (Loop(2, 12), Loop(3, 1), Loop(3, 1)), ())
    small, whole = stored(form), one_pass(form)
    assert small == vector_major((4, 6), 1) and whole == form
    source, sink = from_cyclic(small, form)
    chain = steps(source, sink)
    assert [hop.reorder for hop in chain.hops] == [Reorder(12, (3, 3), (1, 1))]
    assert cost(chain, sink) == Cost(2, 24, 1)
    assert undelivered(chain, source, sink) == ""
    direct = steps(*from_cyclic(whole, form))
    assert not direct and cost(direct, sink) == Cost(0, 18, 1)
    assert source_verdict(24, cost(chain, sink), 18, cost(direct, sink)) == "larger"
    assert store(BeatSequence(form)) == whole


def test_a_one_pass_presented_for_each_consumer_pass_trades_off_against_stored() -> None:
    """The same rows and windows with each window's start read twice (a stride-0 loop of 2
    between the window and its 3 reads), and the whole pass twice: 72 reads.

    - ``one_pass`` is one pass, 36 reads, presented twice: the consumer's form, direct,
      cost (0, 72, 1), 36 elements stored.
    - ``stored`` is the 24-element frame, presented twice: one reorder (frame 12, dims (3,
      2, 3), coefs (1, 0, 1)), cost (3, 48, 1).

    A smaller memory and fewer reads for a live window of 3: a trade-off, which ``stored``
    wins."""
    form = Traversal((4, 6), (Loop(2, 0), Loop(2, 12), Loop(3, 1), Loop(2, 0), Loop(3, 1)), ())
    small, whole = stored(form), one_pass(form)
    assert small == vector_major((4, 6), 1)
    chains = {}
    for name, given in (("stored", small), ("one_pass", whole)):
        source, sink = from_cyclic(given, form)
        chain = steps(source, sink)
        assert undelivered(chain, source, sink) == ""
        chains[name] = chain
    assert [hop.reorder for hop in chains["stored"].hops] == [Reorder(12, (3, 2, 3), (1, 0, 1))]
    assert not chains["one_pass"]
    priced = {name: cost(chain, BeatSequence(form)) for name, chain in chains.items()}
    assert priced == {"stored": Cost(3, 48, 1), "one_pass": Cost(0, 72, 1)}
    assert source_verdict(24, priced["stored"], 36, priced["one_pass"]) == "trade-off"
    assert store(BeatSequence(form)) == small


# (N, MW, MH, PE, SIMD, TH), and how ``stored`` stands to ``one_pass`` over the four fields.
TILED = [
    ((6, 12, 6, 3, 4, 3), "trade-off"),
    ((8, 16, 8, 4, 4, 2), "trade-off"),
    ((6, 12, 4, 2, 6, 3), "trade-off"),
]


@pytest.mark.parametrize(
    ("case", "verdict"), TILED, ids=["6-12-6-3-4-3", "8-16-8-4-4-2", "6-12-4-2-6-3"]
)
def test_the_tiled_weights_are_stored_once(case: tuple[int, ...], verdict: str) -> None:
    """The core's weights (order g, n, k, th; read at PE and SIMD, each beat TH times):
    ``stored`` holds the matrix once, ``one_pass`` with its TH-fold replay, presented once
    for each of the core's N / TH passes: direct. Each case is a trade-off (a memory a
    TH-th of the size against one beat held for the replay), which ``stored`` wins."""
    vectors, depth, outputs, pe, simd, tile = case
    core = Schedule(
        {G: vectors // tile, n: outputs, k: depth, TH: tile}, {n: pe, k: simd}, (G, n, k, TH)
    )
    weights = core.present((depth, outputs), (k, n), lanes=(n, k))
    small, whole = stored(weights), one_pass(weights)
    chains = [
        cost(steps(*from_cyclic(form, weights)), BeatSequence(weights)) for form in (small, whole)
    ]
    depths = [form.beats * form.lanes for form in (small, whole)]
    assert depths == [depth * outputs, depth * outputs * tile]
    assert source_verdict(depths[0], chains[0], depths[1], chains[1]) == verdict
    assert store(BeatSequence(weights)) == small


def test_a_consumer_reading_each_position_once_stores_its_form() -> None:
    form = vector_major((4, 6), 3)
    assert stored(form) == one_pass(form) == form
    assert store(BeatSequence(form)) == form
    assert store(BeatSequence(form.repeated(3))) == form


# Where both candidates are refused, ``store`` once raised the refusal of ``stored``: (4,)
# at one lane, three passes of three windows of 2 at stride 1, each read three times in a
# row, refused from ``stored`` (54 reads, or 18 without the replay, are no whole number of
# the 4-element frame) and from ``one_pass`` (counted once where the consumer reads it
# three times). Both build now: a source is presented once for each pass the consumer
# makes, so ``one_pass``, presented that often, is the consumer's form itself, and no case
# refuses both ways. ``store`` refuses nothing; the test below takes the case and the
# rule in its place.


def test_one_pass_always_builds_directly_so_store_refuses_nothing() -> None:
    """``one_pass`` of any consumer, presented once for each of its passes, is its form:
    direct. The case where both were refused, (4,) as above, now builds both ways:
    ``stored`` (the 4-element frame, three passes) by one reorder (frame 4, dims (3, 2,
    3), coefs (1, 1, 0)), cost (1, 12, 1); ``one_pass`` (18 elements) directly, cost
    (0, 54, 1); a trade-off, which ``stored`` wins. Where ``stored`` itself is refused
    (6 lanes against the 3 elements of (3,), each replayed 4 times: no pass to group),
    ``one_pass`` is chosen."""
    form = Traversal((4,), (Loop(3, 0), Loop(3, 1), Loop(2, 1), Loop(3, 0)), ())
    replays = Grouped((3,), (Loop(3, 1), Loop(4, 0)), 6)
    others = (
        Traversal((4, 6), (Loop(2, 12), Loop(3, 1), Loop(3, 1)), ()),
        vector_major((4, 6), 3).repeated(3),
        Traversal((6,), (Loop(4, 1), Loop(3, 1)), ()),
    )
    for each in (form, replays, *others):
        source, sink = BeatSequence(one_pass(each), Repetition.CYCLIC), BeatSequence(each)
        assert presented(source, sink) == each
        assert not steps(source, sink)
    small_source, sink = from_cyclic(stored(form), form)
    chain = steps(small_source, sink)
    assert [hop.reorder for hop in chain.hops] == [Reorder(4, (3, 2, 3), (1, 1, 0))]
    assert undelivered(chain, small_source, sink) == ""
    assert cost(chain, sink) == Cost(1, 12, 1)
    assert cost(steps(*from_cyclic(one_pass(form), form)), sink) == Cost(0, 54, 1)
    assert source_verdict(4, Cost(1, 12, 1), 18, Cost(0, 54, 1)) == "trade-off"
    assert store(sink) == stored(form) == vector_major((4,), 1)
    with pytest.raises(ValueError, match="6 lanes group no pass"):
        stored(replays)
    assert store(BeatSequence(replays)) == replays
