# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The enumeration oracle ``harness.checks.undelivered`` and its one-wiring test,
``harness.checks._one_wiring``, checked on their own: a chain is accepted only when it
delivers the consumer's positions from what its source presents, and a lane permutation is
wires at an end of the chain only when it is one permutation over every beat.

These guards were the line-independent part of the retired two-package harness tests
(``tests/dataflow_next/test_harness_checks.py`` at the immutable baseline 1ac1616). They
call the kept checker directly on canonical ``steps``; nothing here records, translates or
compares two packages.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest

from dataflow.harness import checks
from dataflow.harness.checks import undelivered
from finn.dataflow.steps import Cost, Hop, Step, Steps, cost, steps
from finn.dataflow.traversal import BeatSequence, Reorder, Traversal, vector_major

ALONG_ROWS = vector_major((4, 4), 4)
ALONG_COLUMNS = Traversal.over((4, 4), ((1, 4, 1),), ((0, 4, 1),))


# -- the checker ---------------------------------------------------------------------------


def _regroup() -> tuple[Steps, BeatSequence, BeatSequence]:
    """The (4, 4) tensor along rows at four lanes, read along columns at four lanes: one
    regroup through one lane, ``steps``' chain and its two ends."""
    source, sink = BeatSequence(ALONG_ROWS), BeatSequence(ALONG_COLUMNS)
    return steps(source, sink), source, sink


def test_the_checker_accepts_the_regroup_and_rejects_what_does_not_deliver() -> None:
    chain, source, sink = _regroup()
    assert undelivered(chain, source, sink) == ""
    assert (cost(chain, sink), chain.describe()) == (
        Cost(20, 4, 1),
        "width_conversion -> reorder -> width_conversion",
    )
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
        assert undelivered(broken, source, sink), name


def test_the_checker_rejects_a_chain_from_beats_the_source_does_not_present() -> None:
    chain, source, sink = _regroup()
    # Start from the source presented twice, though it presents its pass once.
    twice = source.form.repeated(2)
    start = Hop(chain.hops[0].step, BeatSequence(twice), chain.hops[0].sink)
    assert "no whole presentation" in undelivered(Steps((start, *chain.hops[1:])), source, sink)
    other = BeatSequence(Traversal((4, 4), (), ()))
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
) -> tuple[Steps, BeatSequence, BeatSequence, BeatSequence]:
    """A chain of one marker step over the (2, 2) tensor's row-major form, with a source
    and a sink that are other forms of that tensor, and ``checks._rows`` reading the beats
    given for each: the row-major form's are ``SAME``, the source's ``source``, the sink's
    ``sink``. (One ``Traversal`` differs from another in its lanes by one fixed order, so a
    permutation that changes from beat to beat is put in by hand; the forms stand for
    them, and the arrays are what the check reads.) Returns the chain, the
    source, the form the chain runs on and the sink."""
    here = BeatSequence(vector_major((2, 2), 2))
    start = BeatSequence(Traversal.over((2, 2), ((1, 2, 1),), ((0, 2, 1),)))
    end = BeatSequence(vector_major((2, 2), 1))
    beats = {here.form: SAME, start.form: source, end.form: sink}
    monkeypatch.setattr(checks, "_rows", lambda form: beats[form])
    return Steps((Hop(Step.MARKERS, here, here),)), start, here, end


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
