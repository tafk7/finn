# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A cache finds what a change reaches from who read what, as looking at every evaluation does.

``inherit(target, base, seeds)`` gives ``target`` what ``unaffected`` answers over ``base``'s
evaluations, by the readers it keeps where the cache is large (``INDEX_FROM``) and by looking
at each where it is small; ``base`` is left as it was. Each test runs on both paths, the index
used for every cache and then for none, and the threshold between them has its own test.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from random import Random
from typing import Any, cast

import pytest
from support.checked_cache import DIFFERING_READS, CheckedCache

from finn.core.space import _runtime
from finn.core.space._runtime import Evaluation, Readers, Snapshot, fresh, inherit, unaffected
from finn.core.space.results import Available


@pytest.fixture(params=["index", "scan"])
def path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(_runtime, "INDEX_FROM", 0 if request.param == "index" else sys.maxsize)
    return str(request.param)


def entry(reads: set[int]) -> Evaluation:
    return Evaluation(Available(0), decisions=frozenset(reads))


def snapshot() -> Snapshot:
    """A snapshot with a cache and nothing else: the model is never read."""
    return Snapshot(cast(Any, None), {})


def random_snapshot(rng: Random, size: int = 120, decisions: int = 25) -> Snapshot:
    made = snapshot()
    for index in range(size):
        made.cache[index] = entry({rng.randrange(decisions) for _ in range(rng.randrange(4))})
    return made


def readers_of(snap: Snapshot) -> dict[int, set[int]]:
    found: dict[int, set[int]] = {}
    for index, held in snap.cache.items():
        for read in held.decisions:
            found.setdefault(read, set()).add(index)
    return found


def indexed(snap: Snapshot) -> dict[int, set[int]]:
    assert snap.readers is not None
    snap.readers.sync(snap.cache)
    return {read: set(each) for read, each in snap.readers._readers.items() if each}


def inherited(base: Snapshot, seeds: set[int], **kwargs: Any) -> Snapshot:
    made = snapshot()
    inherit(made, base, seeds, **kwargs)
    return made


@pytest.fixture
def own_counts() -> Iterator[None]:
    """The counts this test makes of its own are not the engine's: restore the session's."""
    kept = DIFFERING_READS.copy()
    DIFFERING_READS.clear()
    yield
    DIFFERING_READS.clear()
    DIFFERING_READS.update(kept)


@pytest.fixture(autouse=True)
def cached_as_checked() -> Iterator[None]:
    assert _runtime.cache_factory is CheckedCache  # tests/conftest.py installs it
    yield


@pytest.mark.parametrize("seed", range(20))
def test_the_readers_are_those_of_the_evaluations_held(seed: int) -> None:
    base = random_snapshot(Random(seed))
    readers = Readers()
    readers.sync(base.cache)
    assert {read: set(each) for read, each in readers._readers.items()} == readers_of(base)
    base.cache[1000] = entry({3})  # added after: read at the next sync
    readers.sync(base.cache)
    assert readers._readers[3] >= {1000}


@pytest.mark.parametrize("seed", range(20))
def test_inherit_answers_what_unaffected_answers_and_leaves_its_base_as_it_was(
    seed: int, path: str
) -> None:
    rng = Random(seed)
    base = random_snapshot(rng)
    before = dict(base.cache)
    seeds = {rng.randrange(25) for _ in range(rng.randrange(1, 5))} | {rng.randrange(120)}
    kept = inherited(base, seeds)
    assert dict(kept.cache) == unaffected(base.cache, seeds)
    assert (kept.readers is not None) == (path == "index")
    if path == "index":
        assert indexed(kept) == readers_of(kept)
    assert dict(base.cache) == before
    # Each goes on gaining evaluations without the other seeing them, whichever is read first.
    base.cache[2000] = entry({3})
    kept.cache[1000] = entry({3})
    assert 1000 not in base.cache and 2000 not in kept.cache
    if path == "index":
        assert indexed(base) == readers_of(base)
        assert indexed(kept) == readers_of(kept)
        assert 1000 in indexed(kept)[3] and 2000 not in indexed(kept)[3]
        assert 2000 in indexed(base)[3] and 1000 not in indexed(base)[3]


def test_what_a_cache_gained_since_it_was_made_is_carried_only_where_asked(path: str) -> None:
    base = random_snapshot(Random(1))
    first = inherited(base, {0})
    assert fresh(first) == {}
    first.cache[500] = entry({7})
    first.cache[501] = entry({8})
    carried = inherited(first, {7}, keep_fresh=True)
    assert fresh(carried) == {501: first.cache[501]}
    assert fresh(inherited(first, {7})) == {}


@pytest.mark.parametrize("seed", range(20))
def test_evaluations_given_in_place_of_a_carried_one_are_fresh_and_read_as_given(
    seed: int, path: str
) -> None:
    rng = Random(seed)
    base = random_snapshot(rng)
    base.cache[300] = entry({5})  # not synced when the change asks
    given = {
        rng.randrange(120): entry({rng.randrange(25)}),
        300: entry({6}),
        400: entry({rng.randrange(25)}),
    }
    seeds = {rng.randrange(25), rng.randrange(25)}
    kept = inherited(base, seeds, replacing=given)
    wanted = {**{i: e for i, e in unaffected(base.cache, seeds).items() if i not in given}, **given}
    assert dict(kept.cache) == wanted
    if path == "index":
        assert indexed(kept) == readers_of(kept)
    assert fresh(kept) == given
    carried = inherited(kept, {6}, keep_fresh=True)
    assert fresh(carried) == {i: e for i, e in given.items() if i != 300 and 6 not in e.decisions}
    if path == "index":
        assert indexed(carried) == readers_of(carried)


def test_a_cache_is_indexed_from_the_threshold_up_and_looked_through_below_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_runtime, "INDEX_FROM", 5)
    for size, expect in ((4, False), (5, True), (50, True)):
        base = random_snapshot(Random(size), size=size, decisions=3)
        child = inherited(base, {1})
        assert (child.readers is not None) is expect and (base.readers is not None) is expect
        assert dict(child.cache) == unaffected(base.cache, {1})


def test_an_evaluation_is_only_added_and_giving_an_index_again_is_refused() -> None:
    cache = random_snapshot(Random(3)).cache
    held = dict(cache)
    with pytest.raises(RuntimeError, match="index 7"):
        cache[7] = entry({1})
    with pytest.raises(RuntimeError, match="index 7"):
        cache.update({8000: entry({2}), 7: entry({1})})
    with pytest.raises(RuntimeError, match="index 7"):
        cache |= {7: entry({1})}
    assert dict(cache) == {**held, 8000: cache[8000]}  # what came before the refusal stays
    assert cache.setdefault(7, held[7]) is held[7]  # asking without replacing is allowed
    cache.update({9000: entry({4})})
    cache |= {9001: entry({4})}


@pytest.mark.usefixtures("own_counts")
def test_setdefault_keeps_the_held_one_only_if_the_given_answers_alike_and_reads_no_less() -> None:
    cache = snapshot().cache
    held = Evaluation(Available(1), decisions=frozenset({1}))
    assert cache.setdefault(5, held) is held
    # Equal reads, an equal answer: kept, nothing counted.
    assert cache.setdefault(5, Evaluation(Available(1), decisions=frozenset({1}))) is held
    assert not DIFFERING_READS
    # The given one read more (a trial counts a candidate's admission): kept, counted.
    assert cache.setdefault(5, Evaluation(Available(1), decisions=frozenset({1, 2}))) is held
    assert sum(DIFFERING_READS.values()) == 1
    DIFFERING_READS.clear()
    with pytest.raises(RuntimeError, match="answers differently"):
        cache.setdefault(5, Evaluation(Available(2), decisions=frozenset({1})))
    with pytest.raises(RuntimeError, match="must not have read more"):
        cache.setdefault(5, Evaluation(Available(1), decisions=frozenset()))
    assert cache[5] is held and not DIFFERING_READS
