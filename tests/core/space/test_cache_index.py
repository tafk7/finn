# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A cache finds what a change reaches from who read what, as looking at every evaluation does.

``Cache.without(seeds)`` answers what ``unaffected`` answers over the same evaluations, from
the readers it keeps as evaluations are added; the cache it is made from is left as it was.
"""

from __future__ import annotations

from random import Random

import pytest

from finn.core.space._runtime import Cache, Evaluation, unaffected
from finn.core.space.results import Available


def entry(reads: set[int]) -> Evaluation:
    return Evaluation(Available(0), decisions=frozenset(reads))


def random_cache(rng: Random, size: int = 120, decisions: int = 25) -> Cache:
    cache = Cache()
    for index in range(size):
        cache[index] = entry({rng.randrange(decisions) for _ in range(rng.randrange(4))})
    for _ in range(10):  # an index given again reads as the new entry does
        cache[rng.randrange(size)] = entry({rng.randrange(decisions) for _ in range(3)})
    return cache


def readers_of(cache: Cache) -> dict[int, set[int]]:
    found: dict[int, set[int]] = {}
    for index, held in cache.items():
        for read in held.decisions:
            found.setdefault(read, set()).add(index)
    return found


def indexed(cache: Cache) -> dict[int, set[int]]:
    cache._sync()
    return {read: set(each) for read, each in cache._readers.items() if each}


@pytest.mark.parametrize("seed", range(20))
def test_the_readers_are_those_of_the_evaluations_held(seed: int) -> None:
    cache = random_cache(Random(seed))
    assert indexed(cache) == readers_of(cache)


@pytest.mark.parametrize("seed", range(20))
def test_without_answers_what_unaffected_answers_and_leaves_its_base_as_it_was(seed: int) -> None:
    rng = Random(seed)
    base = random_cache(rng)
    before, before_readers = dict(base), indexed(base)
    seeds = {rng.randrange(25) for _ in range(rng.randrange(1, 5))} | {rng.randrange(120)}
    kept = base.without(seeds)
    assert dict(kept) == unaffected(base, seeds)
    assert indexed(kept) == readers_of(kept)
    assert dict(base) == before and indexed(base) == before_readers
    # Each goes on gaining evaluations without the other seeing them, whichever is read first.
    base[2000] = entry({3})
    kept[1000] = entry({3})
    assert 1000 not in base and 2000 not in kept
    assert indexed(base) == readers_of(base)
    assert indexed(kept) == readers_of(kept)
    assert 1000 in indexed(kept)[3] and 2000 not in indexed(kept)[3]
    assert 2000 in indexed(base)[3] and 1000 not in indexed(base)[3]


def test_what_a_cache_gained_since_it_was_made_is_carried_only_where_asked() -> None:
    base = random_cache(Random(1))
    first = base.without({0})
    assert first.fresh == {}
    first[500] = entry({7})
    first[501] = entry({8})
    carried = first.without({7}, keep_fresh=True)
    assert carried.fresh == {501: first[501]}
    assert first.without({7}).fresh == {}


@pytest.mark.parametrize("seed", range(20))
def test_evaluations_given_in_place_of_a_carried_one_are_fresh_and_read_as_given(seed: int) -> None:
    rng = Random(seed)
    base = random_cache(rng)
    base[300] = entry({5})  # not synced when the change asks
    given = {
        rng.randrange(120): entry({rng.randrange(25)}),
        300: entry({6}),
        400: entry({rng.randrange(25)}),
    }
    seeds = {rng.randrange(25), rng.randrange(25)}
    kept = base.without(seeds, replacing=given)
    wanted = {**{i: e for i, e in unaffected(base, seeds).items() if i not in given}, **given}
    assert dict(kept) == wanted
    assert indexed(kept) == readers_of(kept)
    assert kept.fresh == given
    carried = kept.without({6}, keep_fresh=True)
    assert carried.fresh == {i: e for i, e in given.items() if i != 300 and 6 not in e.decisions}
    assert indexed(carried) == readers_of(carried)
