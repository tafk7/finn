"""Assert every worked example and claim in THEORY.md against the live finn.dataflow code.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <kernel venv python> docs/dataflow-model/check_theory.py
"""

from __future__ import annotations

import random

from finn.dataflow.gemm import k, m, n
from finn.dataflow.plan import Unrealizable, plan
from finn.dataflow.schedule import Index, Schedule
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Repetition,
    Traversal,
    classify,
    period,
    unreplayed,
    vector_major,
)


def loops(nest: tuple[Loop, ...]) -> list[tuple[int, int]]:
    return [(loop.extent, loop.stride) for loop in nest]


def positions(form: Traversal) -> list[tuple[tuple[int, ...], ...]]:
    return list(form.positions())


# §1: position rank
assert Traversal.over((2, 4), ((0, 2, 1), (1, 4, 1)), ()).position(6, 0) == (1, 2)

# §2.1: FINN's default order
v = vector_major((2, 4), 2)
assert loops(v.beat_loops) == [(4, 2)] and loops(v.lane_loops) == [(2, 1)]
assert positions(v) == [((0, 0), (0, 1)), ((0, 2), (0, 3)), ((1, 0), (1, 1)), ((1, 2), (1, 3))]

# §2.2: merging, and canonical form unique per sequence (seeded sample)
assert Traversal((2, 4), (Loop(2, 4), Loop(2, 2)), (Loop(2, 1),)) == v
random.seed(1)


def random_nest(count: int) -> list[Loop]:
    return [Loop(random.choice([1, 2, 3, 4]), random.choice([0, 1, 2, 3, 4, 6])) for _ in range(count)]


forms = []
for _ in range(4000):
    try:
        forms.append(
            Traversal((12,), random_nest(random.randint(0, 3)), random_nest(random.randint(0, 2)))
        )
    except ValueError:
        pass
by_sequence: dict[object, set[Traversal]] = {}
for form in forms:
    by_sequence.setdefault(tuple(positions(form)), set()).add(form)
assert len(by_sequence) == 696
assert all(len(found) == 1 for found in by_sequence.values())

# §3: replay, repetition, unreplayed, period
replayed = v.replayed(2, inner_beats=2)
assert loops(replayed.beat_loops) == [(2, 4), (2, 0), (2, 2)]
assert unreplayed(replayed) == v
repeated = v.repeated(3)
assert loops(repeated.beat_loops) == [(3, 0), (4, 2)]
assert unreplayed(repeated) == repeated and period(repeated) == v

# §3 lemma: position is reuse distance. On the sampled canonical nests, the element offsets a
# stride-0 loop re-presents are a strict subset of one pass inside the moving loop (replay)
# and exactly one pass outside it (repetition).


def offsets(nest: tuple[Loop, ...]) -> set[int]:
    found = {0}
    for loop in nest:
        found = {base + digit * loop.stride for base in found for digit in range(loop.extent)}
    return found


replays = repetitions = 0
for form in forms:
    beats = form.beat_loops
    moving = next((i for i, loop in enumerate(beats) if loop.stride), None)
    if moving is None:
        continue
    lanes = offsets(form.lane_loops)
    one_pass = {b + l for b in offsets(beats[moving:]) for l in lanes}
    for i, loop in enumerate(beats):
        if loop.stride:
            continue
        under = {b + l for b in offsets(beats[i + 1 :]) for l in lanes}
        if i < moving:
            assert under == one_pass
            repetitions += 1
        else:
            assert under < one_pass
            replays += 1
assert (replays, repetitions) == (116, 100), (replays, repetitions)

# §4.3: projection of the dotp toy
s = Schedule({m: 2, n: 2, k: 4}, factors={k: 2}, order=(m, n, k))
x = s.present((2, 4), (m, k), lanes=(k,))
w = s.present((4, 2), (k, n), lanes=(n, k))
y = s.present((2, 2), (m, n), lanes=(n,), reduces=(k,))
assert loops(x.beat_loops) == [(2, 4), (2, 0), (2, 2)] and loops(x.lane_loops) == [(2, 1)]
assert loops(w.beat_loops) == [(2, 0), (2, 1), (2, 4)] and loops(w.lane_loops) == [(2, 2)]
assert loops(y.beat_loops) == [(4, 1)] and loops(y.lane_loops) == []
assert s.beat_count == 8

# §5.1: closing, and the alignment characterization


def aligned(form: Traversal, beats: int) -> bool:
    if form.beats % beats:
        return False
    inner = 1
    for extent in reversed([loop.extent for loop in form.beat_loops]):
        if beats == inner:
            return True
        if beats % inner == 0 and beats // inner <= extent and extent % (beats // inner) == 0:
            return True
        inner *= extent
    return beats == inner


assert s.closing((k,)) == LevelEnd(2)
pairs = [(form, beats) for form in forms for beats in range(1, form.beats + 1)]
assert len(pairs) == 8346
assert all(LevelEnd(beats).aligned(form) == aligned(form, beats) for form, beats in pairs)
try:
    Schedule({m: 2, n: 2, k: 4}, factors={k: 2}, order=(m, k, n)).closing((k,))
    raise AssertionError("closing must refuse a reduction that is not innermost")
except ValueError:
    pass

# §6: classify
columns = Traversal.over((2, 4), ((1, 4, 1), (0, 2, 1)), ())
reorder = classify(vector_major((2, 4), 1), columns).reorder
assert reorder is not None
assert (reorder.frame_beats, reorder.dims, reorder.coefs) == (8, (4, 2), (1, 4))
replay = classify(unreplayed(x), x).reorder
assert replay is not None and (replay.frame_beats, replay.dims, replay.coefs) == (2, (2, 2), (0, 1))
tile_nk = Traversal.over((2, 2), (), ((1, 2, 1), (0, 2, 1)))
tile_kn = Traversal.over((2, 2), (), ((0, 2, 1), (1, 2, 1)))
permuted = classify(tile_nk, tile_kn)
assert permuted.adaptation.value == "lane_permutation" and permuted.lane_permutation == (0, 2, 1, 3)
rows = vector_major((4, 4), 2)
cols = Traversal.over((4, 4), ((1, 4, 1), (0, 2, 2)), ((0, 2, 1),))
assert classify(rows, cols).adaptation.value == "lane_regroup"
assert classify(vector_major((2, 4), 2), vector_major((2, 4), 4)).adaptation.value == "width_conversion"
oh, kh = Index("oh"), Index("kh")
window = Schedule({oh: 4, kh: 3}).present((6,), (oh + kh,))
assert classify(vector_major((6,), 1), window).adaptation.value == "incompatible"

# §7: plans
need = BeatSequence(x, markers=(s.closing((k,)),))
assert plan(BeatSequence(vector_major((2, 4), 2)), need).describe() == "reorder -> markers"
for lanes in (1, 4):
    found = plan(BeatSequence(vector_major((2, 4), lanes)), need).describe()
    assert found == "width_conversion -> reorder -> markers", found
cyclic = plan(
    BeatSequence(vector_major((2, 4), 4), Repetition.CYCLIC),
    BeatSequence(vector_major((2, 4), 2).repeated(2)),
)
assert cyclic.describe() == "width_conversion"
regroup = plan(
    BeatSequence(Traversal.over((4, 6), ((0, 4, 1), (1, 2, 3)), ((1, 3, 1),))),
    BeatSequence(Traversal.over((4, 6), ((1, 3, 2), (0, 2, 2)), ((0, 2, 1), (1, 2, 1)))),
)
assert regroup.describe() == "width_conversion -> reorder -> width_conversion"
try:
    plan(BeatSequence(v), BeatSequence(v, Repetition.CYCLIC))
    raise AssertionError("ONCE must not feed CYCLIC")
except Unrealizable:
    pass

print("THEORY.md: every example and claim holds")
