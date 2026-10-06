# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every realized plan moves the data right, checked against the modules' own semantics.

``plan`` names the steps between two beat sequences and ``realize`` maps them
onto FinnLib modules. The reference here is independent of both: it models
``input_gen`` and ``vpc`` from their RTL headers alone (per frame of
``FM_SIZE`` words, the word at ``f * FM_SIZE + sum(COEFS[k] * i_k)`` over
``DIMS``, with ``olst[d]`` asserted when loops ``d..`` all complete; a ``vpc``
regroups the element sequence), runs the realized chain on the source's
positions, and compares the result with the sink's positions and required
markers, up to the one fixed lane permutation the connection wires. Random
pairs cover mismatched lane counts, lane axes, beat orders, replays and
markers, among them NF other than SF and swapped lane orders.
"""

from __future__ import annotations

import random
from itertools import product
from math import prod

import pytest

from finn.dataflow.plan import Unrealizable, plan
from finn.dataflow.traversal import (
    BeatSequence,
    LevelEnd,
    Loop,
    Position,
    Traversal,
    axis_strides,
    tile,
    vector_major,
)
from finn.kernels.adapters import INPUT_CHAINS, OUTPUT_CHAINS, Convert, Generate, realize

Beats = list[tuple[Position, ...]]


def generate(beats: Beats, module: Generate) -> tuple[Beats, list[tuple[bool, ...]]]:
    """``input_gen``: words by frame and nest; ``olst[d]`` when loops d.. complete."""
    if len(beats) % module.frame:
        raise AssertionError("the input is no whole number of frames")
    out: Beats = []
    marks: list[tuple[bool, ...]] = []
    for start in range(0, len(beats), module.frame):
        for index in product(*(range(extent) for extent in module.dims)):
            offset = sum(i * c for i, c in zip(index, module.coefs))
            assert offset < module.frame
            out.append(beats[start + offset])
            marks.append(
                tuple(
                    all(index[k] == module.dims[k] - 1 for k in range(depth, len(module.dims)))
                    for depth in range(len(module.dims))
                )
            )
    return out, marks


def convert(beats: Beats, module: Convert) -> Beats:
    """``vpc``: the same element sequence, ``lanes_out`` elements a beat."""
    elements = [position for beat in beats for position in beat]
    assert all(len(beat) == module.lanes_in for beat in beats)
    assert len(elements) % module.lanes_out == 0
    return [
        tuple(elements[start : start + module.lanes_out])
        for start in range(0, len(elements), module.lanes_out)
    ]


def check(source: BeatSequence, sink: BeatSequence) -> tuple[str, ...]:
    """Run the realized chain on the source's positions; return its module kinds, both
    sides of the transport."""
    found = plan(source, sink)
    stages = realize(found)
    beats: Beats = list(found.hops[0].source.form.positions()) if found else []
    marks: list[tuple[bool, ...]] = []
    for stage in stages:
        if isinstance(stage.module, Generate):
            beats, marks = generate(beats, stage.module)
            levels = stage.module.levels
        else:
            beats, marks, levels = convert(beats, stage.module), [], ()
    if found:
        wanted = list(sink.form.positions())
        # A lane order that differs is wired by the connection: one fixed
        # permutation of the lanes for every beat.
        wiring = [beats[0].index(position) for position in wanted[0]]
        assert [tuple(beat[lane] for lane in wiring) for beat in beats] == wanted
        for rule in sink.markers:
            if rule.constant:
                continue  # tied high by the connection: asserted on every beat
            depth = levels.index(rule)
            asserted = [mark[depth] for mark in marks]
            assert asserted == [rule.asserted(beat) for beat in range(len(beats))]
    # Each side of the transport is one candidate's chain, and together they are the plan's.
    sides = realize(found.output), realize(found.input)
    assert (*sides[0], *sides[1]) == stages
    for side, chains in zip(sides, (OUTPUT_CHAINS, INPUT_CHAINS)):
        assert not side or tuple(stage.kind for stage in side) in chains, side
    return tuple(stage.kind for stage in stages)


def random_form(rng: random.Random, shape: tuple[int, ...], replay: bool) -> Traversal:
    """Every position once (a replay loop added if asked), lanes from random axes."""
    strides = axis_strides(shape)
    beats: list[Loop] = []
    lanes: list[Loop] = []
    for axis, extent in enumerate(shape):
        factor = rng.choice([f for f in range(1, extent + 1) if extent % f == 0])
        if rng.random() < 0.5 or factor == 1:
            beats.append(Loop(extent, strides[axis]))
        else:
            beats.append(Loop(extent // factor, factor * strides[axis]))
            lanes.append(Loop(factor, strides[axis]))
    rng.shuffle(beats)
    rng.shuffle(lanes)
    if replay:
        beats.insert(rng.randint(0, len(beats)), Loop(rng.choice((2, 3)), 0))
    return Traversal(shape, beats, lanes)


def framed(rng: random.Random, form: Traversal) -> tuple[LevelEnd, ...]:
    """Maybe a marker closing a random innermost suffix of the beat loops."""
    if not form.beat_loops or rng.random() < 0.5:
        return ()
    depth = rng.randrange(len(form.beat_loops))
    return (LevelEnd(prod(loop.extent for loop in form.beat_loops[depth:])),)


def test_known_plans_realize_as_their_candidates():
    rows = vector_major((3, 8), 2)
    replayed = BeatSequence(rows.replayed(2, inner_beats=4), markers=(LevelEnd(4),))
    assert check(BeatSequence(rows), replayed) == ("input_gen",)
    assert check(BeatSequence(vector_major((3, 8), 4)), replayed) == ("vpc", "input_gen")
    assert check(BeatSequence(vector_major((4, 4), 1)), BeatSequence(tile(4, 4, 2, 2))) == (
        "input_gen",
        "vpc",
    )
    rows_as_lanes = Traversal.over((3, 4), ((1, 4, 1),), ((0, 3, 1),))
    assert check(BeatSequence(rows_as_lanes), BeatSequence(vector_major((3, 4), 2))) == (
        "vpc",
        "input_gen",
        "vpc",
    )
    # A frame of one beat (SIMD = K) closes on every beat: its marker is tied high, no
    # step; with a replay, the reorder alone.
    assert check(BeatSequence(rows), BeatSequence(rows, markers=(LevelEnd(1),))) == ()
    one = vector_major((3, 2), 2)
    assert check(
        BeatSequence(one), BeatSequence(one.replayed(3, inner_beats=1), markers=(LevelEnd(1),))
    ) == ("input_gen",)
    # A level wider than one reorder frame groups frames.
    columns = Traversal.over((2, 4), ((1, 4, 1), (0, 2, 1)), ())
    assert check(BeatSequence(columns.replayed(1, inner_beats=1)), BeatSequence(columns)) == ()


@pytest.mark.parametrize("seed", range(4))
def test_random_plans_move_every_element_to_its_place(seed):
    rng = random.Random(seed)
    realized = refused = 0
    for _ in range(150):
        shape = tuple(rng.choice((1, 2, 3, 4, 6)) for _ in range(rng.choice((1, 2, 3))))
        if prod(shape) < 2:
            continue
        source = random_form(rng, shape, replay=False)
        sink_form = random_form(rng, shape, replay=rng.random() < 0.4)
        sink = BeatSequence(sink_form, markers=framed(rng, sink_form))
        try:
            check(BeatSequence(source), sink)
        except Unrealizable:
            refused += 1
            continue
        realized += 1
    assert realized >= 100, (realized, refused)
