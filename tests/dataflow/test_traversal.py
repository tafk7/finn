# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Traversals: the order a stream presents an operand, and what packs and unpacks it.

The stress cases come from baseline FINN: the tiled MVU's two internal
``input_gen`` adapters and the Shuffle op's inner/outer decomposition. Their
parameters are derived here from traversals alone and compared with the
hard-coded values in ``finn-rtllib`` and baseline FINN's ``transpose_decomposition``
(``shuffle_perfect_loopnest_coeffs``, copied here).
"""

import operator
import random
from math import prod

import numpy as np
import pytest
from support.tiled_mvu import MH, MW, NF, PE_T, SF, SIMD_T, R, T

from finn.dataflow import traversal
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Loop,
    Reorder,
    Traversal,
    axis_strides,
    classify,
    offsets,
    pack,
    tile,
    unpack,
    vector_major,
)


def test_tile_is_the_mvau_weight_order_and_packs_the_known_image():
    weights = tile(4, 4, 2, 2)
    assert (weights.lanes, weights.beats, weights.shape) == (4, 4, (4, 4))
    first, second = list(weights.positions())[:2]
    assert first == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert second == ((0, 2), (0, 3), (1, 2), (1, 3))
    matrix = ((-4, -3, -2, -1), (0, 1, 2, 3), (3, 2, 1, 0), (-1, -2, -3, -4))
    # Hand-packed INT3 lanes: p0/s0, p0/s1, p1/s0, p1/s1, low first.
    assert pack(weights, sum(matrix, ()), 3) == (0x22C, 0x6BE, 0xDD3, 0x941)


def test_repetition_and_replay_are_stride_zero_loops():
    vector = vector_major((4,), 2)
    assert list(vector.repeated(2).positions()) == [((0,), (1,)), ((2,), (3,))] * 2
    rows = vector_major((2, 4), 2)
    replayed = rows.replayed(3, inner_beats=2)
    assert replayed.beats == 12
    assert [beat[0] for beat in replayed.positions()][:6] == [(0, 0), (0, 2)] * 3
    with pytest.raises(ValueError, match="divide"):
        vector_major((5,), 2)
    with pytest.raises(ValueError, match="has 4 integers, not 3"):
        pack(vector, (1, 2, 3), 4)
    assert LevelEnd(3).asserted(2) and not LevelEnd(3).asserted(3)


# -- pack and unpack ------------------------------------------------------------


def _packed_by_position(form, integers, bits):
    """``pack`` by its definition: each beat's positions, raveled row-major, low bits first."""
    strides, mask = axis_strides(form.shape), (1 << bits) - 1
    return tuple(
        sum(
            (integers[sum(map(operator.mul, position, strides))] & mask) << (lane * bits)
            for lane, position in enumerate(beat)
        )
        for beat in form.positions()
    )


def _random_form(rng):
    """A traversal of a random operand: vector-major, tiled, or loops of random extents
    along random axes, replay loops (stride 0) among them."""
    shape = tuple(rng.randint(1, 6) for _ in range(rng.randint(1, 3)))
    kind = rng.choice(("vector", "tile", "loops", "loops"))
    if kind == "vector":
        return vector_major(
            shape, rng.choice([d for d in range(1, shape[-1] + 1) if not shape[-1] % d])
        )
    if kind == "tile" and len(shape) == 2:
        rows, cols = shape
        pe = rng.choice([d for d in range(1, rows + 1) if not rows % d])
        return tile(rows, cols, pe, rng.choice([d for d in range(1, cols + 1) if not cols % d]))
    strides = (0, *axis_strides(shape))
    while True:
        loops = [Loop(rng.randint(1, 4), rng.choice(strides)) for _ in range(rng.randint(1, 5))]
        split = rng.randint(0, len(loops))
        try:
            return Traversal(shape, loops[:split], loops[split:])
        except ValueError:
            continue


def test_pack_is_its_definition_at_every_width_signedness_and_lane_count():
    """The words are each beat's lanes at their positions, the low ``bits`` of each
    integer's two's complement, lane zero lowest: for lanes 1..8 bits wide and wider
    than an int64, integers narrower or wider than their lanes, signed or not, and as a
    tuple or an int64 array (packed with numpy) or beyond 64 bits (one at a time)."""
    rng = random.Random(7)
    for _ in range(600):
        form = _random_form(rng)
        bits = rng.choice((*range(1, 18), 31, 32, 33, 63, 64, 65, 100, 128))
        width = rng.choice((1, 2, 3, 4, 7, 8, 9, 16, 32, 62, 63, 64, 65, 80))
        signed = rng.random() < 0.5
        low, high = (
            (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
        )
        integers = tuple(rng.randint(low, high) for _ in range(prod(form.shape)))
        expected = _packed_by_position(form, integers, bits)
        assert len(expected) == form.beats
        assert pack(form, integers, bits) == expected, (form, bits, width, signed)
        if -(2**63) <= low and high < 2**63:
            array = np.array(integers, dtype=np.int64)
            assert pack(form, array, bits) == expected, (form, bits, width, signed)


@pytest.mark.parametrize("bits", (1, 63, 64, 65, 128))
def test_pack_uses_numpy_exactly_up_to_int64_and_python_ints_beyond(bits, monkeypatch):
    """At the int64 bounds pack stays with numpy (an exact-path call fails the test);
    one integer beyond them takes the exact path. Both give the words by definition."""
    form = vector_major((2, 4), 2)

    def refused(*args):
        raise AssertionError("integers within int64 took the exact path")

    edges = (2**63 - 1, -(2**63), -1, 0, 1, 2**62, -(2**62) - 1, 5)
    with monkeypatch.context() as patched:
        patched.setattr(traversal, "_pack_exact", refused)
        assert pack(form, edges, bits) == _packed_by_position(form, edges, bits)
    taken = []
    exact = traversal._pack_exact
    monkeypatch.setattr(traversal, "_pack_exact", lambda *args: taken.append(1) or exact(*args))
    for beyond in (2**63, -(2**63) - 1, 2**64 + 3, -(2**100)):
        found = (*edges[:-1], beyond)
        assert pack(form, found, bits) == _packed_by_position(form, found, bits)
    assert len(taken) == 4


def test_unpack_inverts_pack_at_every_width_signedness_and_lane_count():
    """The operand read back from its words is the operand, for every traversal that
    presents each element (replays among them), lanes of 1 to 64 bits, signed or not."""
    rng = random.Random(11)
    checked = 0
    while checked < 400:
        form = _random_form(rng)
        if set(offsets(form).tolist()) != set(range(prod(form.shape))):
            continue  # an element it never presents: no operand to read back
        signed = rng.random() < 0.5
        bits = rng.choice((1, 2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 62, 63, *((64,) if signed else ())))
        low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        operand = np.array([rng.randint(low, high) for _ in range(prod(form.shape))], np.int64)
        words = pack(form, operand, bits)
        found = unpack(form, words, bits, signed=signed)
        assert found.shape == form.shape and np.array_equal(found.ravel(), operand), (form, bits)
        assert pack(form, found.ravel(), bits) == words
        checked += 1


def test_unpack_refuses_words_that_are_no_operand():
    vector = vector_major((4,), 2)
    assert unpack(vector, (0x1F, 0x23), 4, signed=True).tolist() == [-1, 1, 3, 2]
    assert unpack(vector, (0x1F, 0x23), 4, signed=False).tolist() == [15, 1, 3, 2]
    with pytest.raises(ValueError, match="2 beats, not 3 words"):
        unpack(vector, (0, 0, 0), 4, signed=False)
    with pytest.raises(ValueError, match="wider than its 8 bits"):
        unpack(vector, (0x100, 0), 4, signed=False)
    with pytest.raises(ValueError, match="exceed an int64"):
        unpack(vector, (0, 0), 64, signed=False)
    # A replayed element read twice with two values.
    with pytest.raises(
        ValueError, match=r"element \(1,\) is presented twice .* beat 0 lane 1, .* beat 2 lane 1"
    ):
        unpack(vector.repeated(2), (0x10, 0x00, 0x20, 0x00), 4, signed=False)
    assert unpack(vector.repeated(2), (0x10, 0x32, 0x10, 0x32), 4, signed=False).tolist() == [
        0,
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError, match=r"presents no value of element \(2,\)"):
        unpack(Traversal((4,), (Loop(2, 0),), (Loop(2, 1),)), (0, 0), 4, signed=False)


# -- canonical equality ---------------------------------------------------------


def _random_traversal(rng):
    while True:
        loops = [
            Loop(rng.choice((1, 2, 3)), rng.choice((0, 1, 2, 3, 6, 12)))
            for _ in range(rng.randint(1, 4))
        ]
        split = rng.randint(0, len(loops))
        try:
            return Traversal((4, 6), loops[:split], loops[split:])
        except ValueError:
            continue


def test_canonical_equality_is_equality_of_presented_sequences():
    rng = random.Random(1)
    for _ in range(4000):
        first, second = _random_traversal(rng), _random_traversal(rng)
        same = list(first.positions()) == list(second.positions())
        assert same is (first == second), (first, second)


# -- stress cases: tiled MVU and Shuffle ----------------------------------------


def test_tiled_mvu_input_adapter_is_derived_from_the_two_orders():
    boundary = vector_major((R, MW), SIMD_T)
    core = Traversal.over(
        (R, MW), ((0, R // T, T), (None, NF, 0), (1, SF, SIMD_T), (0, T, 1)), ((1, SIMD_T, 1),)
    )
    verdict = classify(boundary, core)
    # mvu_tiled_axi.sv: input_gen FM_SIZE=SF*TH, DIMS='{NF,SF,TH}, COEFS='{0,1,SF}.
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder == Reorder(SF * T, (NF, SF, T), (0, 1, SF))


def test_tiled_mvu_output_reorder_is_derived_from_the_two_orders():
    core = Traversal.over((R, MH), ((0, R // T, T), (1, NF, PE_T), (0, T, 1)), ((1, PE_T, 1),))
    verdict = classify(core, vector_major((R, MH), PE_T))
    # mvu_tiled_axi.sv genReorder: FM_SIZE=NF*TH, DIMS='{TH,NF}, COEFS='{1,TH}.
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder == Reorder(NF * T, (T, NF), (1, T))


def shuffle_perfect_loopnest_coeffs(shape, perm):
    """Baseline FINN's coefficients of the perfect loop nest of a permutation (its
    ``transpose_decomposition``, the HWCustomOp flow's, deleted): the input's strides
    in the output's order."""
    adjusted_shape = list(shape) + [1]
    input_coeffs = [np.prod(adjusted_shape[i + 1 :]) for i in range(len(shape))]
    return tuple(input_coeffs[i] for i in perm)


def test_outer_shuffle_coefficients_match_finn():
    shape, perm, simd = (2, 3, 4), (1, 0, 2), 2
    moved = Traversal.over(shape, ((1, 3, 1), (0, 2, 1), (2, 2, simd)), ((2, simd, 1),))
    verdict = classify(vector_major(shape, simd), moved)
    finn = [1 if x == 1 else x // simd for x in shuffle_perfect_loopnest_coeffs(shape, perm)]
    assert verdict.adaptation is Adaptation.REORDER
    assert verdict.reorder.coefs == tuple(finn)
    assert verdict.reorder.dims == (3, 2, 2)


def test_inner_shuffle_moves_the_lane_axis():
    transposed = Traversal.over((4, 6), ((1, 6, 1), (0, 2, 2)), ((0, 2, 1),))
    verdict = classify(vector_major((4, 6), 2), transposed)
    assert verdict.adaptation is Adaptation.LANE_REGROUP


def test_different_operands_and_positions_never_presented_are_incompatible():
    assert classify(vector_major((4,), 2), vector_major((8,), 2)).adaptation is (
        Adaptation.INCOMPATIBLE
    )
    half = Traversal.over((4,), ((0, 1, 2),), ((0, 2, 1),))
    assert classify(half, vector_major((4,), 2)).adaptation is Adaptation.INCOMPATIBLE
    # The other way, the sink drops the frame's second beat: a reorder.
    assert classify(vector_major((4,), 2), half).adaptation is Adaptation.REORDER
