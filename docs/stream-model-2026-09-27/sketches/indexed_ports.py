# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Sketch for DESIGN.md: port traversals derived from one indexed nest.

Nothing here is production code. It imports only ``finn.kernels.physical.forms``
(read-only) and checks, by construction and by enumeration, that:

1. the four MVAU stream forms that ``mvau.py`` writes by hand (activation,
   replayed, weights, results) fall out of ONE contraction, ONE folding and ONE
   loop order, including the replay loop, the weight repetition and the frame
   marker period;
2. the per-channel (VVAU) relation ``field = s*PE + p`` falls out of the same
   derivation with a different contraction, and broadcast is derived;
3. thresholding and a PE mismatch between two kernels classify as the adapter
   C6 needs (width conversion), and replay classifies as a reorder;
4. dotp's three B1 checks reduce to one structural admission over the nest;
5. the reduction-order dial is a Decision over the nest, and two legal orders
   are a REORDER apart (so a producer must follow the consumer's choice).

Run: PYTHONPATH=src python docs/stream-model-2026-09-27/sketches/indexed_ports.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

from finn.kernels.physical.forms import (
    Adaptation,
    Every,
    Loop,
    Traversal,
    classify,
    tile,
    vector_major,
)

# --------------------------------------------------------------------------------------
# The objects the design proposes (value types only; no Space engine involved)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One loop of a nest: a temporal (beat) level or a spatial (lane) level."""

    name: str
    extent: int


@dataclass(frozen=True)
class Nest:
    """A kernel's iteration space: beat levels outer to inner, plus lane levels.

    Lane levels are a set: their order inside a beat is a property of each port
    (the hardware's field order), not of the nest.
    """

    beats: tuple[Level, ...]
    lanes: tuple[Level, ...]

    def level(self, name: str) -> Level:
        for level in (*self.beats, *self.lanes):
            if level.name == name:
                return level
        raise KeyError(name)


Index = Mapping[str, int]
"""An affine tensor index: level name -> coefficient (``nf*PE + p`` is {nf: PE, p: 1})."""


@dataclass(frozen=True)
class Operand:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Access:
    """``operand[index_0, index_1, ...]``: which position a nest point touches."""

    operand: Operand
    index: tuple[Index, ...]

    def uses(self, level: str) -> bool:
        return any(axis.get(level, 0) for axis in self.index)

    def stride(self, level: str) -> int:
        shape = self.operand.shape
        axis_strides = [prod(shape[j + 1 :]) for j in range(len(shape))]
        return sum(axis.get(level, 0) * axis_strides[j] for j, axis in enumerate(self.index))


class Refused(ValueError):
    pass


def present(
    nest: Nest, access: Access, *, fields: Sequence[str], reduced: Sequence[str] = ()
) -> Traversal:
    """The traversal a port presents: the nest projected through its access.

    ``fields`` orders the lane levels this port carries (field 0 first).
    ``reduced`` names levels an output port is presented *after* (it emits once
    per completed reduction), so they are not beat loops of the port. A lane
    level the port does not carry must not move its position (broadcast); a
    reduced level must not either (a true reduction).
    """
    for level in nest.lanes:
        if level.name not in fields and access.stride(level.name) != 0:
            raise Refused(f"{access.operand.name}: lane level {level.name} moves the position")
    for name in reduced:
        if access.stride(name) != 0:
            raise Refused(f"{access.operand.name}: {name} is not reduced away")
    beats = [
        Loop(level.extent, access.stride(level.name))
        for level in nest.beats
        if level.name not in reduced
    ]
    lanes = [Loop(nest.level(name).extent, access.stride(name)) for name in fields]
    return Traversal(access.operand.shape, beats, lanes)


def frame_period(nest: Nest, reduced: Sequence[str]) -> Every:
    """The marker closing each reduction: the reduced levels must be the beat suffix."""
    names = [level.name for level in nest.beats]
    suffix = names[len(names) - len(reduced) :]
    if sorted(suffix) != sorted(reduced):
        raise Refused(f"reduction levels {list(reduced)} are not innermost in {names}")
    return Every(prod(nest.level(name).extent for name in reduced))


def once(form: Traversal) -> Traversal:
    """The same traversal with its replay (stride-0) beat loops removed."""
    return Traversal(form.shape, [loop for loop in form.beat_loops if loop.stride], form.lane_loops)


# --------------------------------------------------------------------------------------
# 1. Dense MatMul: Y[r, n] = sum_k X[r, k] W[n, k]
# --------------------------------------------------------------------------------------


def dense(R: int, K: int, N: int, PE: int, SIMD: int, order: str = "r nf kf"):
    levels = {
        "r": Level("r", R),
        "nf": Level("nf", N // PE),
        "kf": Level("kf", K // SIMD),
    }
    nest = Nest(tuple(levels[name] for name in order.split()), (Level("p", PE), Level("s", SIMD)))
    X = Access(Operand("X", (R, K)), ({"r": 1}, {"kf": SIMD, "s": 1}))
    W = Access(Operand("W", (N, K)), ({"nf": PE, "p": 1}, {"kf": SIMD, "s": 1}))
    Y = Access(Operand("Y", (R, N)), ({"r": 1}, {"nf": PE, "p": 1}))
    return nest, X, W, Y


def check_dense(R: int, K: int, N: int, PE: int, SIMD: int) -> None:
    nest, X, W, Y = dense(R, K, N, PE, SIMD)
    reduced = ("kf",)  # the beat levels Y's access does not use
    assert [lv.name for lv in nest.beats if not Y.uses(lv.name)] == list(reduced)

    activation = present(nest, X, fields=("s",))
    weights = present(nest, W, fields=("p", "s"))
    results = present(nest, Y, fields=("p",), reduced=reduced)
    frame = frame_period(nest, reduced)

    # What mvau.py:216-246 writes by hand.
    SF, NF = K // SIMD, N // PE
    hand_activation = vector_major((R, K), SIMD)
    hand_replayed = hand_activation.replayed(NF, inner_beats=SF)
    hand_weights = tile(N, K, PE, SIMD).repeated(R)
    hand_results = vector_major((R, N), PE)
    assert activation == hand_replayed, (activation, hand_replayed)
    assert weights == hand_weights, (weights, hand_weights)
    assert results == hand_results
    assert frame == Every(SF)

    # Broadcast is derived: X does not use the output lane level p.
    assert not X.uses("p")

    # The boundary presents each position once; the replay is a derived adapter.
    boundary = once(activation)
    assert boundary == hand_activation
    verdict = classify(boundary, activation)
    if NF > 1:
        assert verdict.adaptation is Adaptation.REORDER, verdict
        # replay_buffer(LEN=SF, REP=NF) is the special case "insert one stride-0
        # loop above the innermost SF beats"; input_gen realizes it generally.
        # Per frame of SF input beats, emit NF * SF beats; the replay dim has coef 0.
        reorder = verdict.reorder
        assert reorder is not None and reorder.frame_beats == SF, reorder
        assert prod(reorder.dims) == NF * SF and 0 in reorder.coefs, reorder
    else:
        assert verdict.adaptation is Adaptation.IDENTITY

    # External weights: the boundary presents the consumer's order, repetition included
    # (FINN's MVAU in1_V); cyclic delivery produces once(weights) cyclically instead.
    assert once(weights) == tile(N, K, PE, SIMD)


# --------------------------------------------------------------------------------------
# 2. Per-channel (depthwise) MatMul: Y[r, c] = sum_k X[r, c, k] W[c, k]
# --------------------------------------------------------------------------------------


def per_channel(R: int, C: int, K: int, PE: int, SIMD: int, layout: str = "rck"):
    nest = Nest(
        (Level("r", R), Level("cf", C // PE), Level("kf", K // SIMD)),
        (Level("p", PE), Level("s", SIMD)),
    )
    c, k = {"cf": PE, "p": 1}, {"kf": SIMD, "s": 1}
    if layout == "rck":
        X = Access(Operand("X", (R, C, K)), ({"r": 1}, c, k))
    else:  # FINN's NHWC-derived im2col layout: window outer, channel innermost
        X = Access(Operand("X", (R, K, C)), ({"r": 1}, k, c))
    W = Access(Operand("W", (C, K)), (c, k))
    Y = Access(Operand("Y", (R, C)), ({"r": 1}, c))
    return nest, X, W, Y


def check_per_channel(R: int, C: int, K: int, PE: int, SIMD: int) -> None:
    for layout in ("rck", "rkc"):
        nest, X, W, Y = per_channel(R, C, K, PE, SIMD, layout)
        assert X.uses("p")  # not broadcast: each lane is its own channel
        # FinnLib's per-channel activation order: field s*PE + p (p fastest).
        activation = present(nest, X, fields=("s", "p"))
        weights = present(nest, W, fields=("p", "s"))
        results = present(nest, Y, fields=("p",), reduced=("kf",))
        for beat in range(activation.beats):
            r, rest = divmod(beat, (C // PE) * (K // SIMD))
            cf, kf = divmod(rest, K // SIMD)
            for s, p in product(range(SIMD), range(PE)):
                ch, tap = cf * PE + p, kf * SIMD + s
                want = (r, ch, tap) if layout == "rck" else (r, tap, ch)
                assert activation.position(beat, s * PE + p) == want
                assert weights.position(beat, p * SIMD + s) == (ch, tap)
        assert results == vector_major((R, C), PE)
        # No reuse of X across output folds: no stride-0 loop, so no replay.
        assert all(loop.stride for loop in activation.beat_loops)
        # Weights are reused across r: the stride-0 loop is the cyclic repetition.
        assert weights.beat_loops[0].stride == 0 or R == 1


# --------------------------------------------------------------------------------------
# 3. Thresholding and the PE mismatch C6 adapts
# --------------------------------------------------------------------------------------


def thresholding_input(R: int, C: int, PE: int) -> Traversal:
    nest = Nest((Level("r", R), Level("cf", C // PE)), (Level("p", PE),))
    T = Access(Operand("T", (R, C)), ({"r": 1}, {"cf": PE, "p": 1}))
    return present(nest, T, fields=("p",))


def check_adapters() -> None:
    R, K, N = 3, 8, 8
    assert thresholding_input(R, N, 4) == vector_major((R, N), 4)
    nest, _, _, Y = dense(R, K, N, 4, 2)
    produced = present(nest, Y, fields=("p",), reduced=("kf",))
    same = classify(produced, thresholding_input(R, N, 4))
    narrower = classify(produced, thresholding_input(R, N, 2))
    assert same.adaptation is Adaptation.IDENTITY
    assert narrower.adaptation is Adaptation.WIDTH_CONVERSION
    # A consumer walking the channels with another lane axis: inner shuffle.
    transposed = Traversal.over((R, N), ((1, N, 1), (0, 1, R)), ((0, R, 1),))
    assert classify(produced, transposed).adaptation in (
        Adaptation.LANE_REGROUP,
        Adaptation.INCOMPATIBLE,
    )


# --------------------------------------------------------------------------------------
# 4. dotp's admission, as one structural rule over the nest
# --------------------------------------------------------------------------------------


def dotp_admits(nest: Nest, X: Access, W: Access, Y: Access) -> str | None:
    """None when dotp_axi can realize this nest; otherwise the reason.

    Roles are read off the accesses: output lanes P are the lane levels Y uses,
    reduction lanes S are the others; the frame is the beat levels Y does not
    use and must be the innermost suffix. Weights must use every lane level.
    """
    P = [lv.name for lv in nest.lanes if Y.uses(lv.name)]
    S = [lv.name for lv in nest.lanes if not Y.uses(lv.name)]
    if len(P) != 1 or len(S) != 1:
        return "dotp has one output lane level (PE) and one reduction lane level (SIMD)"
    if not all(W.uses(name) for name in (*P, *S)):
        return "weights must differ in every lane"
    if not X.uses(S[0]):
        return "activation lanes must be the reduction lanes"
    reduced = [lv.name for lv in nest.beats if not Y.uses(lv.name)]
    if not reduced:
        return "a frame reduces at least one beat level"
    try:
        frame_period(nest, reduced)
    except Refused as error:
        return str(error)
    return None


def check_admission() -> None:
    assert dotp_admits(*dense(2, 8, 6, 3, 2)) is None
    assert dotp_admits(*per_channel(2, 6, 4, 3, 2)) is None
    # B1's "a frame crossing activation rows": r inside the reduction.
    assert dotp_admits(*dense(2, 8, 6, 3, 2, order="nf kf r")) is not None
    # B1's "results swap frames" has no analogue: results are derived, not supplied.


# --------------------------------------------------------------------------------------
# 5. The reduction-order dial: conv as matmul, K = (kh, kw, c)
# --------------------------------------------------------------------------------------


def check_reduction_order() -> None:
    R, KH, KW, C, N, PE, SIMD = 2, 3, 3, 4, 4, 2, 2
    nest_names = {
        "chw": ("r", "nf", "kh", "kw", "cf"),
        "hcw": ("r", "nf", "kh", "cf", "kw"),
    }
    forms = {}
    for key, order in nest_names.items():
        extents = {"r": R, "nf": N // PE, "kh": KH, "kw": KW, "cf": C // SIMD}
        nest = Nest(tuple(Level(n, extents[n]) for n in order), (Level("p", PE), Level("s", SIMD)))
        X = Access(
            Operand("X", (R, KH, KW, C)), ({"r": 1}, {"kh": 1}, {"kw": 1}, {"cf": SIMD, "s": 1})
        )
        W = Access(
            Operand("W", (N, KH, KW, C)),
            ({"nf": PE, "p": 1}, {"kh": 1}, {"kw": 1}, {"cf": SIMD, "s": 1}),
        )
        Y = Access(Operand("Y", (R, N)), ({"r": 1}, {"nf": PE, "p": 1}))
        assert dotp_admits(nest, X, W, Y) is None  # both orders are legal (inside lambda)
        forms[key] = present(nest, X, fields=("s",))
    assert forms["chw"] != forms["hcw"]
    assert classify(forms["chw"], forms["hcw"]).adaptation is Adaptation.REORDER


if __name__ == "__main__":
    for shape in [(1, 4, 2, 2, 2), (3, 8, 6, 3, 2), (2, 12, 8, 4, 3), (4, 6, 6, 1, 6), (2, 4, 4, 4, 1)]:
        check_dense(*shape)
    for shape in [(1, 4, 4, 2, 2), (3, 6, 4, 3, 2), (2, 8, 9, 4, 3)]:
        check_per_channel(*shape)
    check_adapters()
    check_admission()
    check_reduction_order()
    print("indexed_ports: all checks passed")
