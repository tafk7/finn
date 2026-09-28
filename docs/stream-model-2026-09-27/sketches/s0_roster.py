# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""S0 expressiveness spike (PLAN.md): can one nest model derive the roster's forms?

Sketch only; it reuses ``indexed_ports.py``'s ``Nest``/``Access``/``present`` and
checks each roster member against an independent reference: a hand-written
form already pinned by tests, or a position enumeration from the operation's
own index formula.

1. dense and per-channel MatMul (indexed_ports), and the dense realization of
   per-channel as a reshaped view;
2. tiled MVU (TH > 1): the core's activation, weight-chunk and result forms and
   both ``input_gen`` coefficient sets FINN hard-codes;
3. SWG / conv-as-matmul with stride and dilation;
4. thresholding with PE <= C and PE > C;
5. eltwise with a broadcast operand;
6. a transpose;
7. the boundary rule (G0.2): interior replay is adapted by the receiver,
   whole-pass repetition is part of the interface;
8. markers as loop levels (G0.4): a level is named by the beats it spans and
   must fall on a loop boundary of the presentation;
9. compound plans: width conversion and reorder decompose in either order.

Run: PYTHONPATH=src python docs/stream-model-2026-09-27/sketches/s0_roster.py
"""

from __future__ import annotations

import sys
from itertools import product
from math import gcd, prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from indexed_ports import (  # noqa: E402
    Access,
    Level,
    Nest,
    Operand,
    dotp_admits,
    frame_period,
    once,
    per_channel,
    present,
)

from finn.kernels.physical.forms import (  # noqa: E402
    Adaptation,
    Loop,
    Reorder,
    Traversal,
    classify,
    regrouped,
    split_beats,
    tile,
    vector_major,
)

GAPS: list[str] = []


def enumerate_positions(beats, lanes, index):
    """Positions from the operation's own formula: ``index(beat_values, lane_values)``."""
    out = []
    for b in product(*(range(extent) for _, extent in beats)):
        env_b = dict(zip((name for name, _ in beats), b))
        row = []
        for s in product(*(range(extent) for _, extent in lanes)):
            env = {**env_b, **dict(zip((name for name, _ in lanes), s))}
            row.append(index(env))
        out.append(tuple(row))
    return out


# -- 1. dense realization of per-channel as a reshaped view ---------------------------------


def reshaped(form: Traversal, shape) -> Traversal:
    """The same flat-offset sequence over another row-major shape of equal size."""
    assert prod(shape) == prod(form.shape)
    return Traversal(shape, form.beat_loops, form.lane_loops)


def check_dense_realization() -> None:
    R, K, C, PE, SIMD = 2, 3, 4, 2, 4
    # The datapath reads X[r, (k, c)] as a dense (R, K*C) operand with a
    # block-diagonal W'[c, k*C + c'].
    KC = K * C
    NF, SF = C // PE, KC // SIMD
    nest = Nest(
        (Level("r", R), Level("nf", NF), Level("kf", SF)), (Level("p", PE), Level("s", SIMD))
    )
    X = Access(Operand("X", (R, KC)), ({"r": 1}, {"kf": SIMD, "s": 1}))
    activation = present(nest, X, fields=("s",))
    view = reshaped(activation, (R, K, C))
    # Independent: the dense datapath presents row r's (k, c) elements in (k, c)
    # row-major order, SIMD a beat, the whole row once per output fold.
    want = enumerate_positions(
        (("r", R), ("nf", NF), ("kf", SF)),
        (("s", SIMD),),
        lambda e: (e["r"], *divmod(e["kf"] * SIMD + e["s"], C)),
    )
    assert list(view.positions()) == want
    # The boundary (G0.2) presents each row once over the stream's (R, K, C) tensor.
    assert unreplayed(view) == reshaped(vector_major((R, KC), SIMD), (R, K, C))


# -- 2. tiled MVU (TH > 1) ---------------------------------------------------------------


def check_tiled_mvu() -> None:
    # test_stream_contract.py: R, MW, MH, PE, SIMD, T = 6, 8, 6, 3, 2, 3.
    R, MW, MH, PE, SIMD, T = 6, 8, 6, 3, 2, 3
    SF, NF = MW // SIMD, MH // PE
    # r = rt*T + t: T rows interleaved inside the reduction (T accumulators per lane).
    nest = Nest(
        (Level("rt", R // T), Level("nf", NF), Level("kf", SF), Level("t", T)),
        (Level("p", PE), Level("s", SIMD)),
    )
    X = Access(Operand("X", (R, MW)), ({"rt": T, "t": 1}, {"kf": SIMD, "s": 1}))
    Y = Access(Operand("Y", (R, MH)), ({"rt": T, "t": 1}, {"nf": PE, "p": 1}))
    core_in = present(nest, X, fields=("s",))
    assert core_in == Traversal.over(
        (R, MW), ((0, R // T, T), (None, NF, 0), (1, SF, SIMD), (0, T, 1)), ((1, SIMD, 1),)
    )
    # The result is presented after kf closes, for every t: reduced kf is NOT the
    # innermost suffix. The tiled core admits that; dotp_axi's rule does not.
    core_out = present(nest, Y, fields=("p",), reduced=("kf",))
    assert core_out == Traversal.over(
        (R, MH), ((0, R // T, T), (1, NF, PE), (0, T, 1)), ((1, PE, 1),)
    )
    W = Access(Operand("W", (MH, MW)), ({"nf": PE, "p": 1}, {"kf": SIMD, "s": 1}))
    assert dotp_admits(nest, X, W, Y) is not None  # t inside kf: not a dotp_axi frame
    # Both FINN input_gen parameter sets fall out of classify(boundary, core form).
    assert classify(vector_major((R, MW), SIMD), core_in).reorder == Reorder(
        SF * T, (NF, SF, T), (0, 1, SF)
    )
    assert classify(core_out, vector_major((R, MH), PE)).reorder == Reorder(NF * T, (T, NF), (1, T))
    # Weight chunks: a second split of the lane level p = pt*(PE/T) + pc, pt a beat level.
    for pe, t in ((3, 3), (6, 3), (4, 2)):
        mh = pe * 2
        nest_w = Nest(
            (Level("nf", mh // pe), Level("kf", SF), Level("pt", t)),
            (Level("pc", pe // t), Level("s", SIMD)),
        )
        Wc = Access(
            Operand("W", (mh, MW)), ({"nf": pe, "pt": pe // t, "pc": 1}, {"kf": SIMD, "s": 1})
        )
        chunked = present(nest_w, Wc, fields=("pc", "s"))
        want = enumerate_positions(
            (("nf", mh // pe), ("kf", SF), ("pt", t)),
            (("pc", pe // t), ("s", SIMD)),
            lambda e, pe=pe, t=t: (
                e["nf"] * pe + e["pt"] * (pe // t) + e["pc"],
                e["kf"] * SIMD + e["s"],
            ),
        )
        assert list(chunked.positions()) == want
        # The chunks are the tile's element sequence, fewer lanes a beat.
        assert classify(tile(mh, MW, pe, SIMD), chunked).adaptation is Adaptation.WIDTH_CONVERSION
    # The pinned test form (PE/T = 1) is the same derivation.
    nest_w = Nest(
        (Level("nf", NF), Level("kf", SF), Level("pt", T)), (Level("pc", 1), Level("s", SIMD))
    )
    Wc = Access(Operand("W", (MH, MW)), ({"nf": PE, "pt": 1, "pc": 1}, {"kf": SIMD, "s": 1}))
    assert present(nest_w, Wc, fields=("pc", "s")) == Traversal.over(
        (MH, MW), ((0, NF, PE), (1, SF, SIMD), (0, T, 1)), ((0, PE // T, 1), (1, SIMD, 1))
    )


# -- 3. SWG / conv-as-matmul with stride and dilation ------------------------------------


def check_swg() -> None:
    for H, W, C, KH, KW, S, D, SIMD in ((6, 6, 4, 3, 3, 1, 1, 2), (7, 7, 2, 2, 3, 2, 2, 1)):
        OH, OW = (H - D * (KH - 1) - 1) // S + 1, (W - D * (KW - 1) - 1) // S + 1
        nest = Nest(
            (
                Level("oh", OH),
                Level("ow", OW),
                Level("kh", KH),
                Level("kw", KW),
                Level("cf", C // SIMD),
            ),
            (Level("s", SIMD),),
        )
        # Several levels step one axis: oh*S + kh*D. Still one affine access.
        X = Access(
            Operand("X", (H, W, C)),
            ({"oh": S, "kh": D}, {"ow": S, "kw": D}, {"cf": SIMD, "s": 1}),
        )
        form = present(nest, X, fields=("s",))
        want = enumerate_positions(
            (("oh", OH), ("ow", OW), ("kh", KH), ("kw", KW), ("cf", C // SIMD)),
            (("s", SIMD),),
            lambda e, S=S, D=D, SIMD=SIMD: (
                e["oh"] * S + e["kh"] * D,
                e["ow"] * S + e["kw"] * D,
                e["cf"] * SIMD + e["s"],
            ),
        )
        assert list(form.positions()) == want
        # classify cannot name an overlapping sliding window: the windows revisit
        # positions without a stride-0 loop. input_gen realizes it (FINN conv2d.sv):
        # over a whole-image frame its COEFS are the beat strides of the levels.
        verdict = classify(vector_major((H, W, C), SIMD), form)
        assert verdict.adaptation is Adaptation.INCOMPATIBLE
        coefs = tuple(loop.stride // SIMD for loop in form.beat_loops)
        assert all(loop.stride % SIMD == 0 for loop in form.beat_loops)
        assert (
            sum((loop.extent - 1) * c for loop, c in zip(form.beat_loops, coefs))
            < H * W * C // SIMD
        )
    GAPS.append(
        "SWG: present() derives sliding windows (several levels on one axis), but classify "
        "names no adapter for overlapping windows; input_gen realizes them with whole-frame "
        "COEFS. Extension: a REORDER over a contiguous source with arbitrary nonnegative "
        "beat strides (not needed by S1-S3: no SWG kernel is on streams yet)."
    )


# -- 4. thresholding, PE <= C and PE > C --------------------------------------------------


def check_thresholding() -> None:
    for R, C, PE in ((3, 8, 4), (3, 8, 8), (4, 2, 4), (6, 3, 6), (2, 1, 2)):
        if PE <= C:
            nest = Nest((Level("r", R), Level("cf", C // PE)), (Level("p", PE),))
            T = Access(Operand("T", (R, C)), ({"r": 1}, {"cf": PE, "p": 1}))
            form = present(nest, T, fields=("p",))
            assert form == vector_major((R, C), PE)
        else:
            # PE > C: the fold wraps across rows. A split of r (r = rf*(PE/C) + rs)
            # into a beat level and a lane level: expressible, no padding needed.
            assert PE % C == 0 and R % (PE // C) == 0
            nest = Nest((Level("rf", R // (PE // C)),), (Level("rs", PE // C), Level("c", C)))
            T = Access(Operand("T", (R, C)), ({"rf": PE // C, "rs": 1}, {"c": 1}))
            form = present(nest, T, fields=("rs", "c"))
            # FINN's order: PE consecutive elements of the flat tensor a beat.
            flat = Traversal((R, C), (Loop(R * C // PE, PE),), (Loop(PE, 1),))
            assert form == flat
    GAPS.append(
        "Thresholding PE > C: expressible as a split of the row index (lanes {rs, c}); "
        "needs R divisible by PE/C (otherwise logical padding, S5). Today's "
        "thresholding port refuses it because vector_major needs C % PE == 0."
    )


# -- 5. eltwise with a broadcast operand -------------------------------------------------


def check_eltwise_broadcast() -> None:
    R, C, PE = 3, 4, 2
    nest = Nest((Level("r", R), Level("cf", C // PE)), (Level("p", PE),))
    X = Access(Operand("X", (R, C)), ({"r": 1}, {"cf": PE, "p": 1}))
    B = Access(Operand("B", (C,)), ({"cf": PE, "p": 1},))
    Row = Access(Operand("Brow", (R,)), ({"r": 1},))
    assert present(nest, X, fields=("p",)) == vector_major((R, C), PE)
    # A channel vector: the whole vector per row, as test_stream_contract's rhs_form.
    assert present(nest, B, fields=("p",)) == vector_major((C,), PE).repeated(R)
    # A per-row scalar: one lane (broadcast within the beat), replayed per channel fold.
    row = present(nest, Row, fields=())
    assert row == Traversal((R,), (Loop(R, 1), Loop(C // PE, 0)), ())


# -- 6. transpose ------------------------------------------------------------------------


def check_transpose() -> None:
    I, J, SIMD = 4, 6, 2  # noqa: E741
    nest = Nest((Level("j", J), Level("if", I // SIMD)), (Level("s", SIMD),))
    X = Access(Operand("X", (I, J)), ({"if": SIMD, "s": 1}, {"j": 1}))
    consumer = present(nest, X, fields=("s",))
    assert consumer == Traversal.over((I, J), ((1, J, 1), (0, I // SIMD, SIMD)), ((0, SIMD, 1),))
    assert classify(vector_major((I, J), SIMD), consumer).adaptation is Adaptation.LANE_REGROUP


# -- 7. the boundary rule (G0.2) ---------------------------------------------------------


def unreplayed(form: Traversal) -> Traversal:
    """Remove replay: stride-0 beat loops inside a moving loop. Outer repetition stays."""
    loops = list(form.beat_loops)
    first_moving = next((i for i, loop in enumerate(loops) if loop.stride), len(loops))
    kept = loops[:first_moving] + [loop for loop in loops[first_moving:] if loop.stride]
    return Traversal(form.shape, kept, form.lane_loops)


def check_boundary_rule() -> None:
    R, K, N, PE, SIMD = 3, 8, 6, 3, 2
    SF, NF = K // SIMD, N // PE
    replayed = vector_major((R, K), SIMD).replayed(NF, inner_beats=SF)
    weights = tile(N, K, PE, SIMD).repeated(R)
    # Activations: the module receives each row once and replays inside (receiver adapts).
    assert unreplayed(replayed) == vector_major((R, K), SIMD)
    # External weights: whole-pass repetition is the interface (FINN's in1_V).
    assert unreplayed(weights) == weights
    assert once(weights) == tile(N, K, PE, SIMD)


# -- 8. markers as loop levels (G0.4) ----------------------------------------------------


def level_aligned(form: Traversal, beats: int) -> bool:
    """A level marker closes the innermost loops spanning ``beats`` beats."""
    try:
        split_beats(form, beats)
    except ValueError:
        return False
    return form.beats % beats == 0


def check_levels() -> None:
    R, K, N, PE, SIMD = 2, 8, 6, 3, 2
    SF, NF = K // SIMD, N // PE
    nest = Nest(
        (Level("r", R), Level("nf", NF), Level("kf", SF)), (Level("p", PE), Level("s", SIMD))
    )
    X = Access(Operand("X", (R, K)), ({"r": 1}, {"kf": SIMD, "s": 1}))
    form = present(nest, X, fields=("s",))
    frame = frame_period(nest, ("kf",))
    assert frame.period == SF and level_aligned(form, SF)
    assert level_aligned(form, SF * NF) and not level_aligned(form, 3)
    # Canonical loops merge contiguous levels, so a level cannot be named by
    # position in the canonical nest; it is named by the beats it spans.
    merged = vector_major((R, K), SIMD)
    assert len(merged.beat_loops) == 1 and level_aligned(merged, SF)
    # input_gen's olst[d] closes dims[d:]: its levels are exactly the reorder's nest.
    replay = classify(vector_major((R, K), SIMD), form).reorder
    assert replay is not None
    offered = {prod(replay.dims[d:]) for d in range(len(replay.dims))}
    assert SF in offered


# -- 9. compound plans -------------------------------------------------------------------


def plan_steps(source: Traversal, sink: Traversal) -> tuple[str, ...] | None:
    verdict = classify(source, sink)
    if verdict.adaptation is not Adaptation.INCOMPATIBLE:
        return () if verdict.adaptation is Adaptation.IDENTITY else (verdict.adaptation.value,)
    if source.lanes == sink.lanes:
        return None
    # Width first (over the source's element order), then a reorder at the sink's lanes.
    try:
        widened = regrouped(source, sink.lanes)
        if classify(widened, sink).adaptation is Adaptation.REORDER:
            return ("width_conversion", "reorder")
    except ValueError:
        pass
    # Reorder first at the source's lanes, then width.
    try:
        narrowed = regrouped(sink, source.lanes)
        if (
            classify(source, narrowed).adaptation is Adaptation.REORDER
            and classify(narrowed, sink).adaptation is Adaptation.WIDTH_CONVERSION
        ):
            return ("reorder", "width_conversion")
    except ValueError:
        pass
    common = gcd(source.lanes, sink.lanes)
    try:
        low = regrouped(source, common)
        target = regrouped(sink, common)
        if classify(low, target).adaptation is Adaptation.REORDER:
            return ("width_conversion", "reorder", "width_conversion")
    except ValueError:
        pass
    return None


def check_compound_plans() -> None:
    R, K, N = 2, 8, 8
    # A producer with PE=4 results feeding a dense MatMul consumer with SIMD=2, NF=2.
    produced = vector_major((R, K), 4)
    consumer = vector_major((R, K), 2).replayed(2, inner_beats=K // 2)
    assert classify(produced, consumer).adaptation is Adaptation.INCOMPATIBLE
    assert plan_steps(produced, consumer) == ("width_conversion", "reorder")
    # Narrow producer (SIMD 2) into a wider replayed consumer (SIMD 4).
    produced = vector_major((R, K), 2)
    consumer = vector_major((R, K), 4).replayed(2, inner_beats=K // 4)
    assert plan_steps(produced, consumer) == ("width_conversion", "reorder")
    # A transpose at another width: no two-step chain, but at the common lane
    # count (here 1) every permutation is a reorder, so a regroup always has the
    # fallback width -> reorder -> width (input_gen over single elements: slow,
    # but correct). inner_shuffle is the fast realization of one shape of it.
    transposed = Traversal.over((4, 6), ((1, 6, 1), (0, 2, 2)), ((0, 2, 1),))
    source = vector_major((4, 6), 3)
    assert plan_steps(source, transposed) == ("lane_regroup",)
    common = gcd(source.lanes, transposed.lanes)
    assert classify(regrouped(source, common), regrouped(transposed, common)).adaptation is (
        Adaptation.REORDER
    )
    assert N


if __name__ == "__main__":
    for shape in [(1, 4, 4, 2, 2), (3, 6, 4, 3, 2)]:
        per_channel(*shape)
    check_dense_realization()
    check_tiled_mvu()
    check_swg()
    check_thresholding()
    check_eltwise_broadcast()
    check_transpose()
    check_boundary_rule()
    check_levels()
    check_compound_plans()
    print("s0_roster: all checks passed")
    for gap in GAPS:
        print("GAP:", gap)
