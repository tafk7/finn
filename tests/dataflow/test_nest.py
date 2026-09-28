# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One nest derives every port's traversal: the S0 roster, checked independently.

Each derived traversal is compared with an independent reference: a form
FINN's conventions define (``vector_major``, ``tile``), a traversal pinned by
the kernel tests, or an enumeration of the operation's own index formula.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations, product

import pytest

from finn.dataflow.nest import (
    Access,
    Einsum,
    Level,
    Nest,
    Refused,
    accesses,
    fold,
    frame,
    lane,
    once,
    period,
    present,
)
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Loop,
    Position,
    Reorder,
    Traversal,
    classify,
    tile,
    unreplayed,
    vector_major,
)


def enumerate_positions(
    beats: tuple[tuple[str, int], ...],
    lanes: tuple[tuple[str, int], ...],
    index: Callable[[dict[str, int]], Position],
) -> list[tuple[Position, ...]]:
    """Positions beat by beat from the operation's own formula, lanes innermost last."""
    rows = []
    for values in product(*(range(extent) for _, extent in beats)):
        point = dict(zip((name for name, _ in beats), values))
        rows.append(
            tuple(
                index({**point, **dict(zip((name for name, _ in lanes), inner))})
                for inner in product(*(range(extent) for _, extent in lanes))
            )
        )
    return rows


def matmul(
    contraction: str, extents: dict[str, int], pe: int, simd: int
) -> tuple[Nest, Access, Access, Access]:
    einsum = Einsum(contraction)
    (reduced,) = einsum.reduced
    output = einsum.output[-1]
    nest = fold(einsum, extents, {output: pe, reduced: simd})
    x, w, y = accesses(einsum, nest, extents)
    return nest, x, w, y


# -- einsum and folding ------------------------------------------------------------------


def test_an_einsum_names_its_indices_and_reductions() -> None:
    dense = Einsum("rk,nk->rn")
    assert dense.operands == ("rk", "nk") and dense.output == "rn"
    assert dense.indices == ("r", "k", "n") and dense.reduced == ("k",)
    assert str(Einsum(" rkc, ck -> rc ")) == "rkc,ck->rc"
    for bad in ("rk,nk", "rk,nk->rq", "rr,nk->rn"):
        with pytest.raises(ValueError):
            Einsum(bad)


def test_a_fold_splits_each_folded_index_into_a_beat_and_a_lane_level() -> None:
    nest = fold(Einsum("rk,nk->rn"), {"r": 3, "k": 8, "n": 6}, {"n": 3, "k": 2})
    assert nest.beats == (Level("r", 3), Level("n", 2), Level("k", 4))
    assert set(nest.lanes) == {Level(lane("n"), 3), Level(lane("k"), 2)}
    # An unfolded index named in the lanes keeps a lane level of extent one.
    assert (
        Level(lane("n"), 1) in fold(Einsum("rk,nk->rn"), {"r": 1, "k": 2, "n": 2}, {"n": 1}).lanes
    )
    with pytest.raises(ValueError, match="divide"):
        fold(Einsum("rk,nk->rn"), {"r": 1, "k": 5, "n": 2}, {"k": 2})
    with pytest.raises(ValueError, match="permutes"):
        fold(Einsum("rk,nk->rn"), {"r": 1, "k": 2, "n": 2}, {}, reduction_order=("n",))


# -- MatMul: the four hand-written forms, from one contraction ---------------------------


@pytest.mark.parametrize(
    "R,K,N,PE,SIMD", [(1, 4, 2, 2, 2), (3, 8, 6, 3, 2), (2, 12, 8, 4, 3), (4, 6, 6, 1, 6)]
)
def test_dense_forms_are_mvau_s_hand_written_ones(
    R: int, K: int, N: int, PE: int, SIMD: int
) -> None:
    nest, x, w, y = matmul("rk,nk->rn", {"r": R, "k": K, "n": N}, PE, SIMD)
    p, s = lane("n"), lane("k")
    SF, NF = K // SIMD, N // PE
    activation = present(nest, x, fields=(s,))
    assert activation == vector_major((R, K), SIMD).replayed(NF, inner_beats=SF)
    assert present(nest, w, fields=(p, s)) == tile(N, K, PE, SIMD).repeated(R)
    assert present(nest, y, fields=(p,), reduced=("k",)) == vector_major((R, N), PE)
    assert frame(nest, ("k",)) == LevelEnd(SF)
    # Broadcast is derived: the activations do not use the output lane level.
    assert not x.uses(p)
    # The boundary presents each row once; the replay is the receiver's.
    assert unreplayed(activation) == once(activation) == vector_major((R, K), SIMD)
    # A stored delivery repeats one period of the weights.
    assert period(present(nest, w, fields=(p, s))) == tile(N, K, PE, SIMD)


@pytest.mark.parametrize("R,C,K,PE,SIMD", [(1, 4, 4, 2, 2), (3, 6, 4, 3, 2), (2, 8, 9, 4, 3)])
def test_per_channel_fields_are_finnlib_s_s_times_pe_plus_p(
    R: int, C: int, K: int, PE: int, SIMD: int
) -> None:
    nest, x, w, y = matmul("rkc,ck->rc", {"r": R, "k": K, "c": C}, PE, SIMD)
    p, s = lane("c"), lane("k")
    assert x.uses(p)  # each lane reads its own channel: not broadcast
    activation = present(nest, x, fields=(s, p))
    want = enumerate_positions(
        (("r", R), ("c", C // PE), ("k", K // SIMD)),
        ((s, SIMD), (p, PE)),
        lambda e: (e["r"], e["k"] * SIMD + e[s], e["c"] * PE + e[p]),
    )
    assert list(activation.positions()) == want
    assert all(loop.stride for loop in activation.beat_loops)  # nothing replayed
    weights = present(nest, w, fields=(p, s))
    assert weights == tile(C, K, PE, SIMD).repeated(R)
    assert present(nest, y, fields=(p,), reduced=("k",)) == vector_major((R, C), PE)


def test_a_dense_realization_reads_the_per_channel_operand_as_a_view() -> None:
    R, K, C, PE, SIMD = 2, 3, 4, 2, 4
    nest, x, _, _ = matmul("rk,nk->rn", {"r": R, "k": K * C, "n": C}, PE, SIMD)
    view = x.viewing((R, K, C))
    activation = present(nest, view, fields=(lane("k"),))
    assert activation.shape == (R, K, C)
    want = enumerate_positions(
        (("r", R), ("n", C // PE), ("k", K * C // SIMD)),
        (("s", SIMD),),
        lambda e: (e["r"], *divmod(e["k"] * SIMD + e["s"], C)),
    )
    assert list(activation.positions()) == want
    with pytest.raises(ValueError, match="viewed"):
        x.viewing((R, K, C + 1))


# -- tiled MVU, SWG, thresholding, eltwise, transpose ------------------------------------


def test_tiled_mvu_forms_and_finn_s_input_gen_parameters() -> None:
    R, MW, MH, PE, SIMD, T = 6, 8, 6, 3, 2, 3
    SF, NF = MW // SIMD, MH // PE
    nest = Nest(
        (Level("rt", R // T), Level("nf", NF), Level("kf", SF), Level("t", T)),
        (Level("p", PE), Level("s", SIMD)),
    )
    x = Access((R, MW), ({"rt": T, "t": 1}, {"kf": SIMD, "s": 1}))
    y = Access((R, MH), ({"rt": T, "t": 1}, {"nf": PE, "p": 1}))
    core_in = present(nest, x, fields=("s",))
    core_out = present(nest, y, fields=("p",), reduced=("kf",))
    assert core_in == Traversal.over(
        (R, MW), ((0, R // T, T), (None, NF, 0), (1, SF, SIMD), (0, T, 1)), ((1, SIMD, 1),)
    )
    # mvu_tiled_axi.sv's two input_gen stages.
    assert classify(vector_major((R, MW), SIMD), core_in).reorder == Reorder(
        SF * T, (NF, SF, T), (0, 1, SF)
    )
    assert classify(core_out, vector_major((R, MH), PE)).reorder == Reorder(NF * T, (T, NF), (1, T))
    # t sits inside the reduction: not a dotp_axi frame.
    with pytest.raises(Refused, match="innermost"):
        frame(nest, ("kf",))


def test_sliding_windows_are_one_affine_access() -> None:
    H, W, C, KH, KW, S, D, SIMD = 7, 7, 2, 2, 3, 2, 2, 1
    OH, OW = (H - D * (KH - 1) - 1) // S + 1, (W - D * (KW - 1) - 1) // S + 1
    nest = Nest(
        (Level("oh", OH), Level("ow", OW), Level("kh", KH), Level("kw", KW), Level("cf", C)),
        (Level("s", SIMD),),
    )
    x = Access((H, W, C), ({"oh": S, "kh": D}, {"ow": S, "kw": D}, {"cf": SIMD, "s": 1}))
    want = enumerate_positions(
        (("oh", OH), ("ow", OW), ("kh", KH), ("kw", KW), ("cf", C)),
        (("s", SIMD),),
        lambda e: (e["oh"] * S + e["kh"] * D, e["ow"] * S + e["kw"] * D, e["cf"] + e["s"]),
    )
    assert list(present(nest, x, fields=("s",)).positions()) == want


def test_thresholding_and_a_broadcast_operand() -> None:
    R, C, PE = 3, 8, 4
    nest = Nest((Level("r", R), Level("cf", C // PE)), (Level("p", PE),))
    t = Access((R, C), ({"r": 1}, {"cf": PE, "p": 1}))
    assert present(nest, t, fields=("p",)) == vector_major((R, C), PE)
    # A channel vector broadcast over rows, and a per-row scalar within each beat.
    vector = Access((C,), ({"cf": PE, "p": 1},))
    assert present(nest, vector, fields=("p",)) == vector_major((C,), PE).repeated(R)
    scalar = Access((R,), ({"r": 1},))
    assert present(nest, scalar, fields=()) == Traversal((R,), (Loop(R, 1), Loop(C // PE, 0)), ())
    # A lane level that moves the position must be carried as a field.
    with pytest.raises(Refused, match="field"):
        present(nest, t, fields=())


def test_a_transpose_is_a_lane_regroup() -> None:
    I, J, SIMD = 4, 6, 2  # noqa: E741
    nest = Nest((Level("j", J), Level("if", I // SIMD)), (Level("s", SIMD),))
    columns = present(nest, Access((I, J), ({"if": SIMD, "s": 1}, {"j": 1})), fields=("s",))
    assert classify(vector_major((I, J), SIMD), columns).adaptation is Adaptation.LANE_REGROUP


# -- the reduction-order dial and admission ----------------------------------------------


def test_reduction_orders_are_legal_nests_a_reorder_apart() -> None:
    R, KH, KW, C, N, PE, SIMD = 2, 3, 3, 4, 4, 2, 2
    einsum = Einsum("rhwc,nhwc->rn")
    extents = {"r": R, "h": KH, "w": KW, "c": C, "n": N}
    forms = {}
    for order in permutations(einsum.reduced):
        nest = fold(einsum, extents, {"n": PE, "c": SIMD}, reduction_order=order)
        x, _, y = accesses(einsum, nest, extents)
        assert frame(nest, order) == LevelEnd(KH * KW * C // SIMD)
        forms[order] = present(nest, x, fields=(lane("c"),))
    canonical = forms[einsum.reduced]
    for order, form in forms.items():
        expected = Adaptation.IDENTITY if order == einsum.reduced else Adaptation.REORDER
        assert classify(canonical, form).adaptation is expected, order


def test_a_reduction_presented_before_it_closes_is_refused() -> None:
    nest, _, _, y = matmul("rk,nk->rn", {"r": 2, "k": 4, "n": 2}, 1, 2)
    with pytest.raises(Refused, match="not reduced"):
        present(nest, y, fields=(lane("n"),), reduced=("r",))
    reordered = Nest((nest.beats[1], nest.beats[2], nest.beats[0]), nest.lanes)
    with pytest.raises(Refused, match="innermost"):
        frame(reordered, ("k",))
