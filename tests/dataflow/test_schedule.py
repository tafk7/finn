# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One schedule derives every port's traversal: the kernel roster, checked independently.

Each derived traversal is compared with an independent reference: a form
FINN's conventions define (``vector_major``, ``tile``), a traversal pinned by
the kernel tests, or an enumeration of the operation's own index formula.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations, product

import pytest

from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import Affine, Index, Pace, Refused, Schedule
from finn.dataflow.traversal import (
    Adaptation,
    LevelEnd,
    Loop,
    Position,
    Reorder,
    Traversal,
    classify,
    period,
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


def gemm(M: int, N: int, K: int, pe: int, simd: int) -> Schedule:
    """MatMul's schedule: outputs split by PE, the reduction by SIMD, reduction innermost."""
    return Schedule({m: M, n: N, k: K}, factors={n: pe, k: simd}, order=(m, n, k))


# -- indices, expressions and schedules ---------------------------------------------------


def test_index_arithmetic_builds_affine_expressions() -> None:
    oh, kh = Index("oh"), Index("kh")
    expression = oh * 2 + kh * 3
    assert expression.coefficient(oh) == 2 and expression.coefficient(kh) == 3
    assert expression == 3 * kh + 2 * oh
    assert Affine.of(m) == m * 1 and (m + m).coefficient(m) == 2
    assert Affine.of(m).coefficient(n) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        m * -1
    with pytest.raises(ValueError):
        Index("")


def test_a_schedule_folds_each_index_into_beats_and_lanes() -> None:
    schedule = gemm(3, 6, 8, pe=3, simd=2)
    assert schedule.order == (m, n, k)
    assert (schedule.factor(n), schedule.factor(k), schedule.factor(m)) == (3, 2, 1)
    assert (schedule.steps(m), schedule.steps(n), schedule.steps(k)) == (3, 2, 4)
    assert schedule.beat_count == 24
    # The beats default to the extents' order.
    assert Schedule({k: 4, m: 2}).order == (k, m)
    with pytest.raises(ValueError, match="divide"):
        Schedule({m: 1, k: 5}, factors={k: 2})
    with pytest.raises(ValueError, match="once"):
        Schedule({m: 1, k: 2}, order=(m,))
    with pytest.raises(ValueError, match="no extent"):
        Schedule({m: 1}, factors={k: 2})


def test_the_forms_name_canonical_gemm_operands() -> None:
    assert (Form.DENSE.x, Form.DENSE.w, Form.DENSE.y) == ((m, k), (k, n), (m, n))
    assert Form.DEPTHWISE.x == (m, k, n) and Form.DEPTHWISE.w == (k, n)


# -- MatMul: the hand-written forms, from one schedule ------------------------------------


@pytest.mark.parametrize(
    "M,K,N,PE,SIMD", [(1, 4, 2, 2, 2), (3, 8, 6, 3, 2), (2, 12, 8, 4, 3), (4, 6, 6, 1, 6)]
)
def test_dense_forms_are_mvau_s_hand_written_ones(
    M: int, K: int, N: int, PE: int, SIMD: int
) -> None:
    schedule = gemm(M, N, K, PE, SIMD)
    SF, NF = K // SIMD, N // PE
    activation = schedule.present((M, K), Form.DENSE.x, lanes=(k,))
    assert activation == vector_major((M, K), SIMD).replayed(NF, inner_beats=SF)
    weights = schedule.present((K, N), Form.DENSE.w, lanes=(n, k))
    # Stored (k, n), the weights present MVAU's tile order: the same positions,
    # beat for beat and lane for lane, as the (n, k) tile, transposed.
    reference = tile(N, K, PE, SIMD).repeated(M)
    assert [tuple(p[::-1] for p in beat) for beat in weights.positions()] == list(
        reference.positions()
    )
    assert schedule.present((M, N), Form.DENSE.y, lanes=(n,), reduces=(k,)) == vector_major(
        (M, N), PE
    )
    assert schedule.closing((k,)) == LevelEnd(SF)
    # The boundary presents each row once; the replay is the receiver's.
    assert unreplayed(activation) == vector_major((M, K), SIMD)
    # A stored delivery repeats one period of the weights.
    assert period(weights).beats == NF * SF


def test_dense_activations_are_broadcast_to_the_output_lanes() -> None:
    schedule = gemm(2, 4, 4, 2, 2)
    # X does not read n, so n's lanes need not be carried.
    schedule.present((2, 4), Form.DENSE.x, lanes=(k,))
    # W reads n, so they must be.
    with pytest.raises(Refused, match="carry them as lanes"):
        schedule.present((4, 4), Form.DENSE.w, lanes=(k,))


@pytest.mark.parametrize("M,N,K,PE,SIMD", [(1, 4, 4, 2, 2), (3, 6, 4, 3, 2), (2, 8, 9, 4, 3)])
def test_depthwise_fields_are_finnlib_s_s_times_pe_plus_p(
    M: int, N: int, K: int, PE: int, SIMD: int
) -> None:
    schedule = gemm(M, N, K, PE, SIMD)
    activation = schedule.present((M, K, N), Form.DEPTHWISE.x, lanes=(k, n))
    want = enumerate_positions(
        (("m", M), ("n", N // PE), ("k", K // SIMD)),
        (("s", SIMD), ("p", PE)),
        lambda e: (e["m"], e["k"] * SIMD + e["s"], e["n"] * PE + e["p"]),
    )
    assert list(activation.positions()) == want
    assert all(loop.stride for loop in activation.beat_loops)  # nothing replayed
    weights = schedule.present((K, N), Form.DEPTHWISE.w, lanes=(n, k))
    reference = tile(N, K, PE, SIMD).repeated(M)
    assert [tuple(p[::-1] for p in beat) for beat in weights.positions()] == list(
        reference.positions()
    )
    assert schedule.present((M, N), Form.DEPTHWISE.y, lanes=(n,), reduces=(k,)) == vector_major(
        (M, N), PE
    )


def test_a_dense_realization_reads_the_depthwise_operand_as_a_view() -> None:
    M, K, N, PE, SIMD = 2, 3, 4, 2, 4
    schedule = gemm(M, N, K * N, PE, SIMD)
    activation = schedule.present((M, K, N), Form.DENSE.x, lanes=(k,), view=(M, K * N))
    assert activation.shape == (M, K, N)
    want = enumerate_positions(
        (("m", M), ("n", N // PE), ("k", K * N // SIMD)),
        (("s", SIMD),),
        lambda e: (e["m"], *divmod(e["k"] * SIMD + e["s"], N)),
    )
    assert list(activation.positions()) == want
    with pytest.raises(Refused, match="viewed"):
        schedule.present((M, K, N + 1), Form.DENSE.x, lanes=(k,), view=(M, K * N))


# -- tiled MVU, SWG, thresholding, eltwise, transpose ------------------------------------


def test_tiled_mvu_forms_and_finn_s_input_gen_parameters() -> None:
    M, K, N, PE, SIMD, T = 6, 8, 6, 3, 2, 3
    SF, NF = K // SIMD, N // PE
    mt, t = Index("mt"), Index("t")
    schedule = Schedule(
        {mt: M // T, n: N, k: K, t: T}, factors={n: PE, k: SIMD}, order=(mt, n, k, t)
    )
    core_in = schedule.present((M, K), (mt * T + t, k), lanes=(k,))
    core_out = schedule.present((M, N), (mt * T + t, n), lanes=(n,), reduces=(k,))
    assert core_in == Traversal.over(
        (M, K), ((0, M // T, T), (None, NF, 0), (1, SF, SIMD), (0, T, 1)), ((1, SIMD, 1),)
    )
    # mvu_tiled_axi.sv's two input_gen stages.
    assert classify(vector_major((M, K), SIMD), core_in).reorder == Reorder(
        SF * T, (NF, SF, T), (0, 1, SF)
    )
    assert classify(core_out, vector_major((M, N), PE)).reorder == Reorder(NF * T, (T, NF), (1, T))
    # t sits inside the reduction: not a dotp_axi frame.
    with pytest.raises(Refused, match="innermost"):
        schedule.closing((k,))


def test_sliding_windows_are_one_affine_expression() -> None:
    H, W, C, KH, KW, S, D, SIMD = 7, 7, 2, 2, 3, 2, 2, 1
    OH, OW = (H - D * (KH - 1) - 1) // S + 1, (W - D * (KW - 1) - 1) // S + 1
    oh, ow, kh, kw, c = (Index(name) for name in ("oh", "ow", "kh", "kw", "c"))
    schedule = Schedule({oh: OH, ow: OW, kh: KH, kw: KW, c: C}, factors={c: SIMD})
    x = (oh * S + kh * D, ow * S + kw * D, c)
    want = enumerate_positions(
        (("oh", OH), ("ow", OW), ("kh", KH), ("kw", KW), ("c", C)),
        (("s", SIMD),),
        lambda e: (e["oh"] * S + e["kh"] * D, e["ow"] * S + e["kw"] * D, e["c"] + e["s"]),
    )
    assert list(schedule.present((H, W, C), x, lanes=(c,)).positions()) == want


def test_thresholding_and_a_broadcast_operand() -> None:
    M, C, PE = 3, 8, 4
    c = Index("c")
    schedule = Schedule({m: M, c: C}, factors={c: PE})
    assert schedule.present((M, C), (m, c), lanes=(c,)) == vector_major((M, C), PE)
    # A channel vector broadcast over rows, and a per-row scalar within each beat.
    assert schedule.present((C,), (c,), lanes=(c,)) == vector_major((C,), PE).repeated(M)
    assert schedule.present((M,), (m,)) == Traversal((M,), (Loop(M, 1), Loop(C // PE, 0)), ())
    # An index with lanes that moves the position must be carried as lanes.
    with pytest.raises(Refused, match="carry them as lanes"):
        schedule.present((M, C), (m, c))


def test_a_transpose_is_a_lane_regroup() -> None:
    I, J, SIMD = 4, 6, 2  # noqa: E741
    i, j = Index("i"), Index("j")
    schedule = Schedule({j: J, i: I}, factors={i: SIMD})
    columns = schedule.present((I, J), (i, j), lanes=(i,))
    assert classify(vector_major((I, J), SIMD), columns).adaptation is Adaptation.LANE_REGROUP


# -- the reduction-order dial, holding, and refusals --------------------------------------


def test_reduction_orders_are_legal_schedules_a_reorder_apart() -> None:
    M, KH, KW, C, N, PE, SIMD = 2, 3, 3, 4, 4, 2, 2
    h, w, c = Index("h"), Index("w"), Index("c")
    extents = {m: M, n: N, h: KH, w: KW, c: C}
    forms = {}
    for order in permutations((h, w, c)):
        schedule = Schedule(extents, factors={n: PE, c: SIMD}, order=(m, n, *order))
        assert schedule.closing(order) == LevelEnd(KH * KW * C // SIMD)
        forms[order] = schedule.present((M, KH, KW, C), (m, h, w, c), lanes=(c,))
    canonical = forms[(h, w, c)]
    for order, form in forms.items():
        expected = Adaptation.IDENTITY if order == (h, w, c) else Adaptation.REORDER
        assert classify(canonical, form).adaptation is expected, order


def test_a_held_operand_is_presented_once_per_outer_beat() -> None:
    # Weight-stationary: tiles of n, then of k, rows innermost; each weight tile
    # is presented once, before the rows it serves.
    M, N, K, ROWS, COLS = 4, 6, 8, 2, 3
    schedule = Schedule({m: M, n: N, k: K}, factors={k: ROWS, n: COLS}, order=(n, k, m))
    held = schedule.present((K, N), Form.DENSE.w, lanes=(k, n), holds=(m,))
    assert held.beats == (N // COLS) * (K // ROWS)
    assert all(loop.stride for loop in held.beat_loops)
    with pytest.raises(Refused, match="held"):
        schedule.present((M, K), Form.DENSE.x, lanes=(k,), holds=(m,))


def test_a_reduction_presented_before_it_closes_is_refused() -> None:
    schedule = gemm(2, 2, 4, 1, 2)
    with pytest.raises(Refused, match="reduced"):
        schedule.present((2, 2), Form.DENSE.y, lanes=(n,), reduces=(m,))
    reordered = Schedule({m: 2, n: 2, k: 4}, factors={n: 1, k: 2}, order=(n, k, m))
    with pytest.raises(Refused, match="innermost"):
        reordered.closing((k,))
    with pytest.raises(Refused, match="not an index"):
        schedule.present((2, 2), (m, Index("q")))


def test_an_index_outside_the_schedule_has_no_extent_or_factor() -> None:
    schedule = gemm(2, 2, 4, 1, 2)
    q = Index("q")
    with pytest.raises(Refused, match=r"^q is not an index of the schedule$"):
        schedule.extent(q)
    with pytest.raises(Refused, match=r"^q is not an index of the schedule$"):
        schedule.factor(q)


# -- beat times ---------------------------------------------------------------------------


def _beats_where(schedule: Schedule, keep: Callable[[dict[Index, int]], bool]) -> tuple[int, ...]:
    """The schedule beats, numbered in order, at which ``keep`` holds: an enumeration."""
    order = schedule.order
    steps = [range(schedule.steps(index)) for index in order]
    return tuple(
        beat for beat, values in enumerate(product(*steps)) if keep(dict(zip(order, values)))
    )


def test_a_port_reading_every_beat_is_presented_each_beat() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    assert schedule.beat_times() == tuple(range(schedule.beat_count))


def test_an_output_closing_a_reduction_is_presented_on_its_last_step() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    last = schedule.steps(k) - 1
    assert schedule.beat_times(reduces=(k,)) == _beats_where(schedule, lambda at: at[k] == last)
    assert schedule.beat_times(reduces=(k,))[:3] == (3, 7, 11)


def test_a_held_operand_is_presented_before_its_run() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    assert schedule.beat_times(holds=(k,)) == _beats_where(schedule, lambda at: at[k] == 0)


def test_beat_times_count_the_beats_the_port_presents() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    form = schedule.present((2, 8), (m, n), lanes=(n,), reduces=(k,))
    assert len(schedule.beat_times(reduces=(k,))) == form.beats


def test_beat_times_name_indices_of_the_schedule() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    with pytest.raises(Refused, match="not an index"):
        schedule.beat_times(reduces=(Index("q"),))
    with pytest.raises(Refused, match="not both"):
        schedule.beat_times(reduces=(k,), holds=(k,))


def test_a_pace_is_a_ports_beat_times_and_its_kernels_span() -> None:
    schedule = gemm(2, 8, 12, pe=2, simd=3)
    pace = Pace(schedule, reduces=(k,))
    assert pace.times == schedule.beat_times(reduces=(k,))
    assert pace.span == schedule.beat_count == 32
