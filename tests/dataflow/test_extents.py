# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Extents bound from the tensors the ports read: the rules, coverage, and the kernel roster.

Each roster member states only which index addresses which axis of each
port's tensor (and, where no axis addresses an index alone, the extents its
author gives). The binding must give the extents each member states, and a schedule over the
bound extents must present every position of each tensor whose axes it binds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pytest

from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import Access, Affine, Index, Refused, Schedule, bind_extents
from finn.dataflow.traversal import Traversal

b, c, g, i, j, r, t = (Index(name) for name in ("b", "c", "g", "i", "j", "r", "t"))
oh, ow, kh, kw, mt = (Index(name) for name in ("oh", "ow", "kh", "kw", "mt"))
ZERO = Affine(())
"""The expression of a broadcast axis of extent one: it reads position zero."""


def covers(form: Traversal) -> bool:
    """Whether ``form`` presents every position of its tensor."""
    read = {position for beat in form.positions() for position in beat}
    return read == set(product(*(range(extent) for extent in form.shape)))


def dense(M: int, K: int, N: int) -> tuple[Access, Access, Access]:
    return (
        Access("x", (M, K), Form.DENSE.x),
        Access("w", (K, N), Form.DENSE.w),
        Access("y", (M, N), Form.DENSE.y),
    )


# -- the rules ----------------------------------------------------------------------------


def test_a_plain_index_takes_the_extent_of_the_axis_it_addresses() -> None:
    assert bind_extents(dense(3, 8, 6)) == {m: 3, k: 8, n: 6}
    # An index-valued Affine of coefficient one is the plain index.
    assert bind_extents([Access("x", (3, 8), (Affine.of(m), k))]) == {m: 3, k: 8}
    assert bind_extents([]) == {}


def test_an_index_on_two_axes_must_agree_on_one_port_or_two() -> None:
    assert bind_extents([Access("x", (4, 4), (i, i))]) == {i: 4}
    with pytest.raises(Refused, match=r"^i is 4 \(x axis 0\) and 5 \(x axis 1\)$"):
        bind_extents([Access("x", (4, 5), (i, i))])
    with pytest.raises(Refused, match=r"^n is 6 \(w axis 1\) and 5 \(y axis 1\)$"):
        bind_extents([*dense(3, 8, 6)[:2], Access("y", (3, 5), Form.DENSE.y)])


def test_a_rank_mismatch_is_refused() -> None:
    with pytest.raises(Refused, match=r"^x: 3 indices for a rank-2 tensor$"):
        bind_extents([Access("x", (4, 8), (b, r, c))])


def test_an_index_no_axis_binds_is_refused() -> None:
    with pytest.raises(Refused, match=r"^x axis 0: kh has no extent"):
        bind_extents([Access("x", (7, 4), (oh * 2 + kh, c))], {oh: 3})


def test_an_affine_axis_binds_nothing_and_stays_inside_its_axis() -> None:
    window = Access("x", (7, 4), (oh * 2 + kh, c))
    # Alone, the window gives neither oh nor kh an extent (kh is named first).
    with pytest.raises(Refused, match=r"^x axis 0: kh has no extent"):
        bind_extents([window])
    # Given them, it binds only c, and is checked: reach 2 * 2 + 2 = 6 < 7.
    assert bind_extents([window], {oh: 3, kh: 3}) == {oh: 3, kh: 3, c: 4}
    with pytest.raises(Refused, match=r"^x axis 0: kh \+ oh\*2 reaches 6, beyond extent 6$"):
        bind_extents([Access("x", (6, 4), (oh * 2 + kh, c))], {oh: 3, kh: 3})
    # One index times a stride is not plain either: a subsampling read binds nothing.
    assert bind_extents([Access("x", (5,), (oh * 2,)), Access("y", (3,), (oh,))]) == {oh: 3}
    with pytest.raises(Refused, match=r"^x axis 0: oh\*2 reaches 4, beyond extent 4$"):
        bind_extents([Access("x", (4,), (oh * 2,)), Access("y", (3,), (oh,))])
    # The empty expression reads position zero of an axis of extent one (a broadcast).
    assert bind_extents([Access("x", (1, 4), (ZERO, c))]) == {c: 4}


def test_given_extents_bind_and_must_agree_with_the_axes() -> None:
    assert bind_extents(dense(3, 8, 6), {k: 8, t: 2}) == {m: 3, k: 8, n: 6, t: 2}
    with pytest.raises(Refused, match=r"^k is 4 \(given\) and 8 \(x axis 1\)$"):
        bind_extents(dense(3, 8, 6), {k: 4})
    with pytest.raises(Refused, match="given to k must be a positive integer"):
        bind_extents(dense(3, 8, 6), {k: 0})


def test_a_view_binds_nothing_and_is_checked_against_the_bound_extents() -> None:
    # dotp's dense realization of a depthwise product: X (M, K, N) read as (M, K * N).
    M, K, N = 2, 3, 4
    x = Access("x", (M, K, N), Form.DENSE.x, reshaped=True)
    w, y = Access("w", (K * N, N), Form.DENSE.w), Access("y", (M, N), Form.DENSE.y)
    assert bind_extents([x, w, y]) == {k: K * N, n: N, m: M}
    # Without the ports that bind its indices, a view cannot bind itself.
    with pytest.raises(Refused, match=r"^x axis 0: m has no extent"):
        bind_extents([x, w])
    with pytest.raises(Refused, match=r"^x axis 1: k has no extent"):
        bind_extents([x, y])
    # Its indices given, it binds from them.
    assert bind_extents([x], {m: M, k: K * N}) == {m: M, k: K * N}
    # A view is the tensor's size, or it is refused.
    wider = Access("x", (M, K, N + 1), Form.DENSE.x, reshaped=True)
    with pytest.raises(Refused, match=r"^x: a \(2, 3, 5\) tensor cannot be viewed as \(2, 12\)$"):
        bind_extents([wider, w, y])
    with pytest.raises(Refused, match="^x: a reshaped port reads plain indices$"):
        bind_extents([Access("x", (M, K, N), (m, k * 1 + n), reshaped=True), w, y])


def test_an_access_is_a_checked_value() -> None:
    access = Access("x", [2, 3], [m, k])  # type: ignore[arg-type]
    assert (access.shape, access.index) == ((2, 3), (m, k))
    assert access == Access("x", (2, 3), (m, k)) and hash(access) == hash(
        Access("x", (2, 3), (m, k))
    )
    with pytest.raises(ValueError, match="nonempty name"):
        Access("", (2,), (m,))
    with pytest.raises(ValueError, match="positive"):
        Access("x", (0,), (m,))
    with pytest.raises(TypeError, match="Index"):
        Access("x", (2,), ("m",))  # type: ignore[arg-type]


# -- coverage -----------------------------------------------------------------------------


def test_bound_extents_make_every_bound_port_cover_its_tensor() -> None:
    M, K, N, PE, SIMD = 3, 8, 6, 3, 2
    x, w, y = dense(M, K, N)
    schedule = Schedule(bind_extents([x, w, y]), factors={n: PE, k: SIMD}, order=(m, n, k))
    assert covers(schedule.present(x.shape, x.index, lanes=(k,)))
    assert covers(schedule.present(w.shape, w.index, lanes=(n, k)))
    assert covers(schedule.present(y.shape, y.index, lanes=(n,), reduces=(k,)))


def test_a_tensor_too_wide_for_the_other_ports_is_refused() -> None:
    # x (M, K + 2) against w (K, N) and y (M, N): the ports disagree on k.
    M, K, N = 2, 4, 2
    x, w, y = Access("x", (M, K + 2), Form.DENSE.x), *dense(M, K, N)[1:]
    with pytest.raises(Refused, match=r"^k is 6 \(x axis 1\) and 4 \(w axis 0\)$"):
        bind_extents([x, w, y])
    # Without the binding, a schedule taking k from w presents x silently: two
    # columns of every row are never read.
    schedule = Schedule({m: M, n: N, k: K}, factors={k: 2}, order=(m, n, k))
    assert not covers(schedule.present(x.shape, x.index, lanes=(k,)))


# -- the kernel roster --------------------------------------------------------------------


def plain(access: Access) -> bool:
    """Whether every axis of ``access`` is addressed by a plain index (so binds it)."""
    return all(
        len(Affine.of(axis).terms) == 1 == Affine.of(axis).terms[0][1] for axis in access.index
    )


@dataclass(frozen=True)
class Port:
    """One roster port: its access, and how the schedule presents it."""

    access: Access
    lanes: tuple[Index, ...] = ()
    reduces: tuple[Index, ...] = ()


@dataclass(frozen=True)
class Member:
    """A roster member: its ports, the extents its author gives, and its schedule.

    ``unscheduled`` are accesses that bind but that no schedule presents (a
    table the kernel reads beside its channels). ``uncovered`` names the ports
    that present only part of their tensor: a window's image.
    """

    ports: tuple[Port, ...]
    extents: dict[Index, int]
    given: dict[Index, int]
    factors: dict[Index, int]
    order: tuple[Index, ...]
    unscheduled: tuple[Access, ...] = ()
    uncovered: tuple[str, ...] = ()

    def bound(self) -> dict[Index, int]:
        accesses = [*(port.access for port in self.ports), *self.unscheduled]
        return bind_extents(accesses, self.given)

    def present(self, port: Port) -> Traversal:
        bound = self.bound()
        schedule = Schedule(
            {i: bound[i] for i in self.order}, factors=self.factors, order=self.order
        )
        access = port.access
        view = None
        if access.reshaped:
            view = tuple(schedule.extent(Affine.of(axis).indices[0]) for axis in access.index)
        return schedule.present(
            access.shape, access.index, lanes=port.lanes, reduces=port.reduces, view=view
        )


def gemm_member(
    x: Access, w: Access, y: Access, extents: dict[Index, int], x_lanes: tuple[Index, ...] = (k,)
) -> Member:
    """MatMul's ports under ``m, n, k``, outputs split by 3 and the reduction by 2."""
    ports = (Port(x, x_lanes), Port(w, (n, k)), Port(y, (n,), (k,)))
    return Member(ports, extents, {}, {n: 3, k: 2}, (m, n, k))


H, W, C, KH, KW, S, D = 7, 7, 2, 2, 3, 2, 2
OH, OW = (H - D * (KH - 1) - 1) // S + 1, (W - D * (KW - 1) - 1) // S + 1
WINDOW = (oh * S + kh * D, ow * S + kw * D, c)
M_T, K_T, N_T, T = 6, 8, 6, 3

ROSTER = {
    "dense matmul": gemm_member(
        *dense(3, 8, 6),
        {m: 3, k: 8, n: 6},
    ),
    "per-channel matmul": gemm_member(
        Access("x", (3, 4, 6), Form.DEPTHWISE.x),
        Access("w", (4, 6), Form.DEPTHWISE.w),
        Access("y", (3, 6), Form.DEPTHWISE.y),
        {m: 3, k: 4, n: 6},
        x_lanes=(k, n),
    ),
    "per-channel as a dense view": gemm_member(
        Access("x", (2, 3, 6), Form.DENSE.x, reshaped=True),
        Access("w", (18, 6), Form.DENSE.w),
        Access("y", (2, 6), Form.DENSE.y),
        {m: 2, k: 18, n: 6},
    ),
    # Rows split into tiles, r = mt * T + t: no axis addresses mt or t alone.
    "tiled mvu": Member(
        (
            Port(Access("x", (M_T, K_T), (mt * T + t, k)), (k,)),
            Port(Access("w", (K_T, N_T), Form.DENSE.w), (n, k)),
            Port(Access("y", (M_T, N_T), (mt * T + t, n)), (n,), (k,)),
        ),
        {mt: M_T // T, t: T, k: K_T, n: N_T},
        {mt: M_T // T, t: T},
        {n: 3, k: 2},
        (mt, n, k, t),
    ),
    # A window generator: the windows bind every index; the image is checked.
    "swg": Member(
        (
            Port(Access("x", (H, W, C), WINDOW)),
            Port(Access("y", (OH, OW, KH, KW, C), (oh, ow, kh, kw, c))),
        ),
        {oh: OH, ow: OW, kh: KH, kw: KW, c: C},
        {},
        {},
        (oh, ow, kh, kw, c),
        uncovered=("x",),
    ),
    # FINN's layout, (1, OH, OW, KH * KW * C): the kernel size is its author's fact.
    "swg, finn layout": Member(
        (
            Port(Access("x", (1, H, W, C), (b, *WINDOW))),
            Port(Access("y", (1, OH, OW, KH * KW * C), (b, oh, ow, kh * (KW * C) + kw * C + c))),
        ),
        {b: 1, oh: OH, ow: OW, kh: KH, kw: KW, c: C},
        {kh: KH, kw: KW},
        {},
        (b, oh, ow, kh, kw, c),
        uncovered=("x",),
    ),
    "thresholding": Member(
        (
            Port(Access("x", (3, 8), (r, c)), (c,)),
            Port(Access("y", (3, 8), (r, c)), (c,)),
        ),
        {r: 3, c: 8, g: 2, j: 15},
        {},
        {c: 4},
        (r, c),
        # Its (sets, channels, thresholds) table has its ports' channels: checked.
        unscheduled=(Access("thresholds", (2, 8, 15), (g, c, j)),),
    ),
    "eltwise, broadcast operands": Member(
        (
            Port(Access("lhs", (3, 4), (r, c)), (c,)),
            Port(Access("channels", (4,), (c,)), (c,)),
            Port(Access("row", (1, 4), (ZERO, c)), (c,)),
            Port(Access("column", (3, 1), (r, ZERO))),
            Port(Access("result", (3, 4), (r, c)), (c,)),
        ),
        {r: 3, c: 4},
        {},
        {c: 2},
        (r, c),
    ),
    "transpose": Member(
        (
            Port(Access("x", (4, 6), (i, j)), (i,)),
            Port(Access("y", (6, 4), (j, i)), (i,)),
        ),
        {i: 4, j: 6},
        {},
        {i: 2},
        (j, i),
    ),
}


@pytest.mark.parametrize("name", ROSTER)
def test_the_s0_roster_binds_and_its_bound_ports_cover_their_tensors(name: str) -> None:
    member = ROSTER[name]
    assert member.bound() == member.extents
    for port in member.ports:
        label = port.access.name
        assert covers(member.present(port)) is (label not in member.uncovered), label
        # A port whose every axis is bound (or read through a view) covers.
        if label in member.uncovered:
            assert not plain(port.access)


def test_the_roster_s_windows_are_checked_not_covered() -> None:
    # A strided, dilated window reads only the rows and columns it reaches: the
    # binding checks the reach and, rightly, does not ask for coverage.
    swg = ROSTER["swg"]
    x = swg.ports[0]
    read = {position for beat in swg.present(x).positions() for position in beat}
    assert {row for row, _, _ in read} == {0, 2, 4, 6}
    shorter = Access("x", (H - 1, W, C), WINDOW)
    with pytest.raises(Refused, match=r"^x axis 0: kh\*2 \+ oh\*2 reaches 6, beyond extent 6$"):
        bind_extents([shorter, swg.ports[1].access])


def test_the_tiled_mvu_binds_only_with_its_tiles_given() -> None:
    tiled = ROSTER["tiled mvu"]
    accesses = [port.access for port in tiled.ports]
    with pytest.raises(Refused, match=r"^x axis 0: mt has no extent"):
        bind_extents(accesses, {t: T})
    # A known limit: its rows are a window's axis, checked and not bound, so
    # tiles given too few pass the check and leave rows unread.
    short = Member(tiled.ports, {}, {mt: 1, t: T}, tiled.factors, tiled.order)
    assert short.bound() == {mt: 1, t: T, k: K_T, n: N_T}
    assert not covers(short.present(tiled.ports[0]))
