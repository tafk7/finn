"""P0.2: a kernel's `schedule` reads extents bound from its own ports (plan S5, D3, D4).

`sketches/bound_probe.py` as a test, with the disagreement a `reject(...)` rather
than a raised exception, plus the view case (dotp `reshaped`) and an
Affine-addressed axis (a sliding window), which binds nothing.

    cd docs/kernel-authoring-2026-09-29/p0 && PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH=../../../src:../../../tests <venv python> -m pytest -q -p no:cacheprovider .
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from _bind import Access, BoundKernel, bind_extents, extent_of, port_names
from _pool import Pool, b, c, placed_pool, s, stream

from finn.core.space import Decision, Param, Rejected, Space, derived, design_space, divisors_of
from finn.dataflow.gemm import Form, k, m, n
from finn.dataflow.schedule import SCHEDULE, Index, Refused, Schedule
from finn.dataflow.stream import Stream
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.configure import commit
from finn.kernels.dotp import DotpAxiKernel, PackedDotpKernel
from finn.kernels.port import ScheduledPort
from finn.kernels.target import DspBlock


def codes_and_messages(result: object) -> set[tuple[str, str]]:
    assert isinstance(result, Rejected), result
    return {(f.code, f.message) for f in result.findings}


# -- the probe, as a test -----------------------------------------------------------------


def test_the_schedule_reads_extents_bound_from_its_ports() -> None:
    assert port_names(Pool) == ("x", "y")  # read from the class, never the graph
    point = placed_pool((1, 4, 8))
    assert point.pool.extents == {b: 1, s: 4, c: 8}
    point = commit(point, {"pool.pe": 4})
    x, y = point.pool.x.sequence.form, point.pool.y.sequence.form
    assert (x.beats, x.lanes, y.beats, y.lanes) == (8, 4, 2, 4)


def test_a_disagreement_is_a_rejection_not_an_exception() -> None:
    point = placed_pool((1, 4, 6))  # x says c = 6, y says c = 8
    expected = {("kernel-extents", "c is 6 (x axis 2) and 8 (y axis 1)")}
    assert codes_and_messages(point.pool.query(Pool.extents)) == expected
    # It propagates as the same finding to everything that reads the extents.
    assert codes_and_messages(point.pool.query(Pool.channels)) == expected
    assert codes_and_messages(point.pool.query(Pool.build_requirements)) == expected
    # Committing a fold over a refused domain is refused by the commit helper.
    with pytest.raises(ValueError, match="pool.extents: kernel-extents: c is 6"):
        commit(point, {"pool.pe": 2})


def test_a_rank_mismatch_is_a_rejection() -> None:
    point = placed_pool((4, 8))
    assert codes_and_messages(point.pool.query(Pool.extents)) == {
        ("kernel-extents", "x: 3 indices for a rank-2 tensor")
    }


# -- the view case: dotp reads (M, K, N) activations as (M, K * N) ------------------------


def placed_dotp(x_shape: tuple[int, ...]) -> Space:
    class Placed(Space):
        x = stream(x_shape, "INT3", "in0_V")
        w = stream((12, 4), "INT3", "in1_V")
        y = stream((2, 4), "INT9", "out0_V")
        compute = PackedDotpKernel(
            form=Form.DENSE,
            reshape_activations=True,
            target_dsp=DspBlock.DSP58,
            target_period_ns=5.0,
            x_stream=x,
            w_stream=w,
            y_stream=y,
        )

    return commit(
        design_space(Placed()),
        {"compute.pe": 2, "compute.simd": 4, "compute.compute_pumping": False},
    )


def dotp_accesses(kernel: DotpAxiKernel) -> list[Access]:
    return [
        Access(name, port.stream.tensor.shape, tuple(port.index), port.reshaped)
        for name in port_names(type(kernel))
        for port in (getattr(kernel, name),)
    ]


def test_a_view_binds_nothing_and_is_checked_against_the_bound_extents() -> None:
    kernel = placed_dotp((2, 3, 4)).compute
    accesses = dotp_accesses(kernel)
    assert [(a.name, a.index, a.reshaped) for a in accesses] == [
        ("x", (m, k), True),
        ("w", (k, n), False),
        ("y", (m, n), False),
    ]
    # The binding agrees with today's hand-written getters.
    assert bind_extents(accesses) == {k: 12, n: 4, m: 2}
    assert (kernel.rows, kernel.outputs, kernel.reduction) == (2, 4, 12)
    assert kernel.x.sequence.form.shape == (2, 3, 4)
    # Without the other ports the view's indices have no extent: it cannot bind itself.
    with pytest.raises(Refused, match="x: m has no extent"):
        bind_extents([accesses[0]])


def test_a_view_of_another_size_is_refused() -> None:
    kernel = placed_dotp((2, 3, 5)).compute  # 30 elements, viewed as (M, K) = (2, 12)
    with pytest.raises(Refused, match=r"x: a \(2, 3, 5\) tensor cannot be viewed as \(2, 12\)"):
        bind_extents(dotp_accesses(kernel))


# -- an Affine-addressed axis: a sliding window --------------------------------------------

oh, kh = Index("oh"), Index("kh")


class Window(BoundKernel):
    """A 1-D sliding window generator: X[oh * S + kh, c] -> Y[oh, kh, c]."""

    id = "probe.window"
    module = "window"
    stride: ClassVar[int] = 2
    x_stream: Stream = Param(required=False)
    y_stream: Stream = Param(required=False)

    channels = extent_of(c)
    simd: int = Decision(domain=divisors_of(channels))

    @derived(semantics=SCHEDULE)
    def schedule(self) -> Schedule:
        return Schedule(self.extents, folds={c: self.simd}, beats=(oh, kh, c))

    x = ScheduledPort(
        name="s_axis_input",
        endpoint=Endpoint.TARGET,
        stream=x_stream,
        schedule=schedule,
        index=(oh * 2 + kh, c),
        lanes=(c,),
    )
    y = ScheduledPort(
        name="m_axis_output",
        endpoint=Endpoint.INITIATOR,
        stream=y_stream,
        schedule=schedule,
        index=(oh, kh, c),
        lanes=(c,),
    )


def placed_window(height: int) -> Space:
    class Placed(Space):
        x = stream((height, 4), "INT4", "in0_V")
        y = stream((3, 3, 4), "INT4", "out0_V")  # OH = 3 windows of KH = 3, stride 2
        window = Window(x_stream=x, y_stream=y)

    return design_space(Placed())


def test_an_affine_axis_binds_nothing() -> None:
    x = Access("x", (7, 4), (oh * 2 + kh, c))
    # Alone, the window axis gives neither oh nor kh an extent.
    with pytest.raises(Refused, match="x: kh has no extent"):
        bind_extents([x])
    # Given them (bound_schedule(extents=...)), it is checked: reach 2 * 2 + 2 = 6 < 7.
    assert bind_extents([x], {oh: 3, kh: 3}) == {oh: 3, kh: 3, c: 4}


def test_a_window_kernel_binds_oh_and_kh_from_its_output_and_checks_the_reach() -> None:
    point = placed_window(7)
    assert point.window.extents == {oh: 3, kh: 3, c: 4}  # x's axis 0 contributed nothing
    point = commit(point, {"window.simd": 2})
    assert point.window.x.sequence.form.shape == (7, 4)
    short = placed_window(6)  # the last window reaches row 6
    assert codes_and_messages(short.window.query(Window.extents)) == {
        ("kernel-extents", "x axis 0: kh + oh*2 reaches 6, beyond extent 6")
    }
