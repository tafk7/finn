# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A kernel's initiation interval (L6): one value, owned where the pace is.

A kernel with a module states its ``interval``: L6's 1 by default, its own, with its
standing and source, where its hardware differs (Pool's 1, assumed). A composite has none.
Each port carries its own kernel's, through any alias, with its pace (``timed_pace``). One
conversion (``finn.kernels.laws.in_cycles``) gives a kernel's cycles a frame and FIFO
sizing's beat times. A kernel with a module never restates its cycles
(``kernel-cycles-restated``), MemStream's restatement as the base's own aside; a port of
its body states no interval of its own (``port-interval``), and a composite none
(``kernel-interval-composite``).
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2
from support.refusals import refused

from finn.core.space import (
    DefinitionError,
    Inapplicable,
    Param,
    derived,
    design_space,
    view,
)
from finn.dataflow.gemm import c, kh, kw
from finn.dataflow.schedule import Pace
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel
from finn.kernels.channels import Neighbours
from finn.kernels.fifo_sizing import assumed, pattern
from finn.kernels.laws import (
    NATIVE_PROTOCOL,
    RATE_INTERVAL,
    Axis,
    AxisSpelling,
    Interval,
    Standing,
    TimedPace,
    in_cycles,
)
from finn.kernels.library import ModuleSpec
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.pool import ELEMENTS, PoolKernel
from finn.kernels.port_model import InPort, OutPort
from finn.platform.capabilities import Platform, Resources

IMAGE = Tensor((4, 4, 2), ScalarEncoding(DataType["INT4"]))
POOL: dict[str, Any] = dict(window=(2, 2), stride=(2, 2), platform=FULL_DSP48E2, input_tensor=IMAGE)
TWO = Interval(2, Standing.MEASURED, "a test's statement")
PROBE = CopiedSource("finnlib", "rtl/probe.sv", provides=("module:probe",))
TENSOR = Tensor((4, 1), ScalarEncoding(DataType["INT4"]))
ARRIVING = BeatSequence(vector_major((4, 1), 1))


class SlowPool(PoolKernel):
    """Pool, its iterations two cycles each."""

    id = "test.interval.slow_pool"

    @derived
    def interval(self) -> Interval:
        return TWO


class Wrapped(Kernel):
    """A slow Pool, its ports exposed as the wrapper's own."""

    id = "test.interval.wrapped"
    window: tuple[int, int] = Param()
    stride: tuple[int, int] = Param()
    platform: Platform = Param()
    input_tensor: Tensor = Param()

    inner = SlowPool(window=window, stride=stride, platform=platform, input_tensor=input_tensor)
    x = inner.input
    y = inner.output


def pool(space_type: type[Kernel]) -> Any:
    """``space_type`` on ``POOL``, its PE 1 (a wrapper's, its member's)."""
    pe = Wrapped.inner.pe if space_type is Wrapped else getattr(space_type, "pe")
    return design_space(space_type(**POOL)).with_choices({pe: 1})


def test_an_interval_is_l6_s_or_stated_with_its_standing_and_source() -> None:
    assert RATE_INTERVAL == Interval() and str(RATE_INTERVAL) == "1 (L6)"
    assert str(TWO) == "2, measured: a test's statement"
    assert Interval(1, Standing.ASSUMED, "a pragma").standing is Standing.ASSUMED
    for given in (
        dict(cycles=0),
        dict(cycles=2),  # another than L6's states its standing and source
        dict(cycles=1, source="a pragma"),
        dict(cycles=2, standing=Standing.MEASURED),
    ):
        with pytest.raises(ValueError):
            Interval(**given)


def test_one_conversion_gives_cycles_and_a_pace_s_beat_times() -> None:
    assert in_cycles(9, RATE_INTERVAL) == 9 and in_cycles(9, TWO) == 18
    point = pool(PoolKernel)
    pace = point.output.pace
    assert isinstance(pace, Pace)
    timed = TimedPace(pace, TWO)
    assert timed.times == tuple(2 * each for each in pace.times)
    assert timed.span == 2 * pace.span
    assert pattern(timed).span == timed.span


def test_a_kernel_with_a_module_states_l6_s_interval_and_its_ports_carry_it() -> None:
    point = pool(PoolKernel)
    # Pool's is stated: 1, assumed, on the pragmas its top requests.
    assert point.interval.cycles == 1 and point.interval.standing is Standing.ASSUMED
    assert "II=1" in point.interval.source
    assert point.frame_cycles == point.schedule.iterations == 2 * 2 * 2 * 2 * 2
    for port in (point.input, point.output):
        assert port.interval == point.interval
        assert port.timed_pace == TimedPace(port.pace, point.interval)
    fifo = design_space(_Stage(input_tensor=TENSOR, arriving=ARRIVING))
    assert fifo.interval == RATE_INTERVAL and fifo.input.interval == RATE_INTERVAL
    # A port of a given sequence has no pace, so none in cycles.
    assert isinstance(fifo.query(_Stage.input.timed_pace), Inapplicable)


def test_a_slower_interval_scales_the_kernel_s_cycles_and_its_ports_pace() -> None:
    fast, slow = pool(PoolKernel), pool(SlowPool)
    assert slow.frame_cycles == 2 * fast.frame_cycles == 64
    assert slow.output.timed_pace.span == 64
    assert slow.output.timed_pace.times == tuple(2 * t for t in fast.output.pace.times)


def test_the_interval_travels_with_the_pace_through_a_wrapper() -> None:
    """The prototype's ``slow`` case: an interval of 2 at the edge and through a wrapper.
    The wrapper has none of its own; its exposed port's is its kernel's, and its cycles a
    frame its member's, not a parent's guess at 1."""
    point = pool(Wrapped)
    assert isinstance(point.query(Wrapped.interval), Inapplicable)
    assert point.x.interval == point.y.interval == TWO
    assert point.y.timed_pace.span == 64
    assert point.frame_cycles == point.inner.frame_cycles == 64


def test_a_port_of_a_kernel_s_body_states_no_interval_of_its_own() -> None:
    """A port's interval is its kernel's: a port of its body stating its own is refused
    when the class is made (``port-interval``), so that its pace in cycles and its kernel's
    cycles never read two intervals (Pool's output at 3, its cycles at 1)."""
    with refused("port-interval", error=DefinitionError, message=r"Odd\.output states its own"):

        class Odd(PoolKernel):
            id = "test.interval.odd_pool"
            output = OutPort(
                transport=Axis(bus="m_axis", spelling=AxisSpelling.UPPER),
                tensor=PoolKernel.output_tensor,
                admits=ELEMENTS,
                schedule=PoolKernel.schedule,
                index=PoolKernel.output_index,
                lanes=(c,),
                reduces=(kh, kw),
                interval=TWO,
            )

    with refused("port-interval", error=DefinitionError, message=r"Own\.input states its own"):

        class Own(_Stage):
            id = "test.interval.own"
            input = InPort(tensor=_Stage.input_tensor, sequence=_Stage.arriving, interval=TWO)


def test_a_kernel_with_no_module_states_no_interval() -> None:
    """A composite has none: its members' ports carry their own kernels', and its
    cycles a frame are its members'; one it stated would be read and ignored."""
    with refused(
        "kernel-interval-composite", error=DefinitionError, message=r"Stating\.interval: "
    ):

        class Stating(Wrapped):
            id = "test.interval.stating"

            @derived
            def interval(self) -> Interval:
                return TWO


def test_fifo_sizing_says_where_it_rests_on_an_assumed_interval() -> None:
    point = pool(PoolKernel)
    timed = point.output.timed_pace
    assert assumed(Neighbours()) == ""
    assert assumed(Neighbours(producer_pace=TimedPace(timed.pace, TWO))) == ""
    note = assumed(Neighbours(producer_pace=timed, consumer_pace=timed))
    assert note.startswith("; at the producer's interval 1, assumed: ")
    assert " and the consumer's interval 1, assumed: " in note


class _Stage(Kernel):
    """A model-only stage on a library module."""

    id = "test.interval.stage"
    interface_protocol = NATIVE_PROTOCOL
    input_tensor: Tensor = Param()
    arriving: BeatSequence = Param()
    input = InPort(tensor=input_tensor, sequence=arriving)

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}

    @view
    def resource_use(self) -> Resources:
        return Resources()

    module_spec = ModuleSpec("probe", (PROBE,))


def test_a_stage_s_cycles_are_its_beats_at_its_interval() -> None:
    class SlowStage(_Stage):
        id = "test.interval.slow_stage"

        @derived
        def interval(self) -> Interval:
            return TWO

    assert design_space(_Stage(input_tensor=TENSOR, arriving=ARRIVING)).frame_cycles == 4
    slow = design_space(SlowStage(input_tensor=TENSOR, arriving=ARRIVING))
    assert slow.frame_cycles == 8 and slow.input.interval == TWO


def test_a_kernel_with_a_module_does_not_restate_its_cycles() -> None:
    with refused("kernel-cycles-restated", error=DefinitionError):

        class Restating(_Stage):
            id = "test.interval.restating"

            @view
            def frame_cycles(self) -> int:
                return 1

    # Nor the base's private rate law behind them (``_frame_cycles``).
    with refused("kernel-cycles-restated", error=DefinitionError, message=r"\._frame_cycles: "):

        class Behind(_Stage):
            id = "test.interval.behind"

            def _frame_cycles(self) -> int:
                return 1

    with refused(
        "kernel-cycles-restated", error=DefinitionError, message=r"\._rate_frame_cycles: "
    ):

        class Rated(_Stage):
            id = "test.interval.rated"

            def _rate_frame_cycles(self) -> int:
                return 1

    # MemStream's restatement is the base's own, so that its channel counts it by kind.
    assert MemStreamKernel.__dict__["frame_cycles"].function is Kernel._frame_cycles

    class Counted(_Stage):
        id = "test.interval.counted"
        frame_cycles = view(Kernel._frame_cycles)

    assert design_space(Counted(input_tensor=TENSOR, arriving=ARRIVING)).frame_cycles == 4


class WrappedPool(Kernel):
    """Pool itself, wrapped: its assumption reported below it."""

    id = "test.interval.wrapped_pool"
    window: tuple[int, int] = Param()
    stride: tuple[int, int] = Param()
    platform: Platform = Param()
    input_tensor: Tensor = Param()

    inner = PoolKernel(window=window, stride=stride, platform=platform, input_tensor=input_tensor)
    x = inner.input
    y = inner.output


def test_what_a_kernel_s_cycles_assume_is_reported_below_each_member() -> None:
    """Cycles that rest on an assumed interval say so (``Kernel.assumed``): Pool's own,
    a wrapper's below its member; a measured or L6's interval assumes nothing."""
    (own,) = pool(PoolKernel).assumed
    assert own.startswith("interval 1, assumed: the II=1 its HLS top requests")
    point = design_space(WrappedPool(**POOL)).with_choices({WrappedPool.inner.pe: 1})
    assert point.assumed == (f"inner: {own}",)
    assert pool(SlowPool).assumed == pool(Wrapped).assumed == ()
    assert design_space(_Stage(input_tensor=TENSOR, arriving=ARRIVING)).assumed == ()
