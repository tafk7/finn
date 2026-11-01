# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""A kernel's rate (L6): its initiation interval, one value owned where the pace is, and the
rate a frame it states where the derived one cannot express it.

A kernel with a module states its ``interval``: L6's 1 by default, its own, with its
standing and source, where its hardware differs (Pool's 1, assumed). A composite has none.
Each port carries its own kernel's, through any alias, with its pace (``timed_pace``). One
conversion (``finn.kernels.laws.in_cycles``) gives a kernel's cycles a frame and FIFO
sizing's beat times. A port of its body states no interval of its own (``port-interval``),
and a composite none (``kernel-interval-composite``). Where its schedule and ports at its
interval cannot express its rate (a memory mover), a kernel with a module states its rate a
frame (``StatedRate``): a port's beat times and its span, with its reason, standing and
source; its cycles a frame are its span. A composite states none
(``kernel-rate-composite``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, in_order, stage_tensor
from support.refusals import refused

from finn.core.space import (
    Available,
    DefinitionError,
    Inapplicable,
    Param,
    Rejected,
    Select,
    derived,
    design_space,
    view,
)
from finn.dataflow.gemm import c, kh, kw
from finn.dataflow.schedule import Pace
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.base import Kernel, neighbours
from finn.kernels.channels import Channel, Neighbours
from finn.kernels.configure import commit
from finn.kernels.fifo import FifoKernel
from finn.kernels.fifo_sizing import assumed, pattern
from finn.kernels.laws import (
    NATIVE_PROTOCOL,
    RATE_INTERVAL,
    Axis,
    AxisSpelling,
    Interval,
    Standing,
    StatedRate,
    TimedPace,
    in_cycles,
)
from finn.kernels.library import ModuleSpec
from finn.kernels.pool import ELEMENTS, PoolKernel
from finn.kernels.port_model import InPort, OutPort
from finn.kernels.sides import Side
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
    assert fifo.input.timed_pace is None


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


def test_fifo_sizing_says_where_it_rests_on_an_assumed_interval_or_rate() -> None:
    point = pool(PoolKernel)
    timed = point.output.timed_pace
    assert assumed(Neighbours()) == ""
    assert assumed(Neighbours(producer_pace=MOVED)) == ""
    stated = replace(MOVED, standing=Standing.ASSUMED)
    assert assumed(Neighbours(consumer_pace=stated)) == f"; at the consumer's rate {stated}"
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


#: A memory mover's rate, as such a kernel states it: four beats as their words arrive, two
#: words a beat, and a call's three cycles after them.
MOVED = StatedRate(
    "input",
    (1, 3, 5, 7),
    11,
    Standing.MEASURED,
    "a test's measurement",
    "its memory words and each call take cycles its stream does not show",
)


class _Mover(_Stage):
    """A stage that states its rate a frame."""

    id = "test.interval.mover"
    rate: StatedRate = Param()

    @derived
    def stated_rate(self) -> StatedRate:
        stated: StatedRate = self.rate
        return stated


def test_a_stated_rate_times_each_beat_within_its_span_with_its_reason_and_standing() -> None:
    assert str(MOVED) == "11 cycles a frame, measured: a test's measurement"
    for given in (
        dict(port=""),
        dict(standing=None),
        dict(source=""),
        dict(reason=""),
        dict(span=0),
        dict(times=()),
        dict(times=[1, 3, 5, 7]),  # not a tuple
        dict(times=(1, 3, 2, 7)),  # out of order
        dict(times=(0, 0, 0, 0)),  # more than one beat a cycle
        dict(times=(1, 3, 5, 11)),  # past the span
        dict(times=(-1, 3, 5, 7)),
    ):
        with pytest.raises(ValueError):
            replace(MOVED, **given)


def test_a_kernel_with_a_module_may_state_its_rate_in_place_of_the_derived_one() -> None:
    """Its span, not its four beats at its interval; its port's pace in cycles, the rate's
    beat times, though the port follows its other side and has no pace of its own."""
    point = design_space(_Mover(input_tensor=TENSOR, arriving=ARRIVING, rate=MOVED))
    assert point.frame_cycles == 11
    assert point.checked_rate == point.stated_rate == MOVED and point.assumed == ()
    assert point.input.timed_pace == MOVED
    assumed_rate = replace(MOVED, standing=Standing.ASSUMED)
    point = design_space(_Mover(input_tensor=TENSOR, arriving=ARRIVING, rate=assumed_rate))
    assert point.assumed == (f"rate {assumed_rate}",)
    # Every other kernel states none: its cycles are derived.
    assert pool(PoolKernel).stated_rate is None and pool(PoolKernel).checked_rate is None


class _Sometimes(_Stage):
    """A stage that states its rate at some design points only."""

    id = "test.interval.sometimes"
    stating: bool = Param()

    @derived
    def stated_rate(self) -> StatedRate | None:
        return MOVED if self.stating else None


def test_a_kernel_states_its_rate_where_it_must_and_its_rate_is_derived_elsewhere() -> None:
    def point(stating: bool) -> Any:
        return design_space(_Sometimes(input_tensor=TENSOR, arriving=ARRIVING, stating=stating))

    assert point(True).frame_cycles == 11 and point(True).input.timed_pace == MOVED
    assert point(False).frame_cycles == 4 and point(False).input.timed_pace is None


#: Pool's eight output beats on ``POOL`` (its PE 1), stalled: a beat every three cycles of a
#: frame of 100.
STALLED = replace(MOVED, port="output", times=(1, 4, 7, 10, 13, 16, 19, 22), span=100)


class Stalling(PoolKernel):
    """Pool, stating its output's rate."""

    id = "test.interval.stalling"

    @derived
    def stated_rate(self) -> StatedRate:
        return STALLED


class WrappedStalling(Kernel):
    """A stalling Pool, its ports exposed as the wrapper's own."""

    id = "test.interval.wrapped_stalling"
    window: tuple[int, int] = Param()
    stride: tuple[int, int] = Param()
    platform: Platform = Param()
    input_tensor: Tensor = Param()

    inner = Stalling(window=window, stride=stride, platform=platform, input_tensor=input_tensor)
    x = inner.input
    y = inner.output


def test_a_kernel_with_a_schedule_may_state_its_rate_too() -> None:
    """Its span, in place of its schedule's iterations at its interval, and its named port's
    pace in cycles, through a wrapper's alias too; its other port keeps its derived one."""
    point = pool(Stalling)
    assert point.frame_cycles == 100
    assert point.output.timed_pace == STALLED
    assert point.input.timed_pace == TimedPace(point.input.pace, point.interval)
    wrapped = design_space(WrappedStalling(**POOL)).with_choices({WrappedStalling.inner.pe: 1})
    assert wrapped.y.timed_pace == STALLED and wrapped.frame_cycles == 100


def test_a_stated_rate_times_a_port_of_its_kernel_each_beat() -> None:
    """Checked where it is read (``checked_rate``): its cycles, its port's pace in cycles
    and its assumptions are refused with it."""
    for rate, message in (
        (replace(MOVED, port="output"), "states the rate of 'output', no port of it"),
        (replace(MOVED, times=(1, 3, 5)), "times 3 beats of input, which carries 4 a frame"),
    ):
        point = design_space(_Mover(input_tensor=TENSOR, arriving=ARRIVING, rate=rate))
        for read in (_Mover.checked_rate, _Mover.frame_cycles, _Mover.input.timed_pace):
            result = point.query(read)
            assert isinstance(result, Rejected), result
            [finding] = result.findings
            assert (finding.code, finding.message) == ("kernel-rate-port", f"_Mover {message}")


def test_a_kernel_stating_its_rate_restates_no_cycles() -> None:
    """Its cycles are its rate's span: restating them, or the base's reading of them, could
    only contradict it. A kernel stating no rate may restate its cycles (CONTRACT.md L6: no
    check guards it)."""
    for name in ("frame_cycles", "checked_rate"):
        with refused(
            "kernel-rate-restated", error=DefinitionError, message=rf"Restating\.{name}: "
        ):
            type(
                "Restating",
                (_Mover,),
                {"id": f"test.interval.restating_{name}", name: view(lambda self: 99)},
            )

    class Counted(_Stage):
        id = "test.interval.counted"

        @view
        def frame_cycles(self) -> int:
            return 99

    assert design_space(Counted(input_tensor=TENSOR, arriving=ARRIVING)).frame_cycles == 99


def test_a_port_of_a_kernel_s_body_states_no_rate_of_its_own() -> None:
    with refused("port-rate", error=DefinitionError, message=r"Own\.input states its own rate"):

        class Own(_Stage):
            id = "test.interval.own_rate"
            input = InPort(tensor=_Stage.input_tensor, sequence=_Stage.arriving, rate=MOVED)


FIFO_TENSOR = stage_tensor(4, words=4)
FIFO: dict[str, Any] = dict(
    input_tensor=FIFO_TENSOR, arriving=in_order(FIFO_TENSOR), depth=8, platform=FULL_DSP48E2
)


class TimedFifo(FifoKernel):
    """A FIFO stating its input's rate: its four beats as ``MOVED`` times them."""

    id = "test.interval.timed_fifo"

    @derived
    def stated_rate(self) -> StatedRate:
        return MOVED


class Holding(Kernel):
    """A timed FIFO between two boundary channels."""

    id = "test.interval.holding"
    compute = TimedFifo(**FIFO)
    incoming = Channel(
        tensor=FIFO_TENSOR,
        platform=FULL_DSP48E2,
        producer=Side.boundary("s_axis_0"),
        consumer=compute.input.side,
    )
    outgoing = Channel(
        tensor=FIFO_TENSOR,
        platform=FULL_DSP48E2,
        producer=compute.output.side,
        consumer=Side.boundary("m_axis_0"),
    )
    x = incoming.outer_in
    y = outgoing.outer_out


class Selecting(Kernel):
    """A timed FIFO or a plain one, selected, between two boundary channels."""

    id = "test.interval.selecting"
    compute: FifoKernel = Select(timed=TimedFifo(**FIFO), plain=FifoKernel(**FIFO))
    incoming = Channel(
        tensor=FIFO_TENSOR,
        platform=FULL_DSP48E2,
        producer=Side.boundary("s_axis_0"),
        consumer=compute.input.side,
    )
    outgoing = Channel(
        tensor=FIFO_TENSOR,
        platform=FULL_DSP48E2,
        producer=compute.output.side,
        consumer=Side.boundary("m_axis_0"),
    )
    x = incoming.outer_in
    y = outgoing.outer_out


DIRECT = {"incoming.transport": "direct", "outgoing.transport": "direct"}


def test_the_port_beside_a_channel_is_paced_by_its_kernel_s_stated_rate() -> None:
    """What FIFO sizing reads of it (``Neighbours``), a member's port and a selected case's;
    the port its rate does not name, and the plain case's, by none."""
    point = commit(design_space(Holding()), {"compute.ram_style": "auto", **DIRECT})
    around = point.query(neighbours(Holding, "incoming"))
    assert isinstance(around, Available) and around.value.consumer_pace == MOVED
    around = point.query(neighbours(Holding, "outgoing"))
    assert isinstance(around, Available) and around.value.producer_pace is None
    for case, pace in (("timed", MOVED), ("plain", None)):
        selected = commit(
            design_space(Selecting()),
            {"compute": case, f"compute.{case}.ram_style": "auto", **DIRECT},
        )
        around = selected.query(neighbours(Selecting, "incoming"))
        assert isinstance(around, Available) and around.value.consumer_pace == pace


def test_a_kernel_with_no_module_states_no_rate() -> None:
    """A composite's cycles are its members': a member states its own."""
    with refused(
        "kernel-rate-composite", error=DefinitionError, message=r"Composite\.stated_rate: "
    ):

        class Composite(Wrapped):
            id = "test.interval.composite"

            @derived
            def stated_rate(self) -> StatedRate:
                return MOVED


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
