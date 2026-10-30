# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual scheduled child projection/access and native Transpose demand."""

from __future__ import annotations

import re

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import (
    Available,
    Decision,
    Inapplicable,
    Rejected,
    Space,
    Unresolved,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.dataflow.schedule import Access, Index, Pace, Schedule
from finn.dataflow.streams import Stream
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, LevelEnd
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel, bound_extents, bound_schedule, spanned
from finn.kernels.laws import AXIS, AXIS_CLOCKING, Axis, Native
from finn.kernels.port import LoopEnds
from finn.kernels.port_model import StreamPort
from finn.kernels.transpose import TransposeKernel, i

UNNAMED_ENDS = LoopEnds((LevelEnd(1),))
"""A loop-completion vector, which a native port named in full has no pin for: each port
so declared refuses its pins directly."""

R, C = Index("r"), Index("c")
TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SCHEDULE = Schedule({R: 2, C: 4}, {C: 2})


class Projection(Space):
    fold: int = Decision(domain=(1, 2, 4))

    @derived
    def scheduled(self) -> Schedule:
        return Schedule({R: 2, C: 4}, {C: self.fold})

    input = StreamPort(
        clocking=AXIS_CLOCKING,
        label="input",
        tensor=TENSOR,
        schedule=scheduled,
        index=(R, C),
        lanes=(C,),
        transport=Axis(bus="in"),
    )
    output = StreamPort(
        clocking=AXIS_CLOCKING,
        label="output",
        transport=Axis(bus="out"),
        endpoint=Endpoint.INITIATOR,
        tensor=Tensor((2,), TENSOR.element),
        schedule=scheduled,
        index=(R,),
        reduces=(C,),
    )


def test_access_and_element_are_early_and_output_never_binds_parent_extents() -> None:
    point = design_space(Projection())
    assert point.input.element == TENSOR.element
    assert point.input.access == Access("input", TENSOR.shape, (R, C), False)
    assert point.output.binds is False  # an output never binds its kernel's extents
    assert isinstance(point.query(type(point).output.access), Inapplicable)
    assert isinstance(point.query(Projection.input.stream), Unresolved)
    assert not inspection.decisions(StreamPort)


@pytest.mark.parametrize("fold", (1, 2, 4))
def test_projection_reduction_and_pace_are_the_current_schedule_values(fold: int) -> None:
    point = design_space(Projection()).with_choices(fold=fold)
    schedule = point.scheduled
    expected_input = BeatSequence(schedule.present(TENSOR.shape, (R, C), lanes=(C,)))
    expected_output = BeatSequence(schedule.present((2,), (R,), reduces=(C,)))
    assert point.input.stream == Stream(TENSOR.element, expected_input)
    assert point.output.stream == Stream(TENSOR.element, expected_output)
    assert point.input.pace == Pace(schedule, (), ())
    assert point.output.pace == Pace(schedule, (C,), ())
    assert point.output.contract.label == "output"


def test_schedule_and_sequence_conflict_remains_a_local_nonvalue() -> None:
    port = StreamPort(
        clocking=AXIS_CLOCKING,
        transport=AXIS,
        label="p",
        tensor=TENSOR,
        schedule=SCHEDULE,
        sequence=BeatSequence(SCHEDULE.present(TENSOR.shape, (R, C), lanes=(C,))),
        index=(R, C),
    )
    point = design_space(port)
    answer = point.query(StreamPort.stream)
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["port-presentation"]
    assert inspection.validity(point).status == "valid"


@pytest.mark.parametrize("suppressed", ("reduces", "holds"))
def test_closing_markers_and_suppressed_axes_keep_their_actual_pace(suppressed: str) -> None:
    source = design_space(
        StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="source",
            tensor=TENSOR,
            schedule=SCHEDULE,
            index=(R, C),
            lanes=(C,),
            closes=(C,),
        )
    )
    assert source.stream.sequence.markers == (SCHEDULE.closing((C,)),)
    assert source.contract.contract.markers[0][1] == SCHEDULE.closing((C,))
    reduces = (C,) if suppressed == "reduces" else ()
    holds = (C,) if suppressed == "holds" else ()
    output = design_space(
        StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="output",
            endpoint=Endpoint.INITIATOR,
            tensor=Tensor((2,), TENSOR.element),
            schedule=SCHEDULE,
            index=(R,),
            reduces=reduces,
            holds=holds,
        )
    )
    assert output.stream.sequence.form == SCHEDULE.present((2,), (R,), reduces=reduces, holds=holds)
    assert output.pace == Pace(SCHEDULE, reduces, holds)


def test_a_reshaped_affine_axis_preserves_the_local_schedule_refusal() -> None:
    point = design_space(
        StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            label="p",
            tensor=TENSOR,
            schedule=SCHEDULE,
            index=(2 * R, C),
            lanes=(C,),
            reshaped=True,
        )
    )
    result = point.query(StreamPort.stream)
    assert isinstance(result, Rejected)
    assert [finding.code for finding in result.findings] == ["port-schedule"]
    assert "reshaped port reads plain indices" in result.findings[0].message


@pytest.mark.parametrize("enabled", (None, False, True))
def test_scheduled_child_presence_is_checked_before_local_work(enabled: bool | None) -> None:
    class Guarded(Space):
        enabled: bool = Decision(domain=(False, True))
        port = StreamPort(
            clocking=AXIS_CLOCKING,
            transport=AXIS,
            when=enabled,
            label="p",
            tensor=TENSOR,
            schedule=SCHEDULE,
            index=(R, C),
            lanes=(C,),
        )

    point = design_space(Guarded())
    if enabled is not None:
        point = point.with_choices(enabled=enabled)
    result = point.query(Guarded.port.contract)
    expected = Unresolved if enabled is None else Available if enabled else Inapplicable
    assert isinstance(result, expected)


def test_a_kernel_states_its_extents_and_schedule_by_dataflow_s_rules() -> None:
    class Explicit(Kernel):
        id = "test.scheduled.explicit_access"
        fold: int = Decision(domain=(1, 2, 4))

        @derived
        def extents(self) -> dict[Index, int] | Rejected:
            return bound_extents([Access("actual", TENSOR.shape, (R, C), False)])

        @derived
        def schedule(self) -> Schedule | Rejected:
            return bound_schedule(self.extents, (R, C), {C: self.fold})

    class Inherited(Explicit):
        pass

    for kind in (Explicit, Inherited):
        point = design_space(kind()).with_choices(fold=2)
        assert point.extents == {R: 2, C: 4}
        assert point.schedule == Schedule({R: 2, C: 4}, {C: 2}, (R, C))


def test_a_kernel_with_no_scheduled_port_has_no_schedule_and_no_extents() -> None:
    class Empty(Kernel):
        id = "test.scheduled.empty"

    # The base holds no schedule mechanics: no extents, accesses, or schedule builders.
    assert isinstance(design_space(Empty()).query(Empty.schedule), Inapplicable)
    for gone in ("extents", "_accesses", "_bound", "bound_schedule", "spanned"):
        assert not hasattr(Kernel, gone), gone


@pytest.mark.parametrize(
    ("order", "factors", "code", "message"),
    (
        ((R, C, Index("t")), {}, "kernel-extents", r"^\[t\] are bound by no placed port$"),
        ((R, C), {C: 3}, "kernel-schedule", "^a folding factor of 3 does not divide c's extent 4$"),
        ((R, C, C), {}, "kernel-schedule", "names each index"),
    ),
)
def test_a_schedule_dataflow_refuses_is_the_kernel_s_refusal(
    order: tuple[Index, ...], factors: dict[Index, int], code: str, message: str
) -> None:
    found = bound_schedule({R: 2, C: 4}, order, factors)
    assert isinstance(found, Rejected)
    ((finding),) = found.findings
    assert finding.code == code and re.search(message, finding.message)


def test_extents_and_spans_dataflow_refuses_are_the_kernel_s_refusals() -> None:
    conflict = bound_extents([Access("x", (2, 3), (R, R))])
    assert isinstance(conflict, Rejected)
    assert [(f.code, f.message) for f in conflict.findings] == [
        ("kernel-extents", "r is 2 (x axis 0) and 3 (x axis 1)")
    ]
    assert bound_extents([], {R: 2}) == {R: 2}
    unbound = spanned((R, C + Index("t")), {R: 2, C: 4})
    assert isinstance(unbound, Rejected)
    assert [(f.code, f.message) for f in unbound.findings] == [
        ("kernel-extents", "[t] are bound by no placed port")
    ]
    assert spanned((R, C * 2 + R), {R: 2, C: 4}) == (2, 8)


class SerialTranspose(TransposeKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    # Each port's lanes wait on a choice of its own: the first port's demand comes first.
    @derived
    def first(self) -> tuple[Index, ...] | Rejected:
        if self.first_state == "refused":
            return reject("first-lanes", "first input blocks")
        return (i,)

    @derived
    def second(self) -> tuple[Index, ...]:
        _ = self.second_state
        return (i,)

    input = StreamPort(
        label="input",
        tensor=TransposeKernel.input_tensor,
        schedule=TransposeKernel.schedule,
        index=TransposeKernel.columns,
        lanes=first,
        reshaped=True,
        transport=Native(names=("idat", "ivld", "irdy")),
        loop_ends=UNNAMED_ENDS,
    )
    output = StreamPort(
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=TransposeKernel.output_tensor,
        schedule=TransposeKernel.schedule,
        index=TransposeKernel.indices,
        lanes=second,
        transport=Native(names=("odat", "ovld", "ordy")),
        loop_ends=UNNAMED_ENDS,
    )


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize(
    "first,second", ((None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready"))
)
def test_transpose_assembly_has_real_serial_schedule_and_physical_demand(
    permuted: bool,
    eager: bool,
    first: str | None,
    second: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    tensor = Tensor((2, 6, 4), TENSOR.element)
    point = design_space(SerialTranspose(input_tensor=tensor, platform=FULL_DSP48E2))
    changes: dict[object, object] = {SerialTranspose.simd: 2, SerialTranspose.ram_style: "auto"}
    if first is not None:
        changes[SerialTranspose.first_state] = first
    if second is not None:
        changes[SerialTranspose.second_state] = second
    point = point.with_choices(changes)
    evidence = inspection.explain(point, SerialTranspose.netlist)
    if first is None or second is None and first != "refused":
        assert isinstance(evidence.result, Unresolved)
    elif first == "refused":
        assert isinstance(evidence.result, Rejected)
        assert [f.code for f in evidence.result.findings] == ["first-lanes"]
    else:
        assert isinstance(evidence.result, Rejected)
        assert [f.message for f in evidence.result.findings] == [
            "input: a native port named in full has no loop-completion pin",
            "output: a native port named in full has no loop-completion pin",
        ]
    keys = {node.declaration.key for node in evidence.nodes}
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "output.stream", "output.transport"}
