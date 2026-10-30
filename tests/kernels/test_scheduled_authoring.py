# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Actual scheduled child projection/access and native Transpose demand."""

from __future__ import annotations

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
from finn.dataflow.traversal import BeatSequence
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.base import Kernel
from finn.kernels.port_model import IndexedPort, ScheduledPort
from finn.kernels.transpose import TransposeKernel, i

R, C = Index("r"), Index("c")
TENSOR = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
SCHEDULE = Schedule({R: 2, C: 4}, {C: 2})


class Projection(Space):
    fold: int = Decision(domain=(1, 2, 4))

    @derived
    def scheduled(self) -> Schedule:
        return Schedule({R: 2, C: 4}, {C: self.fold})

    input = IndexedPort(
        name="in", label="input", tensor=TENSOR, schedule=scheduled, index=(R, C), lanes=(C,)
    )
    output = ScheduledPort(
        name="out",
        label="output",
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
    assert not hasattr(type(point.output), "binds")
    assert not hasattr(type(point.output), "access")
    assert isinstance(point.query(Projection.input.stream), Unresolved)
    assert not inspection.decisions(ScheduledPort) and not inspection.decisions(IndexedPort)


@pytest.mark.parametrize("fold", (1, 2, 4))
def test_projection_reduction_and_pace_are_the_current_schedule_values(fold: int) -> None:
    point = design_space(Projection()).with_choices(fold=fold)
    schedule = point.scheduled
    expected_input = BeatSequence(schedule.present(TENSOR.shape, (R, C), lanes=(C,)))
    expected_output = BeatSequence(schedule.present((2,), (R,), reduces=(C,)))
    assert point.input.stream == Stream(TENSOR.element, expected_input)
    assert point.output.stream == Stream(TENSOR.element, expected_output)
    assert point.input.contract.contract.pace == Pace(schedule, (), ())
    assert point.output.contract.contract.pace == Pace(schedule, (C,), ())
    assert point.output.contract.label == "output"


def test_schedule_and_sequence_conflict_remains_a_local_nonvalue() -> None:
    port = ScheduledPort(
        name="p",
        label="p",
        tensor=TENSOR,
        schedule=SCHEDULE,
        sequence=BeatSequence(SCHEDULE.present(TENSOR.shape, (R, C), lanes=(C,))),
        index=(R, C),
    )
    point = design_space(port)
    answer = point.query(ScheduledPort.stream)
    assert isinstance(answer, Rejected)
    assert [finding.code for finding in answer.findings] == ["port-presentation"]
    assert inspection.validity(point).status == "valid"


@pytest.mark.parametrize("suppressed", ("reduces", "holds"))
def test_closing_markers_and_suppressed_axes_keep_their_actual_pace(suppressed: str) -> None:
    source = design_space(
        ScheduledPort(
            name="source",
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
        ScheduledPort(
            name="output",
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
    assert output.contract.contract.pace == Pace(SCHEDULE, reduces, holds)


def test_a_reshaped_affine_axis_preserves_the_local_schedule_refusal() -> None:
    point = design_space(
        ScheduledPort(
            name="p",
            label="p",
            tensor=TENSOR,
            schedule=SCHEDULE,
            index=(2 * R, C),
            lanes=(C,),
            reshaped=True,
        )
    )
    result = point.query(ScheduledPort.stream)
    assert isinstance(result, Rejected)
    assert [finding.code for finding in result.findings] == ["port-schedule"]
    assert "reshaped port reads plain indices" in result.findings[0].message


@pytest.mark.parametrize("enabled", (None, False, True))
def test_scheduled_child_presence_is_checked_before_local_work(enabled: bool | None) -> None:
    class Guarded(Space):
        enabled: bool = Decision(domain=(False, True))
        port = ScheduledPort(
            when=enabled,
            name="p",
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


def test_authored_access_method_is_preserved_through_inheritance() -> None:
    class Explicit(Kernel):
        id = "test.scheduled.explicit_access"

        def _accesses(self) -> list[Access]:
            return [Access("actual", TENSOR.shape, (R, C), False)]

    class Inherited(Explicit):
        pass

    for kind in (Explicit, Inherited):
        assert design_space(kind()).extents == {R: 2, C: 4}
        assert kind._accesses is Explicit._accesses


def test_empty_accesses_are_inherited_without_a_generated_method() -> None:
    class Empty(Kernel):
        id = "test.scheduled.empty"

    class Subclass(Empty):
        pass

    # Indexed kernels author access reads over actual Ports; no lowering hook generates
    # per-class methods. The empty base method is inherited here unchanged.
    assert Subclass._accesses is Empty._accesses
    assert design_space(Empty()).extents == design_space(Subclass()).extents == {}


class SerialTranspose(TransposeKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    @derived
    def first(self) -> Schedule | Rejected:
        if self.first_state == "refused":
            return reject("first-schedule", "first input blocks")
        return self.columns_out

    @derived
    def second(self) -> Schedule:
        _ = self.second_state
        return self.columns_out

    input = IndexedPort(
        name="input",
        label="input",
        tensor=TransposeKernel.input_tensor,
        schedule=first,
        index=TransposeKernel.columns,
        lanes=(i,),
        reshaped=True,
        signals=("idat", "ivld"),
        clocking=TransposeKernel.port_clocking,
    )
    output = ScheduledPort(
        name="output",
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=TransposeKernel.output_tensor,
        schedule=second,
        index=TransposeKernel.indices,
        lanes=(i,),
        signals=("odat", "ovld"),
        clocking=TransposeKernel.port_clocking,
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
        assert [f.code for f in evidence.result.findings] == ["first-schedule"]
    else:
        assert isinstance(evidence.result, Rejected)
        assert [f.message for f in evidence.result.findings] == [
            "input: signals are data, valid and ready",
            "output: signals are data, valid and ready",
        ]
    keys = {node.declaration.key for node in evidence.nodes}
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "output.stream", "output.signals"}
