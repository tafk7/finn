# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Eltwise's actual child model access and three-contributor serial assembly."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2

from finn.core.space import (
    Available,
    Decision,
    Rejected,
    Unresolved,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.dataflow.schedule import Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.eltwise import EltwiseKernel, c
from finn.kernels.port_model import IndexedPort, ScheduledPort

LHS = Tensor((2, 4), ScalarEncoding(DataType["INT3"]))
RHS = Tensor((4,), LHS.element)


def test_actual_eltwise_inputs_bind_early_and_result_is_structurally_nonbinding() -> None:
    point = design_space(
        EltwiseKernel(
            operation="ADD", lhs_tensor=LHS, rhs_tensor=RHS, b_scale=1.0, platform=FULL_DSP48E2
        )
    )
    assert point.lhs_dtype == point.rhs_dtype == LHS.element.dtype
    assert point.extents[c] == 4
    assert point.rhs.access.shape == (4,)
    assert point.result_tensor.shape == LHS.shape
    assert isinstance(point.query(EltwiseKernel.lhs.stream), Unresolved)
    assert not hasattr(type(point.result), "access")
    assert not hasattr(type(point.result), "binds")
    assert not inspection.decisions(IndexedPort) and not inspection.decisions(ScheduledPort)


def test_parent_arithmetic_refusal_keeps_intrinsic_child_input_characterization() -> None:
    bad = design_space(
        EltwiseKernel(
            operation="DIV", lhs_tensor=LHS, rhs_tensor=RHS, b_scale=1.0, platform=FULL_DSP48E2
        )
    ).with_choices(pe=2)
    assessment = bad.inspect(EltwiseKernel.netlist)
    assert isinstance(assessment.result, Rejected) and assessment.output_result is None
    assert isinstance(bad.query(EltwiseKernel.lhs.contract), Available)


class SerialEltwise(EltwiseKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    @derived
    def first(self) -> Schedule | Rejected:
        if self.first_state == "refused":
            return reject("first-schedule", "first model input blocks")
        return self.schedule

    @derived
    def second(self) -> Schedule:
        _ = self.second_state
        return self.schedule

    lhs = IndexedPort(
        name="lhs",
        label="lhs",
        tensor=EltwiseKernel.lhs_tensor,
        schedule=first,
        index=EltwiseKernel.indices,
        lanes=(c,),
        signals=("adat", "avld"),
        clocking=EltwiseKernel.port_clocking,
    )
    rhs = IndexedPort(
        name="rhs",
        label="rhs",
        tensor=EltwiseKernel.rhs_tensor,
        schedule=second,
        index=EltwiseKernel.rhs_indices,
        lanes=(c,),
        signals=("bdat", "bvld"),
        clocking=EltwiseKernel.port_clocking,
    )
    result = ScheduledPort(
        name="result",
        label="result",
        endpoint=Endpoint.INITIATOR,
        tensor=EltwiseKernel.result_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.indices,
        lanes=(c,),
        signals=("odat", "ovld"),
        clocking=EltwiseKernel.port_clocking,
    )


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize(
    "first,second", ((None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready"))
)
def test_actual_three_port_assembly_halts_or_accumulates_in_source_order(
    first: str | None,
    second: str | None,
    permuted: bool,
    eager: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(
        SerialEltwise(
            operation="ADD", lhs_tensor=LHS, rhs_tensor=RHS, b_scale=1.0, platform=FULL_DSP48E2
        )
    )
    choices: dict[object, object] = {SerialEltwise.pe: 2}
    if first is not None:
        choices[SerialEltwise.first_state] = first
    if second is not None:
        choices[SerialEltwise.second_state] = second
    point = point.with_choices(choices)
    trace = inspection.explain(point, SerialEltwise.netlist)
    if first is None or second is None and first != "refused":
        assert isinstance(trace.result, Unresolved)
    elif first == "refused":
        assert isinstance(trace.result, Rejected)
        assert [item.code for item in trace.result.findings] == ["first-schedule"]
    else:
        assert isinstance(trace.result, Rejected)
        # Rejected canonicalizes diagnostics by finding key. Actual source
        # order is lhs/rhs/result; demanded-node boundaries are checked below.
        assert [item.message for item in trace.result.findings] == [
            "lhs: signals are data, valid and ready",
            "result: signals are data, valid and ready",
            "rhs: signals are data, valid and ready",
        ]
    keys = {node.declaration.key for node in trace.nodes}
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "rhs.stream", "rhs.signals", "result.stream"}
    if second is None and first == "ready":
        assert not keys & {"result.stream", "result.signals"}
