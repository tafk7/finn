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
    Inapplicable,
    Rejected,
    Unresolved,
    derived,
    design_space,
    inspection,
    reject,
)
from finn.dataflow.schedule import Index
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import LevelEnd
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.eltwise import EltwiseKernel, c
from finn.kernels.laws import Native
from finn.kernels.port import LoopEnds
from finn.kernels.port_model import StreamPort

UNNAMED_ENDS = LoopEnds((LevelEnd(1),))
"""A loop-completion vector, which a native port named in full has no pin for: each port
so declared refuses its pins directly."""

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
    assert point.result.binds is False  # an output never binds its kernel's extents
    assert isinstance(point.query(type(point).result.access), Inapplicable)
    assert not inspection.decisions(StreamPort)


def test_parent_arithmetic_refusal_keeps_intrinsic_child_input_characterization() -> None:
    bad = design_space(
        EltwiseKernel(
            operation="DIV", lhs_tensor=LHS, rhs_tensor=RHS, b_scale=1.0, platform=FULL_DSP48E2
        )
    ).with_choices(pe=2)
    assessment = bad.inspect(EltwiseKernel.netlist)
    assert isinstance(assessment.result, Rejected) and assessment.output_result is None
    assert isinstance(bad.query(EltwiseKernel.lhs.stream_pins), Available)


class SerialEltwise(EltwiseKernel):
    first_state: str = Decision(domain=("ready", "refused"))
    second_state: str = Decision(domain=("ready",))

    # Each port's lanes wait on a choice of its own: the first port's demand comes first.
    @derived
    def first(self) -> tuple[Index, ...] | Rejected:
        if self.first_state == "refused":
            return reject("first-lanes", "first model input blocks")
        return (c,)

    @derived
    def second(self) -> tuple[Index, ...]:
        _ = self.second_state
        return (c,)

    lhs = StreamPort(
        label="lhs",
        tensor=EltwiseKernel.lhs_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.indices,
        lanes=first,
        transport=Native(names=("adat", "avld", "ardy")),
        loop_ends=UNNAMED_ENDS,
    )
    rhs = StreamPort(
        label="rhs",
        tensor=EltwiseKernel.rhs_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.rhs_indices,
        lanes=second,
        transport=Native(names=("bdat", "bvld", "brdy")),
        loop_ends=UNNAMED_ENDS,
    )
    result = StreamPort(
        label="result",
        endpoint=Endpoint.INITIATOR,
        tensor=EltwiseKernel.result_tensor,
        schedule=EltwiseKernel.schedule,
        index=EltwiseKernel.indices,
        lanes=(c,),
        transport=Native(names=("odat", "ovld", "ordy")),
        loop_ends=UNNAMED_ENDS,
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
        assert [item.code for item in trace.result.findings] == ["first-lanes"]
    else:
        assert isinstance(trace.result, Rejected)
        # Rejected canonicalizes diagnostics by finding key. Actual source
        # order is lhs/rhs/result; demanded-node boundaries are checked below.
        assert [item.message for item in trace.result.findings] == [
            "lhs: a native port named in full has no loop-completion pin",
            "result: a native port named in full has no loop-completion pin",
            "rhs: a native port named in full has no loop-completion pin",
        ]
    keys = {node.declaration.key for node in trace.nodes}
    if first is None or first == "refused":
        assert not keys & {"second", "second_state", "rhs.stream", "rhs.transport", "result.stream"}
    if second is None and first == "ready":
        assert not keys & {"result.stream", "result.transport"}
