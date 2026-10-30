# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Remaining leaf model bindings, serial physical calls and source-stage boundaries."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import FULL_DSP48E2, FULL_DSP58

from finn.core.space import (
    Decision,
    Inapplicable,
    Rejected,
    Unresolved,
    derived,
    design_space,
    inspection,
    reject,
    view,
)
from finn.dataflow.schedule import Schedule
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import LevelEnd
from finn.kernels.artifacts.abi import Endpoint
from finn.kernels.laws import Native
from finn.kernels.pool import PoolKernel, c, kh, kw
from finn.kernels.port import LoopEnds
from finn.kernels.port_model import StreamPort
from finn.kernels.pwpolyf import PwPolyKernel

UNNAMED_ENDS = LoopEnds((LevelEnd(1),))
"""A loop-completion vector, which a native port named in full has no pin for: each port
so declared refuses its pins directly."""

TENSOR = Tensor((1, 3, 3, 4), ScalarEncoding(DataType["INT4"]))


def test_actual_pool_model_remains_usable_before_pe_and_has_no_output_binding() -> None:
    point = design_space(
        PoolKernel(input_tensor=TENSOR, window=(2, 2), stride=(1, 1), platform=FULL_DSP48E2)
    )
    assert point.input.element == TENSOR.element
    assert point.channels == 4
    assert point.output_tensor.shape == (1, 2, 2, 4)
    assert isinstance(point.query(PoolKernel.input.stream), Unresolved)
    assert point.output.binds is False  # an output never binds its kernel's extents
    assert isinstance(point.query(type(point).output.access), Inapplicable)
    assert not hasattr(PoolKernel, "input_stream")


def test_actual_pwpoly_model_remains_usable_before_pe() -> None:
    tensor = Tensor((2, 4), ScalarEncoding(DataType["FLOAT32"]))
    point = design_space(PwPolyKernel(function="gelu", input_tensor=tensor, platform=FULL_DSP58))
    assert point.input.element == tensor.element
    assert point.output_tensor == tensor
    assert isinstance(point.query(PwPolyKernel.output.stream), Unresolved)
    assert not hasattr(PwPolyKernel, "output_stream")


class SerialPool(PoolKernel):
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

    input = StreamPort(
        label="input",
        tensor=PoolKernel.input_tensor,
        schedule=first,
        index=PoolKernel.input_index,
        lanes=(c,),
        closes=(kh, kw),
        transport=Native(names=("data", "valid", "ready")),
        loop_ends=UNNAMED_ENDS,
    )
    output = StreamPort(
        label="output",
        endpoint=Endpoint.INITIATOR,
        tensor=PoolKernel.output_tensor,
        schedule=second,
        index=PoolKernel.output_index,
        lanes=(c,),
        reduces=(kh, kw),
        transport=Native(names=("result", "valid", "ready")),
        loop_ends=UNNAMED_ENDS,
    )

    @view
    def parameters(self) -> dict[str, int | str]:
        return {}


@pytest.mark.parametrize("permuted,eager", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize(
    "first,second", ((None, "ready"), ("ready", "ready"), ("ready", None), ("refused", "ready"))
)
def test_actual_pool_native_source_keeps_serial_physical_inputs_before_hls_source(
    first: str | None,
    second: str | None,
    permuted: bool,
    eager: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINN_SPACE_PERMUTE", "1" if permuted else "0")
    monkeypatch.setenv("FINN_SPACE_EAGER", "1" if eager else "0")
    point = design_space(
        SerialPool(input_tensor=TENSOR, window=(2, 2), stride=(1, 1), platform=FULL_DSP48E2)
    )
    choices: dict[str, object] = {"pe": 2}
    if first is not None:
        choices["first_state"] = first
    if second is not None:
        choices["second_state"] = second
    point = point.with_choices(**choices)
    trace = inspection.explain(point, SerialPool.netlist)
    if first is None or first == "ready" and second is None:
        assert isinstance(trace.result, Unresolved)
    elif first == "refused":
        assert isinstance(trace.result, Rejected)
        assert [f.code for f in trace.result.findings] == ["first-schedule"]
    else:
        assert isinstance(trace.result, Rejected)
        assert [f.message for f in trace.result.findings] == [
            "input: a native port named in full has no loop-completion pin",
            "output: a native port named in full has no loop-completion pin",
        ]
    visited = {n.declaration.key for n in trace.nodes}
    assert "module_sources" not in visited
    if first is None or first == "refused":
        assert not visited & {"second", "second_state", "output.stream", "output.transport"}
