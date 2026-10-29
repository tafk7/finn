# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical logical plans stay usable without timing; physical coverage stays explicit."""

from __future__ import annotations

import pytest

from finn.core.space import Rejected, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.logical import (
    Available,
    Edge,
    LogicalPlan,
    Node,
    Port,
    Source,
    adapters,
    packing,
)
from finn.dataflow.profiles import Attachment, TimedPlan, Unavailable, readiness
from finn.dataflow.steps import Step, Steps, Unrealizable, steps
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, Grouped, LevelDone, Loop, vector_major
from finn.kernels.adapters import realize
from finn.kernels.port import stream_refusal
from finn.kernels.vpc import VpcKernel

ELEMENT = ScalarEncoding(resolve_qonnx_datatype_name("INT8"))
TENSOR = Tensor((2, 6), ELEMENT)


def test_untimed_logical_width_requirement_has_a_real_hardware_recipe() -> None:
    source = BeatSequence(vector_major(TENSOR.shape, 2))
    sink = BeatSequence(vector_major(TENSOR.shape, 3))
    plan = LogicalPlan(
        "width",
        (Node("producer"), Node("consumer")),
        (
            Port("out", "producer", "write", "value", TENSOR, source),
            Port("in", "consumer", "read", "value", TENSOR, sink),
        ),
        (Edge("edge", "out", "in"),),
        (Source("value", TENSOR),),
    )
    packed = packing(plan, "in")
    assert isinstance(packed, Available)
    assert (packed.value.payload_bits, packed.value.beats_per_pass) == (24, 4)
    requirement = adapters(plan, "edge")
    assert isinstance(requirement, Available)
    assert isinstance(requirement.value, Steps)
    assert requirement.value.kinds == (Step.WIDTH,)
    (recipe,) = realize(requirement.value)
    assert recipe.kind == "vpc"
    point = design_space(VpcKernel(input_tensor=TENSOR, arriving=source, leaving=sink))
    assert point.parameters["PI"] == 2 and point.parameters["PO"] == 3
    assert isinstance(readiness(TimedPlan(plan, Attachment(plan.digest))), Unavailable)
    # Packing and the physical recipe were available; no clock/rate was invented.


@pytest.mark.parametrize("unsupported", ("grouped", "held-marker"))
def test_logically_supported_forms_do_not_silently_add_physical_capability(
    unsupported: str,
) -> None:
    if unsupported == "grouped":
        tensor = Tensor((6, 2), ELEMENT)
        source = BeatSequence(Grouped(tensor.shape, (Loop(2, 1), Loop(6, 2)), 4))
        sink = BeatSequence(vector_major(tensor.shape, 2))
    else:
        tensor = TENSOR
        source = BeatSequence(vector_major(tensor.shape, 2).repeated(2))
        sink = BeatSequence(source.form, markers=(LevelDone(6, 3),))
    logical = steps(source, sink)
    with pytest.raises(Unrealizable, match="Grouped|held LevelDone"):
        realize(logical)
    point = design_space(VpcKernel(input_tensor=tensor, arriving=source, leaving=sink))
    port = point.input if unsupported == "grouped" else point.output
    result = port.query(type(port).stream)
    assert isinstance(result, Rejected)
    assert any(
        f.code in {"port-presentation-unsupported", "port-marker-unsupported"}
        for f in result.findings
    )
    # Direct pure protocol helpers enforce the same boundary, not just native authoring.
    from finn.dataflow.streams import Stream  # noqa: PLC0415

    assert isinstance(
        stream_refusal(Stream(ELEMENT, source if unsupported == "grouped" else sink)), Rejected
    )
