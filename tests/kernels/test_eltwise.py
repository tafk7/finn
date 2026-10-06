# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The elementwise kernel: its result type, its native scale, its sources and its refusals."""

from __future__ import annotations

import pytest

from finn.core.space import Available, Param, Rejected, Space, Unresolved, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.target import DspBlock, Platform
from kernels.helpers import eltwise


@pytest.mark.parametrize(
    ("operation", "lhs", "rhs", "result"),
    (
        ("ADD", "INT3", "INT3", "INT4"),
        ("ADD", "UINT3", "UINT3", "UINT4"),
        ("SUB", "UINT3", "UINT3", "INT4"),
        ("SBR", "UINT3", "UINT3", "INT4"),
        ("MUL", "UINT3", "UINT3", "UINT6"),
        ("MUL", "INT3", "INT3", "INT6"),
        ("ADD", "INT9", "FLOAT32", "FLOAT32"),
    ),
)
def test_eltwise_result_grows_for_integers_and_converts_to_float(
    operation: str, lhs: str, rhs: str, result: str
) -> None:
    point = eltwise(operation=operation, lhs=lhs, rhs=rhs)
    requirements = point.module
    assert point.result_dtype.name == result
    widths = {port.name: port.width for port in requirements.pins.pins if isinstance(port, Signal)}
    assert widths["adat"] == 2 * resolve_qonnx_datatype_name(lhs).bitwidth()
    assert widths["bdat"] == 2 * resolve_qonnx_datatype_name(rhs).bitwidth()
    assert widths["odat"] == 2 * resolve_qonnx_datatype_name(result).bitwidth()


def test_eltwise_rounds_its_scale_to_binary32_and_lists_its_sources_in_order() -> None:
    integer = eltwise(scale=1.0 + 2**-30)
    assert integer.native_scale == 1.0
    requirements = integer.module
    assert dict(requirements.parameters)["B_SCALE"] == "1.0"
    assert all(isinstance(source, CopiedSource) for source in requirements.sources)
    assert [source.path for source in requirements.sources if isinstance(source, CopiedSource)] == [
        "rtl/arith/binopi.sv",
        "rtl/arith/binopf.sv",
        "rtl/arith/int_to_fp32.sv",
        "rtl/infra/fifo.sv",
        "rtl/arith/eltwise.sv",
    ]
    floating = eltwise(lhs="FLOAT32", rhs="FLOAT32", scale=0.25)
    assert dict(floating.module.parameters)["B_SCALE"] == "0.25"


@pytest.mark.parametrize("scale", (1e100, float("inf"), float("nan")))
def test_eltwise_refuses_a_nonfinite_or_unrepresentable_native_scale(
    scale: float,
) -> None:
    point = eltwise(scale=scale)
    assert isinstance(point.query(EltwiseKernel.native_scale), Rejected)
    assert isinstance(point.inspect(EltwiseKernel.module).accepted_result, Rejected)


def test_eltwise_refuses_unsupported_profiles() -> None:
    refused = (
        eltwise(pe=0),
        eltwise(operation="DIV"),
        eltwise(lhs="INT4"),
        eltwise(lhs="BIPOLAR"),
        eltwise(scale=0.5),
        eltwise(operation="MUL", lhs="FLOAT32", scale=0.5),
        eltwise(lhs="FLOAT32", target=DspBlock.DSP48E2),
    )
    assert all(
        isinstance(point.inspect(EltwiseKernel.module).accepted_result, Rejected)
        for point in refused
    )


def test_eltwise_result_is_known_while_an_optional_parent_platform_is_unset() -> None:
    # The optional formal is the parent's own, bound to the child by name.
    class OptionalPlatform(Space):
        platform: Platform = Param(required=False)
        arithmetic = EltwiseKernel(
            operation="ADD",
            pe=2,
            lhs_dtype=resolve_qonnx_datatype_name("INT3"),
            rhs_dtype=resolve_qonnx_datatype_name("INT3"),
            b_scale=1.0,
            platform=platform,
        )

    point = design_space(OptionalPlatform())
    assert point.arithmetic.result_dtype.name == "INT4"
    assessment = point.arithmetic.inspect(EltwiseKernel.module)
    assert isinstance(assessment.output_result, Available)
    assert isinstance(assessment.accepted_result, Unresolved)
