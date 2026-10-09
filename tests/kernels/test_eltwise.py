# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The elementwise kernel: its result type, its native scale, its sources and its refusals;
its test-side reference, and what its numeric sweep covers of its design space."""

from __future__ import annotations

import numpy as np
import pytest

from finn.core.space import Available, Param, Rejected, Space, Unresolved, design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.target import DspBlock, Platform
from kernels.helpers import eltwise
from kernels.specs.eltwise import computed
from kernels.sweeps import eltwise_numeric


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
    widths = {port.name: port.width for port in requirements.abi.pins if isinstance(port, Signal)}
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


def test_the_reference_computes_each_operation_with_its_operand_in_turn() -> None:
    lhs, rhs = np.array([[0, 7], [3, 1]]), np.array([7, 2])  # rhs broadcast over the rows
    assert computed("ADD", lhs, rhs).tolist() == [[7, 9], [10, 3]]
    assert computed("SUB", lhs, rhs).tolist() == [[-7, 5], [-4, -1]]
    assert computed("SBR", lhs, rhs).tolist() == [[7, -5], [4, 1]]
    assert computed("MUL", lhs, rhs).tolist() == [[0, 14], [21, 2]]


def test_the_reference_scales_and_rounds_in_binary32() -> None:
    """rhs scaled first, then the operation rounded: 1 - 0.5 * 2**26 is 1 - 2**25, halfway
    between two binary32 neighbours, rounded to the even one, -2**25."""
    rhs = np.array([2.0**26], dtype=np.float32)
    found = computed("SUB", np.array([1]), rhs, scale=0.5)
    assert found.dtype == np.float32 and found.tolist() == [-(2.0**25)]
    with pytest.raises(ValueError, match="2\\*\\*24"):
        computed("ADD", np.array([(1 << 24) + 1]), rhs)
    with pytest.raises(ValueError, match="float arithmetic only"):
        computed("ADD", np.array([1]), np.array([1]), scale=0.5)


def test_the_numeric_sweep_covers_each_operation_a_broadcast_and_float_arithmetic() -> None:
    """Lean (decision KT10): every operation the kernel implements, signed and unsigned
    integers, an rhs of fewer axes, both float paths (an integer converted, and FLOAT32
    alone), a scale, and PE at 1 and the whole innermost extent. Each case places."""
    cases = eltwise_numeric.CASES
    assert {case.operation for case in cases} == {"ADD", "SUB", "SBR", "MUL"}
    assert {case.lhs.startswith("U") for case in cases if case.lhs != "FLOAT32"} == {True, False}
    assert any(len(case.rhs_shape) < len(case.lhs_shape) for case in cases)
    floating = {(case.lhs, case.rhs) for case in cases if "FLOAT32" in (case.lhs, case.rhs)}
    assert ("FLOAT32", "FLOAT32") in floating and len(floating) == 2
    assert any(case.scale != 1.0 for case in cases)
    assert {1, 8} <= {case.pe for case in cases if case.lhs_shape[-1] == 8}
    for case in cases:
        point = eltwise_numeric.placed(case)
        assert not isinstance(point.query(type(point).module), Rejected), case.label
