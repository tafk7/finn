# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public authoring behavior of FIFO, conversion and elementwise kernels."""

from __future__ import annotations

from typing import TypeVar

import pytest

from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.core.space import (
    Available,
    Param,
    QueryResult,
    Rejected,
    Space,
    Unresolved,
    design_space,
)
from finn.core.space.errors import ValueUnavailableError
from finn.kernels.target import DspBlock, Platform
from kernels.helpers import full_platform

T = TypeVar("T")


def decided(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def test_fifo_start_commit_and_view_read_keep_opaque_word_geometry() -> None:
    base = design_space(FifoKernel(word_bits=13, depth=8))
    assert isinstance(base.inspect(FifoKernel.module).accepted_result, Unresolved)
    with pytest.raises(ValueUnavailableError):
        _ = base.module
    assert base.field(FifoKernel.word_bits).get() == 13
    assert base.field(FifoKernel.word_bits).query() == Available(13)
    assert decided(base.field(FifoKernel.ram_style).state).status == "unassigned"
    chosen = base.with_choices(ram_style="auto")
    requirements = chosen.module
    assert requirements.parameters == (("DATA_WIDTH", 13), ("DEPTH", 8), ("RAM_STYLE", '"auto"'))
    assert [
        (port.name, port.direction, port.width)
        for port in requirements.pins.ports
        if isinstance(port, Signal)
    ] == [
        ("clk", Direction.IN, 1),
        ("rst", Direction.IN, 1),
        ("idat", Direction.IN, 13),
        ("ivld", Direction.IN, 1),
        ("irdy", Direction.OUT, 1),
        ("odat", Direction.OUT, 13),
        ("ovld", Direction.OUT, 1),
        ("ordy", Direction.IN, 1),
    ]
    assert chosen.inspect(FifoKernel.module).accepted_result == Available(requirements)
    assert chosen.field(FifoKernel.module).get() == requirements
    assert chosen.field(FifoKernel.module).query() == Available(requirements)
    assert chosen.query(FifoKernel.module) == Available(requirements)
    assert isinstance(base.query(FifoKernel.ram_style), Unresolved)


@pytest.mark.parametrize("style", ("auto", "shift", "distributed", "block", "ultra"))
def test_fifo_ram_styles_remain_explicit_and_preserve_native_parameter_values(style: str) -> None:
    point = design_space(FifoKernel(word_bits=17, depth=64)).with_choices(ram_style=style)
    assert dict(point.module.parameters)["RAM_STYLE"] == f'"{style}"'


@pytest.mark.parametrize(("bits", "depth"), ((0, 8), (13, 1), (1 << 32, 8), (13, 1 << 32)))
def test_fifo_geometry_refusal_remains_visible_before_and_after_ram_choice(
    bits: int, depth: int
) -> None:
    base = design_space(FifoKernel(word_bits=bits, depth=depth))
    assert base.inspect(FifoKernel.module).constraints.refused == ("geometry_supported",)
    assert isinstance(base.inspect(FifoKernel.module).accepted_result, Unresolved)
    chosen = base.with_choices(ram_style="auto")
    assert isinstance(chosen.inspect(FifoKernel.module).accepted_result, Rejected)
    with pytest.raises(ValueUnavailableError) as error:
        _ = chosen.module
    assert isinstance(error.value.result, Rejected)


def eltwise(
    *,
    operation: str = "ADD",
    pe: int = 2,
    lhs: str = "INT3",
    rhs: str = "INT3",
    scale: float = 1.0,
    target: DspBlock = DspBlock.DSP58,
) -> EltwiseKernel:
    return design_space(
        EltwiseKernel(
            operation=operation,
            pe=pe,
            lhs_dtype=resolve_qonnx_datatype_name(lhs),
            rhs_dtype=resolve_qonnx_datatype_name(rhs),
            b_scale=scale,
            platform=full_platform(target),
        )
    )


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
def test_eltwise_preserves_integer_growth_unsigned_subtraction_and_float_conversion(
    operation: str, lhs: str, rhs: str, result: str
) -> None:
    point = eltwise(operation=operation, lhs=lhs, rhs=rhs)
    requirements = point.module
    assert point.result_dtype.name == result
    widths = {port.name: port.width for port in requirements.pins.ports if isinstance(port, Signal)}
    assert widths["adat"] == 2 * resolve_qonnx_datatype_name(lhs).bitwidth()
    assert widths["bdat"] == 2 * resolve_qonnx_datatype_name(rhs).bitwidth()
    assert widths["odat"] == 2 * resolve_qonnx_datatype_name(result).bitwidth()


def test_eltwise_preserves_binary32_scale_rounding_and_source_order() -> None:
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
def test_eltwise_nonfinite_or_unrepresentable_native_scale_is_an_explicit_refusal(
    scale: float,
) -> None:
    point = eltwise(scale=scale)
    assert isinstance(point.query(EltwiseKernel.native_scale), Rejected)
    assert isinstance(point.inspect(EltwiseKernel.module).accepted_result, Rejected)


def test_eltwise_retains_supported_profile_restrictions() -> None:
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


def test_eltwise_narrow_result_stays_known_with_optional_parent_platform_omission() -> None:
    # Replaces an inline exposed Param child binding: the optional formal is the
    # parent's own, bound to the child by name.
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
