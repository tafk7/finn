# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public authoring behavior of FIFO, conversion and elementwise kernels."""

from __future__ import annotations

from typing import TypeVar

import pytest

from finn.kernels.artifacts.abi import Direction, Signal
from finn.kernels.artifacts.contribution_types import CopiedSource
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.fifo import FifoKernel
from finn.kernels.int_to_fp32 import IntToFp32Kernel
from finn.core.space import QueryResult, Available, Param, Rejected, Space, Subspace, Unresolved
from finn.core.space.errors import RequestError, ValueUnavailableError
from finn.kernels.target import DspBlock

T = TypeVar("T")


def decided(answer: QueryResult[T]) -> T:
    assert isinstance(answer, Available), answer
    return answer.value


def test_fifo_start_commit_and_callable_view_keep_opaque_word_geometry() -> None:
    base = FifoKernel(word_bits=13, depth=8)
    assert isinstance(base.build_requirements.inspect().accepted_result, Unresolved)
    with pytest.raises(ValueUnavailableError):
        base.build_requirements()
    assert base.field(FifoKernel.word_bits).get() == 13
    assert base.field(FifoKernel.word_bits).query() == Available(13)
    assert decided(base.field(FifoKernel.ram_style).state).status == "unassigned"
    chosen = base.with_choices(ram_style="auto")
    requirements = chosen.build_requirements()
    assert requirements.parameters == (("DATA_WIDTH", 13), ("DEPTH", 8), ("RAM_STYLE", '"auto"'))
    assert [
        (port.name, port.direction, port.width)
        for port in requirements.abi.ports
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
    assert chosen.inspect(FifoKernel.build_requirements) == chosen.build_requirements.inspect()
    assert chosen.view(FifoKernel.build_requirements)() == requirements
    assert chosen.build_requirements.query() == Available(requirements)
    assert isinstance(base.query(FifoKernel.ram_style), Unresolved)


@pytest.mark.parametrize("style", ("auto", "shift", "distributed", "block", "ultra"))
def test_fifo_ram_styles_remain_explicit_and_preserve_native_parameter_values(style: str) -> None:
    point = FifoKernel(word_bits=17, depth=64).with_choices(ram_style=style)
    assert dict(point.build_requirements().parameters)["RAM_STYLE"] == f'"{style}"'


@pytest.mark.parametrize(("bits", "depth"), ((0, 8), (13, 1), (1 << 32, 8), (13, 1 << 32)))
def test_fifo_geometry_refusal_remains_visible_before_and_after_ram_choice(
    bits: int, depth: int
) -> None:
    base = FifoKernel(word_bits=bits, depth=depth)
    assert base.build_requirements.inspect().constraints.refused == ("geometry_supported",)
    assert isinstance(base.build_requirements.inspect().accepted_result, Unresolved)
    chosen = base.with_choices(ram_style="auto")
    assert isinstance(chosen.build_requirements.inspect().accepted_result, Rejected)
    with pytest.raises(ValueUnavailableError) as error:
        chosen.build_requirements()
    assert isinstance(error.value.result, Rejected)


@pytest.mark.parametrize(
    ("name", "width", "signed"), (("INT9", 9, 1), ("BINARY", 1, 0), ("INT128", 128, 1))
)
def test_converter_has_only_native_combinational_pins(name: str, width: int, signed: int) -> None:
    point = IntToFp32Kernel(input_dtype=resolve_qonnx_datatype_name(name))
    assert point.result_dtype.name == "FLOAT32"
    requirements = point.build_requirements()
    assert requirements.parameters == (("SIGNED", signed), ("WIDTH", width))
    assert [
        (port.name, port.direction, port.width)
        for port in requirements.abi.ports
        if isinstance(port, Signal)
    ] == [("ival", Direction.IN, width), ("fval", Direction.OUT, 32)]


@pytest.mark.parametrize("name", ("FLOAT32", "BIPOLAR", "TERNARY", "INT129"))
def test_converter_refuses_unsupported_encodings_and_widths(name: str) -> None:
    point = IntToFp32Kernel(input_dtype=resolve_qonnx_datatype_name(name))
    assert isinstance(point.build_requirements.inspect().accepted_result, Rejected)


def test_required_root_inputs_fail_binding_and_optional_parent_exposure_keeps_partial_read() -> (
    None
):
    with pytest.raises(RequestError):
        FifoKernel(word_bits=13)
    with pytest.raises(RequestError):
        IntToFp32Kernel()

    class OptionalConverter(Space):
        converter = Subspace(
            IntToFp32Kernel,
            input_dtype=Param(QONNX_DATATYPE_VALUE_SEMANTICS, required=False),
        )

    point = OptionalConverter()
    assert point.converter.result_dtype.name == "FLOAT32"
    assert isinstance(point.converter.build_requirements.inspect().accepted_result, Unresolved)


def eltwise(
    *,
    operation: str = "ADD",
    pe: int = 2,
    lhs: str = "INT3",
    rhs: str = "INT3",
    scale: float = 1.0,
    target: DspBlock = DspBlock.DSP58,
) -> EltwiseKernel:
    return EltwiseKernel(
        {
            EltwiseKernel.operation: operation,
            EltwiseKernel.pe: pe,
            EltwiseKernel.lhs_dtype: resolve_qonnx_datatype_name(lhs),
            EltwiseKernel.rhs_dtype: resolve_qonnx_datatype_name(rhs),
            EltwiseKernel.b_scale: scale,
            EltwiseKernel.target_dsp: target,
        }
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
    requirements = point.build_requirements()
    assert point.result_dtype.name == result
    widths = {port.name: port.width for port in requirements.abi.ports if isinstance(port, Signal)}
    assert widths["adat"] == 2 * resolve_qonnx_datatype_name(lhs).bitwidth()
    assert widths["bdat"] == 2 * resolve_qonnx_datatype_name(rhs).bitwidth()
    assert widths["odat"] == 2 * resolve_qonnx_datatype_name(result).bitwidth()


def test_eltwise_preserves_binary32_scale_rounding_and_source_order() -> None:
    integer = eltwise(scale=1.0 + 2**-30)
    assert integer.native_scale == 1.0
    requirements = integer.build_requirements()
    assert dict(requirements.parameters)["B_SCALE"] == "1.0"
    assert all(isinstance(source, CopiedSource) for source in requirements.contributions)
    assert [
        source.path for source in requirements.contributions if isinstance(source, CopiedSource)
    ] == [
        "rtl/binopi.sv",
        "rtl/binopf.sv",
        "rtl/int_to_fp32.sv",
        "rtl/queue.sv",
        "rtl/eltwise.sv",
    ]
    floating = eltwise(lhs="FLOAT32", rhs="FLOAT32", scale=0.25)
    assert dict(floating.build_requirements().parameters)["B_SCALE"] == "0.25"


@pytest.mark.parametrize("scale", (1e100, float("inf"), float("nan")))
def test_eltwise_nonfinite_or_unrepresentable_native_scale_is_an_explicit_refusal(
    scale: float,
) -> None:
    point = eltwise(scale=scale)
    assert isinstance(point.query(EltwiseKernel.native_scale), Rejected)
    assert isinstance(point.build_requirements.inspect().accepted_result, Rejected)


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
        isinstance(point.build_requirements.inspect().accepted_result, Rejected)
        for point in refused
    )


def test_eltwise_narrow_result_stays_known_with_optional_parent_target_omission() -> None:
    class OptionalTarget(Space):
        arithmetic = Subspace(
            EltwiseKernel,
            operation="ADD",
            pe=2,
            lhs_dtype=resolve_qonnx_datatype_name("INT3"),
            rhs_dtype=resolve_qonnx_datatype_name("INT3"),
            b_scale=1.0,
            target_dsp=Param(DspBlock, required=False),
        )

    point = OptionalTarget()
    assert point.arithmetic.result_dtype.name == "INT4"
    assessment = point.arithmetic.build_requirements.inspect()
    assert isinstance(assessment.output_result, Available)
    assert isinstance(assessment.accepted_result, Unresolved)
