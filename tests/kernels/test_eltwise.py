# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The elementwise kernel: its result type, its native scale, its sources and its refusals;
its test-side reference, and what its numeric sweep covers of its design space."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from support.helpers import eltwise, module_of
from support.kernel_specs.eltwise import computed
from support.refusals import refused

from finn.core.space import (
    Decision,
    DefinitionError,
    Rejected,
    RequestError,
    inspection,
)
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.kernels.artifacts.abi import Signal
from finn.kernels.artifacts.contributions import CopiedSource
from finn.kernels.configure import refusal
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.library import library_module
from finn.platform.capabilities import DspBlock
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
    requirements = library_module(point)
    assert point.result_dtype.name == result
    widths = {port.name: port.width for port in requirements.abi.pins if isinstance(port, Signal)}
    assert widths["adat"] == 2 * resolve_qonnx_datatype_name(lhs).bitwidth()
    assert widths["bdat"] == 2 * resolve_qonnx_datatype_name(rhs).bitwidth()
    assert widths["odat"] == 2 * resolve_qonnx_datatype_name(result).bitwidth()


def test_eltwise_rounds_its_scale_to_binary32_and_lists_its_sources_in_order() -> None:
    integer = eltwise(scale=1.0 + 2**-30)
    assert integer.native_scale == 1.0
    requirements = library_module(integer)
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
    assert dict(library_module(floating).parameters)["B_SCALE"] == "0.25"


@pytest.mark.parametrize("scale", (1e100, float("inf")))
def test_eltwise_refuses_a_nonfinite_or_unrepresentable_native_scale(
    scale: float,
) -> None:
    """The scale is refused on its own value, and the kernel through it, at the origin,
    its PE still open (the kernel's rule reads facts alone). Setting PE breaks no rule
    that was not broken before, so it is accepted, and the kernel stays refused."""
    point = eltwise(scale=scale, pe=None)
    assert isinstance(point.query(EltwiseKernel.native_scale), Rejected)
    assert isinstance(dict(inspection.states(point))["pe"], inspection.Open)
    assert refused_by(point) == {("native_scale", "eltwise-scale")}
    report = point.try_with_choices(pe=2)
    assert report.accepted
    assert refused_by(report.instance) == {("native_scale", "eltwise-scale")}


def test_a_nan_scale_is_refused_where_it_enters_not_by_the_kernel() -> None:
    """NaN is no value (``==`` does not hold it equal to itself): refused where it enters
    the design space, naming the fact, given at the call or as a change of facts. The
    kernel's own ``eltwise-scale`` never reads it."""
    with refused("literal-not-value", error=DefinitionError, message="NaN is no value"):
        eltwise(scale=float("nan"))
    point = eltwise(scale=1.0, pe=None)
    with pytest.raises(RequestError, match="b_scale: NaN is no value"):
        point.with_facts({EltwiseKernel.b_scale: float("nan")})


def refused_by(point: EltwiseKernel) -> set[tuple[str, str]]:
    """The owner and code of each finding that refuses ``point``'s configuration (its
    validity's false pieces)."""
    return {(finding.owner, finding.code) for finding in refusal(inspection.validity(point))}


@pytest.mark.parametrize(
    ("profile", "code"),
    (
        ({"operation": "DIV"}, "eltwise-operation"),
        ({"lhs": "INT4"}, "eltwise-integer-pair"),
        ({"lhs": "BIPOLAR"}, "eltwise-integer-pair"),
        ({"scale": 0.5}, "eltwise-scale"),
        ({"operation": "MUL", "lhs": "FLOAT32", "scale": 0.5}, "eltwise-scale"),
        ({"lhs": "FLOAT32", "target": DspBlock.DSP48E2}, "eltwise-target"),
    ),
)
def test_eltwise_refuses_unsupported_profiles(profile: dict[str, Any], code: str) -> None:
    """A profile the RTL does not implement is refused on its facts, its PE still open:
    the kernel's rule reads facts alone, so the kernel reads refused from the start,
    naming it. Setting PE breaks no rule that was not broken before, so it is accepted,
    and the kernel stays refused by the same rule; a PE outside its domain is refused by
    the domain."""
    with pytest.raises(ValueError, match="domain-membership"):
        eltwise(pe=0)
    point = eltwise(pe=None, **profile)
    assert isinstance(dict(inspection.states(point))["pe"], inspection.Open)
    assert inspection.validity(point).status == "refused"
    # BIPOLAR is no ordinary integer either: operands_supported refuses it beside.
    assert ("implementation_supported", code) in refused_by(point)
    report = point.try_with_choices(pe=2)
    assert report.accepted
    assert refused_by(report.instance) == refused_by(point)


def test_eltwise_pe_is_a_divisor_of_the_channels_its_parameter_holds() -> None:
    """PE's domain lists the channels' divisors, and admits none past the RTL's PE
    parameter, a 32-bit integer: a type, not a rule."""
    point = eltwise(pe=None)
    (pe,) = inspection.viable(point, [EltwiseKernel.pe])
    assert pe.values is not None and [item.value for item in pe.values] == [1, 2, 3, 6]
    declared = EltwiseKernel.__dict__["pe"]
    assert isinstance(declared, Decision)
    assert declared.domain.accepts(candidate=2**31, extent=2**32) is True
    assert declared.domain.accepts(candidate=2**32, extent=2**33) is False


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
        assert not isinstance(module_of(point, eltwise_numeric.stem(case)), Rejected), case.label
