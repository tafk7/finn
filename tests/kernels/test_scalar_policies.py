# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatype choices, supplied encodings, and partial rejection share one policy."""

import pytest

from finn.core.space import Available, Decision, Param, Rejected, Space, Unresolved, design_space
from finn.kernels.datatypes.domains import Integer, SignedInteger
from finn.kernels.datatypes.scalar import integer_scalar
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.datatypes import QONNXDataType
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dtype
from finn.kernels.dotp import DotpAxiKernel
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.int_to_fp32 import IntToFp32Kernel
from finn.kernels.memstream_hls import MemStreamHlsKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.test_dotp import kernel as dotp
from kernels.test_migrated_rich import threshold_base
from kernels.test_migrated_simple import eltwise


class Precision(Space):
    limit: int = Param()
    dtype: QONNXDataType = Decision(
        domain=SignedInteger(2, limit).domain(), semantics=QONNX_DATATYPE_VALUE_SEMANTICS
    )
    scalar = integer_scalar(dtype, SignedInteger(2, limit))


def test_dtype_choice_and_scalar_admission_share_dynamic_limits():
    base = design_space(Precision(limit=5))
    candidates = base.field(Precision.dtype).candidates()
    assert isinstance(candidates, Available)
    assert [value.name for value in candidates.value] == ["INT2", "INT3", "INT4", "INT5"]
    assert isinstance(base.query(Precision.scalar.encoding), Unresolved)
    for name in ("INT2", "INT5"):
        selected = base.with_choices(dtype=dtype(name))
        encoding = selected.query(Precision.scalar.encoding)
        assert isinstance(encoding, Available)
        assert encoding.value.dtype.name == name
        assert encoding.value.bits == int(name[3:])
    for name in ("INT1", "INT6", "UINT3", "TERNARY", "FLOAT32"):
        report = base.try_with_choices(dtype=dtype(name))
        assert not report.accepted
        assert report.instance is base


def test_invalid_bound_is_a_domain_refusal():
    base = design_space(Precision(limit=0))
    assert not base.try_with_choices(dtype=dtype("INT3")).accepted


def test_scalar_encoding_detaches_qonnx_values():
    selected = design_space(Precision(limit=5)).with_choices(dtype=dtype("INT3"))
    first = selected.scalar.encoding
    mutable = first.dtype
    mutable._bitwidth = 100
    assert first.bits == 3
    assert selected.scalar.encoding.bits == 3


@pytest.mark.parametrize("name", ("INT0", "UINT0", "BIPOLAR", "TERNARY", "FLOAT16", "INT129"))
def test_converter_unsupported_encodings_refuse_without_pin_construction_errors(name):
    point = design_space(IntToFp32Kernel(input_dtype=dtype(name)))
    assert isinstance(point.query(IntToFp32Kernel.build_requirements), Rejected)


@pytest.mark.parametrize("name", ("INT0", "UINT0"))
def test_zero_width_encodings_refuse_across_consumers(name):
    assert isinstance(eltwise(lhs=name, rhs=name).query(EltwiseKernel.build_requirements), Rejected)
    assert isinstance(
        design_space(MemStreamHlsKernel(element_dtype=dtype(name), depth=3)).query(
            MemStreamHlsKernel.build_requirements
        ),
        Rejected,
    )
    assert isinstance(
        threshold_base(input_dtype=name)
        .with_choices(use_axilite=False, deep_pipeline=False)
        .query(ThresholdingAxiKernel.build_requirements),
        Rejected,
    )


def test_threshold_type_refusal_precedes_unrelated_configuration_choices():
    base = threshold_base(input_dtype="BIPOLAR")
    assessment = base.inspect(ThresholdingAxiKernel.build_requirements)
    assert isinstance(assessment.constraints.results["types_supported"], Rejected)
    assert isinstance(assessment.accepted_result, Unresolved)


@pytest.mark.parametrize("field", ("pe", "simd"))
def test_dotp_rejects_native_parameter_overflow(field):
    assert isinstance(dotp(**{field: 2**32}).query(DotpAxiKernel.build_requirements), Rejected)


def test_dotp_rejects_packed_width_overflow_even_when_dimensions_fit():
    assert isinstance(dotp(pe=2**30, simd=2).query(DotpAxiKernel.build_requirements), Rejected)


def test_integer_policy_check_agrees_with_domain_admission():
    class Types(Space):
        value: QONNXDataType = Decision(
            domain=Integer(1, 4).domain(), semantics=QONNX_DATATYPE_VALUE_SEMANTICS
        )

    base = design_space(Types())
    for name in ("BINARY", "INT1", "INT4", "UINT4", "INT5", "INT0", "TERNARY"):
        offered = dtype(name)
        assert base.try_with_choices(value=offered).accepted is (
            Integer(1, 4).check(offered) is True
        )
