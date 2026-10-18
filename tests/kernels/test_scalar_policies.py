# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels refuse the encodings and widths their hardware does not take."""

from support.helpers import eltwise, threshold_base

from finn.core.space import Rejected, inspection
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.thresholding import ThresholdingAxiKernel


def test_an_encoding_a_kernel_does_not_take_is_refused():
    assert isinstance(eltwise(lhs="FLOAT16", rhs="FLOAT16").query(EltwiseKernel.module), Rejected)
    # Refused by its validity while PE is still open.
    assert inspection.validity(threshold_base(input_dtype="FLOAT32")).status == "refused"


def test_threshold_type_refusal_precedes_unrelated_configuration_choices():
    base = threshold_base(input_dtype="BIPOLAR")
    assessment = base.inspect(ThresholdingAxiKernel.module)
    assert isinstance(assessment.constraints.results["types_supported"], Rejected)
    assert isinstance(assessment.accepted_result, Rejected)
