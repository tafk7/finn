# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels refuse the encodings and widths their hardware does not take."""

import pytest

from finn.core.space import Rejected, Unresolved, inspection
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import eltwise, threshold_base


@pytest.mark.parametrize("name", ("INT0", "UINT0"))
def test_zero_width_encodings_refuse_across_consumers(name):
    assert isinstance(eltwise(lhs=name, rhs=name).query(EltwiseKernel.module), Rejected)
    # Refused by its admission while PE is still open.
    assert isinstance(inspection.admission(threshold_base(input_dtype=name)), Rejected)


def test_threshold_type_refusal_precedes_unrelated_configuration_choices():
    base = threshold_base(input_dtype="BIPOLAR")
    assessment = base.inspect(ThresholdingAxiKernel.module)
    assert isinstance(assessment.constraints.results["types_supported"], Rejected)
    assert isinstance(assessment.accepted_result, Unresolved)
