# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels refuse the encodings and widths their hardware does not take."""

from support.helpers import eltwise, threshold_base

from finn.core.space import Rejected, inspection
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.thresholding import ThresholdingAxiKernel


def test_an_encoding_a_kernel_does_not_take_is_refused():
    assert isinstance(eltwise(lhs="FLOAT16", rhs="FLOAT16").query(EltwiseKernel.netlist), Rejected)
    # Refused by its validity while PE is still open.
    assert inspection.validity(threshold_base(input_dtype="FLOAT32")).status == "refused"


def test_threshold_type_refusal_precedes_unrelated_configuration_choices():
    base = threshold_base(input_dtype="BIPOLAR")
    validity = inspection.validity(base)
    assert validity.status == "refused"
    assert "types_supported" in {piece.owner for piece in validity.pieces}
    assessment = base.inspect(ThresholdingAxiKernel.netlist)
    assert isinstance(assessment.validity, Rejected)
    assert isinstance(assessment.accepted_result, Rejected)
