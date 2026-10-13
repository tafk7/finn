# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernels refuse the encodings and widths their hardware does not take."""

import pytest

from finn.core.space import Rejected, Unresolved, inspection
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.eltwise import EltwiseKernel
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import eltwise, threshold_base


@pytest.mark.parametrize("name", ("INT0", "UINT0"))
def test_zero_width_encodings_are_no_tensors_element(name):
    # A kernel's input encodings are its tensors': a zero-width one is refused before
    # any kernel reads it.
    with pytest.raises(ValueError, match="positive width"):
        ScalarEncoding(resolve_qonnx_datatype_name(name))


def test_an_encoding_a_kernel_does_not_take_is_refused():
    assert isinstance(eltwise(lhs="FLOAT16", rhs="FLOAT16").query(EltwiseKernel.module), Rejected)
    # Refused by its admission while PE is still open.
    assert isinstance(inspection.admission(threshold_base(input_dtype="FLOAT32")), Rejected)


def test_threshold_type_refusal_precedes_unrelated_configuration_choices():
    base = threshold_base(input_dtype="BIPOLAR")
    assessment = base.inspect(ThresholdingAxiKernel.module)
    assert isinstance(assessment.constraints.results["types_supported"], Rejected)
    assert isinstance(assessment.accepted_result, Unresolved)
