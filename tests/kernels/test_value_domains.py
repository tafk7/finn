# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernels' datatype refusals (``finn.kernels.values.domains``)."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange

from finn.core.space import Rejected
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.values.domains import admit_element


def test_an_admitted_element_is_the_encoding() -> None:
    narrow = TensorRange(-7, 7)
    assert admit_element(DataType["INT4"], narrow) == ScalarEncoding(DataType["INT4"], narrow)
    assert admit_element(DataType["FLOAT32"]) == ScalarEncoding(DataType["FLOAT32"])


@pytest.mark.parametrize(
    ("name", "stated"),
    (
        ("INT3", TensorRange(-5, 3)),
        ("UINT4", TensorRange(0, 16)),
        ("INT8", TensorRange(0, 1, nan=True)),
    ),
)
def test_an_element_the_datatype_cannot_hold_is_refused_as_storage(
    name: str, stated: TensorRange
) -> None:
    refused = admit_element(DataType[name], stated)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-storage"}
