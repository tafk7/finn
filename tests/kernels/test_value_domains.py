# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernels' datatype refusals (``finn.kernels.values.domains``)."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType

from finn.core.space import Rejected
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.values.domains import admit_element


def test_an_admitted_element_is_the_encoding() -> None:
    assert admit_element(DataType["INT4"], (-7, 7)) == ScalarEncoding(DataType["INT4"], (-7, 7))
    assert admit_element(DataType["FLOAT32"]) == ScalarEncoding(DataType["FLOAT32"])


@pytest.mark.parametrize(
    ("name", "bounds"),
    (("INT3", (-5, 3)), ("INT3", (2, 1)), ("UINT4", (0, 16)), ("FLOAT32", (0, 1))),
)
def test_an_element_the_datatype_cannot_hold_is_refused_as_storage(
    name: str, bounds: tuple[int, int]
) -> None:
    refused = admit_element(DataType[name], bounds)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"dtype-storage"}
