############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""DatatypeSupport.accepts — the declarative per-port datatype gate (category + bits)."""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport


def test_integer_kind_accepts_integer_rejects_float():
    s = DatatypeSupport(kind=DatatypeKind.INTEGER)
    assert s.accepts(DataType["INT8"]) is None
    assert s.accepts(DataType["UINT4"]) is None
    assert s.accepts(DataType["FLOAT32"]) is not None


def test_float_kind_accepts_float_rejects_integer():
    s = DatatypeSupport(kind=DatatypeKind.FLOAT)
    assert s.accepts(DataType["FLOAT32"]) is None
    assert s.accepts(DataType["INT8"]) is not None


def test_any_kind_accepts_any_category_within_bits():
    s = DatatypeSupport(kind=DatatypeKind.ANY, min_bits=1, max_bits=32)
    assert s.accepts(DataType["INT8"]) is None
    assert s.accepts(DataType["FLOAT32"]) is None


def test_bitwidth_range_gates_first():
    s = DatatypeSupport(kind=DatatypeKind.INTEGER, min_bits=4, max_bits=8)
    assert s.accepts(DataType["INT4"]) is None
    assert s.accepts(DataType["INT8"]) is None
    # in-category but out of bit range → rejected with a bitwidth reason.
    reason = s.accepts(DataType["INT16"])
    assert reason is not None and "bitwidth" in reason


def test_invalid_ranges_rejected_at_construction():
    with pytest.raises(ValueError):
        DatatypeSupport(min_bits=0)
    with pytest.raises(ValueError):
        DatatypeSupport(min_bits=8, max_bits=4)
