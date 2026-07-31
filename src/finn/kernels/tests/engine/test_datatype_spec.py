############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""resolve_datatype_spec — the DatatypeSpec union resolver (derivation half)."""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.datatype_spec import (
    VALUE_OPTIMIZED,
    resolve_datatype_spec,
    value_optimized,
)
from finn.kernels.engine.point import Point


def _point():
    return Point({})


def _ctx(**kw):
    return Context(**kw)


def test_none_falls_back_to_graph_dtype():
    ctx = _ctx(datatypes={"out": DataType["INT32"]})
    dt = resolve_datatype_spec(None, iface="out", point=_point(), context=ctx)
    assert dt == DataType["INT32"]


def test_fixed_datatype_used_as_is():
    ctx = _ctx(datatypes={"out": DataType["INT32"]})
    dt = resolve_datatype_spec(DataType["FLOAT32"], iface="out", point=_point(), context=ctx)
    assert dt == DataType["FLOAT32"]


def test_str_copies_named_interface_dtype():
    ctx = _ctx(datatypes={"inp": DataType["INT8"], "out": DataType["INT32"]})
    dt = resolve_datatype_spec("inp", iface="out", point=_point(), context=ctx)
    assert dt == DataType["INT8"]


def test_value_optimized_narrows_from_static_values():
    # weights in [-3, 2] → smallest signed type that fits (INT3), narrower than INT8.
    w = np.array([[-3, 2], [1, 0]], dtype=np.float32)
    ctx = _ctx(datatypes={"weights": DataType["INT8"]}, initializers={"weights": w})
    dt = resolve_datatype_spec(
        VALUE_OPTIMIZED, iface="weights", point=_point(), context=ctx
    )
    assert dt.bitwidth() < DataType["INT8"].bitwidth()
    assert dt.min() <= -3 and dt.max() >= 2


def test_value_optimized_falls_back_to_graph_dtype_when_dynamic():
    # No initializer ⇒ dynamic ⇒ graph dtype (no narrowing).
    ctx = _ctx(datatypes={"weights": DataType["INT8"]})
    dt = resolve_datatype_spec(
        VALUE_OPTIMIZED, iface="weights", point=_point(), context=ctx
    )
    assert dt == DataType["INT8"]


def test_value_optimized_unsigned_range():
    w = np.array([0, 5, 3], dtype=np.float32)
    ctx = _ctx(datatypes={"w": DataType["UINT8"]}, initializers={"w": w})
    dt = value_optimized("w")(_point(), ctx)
    assert dt.min() >= 0 and dt.max() >= 5
    assert dt.bitwidth() < DataType["UINT8"].bitwidth()


def test_callable_spec_is_called_with_point_and_context():
    ctx = _ctx(datatypes={"out": DataType["INT32"]})
    spec = lambda p, c: DataType["UINT16"]
    dt = resolve_datatype_spec(spec, iface="out", point=_point(), context=ctx)
    assert dt == DataType["UINT16"]


def test_invalid_spec_raises():
    ctx = _ctx(datatypes={"out": DataType["INT32"]})
    with pytest.raises(ValueError):
        resolve_datatype_spec(123, iface="out", point=_point(), context=ctx)
