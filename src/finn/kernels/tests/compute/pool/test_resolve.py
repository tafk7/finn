############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Pool resolve — the windowing-geometry op (spec-anchored).

The structure no other kernel has: the input→output spatial-shape transformation
(``out = floor/ceil((in + pad − dil*(k−1) − 1)/stride + 1)``), graph-vs-derived shape
VALIDATION (a mislabeled graph output is rejected), MaxPool's optional Indices interface,
the Function-driven output-dtype rule (MaxPool equal dtypes; AvgPool signedness), and
PE = divisor(Channels). Geometry cases checked against the ONNX MaxPool spec.
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.engine.context import Context
from finn.kernels.engine.point import Illegal
from finn.kernels.compute.pool import pool_kernel
from finn.kernels.compute.pool.geometry import pool_output_dim, pool_output_spatial
from finn.kernels.compute.pool.names import POOL_HLS

C = 32  # channels


def _ctx(in_spatial, out_spatial, *, ch=C, idt="INT8", odt="INT8", indices=False):
    shapes = {"inp": (1, *in_spatial, ch), "out": (1, *out_spatial, ch)}
    datatypes = {"inp": DataType[idt], "out": DataType[odt]}
    if indices:
        shapes["indices"] = (1, *out_spatial, ch)
        datatypes["indices"] = DataType["INT64"]
    return Context(shapes=shapes, datatypes=datatypes, fpgapart="xcvc1902-vsva2197-2MP-e-S")


# --- the geometry formula (pure math, ONNX spec) ---------------------------


@pytest.mark.parametrize(
    "in_dim,k,s,pb,pe,dil,ceil,expected",
    [
        (16, 2, 2, 0, 0, 1, False, 8),
        (16, 3, 1, 0, 0, 1, False, 14),
        (16, 3, 1, 1, 1, 1, False, 16),
        (7, 2, 2, 0, 0, 1, False, 3),
        (7, 2, 2, 0, 0, 1, True, 4),
        (16, 3, 1, 0, 0, 2, False, 12),
    ],
)
def test_pool_output_dim(in_dim, k, s, pb, pe, dil, ceil, expected):
    assert pool_output_dim(in_dim, k, s, pb, pe, dil, ceil) == expected


def test_pool_output_spatial_flat_pads():
    out = pool_output_spatial((16, 16), (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), ceil_mode=False)
    assert out == (16, 16)


# --- derived output shape validated against the graph ----------------------


def test_valid_geometry_resolves():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    pt = op.configure(_ctx((16, 16), (8, 8)), {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)


def test_wrong_graph_output_shape_is_rejected():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    result = op.configure(_ctx((16, 16), (7, 7)), {"backend": POOL_HLS, "PE": 8})
    assert isinstance(result, Illegal)
    assert any("derived pooling geometry" in r for r in result.reasons)


def test_normal_and_folded_shapes():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    ctx = _ctx((16, 16), (8, 8))
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": 8})
    assert op.get_normal_input_shape(ctx, 0) == (1, 16, 16, C)
    assert op.get_normal_output_shape(ctx, 0) == (1, 8, 8, C)
    assert op.get_folded_input_shape(pt, ctx, 0) == (1, 16, 16, C // 8, 8)
    assert op.get_folded_output_shape(pt, ctx, 0) == (1, 8, 8, C // 8, 8)


@pytest.mark.parametrize("pe", [1, 4, 8, 32])
def test_stream_widths(pe):
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    ctx = _ctx((16, 16), (8, 8))
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": pe})
    assert op.get_instream_width(pt, ctx, 0) == 8 * pe
    assert op.get_outstream_width(pt, ctx, 0) == 8 * pe


@pytest.mark.parametrize("pe", [1, 4, 8, 32])
def test_exp_cycles(pe):
    op = pool_kernel(function="MaxPool", kernel_shape=(3, 3), strides=(1, 1))
    ctx = _ctx((16, 16), (14, 14))
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": pe})
    assert op.get_exp_cycles(pt, ctx) == 1 * (14 * 14) * (C * 9) // pe


# --- MaxPool optional Indices interface ------------------------------------


def test_no_indices_by_default():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    assert len(op.outputs()) == 1


def test_indices_output_present_when_requested():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2), has_indices=True)
    ctx = _ctx((16, 16), (8, 8), indices=True)
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    assert len(op.outputs()) == 2
    assert op.get_folded_output_shape(pt, ctx, 1) == (1, 8, 8, C // 8, 8)
    assert op.get_outstream_width(pt, ctx, 1) == 64 * 8  # INT64 * PE


# --- Function-driven output-dtype rule -------------------------------------


def test_maxpool_requires_equal_dtypes():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    bad = op.configure(_ctx((16, 16), (8, 8), idt="INT8", odt="INT16"), {"backend": POOL_HLS, "PE": 8})
    assert isinstance(bad, Illegal)


def test_avgpool_signedness_rule():
    op = pool_kernel(function="AveragePool", kernel_shape=(2, 2), strides=(2, 2))
    bad = op.configure(_ctx((16, 16), (8, 8), idt="INT8", odt="UINT8"), {"backend": POOL_HLS, "PE": 8})
    assert isinstance(bad, Illegal)
    ok = op.configure(_ctx((16, 16), (8, 8), idt="UINT8", odt="UINT4"), {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(ok, Illegal)


def test_non_dividing_pe_illegal():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    assert isinstance(
        op.configure(_ctx((16, 16), (8, 8)), {"backend": POOL_HLS, "PE": 5}), Illegal
    )  # 5 ∤ 32
