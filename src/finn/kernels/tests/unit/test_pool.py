############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""ONNX-level pooling (``ops/pool``) — the windowing-geometry stress test.

Exercises the structure no prior Kernel did: the input→output **spatial-shape
transformation** (`out = floor/ceil((in + pad − dil*(k−1) − 1)/stride + 1)`), the
graph-vs-derived shape VALIDATION, MaxPool's optional Indices output, and the
Function-driven output-dtype rule. Geometry cases checked against the ONNX spec
(https://onnx.ai/onnx/operators/onnx__MaxPool.html).
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import Context, Illegal
from finn.kernels.ops.pool import pool_kernel
from finn.kernels.ops.pool.geometry import pool_output_dim, pool_output_spatial
from finn.kernels.ops.pool.names import POOL_HLS

C = 32  # channels


def _ctx(in_spatial, out_spatial, *, ch=C, idt="INT8", odt="INT8", indices=False):
    # NHWC: (N, *spatial, C). Indices (if present) mirror the output spatial shape.
    shapes = {
        "inp": (1, *in_spatial, ch),
        "out": (1, *out_spatial, ch),
    }
    datatypes = {"inp": DataType[idt], "out": DataType[odt]}
    if indices:
        shapes["indices"] = (1, *out_spatial, ch)
        datatypes["indices"] = DataType["INT64"]
    return Context(shapes=shapes, datatypes=datatypes, fpgapart="xcvc1902-vsva2197-2MP-e-S")


# ---------------------------------------------------------------------------
# The geometry formula itself (pure math, ONNX spec).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_dim,k,s,pb,pe,dil,ceil,expected",
    [
        (16, 2, 2, 0, 0, 1, False, 8),    # classic 2x2 stride-2 halving
        (16, 3, 1, 0, 0, 1, False, 14),   # 3x3 stride-1 valid -> 14
        (16, 3, 1, 1, 1, 1, False, 16),   # 3x3 stride-1 SAME (pad 1 each side) -> 16
        (7, 2, 2, 0, 0, 1, False, 3),     # odd input, floor -> 3
        (7, 2, 2, 0, 0, 1, True, 4),      # odd input, ceil_mode -> 4
        (16, 3, 1, 0, 0, 2, False, 12),   # dilation 2: effective kernel 5 -> 12
    ],
)
def test_pool_output_dim(in_dim, k, s, pb, pe, dil, ceil, expected):
    assert pool_output_dim(in_dim, k, s, pb, pe, dil, ceil) == expected


def test_pool_output_spatial_flat_pads():
    # 2D, pads in ONNX flat form [begin_h, begin_w, end_h, end_w].
    out = pool_output_spatial((16, 16), (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), ceil_mode=False)
    assert out == (16, 16)


# ---------------------------------------------------------------------------
# The Kernel: derived output shape validated against the graph.
# ---------------------------------------------------------------------------


def test_valid_geometry_resolves():
    # 2x2 stride-2 on 16x16 -> 8x8; graph declares 8x8 -> legal.
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    pt = op.configure(_ctx((16, 16), (8, 8)), {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)


def test_wrong_graph_output_shape_is_rejected():
    # Same pool, but the graph mislabels the output as 7x7 -> the geometry predicate fires.
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
    # PE folds the channel (last) axis on both interfaces.
    assert op.get_folded_input_shape(pt, ctx, 0) == (1, 16, 16, C // 8, 8)
    assert op.get_folded_output_shape(pt, ctx, 0) == (1, 8, 8, C // 8, 8)


@pytest.mark.parametrize("pe", [1, 4, 8, 32])
def test_stream_widths(pe):
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    ctx = _ctx((16, 16), (8, 8))
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": pe})
    assert op.get_instream_width(pt, ctx, 0) == 8 * pe
    assert op.get_outstream_width(pt, ctx, 0) == 8 * pe


# ---------------------------------------------------------------------------
# Cost: the K-window reduction product (N * OutH*OutW * C*K² / PE).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pe", [1, 4, 8, 32])
def test_exp_cycles(pe):
    op = pool_kernel(function="MaxPool", kernel_shape=(3, 3), strides=(1, 1))
    ctx = _ctx((16, 16), (14, 14))  # 3x3 stride-1 valid -> 14x14
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": pe})
    expected = 1 * (14 * 14) * (C * 9) // pe
    assert op.get_exp_cycles(pt, ctx) == expected


# ---------------------------------------------------------------------------
# MaxPool's OPTIONAL Indices output — the guarded/optional interface shape.
# ---------------------------------------------------------------------------


def test_no_indices_by_default():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    assert len(op.outputs()) == 1


def test_indices_output_present_when_requested():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2), has_indices=True)
    ctx = _ctx((16, 16), (8, 8), indices=True)
    pt = op.configure(ctx, {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(pt, Illegal), getattr(pt, "reasons", None)
    assert len(op.outputs()) == 2
    # The Indices output (index 1) folds by PE and carries its own (int64) dtype.
    assert op.get_folded_output_shape(pt, ctx, 1) == (1, 8, 8, C // 8, 8)
    assert op.get_outstream_width(pt, ctx, 1) == 64 * 8  # INT64 * PE


# ---------------------------------------------------------------------------
# Function-driven output-dtype rule (op-semantic).
# ---------------------------------------------------------------------------


def test_maxpool_requires_equal_dtypes():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    bad = op.configure(_ctx((16, 16), (8, 8), idt="INT8", odt="INT16"), {"backend": POOL_HLS, "PE": 8})
    assert isinstance(bad, Illegal)


def test_avgpool_signedness_rule():
    op = pool_kernel(function="AveragePool", kernel_shape=(2, 2), strides=(2, 2))
    # signed in / unsigned out -> Illegal
    bad = op.configure(_ctx((16, 16), (8, 8), idt="INT8", odt="UINT8"), {"backend": POOL_HLS, "PE": 8})
    assert isinstance(bad, Illegal)
    # matching signedness, out may requantize narrower -> legal
    ok = op.configure(_ctx((16, 16), (8, 8), idt="UINT8", odt="UINT4"), {"backend": POOL_HLS, "PE": 8})
    assert not isinstance(ok, Illegal)


def test_non_dividing_pe_illegal():
    op = pool_kernel(function="MaxPool", kernel_shape=(2, 2), strides=(2, 2))
    assert isinstance(op.configure(_ctx((16, 16), (8, 8)), {"backend": POOL_HLS, "PE": 5}), Illegal)  # 5 ∤ 32
