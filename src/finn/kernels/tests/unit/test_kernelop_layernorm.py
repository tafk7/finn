############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""LayerNorm — the Tier-3 (estimate-only) Kernel worked example
(kernelop-tensor-block-stream.md §4, §7).

The cleanest op to prove the vertical slice: 1-in / 1-out, last-axis (channel) fold by
SIMD, no weights, no reduction-across-interfaces. The op declares interfaces + shape/
dtype; each Backend owns the ``{input: "SIMD", output: "SIMD"}`` tiling. We
validate the projected getters against FINN's own LayerNorm formulas
(``src/finn/custom_op/fpgadataflow/layernorm.py``):

  normal_input_shape = ifm_dim
  folded_input_shape = ishape[:-1] + [n/SIMD, SIMD]
  instream_width     = i_bits * SIMD
  outstream_width    = o_bits * SIMD

The rough op-level ``get_exp_cycles`` is ``prod(ifm_dim)//SIMD`` — FINN's rtl leaf adds
a ``val_queue_len`` pipeline-fill term that is microarchitecture-specific (the per-impl
cost OVERRIDE, deliberately NOT built here).
"""

import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Context,
    Illegal,
    Backend,
    Interface,
    Kernel,
    KernelError,
    Role,
)


# ---------------------------------------------------------------------------
# The LayerNorm Kernel: two impls (hls/rtl), both fold the channel dim by SIMD.
# ---------------------------------------------------------------------------

CHANNELS = 64
IFM = (1, 56, CHANNELS)  # (batch, spatial, channels) — NHWC-ish, channels last


def _layernorm_op() -> Kernel:
    # SIMD folds the last (channel) dim. The op declares the block (channel dim FULL — the
    # normalization spans all channels); the impls declare the SIMD stream fold. The engine
    # derives the SIMD dial (divisors of the channel count), divisibility, and widths.
    from finn.kernels.space import FULL

    # rank-3 NHWC-ish (batch, spatial, channels): block spans the channel dim.
    channel_block = [1, 1, FULL]
    channel_stream = [1, 1, "SIMD"]
    hls = Backend(
        name="layernorm_hls",
        stream={"inp": list(channel_stream), "out": list(channel_stream)},
    )
    rtl = Backend(
        name="layernorm_rtl",
        stream={"inp": list(channel_stream), "out": list(channel_stream)},
    )

    return Kernel(
        name="LayerNorm",
        interfaces=(
            Interface("inp", Role.DATA_IN, block=list(channel_block)),
            Interface("out", Role.DATA_OUT, block=list(channel_block)),
        ),
        pool=(hls, rtl),
    )


def _ctx() -> Context:
    return Context(
        shapes={"inp": IFM, "out": IFM},
        datatypes={"inp": DataType["INT8"], "out": DataType["FLOAT32"]},
        fpgapart="xcvc1902-vsva2197-2MP-e-S",
    )


# ---------------------------------------------------------------------------
# Resolve + normal (TENSOR) getters — no backend needed.
# ---------------------------------------------------------------------------


def test_configures_to_a_point():
    op = _layernorm_op()
    pt = op.configure(_ctx(), {"SIMD": 16})
    assert not isinstance(pt, Illegal)
    assert pt["SIMD"] == 16
    assert pt["implementation"] == "layernorm_hls"  # first registered = default


def test_normal_shapes_and_dtypes_need_no_point():
    op, ctx = _layernorm_op(), _ctx()
    assert op.get_normal_input_shape(ctx, 0) == IFM
    assert op.get_normal_output_shape(ctx, 0) == IFM
    assert op.get_input_datatype(ctx, 0) == DataType["INT8"]
    assert op.get_output_datatype(ctx, 0) == DataType["FLOAT32"]


# ---------------------------------------------------------------------------
# Folded (STREAM) getters — validated against FINN's LayerNorm formulas.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simd", [1, 2, 8, 16, 32, 64])
def test_folded_input_shape_matches_finn(simd):
    op, ctx = _layernorm_op(), _ctx()
    pt = op.configure(ctx, {"SIMD": simd})
    # FINN: folded = ishape[:-1] + [n/simd, simd]
    expected = IFM[:-1] + (CHANNELS // simd, simd)
    assert op.get_folded_input_shape(pt, ctx, 0) == expected
    assert op.get_folded_output_shape(pt, ctx, 0) == expected


@pytest.mark.parametrize("simd", [1, 8, 16, 64])
def test_stream_widths_match_finn(simd):
    op, ctx = _layernorm_op(), _ctx()
    pt = op.configure(ctx, {"SIMD": simd})
    assert op.get_instream_width(pt, ctx, 0) == 8 * simd  # INT8
    assert op.get_outstream_width(pt, ctx, 0) == 32 * simd  # FLOAT32


@pytest.mark.parametrize("simd", [1, 8, 16, 64])
def test_rough_exp_cycles_is_throughput_floor(simd):
    op, ctx = _layernorm_op(), _ctx()
    pt = op.configure(ctx, {"SIMD": simd})
    # prod(ifm) // simd — the monotone throughput floor (FINN's leading term).
    expected = (1 * 56 * CHANNELS) // simd
    assert op.get_exp_cycles(pt, ctx) == expected


def test_exp_cycles_monotone_in_simd():
    # More parallelism -> fewer cycles (never increases). The SetFolding property.
    op, ctx = _layernorm_op(), _ctx()
    prev = None
    for simd in [1, 2, 4, 8, 16, 32, 64]:
        pt = op.configure(ctx, {"SIMD": simd})
        c = op.get_exp_cycles(pt, ctx)
        if prev is not None:
            assert c <= prev
        prev = c


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------


def test_bad_index_raises():
    op, ctx = _layernorm_op(), _ctx()
    pt = op.configure(ctx, {"SIMD": 16})
    with pytest.raises(KernelError, match="input index 1 out of range"):
        op.get_folded_input_shape(pt, ctx, 1)


def test_non_dividing_simd_is_illegal_at_resolve():
    # SIMD must be a divisor of channels — the divisor_axis domain rejects 5 at resolve.
    op = _layernorm_op()
    result = op.configure(_ctx(), {"SIMD": 5})
    assert isinstance(result, Illegal)
