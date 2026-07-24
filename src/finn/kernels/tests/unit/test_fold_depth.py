############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Unit tests for the topology-independent fold-depth queries (F1 groundwork).

weight_fold_depth / threshold_fold_depth must reproduce the op's WMEM/TMEM
geometry aliases EXACTLY, across embedded AND decoupled fixture points — proving
one source can replace the two names (point.WMEM read in emit vs depth_key)."""

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.space import Context, resolve, weight_fold_depth, threshold_fold_depth
from finn.kernels.ops.mvau import MVAU_DSP_SOFTVEC, MVAU_HLS, mvau_schema
from finn.kernels.ops.parameters.names import DECOUPLED, EMBEDDED, WEIGHTS
from finn.kernels.space.param_names import topology_key

THRESHOLDS = "thresholds"
TOPOLOGY = topology_key(WEIGHTS)
VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _ctx(mw=6, mh=8):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType["INT8"], "inp": DataType["INT8"], "out": DataType["INT16"]},
        initializers={"weights": w},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def _thresh_ctx(mw=6, mh=8, steps=7):
    w = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    thr = np.sort(
        np.random.RandomState(1).randint(0, 100, size=(mh, steps)).astype(np.float32), axis=1
    )
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh), "thresholds": (mh, steps)},
        datatypes={
            "weights": DataType["INT4"], "inp": DataType["INT4"],
            "out": DataType["INT16"], "thresholds": DataType["INT16"],
        },
        initializers={"weights": w, "thresholds": thr},
        fpgapart=VERSAL, clk_ns=5.0,
    )


def test_weight_fold_depth_matches_wmem_decoupled():
    ctx = _ctx()
    p = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_DSP_SOFTVEC, "PE": 2, "SIMD": 2,
        "resType": "dsp", TOPOLOGY: DECOUPLED,
    })
    # WMEM = MW*MH/(PE*SIMD) = 6*8/(2*2). The decoupled depth_key traces to the same math.
    assert weight_fold_depth(p, ctx, WEIGHTS) == 12
    from finn.kernels.space.param_names import depth_key
    assert weight_fold_depth(p, ctx, WEIGHTS) == p[depth_key(WEIGHTS)]


def test_weight_fold_depth_matches_wmem_embedded():
    ctx = _ctx()
    p = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    # embedded HLS has no depth_key, but the topology-independent geometry query works
    assert weight_fold_depth(p, ctx, WEIGHTS) == 12


def test_threshold_fold_depth_matches_tmem_present():
    ctx = _thresh_ctx()
    p = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    assert threshold_fold_depth(p, ctx, THRESHOLDS) == 4  # MH//PE = 8//2


def test_threshold_fold_depth_zero_when_absent():
    ctx = _ctx()  # no thresholds
    p = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_HLS, "PE": 2, "SIMD": 2,
        "resType": "lut", TOPOLOGY: EMBEDDED,
    })
    assert threshold_fold_depth(p, ctx, THRESHOLDS) == 0
