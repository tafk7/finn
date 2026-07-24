############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Emit tests: MVAU compute-core codegen (RTL softvec+packed + HLS), embedded mode.

Compute-half only — the decoupled weight-delivery path (memstream / IPI stitch) is
the deferred composition arc and is NOT exercised here.
"""

import re

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import Artifacts, Context, GeneratedFile, emit_point, resolve
from finn.kernels.ops.mvau import (
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    mvau_pool,
    mvau_schema,
)
from finn.kernels.ops.parameters.names import DECOUPLED, EMBEDDED, WEIGHTS
from finn.kernels.space.param_names import topology_key

# Composed for the ``weights`` interface -> ``parameters.weights.topology``.
TOPOLOGY = topology_key(WEIGHTS)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
_UNFILLED = re.compile(r"\$[A-Z][A-Z0-9_]*\$")


def make_context(mw=6, mh=8, wdt="INT8", idt="INT8", weights=None):
    if weights is None:
        weights = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": DataType[wdt], "inp": DataType[idt], "out": DataType["INT16"]},
        initializers={"weights": weights},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def dsp_point(schema, ctx, impl, **overrides):
    # The DSP/RTL core is a streamed-weight core (embedded is illegal for it — base FINN has
    # no embedded-weight RTL path), so its delivery topology is `decoupled`. The COMPUTE-half
    # emit (emit_point over the compute pool) is topology-independent — the memstream is the
    # separate delivery half — so these compute-core golden tests are unaffected by the mode.
    a = {
        "implementation": impl,
        "PE": 2,
        "SIMD": 2,
        "resType": "dsp",
        TOPOLOGY: DECOUPLED,
    }
    a.update(overrides)
    return resolve(schema, ctx, a)


def hls_point(schema, ctx, **overrides):
    a = {
        "implementation": MVAU_HLS,
        "PE": 2,
        "SIMD": 2,
        "resType": "lut",
        TOPOLOGY: EMBEDDED,
    }
    a.update(overrides)
    return resolve(schema, ctx, a)


# --- RTL wrapper golden ----------------------------------------------------


def test_rtl_softvec_wrapper_golden():
    schema = mvau_schema()
    ctx = make_context()
    p = dsp_point(schema, ctx, MVAU_DSP_SOFTVEC)
    arts = emit_point(mvau_pool(), p, ctx)

    assert isinstance(arts, Artifacts)
    assert len(arts.generated) == 1
    c = arts.generated[0].content()
    assert "parameter\tVERSION = 3" in c  # forced DSP58
    assert "parameter\tMW = 6" in c
    assert "parameter\tMH = 8" in c
    assert "parameter\tPE = 2" in c
    assert "SEGMENTLEN =" in c
    assert "NARROW_WEIGHTS =" in c
    assert not _UNFILLED.search(c)  # every slot filled
    # embedded RTL needs NO weight file (weights arrive via in1_V stream)
    assert arts.data_files == ()
    # softvec ships mvu.sv (not the packed core)
    static = [s.resource.split("/")[-1] for s in arts.static_files]
    assert "mvu.sv" in static
    assert "mvu_vvu_8sx9_dsp58.sv" not in static


def test_rtl_packed_ships_packed_core():
    schema = mvau_schema()
    ctx = make_context()
    p = dsp_point(schema, ctx, MVAU_DSP_PACKED)
    arts = emit_point(mvau_pool(), p, ctx)
    static = [s.resource.split("/")[-1] for s in arts.static_files]
    assert "mvu_vvu_8sx9_dsp58.sv" in static
    assert "mvu.sv" not in static


def test_dsp_bundles_share_identical_wrapper():
    # softvec and packed emit the SAME wrapper on the same context; they diverge only
    # in the static .sv list (the documented 2c non-separation).
    schema = mvau_schema()
    ctx = make_context()
    a_sv = emit_point(mvau_pool(), dsp_point(schema, ctx, MVAU_DSP_SOFTVEC), ctx)
    a_pk = emit_point(mvau_pool(), dsp_point(schema, ctx, MVAU_DSP_PACKED), ctx)
    assert a_sv.generated[0].content() == a_pk.generated[0].content()
    assert {s.resource for s in a_sv.static_files} != {s.resource for s in a_pk.static_files}


# --- HLS golden ------------------------------------------------------------


def test_hls_cpp_golden():
    schema = mvau_schema()
    ctx = make_context()
    p = hls_point(schema, ctx, PE=2, SIMD=2, resType="lut")
    arts = emit_point(mvau_pool(), p, ctx)

    c = arts.generated[0].content()
    assert "Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1," in c
    assert "#define PE1 2" in c
    assert "#define MW1 6" in c
    assert "ap_resource_lut()" in c  # resType=lut
    assert 'PassThroughActivation<ap_int<14>>()' in c  # accDataType=INT14
    assert '#include "params.h"' in c
    assert not _UNFILLED.search(c)


def test_hls_restype_dsp_maps_to_dsp_resource():
    schema = mvau_schema()
    ctx = make_context()
    p = hls_point(schema, ctx, resType="dsp")
    c = emit_point(mvau_pool(), p, ctx).generated[0].content()
    assert "ap_resource_dsp()" in c


def test_hls_params_header_is_valid_fixedpoint_initializer():
    schema = mvau_schema()
    ctx = make_context(mw=6, mh=8, wdt="INT8")
    p = hls_point(schema, ctx, PE=2, SIMD=2)
    arts = emit_point(mvau_pool(), p, ctx)
    params = [d for d in arts.data_files if d.filename == "params.h"]
    assert len(params) == 1
    text = params[0].content
    # WMEM = MW*MH/(PE*SIMD) = 6*8/(2*2) = 12
    assert text.startswith("const FixedPointWeights<2,ap_int<8>,2,12> weights = ")
    assert text.count("{") == text.count("}")  # balanced C++ initializer
    assert text.rstrip().endswith(";")
    assert "ap_uint<16>" in text  # SIMD(2)*8 = 16-bit packed innermost


# --- hermeticity by construction -------------------------------------------


def test_emit_needs_only_point_and_dict_context():
    schema = mvau_schema()
    ctx = make_context()  # plain dicts, never a ModelWrapper
    for p in (dsp_point(schema, ctx, MVAU_DSP_SOFTVEC), hls_point(schema, ctx)):
        arts = emit_point(mvau_pool(), p, ctx)
        assert arts.generated[0].content()
    assert not hasattr(ctx, "graph") and not hasattr(ctx, "model")


# --- dispatch across the whole pool ----------------------------------------


def test_all_three_bundles_emit():
    schema = mvau_schema()
    ctx = make_context()
    points = {
        MVAU_DSP_SOFTVEC: dsp_point(schema, ctx, MVAU_DSP_SOFTVEC),
        MVAU_DSP_PACKED: dsp_point(schema, ctx, MVAU_DSP_PACKED),
        MVAU_HLS: hls_point(schema, ctx),
    }
    for impl, p in points.items():
        arts = emit_point(mvau_pool(), p, ctx)
        assert isinstance(arts, Artifacts), impl
        assert len(arts.generated) == 1, impl


# --- thresholds bake (Part C) ----------------------------------------------


def _thresh_context(mw=6, mh=8, steps=7, tdt="INT16"):
    weights = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    thr = np.sort(
        np.random.RandomState(1).randint(0, 100, size=(mh, steps)).astype(np.float32), axis=1
    )
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh), "thresholds": (mh, steps)},
        datatypes={
            "weights": DataType["INT4"], "inp": DataType["INT4"],
            "out": DataType["INT16"], "thresholds": DataType[tdt],
        },
        initializers={"weights": weights, "thresholds": thr},
        fpgapart=VERSAL, clk_ns=5.0,
    )


def test_hls_thresholded_node_bakes_thresh_h():
    # A 3-input HLS node bakes a ThresholdsActivation into thresh.h and uses `threshs` in the
    # docompute (not PassThrough); the core includes thresh.h. Thresholds are constant → no port.
    schema = mvau_schema()
    ctx = _thresh_context(steps=7)
    p = hls_point(schema, ctx, PE=2, SIMD=2, resType="lut")
    arts = emit_point(mvau_pool(), p, ctx)
    files = {d.filename for d in arts.data_files}
    assert "thresh.h" in files
    thresh = next(d for d in arts.data_files if d.filename == "thresh.h").content
    assert "ThresholdsActivation<" in thresh
    assert "comp::less_equal" in thresh
    cpp = arts.generated[0].content()
    assert '#include "thresh.h"' in cpp
    assert "threshs," in cpp
    assert "PassThroughActivation" not in cpp
    assert not _UNFILLED.search(cpp)


def test_hls_no_threshold_node_has_no_thresh_h():
    # A 2-input node bakes no thresh.h and keeps the PassThrough activation.
    schema = mvau_schema()
    ctx = make_context()
    p = hls_point(schema, ctx, PE=2, SIMD=2, resType="lut")
    arts = emit_point(mvau_pool(), p, ctx)
    assert "thresh.h" not in {d.filename for d in arts.data_files}
    cpp = arts.generated[0].content()
    assert "PassThroughActivation" in cpp
    assert '#include "thresh.h"' not in cpp
