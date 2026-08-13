############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU compute-core emit — structural (M3, M7, T5).

Asserts STRUCTURE: every template slot is bound (no unfilled $UPPERCASE$), port widths
equal the resolved stream widths, the right static source set ships per backend variant,
softvec vs packed differ ONLY in the instantiated core module, and the HLS thresholds
bake lands in thresh.h. Byte-for-byte parity vs real FINN is the hw diff tier's job — here
we guard the emit contract, not the exact codegen text.
"""

import re

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.model.artifacts import Artifacts
from finn.kernels.engine.context import Context
from finn.kernels.engine.resolve import resolve
from finn.kernels.model.backend import emit_point
from finn.kernels.compute.mvau import (
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    MvauDataflowOp,
)
from finn.kernels.dataflow.parameters.names import DECOUPLED, EMBEDDED, WEIGHTS
from finn.kernels.model.param_names import topology_key

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
        fpgapart=VERSAL, clk_ns=5.0,
    )


def dsp_point(schema, ctx, backend, **overrides):
    a = {"backend": backend, "PE": 2, "SIMD": 2, "resType": "dsp", TOPOLOGY: DECOUPLED}
    a.update(overrides)
    return resolve(schema, ctx, a)


def hls_point(schema, ctx, **overrides):
    a = {"backend": MVAU_HLS, "PE": 2, "SIMD": 2, "resType": "lut", TOPOLOGY: EMBEDDED}
    a.update(overrides)
    return resolve(schema, ctx, a)


# --- RTL wrapper: all slots filled, right core + static set ----------------


def test_rtl_softvec_wrapper_all_slots_filled_ships_softvec_core():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    arts = emit_point(MvauDataflowOp.pool, dsp_point(schema, ctx, MVAU_DSP_SOFTVEC), ctx)
    assert isinstance(arts, Artifacts)
    assert len(arts.generated) == 1
    c = arts.generated[0].content()
    assert not _UNFILLED.search(c)  # every slot bound
    assert "mvu_vvu_axi_softvec #(" in c  # softvec per-core wrapper (2c split)
    assert arts.data_files == ()  # decoupled RTL: weights via stream, no weight file
    static = [s.resource.split("/")[-1] for s in arts.static_files]
    assert "mvu_vvu_axi_softvec.sv" in static and "mvu.sv" in static
    assert "mvu_vvu_axi_packed.sv" not in static
    assert "mvu_vvu_axi.sv" not in static  # retired fused wrapper gone


def test_rtl_packed_ships_packed_core():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    arts = emit_point(MvauDataflowOp.pool, dsp_point(schema, ctx, MVAU_DSP_PACKED), ctx)
    assert "mvu_vvu_axi_packed #(" in arts.generated[0].content()
    static = [s.resource.split("/")[-1] for s in arts.static_files]
    assert "mvu_vvu_axi_packed.sv" in static and "mvu_vvu_8sx9_dsp58.sv" in static
    assert "mvu.sv" not in static


def test_dsp_bundles_wrapper_differs_only_in_core_module():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    a_sv = emit_point(MvauDataflowOp.pool, dsp_point(schema, ctx, MVAU_DSP_SOFTVEC), ctx)
    a_pk = emit_point(MvauDataflowOp.pool, dsp_point(schema, ctx, MVAU_DSP_PACKED), ctx)
    c_sv, c_pk = a_sv.generated[0].content(), a_pk.generated[0].content()
    assert c_sv != c_pk
    assert c_sv.replace("mvu_vvu_axi_softvec", "X") == c_pk.replace("mvu_vvu_axi_packed", "X")
    assert {s.resource for s in a_sv.static_files} != {s.resource for s in a_pk.static_files}


# --- HLS: all slots filled, restype maps to resource, params.h -------------


def test_hls_cpp_all_slots_filled():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    c = emit_point(MvauDataflowOp.pool, hls_point(schema, ctx, PE=2, SIMD=2, resType="lut"), ctx).generated[0].content()
    assert "Matrix_Vector_Activate_Batch<" in c
    assert "ap_resource_lut()" in c
    assert '#include "params.h"' in c
    assert not _UNFILLED.search(c)


def test_hls_restype_dsp_maps_to_dsp_resource():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    c = emit_point(MvauDataflowOp.pool, hls_point(schema, ctx, resType="dsp"), ctx).generated[0].content()
    assert "ap_resource_dsp()" in c


def test_hls_params_header_geometry():
    schema, ctx = MvauDataflowOp.compile(), make_context(mw=6, mh=8, wdt="INT8")
    arts = emit_point(MvauDataflowOp.pool, hls_point(schema, ctx, PE=2, SIMD=2), ctx)
    params = [d for d in arts.data_files if d.filename == "params.h"]
    assert len(params) == 1
    text = params[0].content
    # WMEM = MW*MH/(PE*SIMD) = 6*8/(2*2) = 12
    assert text.startswith("const FixedPointWeights<2,ap_int<8>,2,12> weights = ")
    assert text.count("{") == text.count("}")  # balanced C++ initializer
    assert text.rstrip().endswith(";")


# --- T5: port widths equal the resolved stream widths ----------------------


def test_emitted_port_widths_match_stream_widths():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    p = dsp_point(schema, ctx, MVAU_DSP_SOFTVEC)
    arts = emit_point(MvauDataflowOp.pool, p, ctx)
    by_role = {port.role.value: port for port in arts.ports}
    # DATA_IN carries the input stream width; DATA_OUT the output stream width.
    assert by_role["data_in"].width == p["stream_width.inp"]
    assert by_role["data_out"].width == p["stream_width.out"]


# --- dispatch + hermeticity ------------------------------------------------


def test_all_three_bundles_emit():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    points = {
        MVAU_DSP_SOFTVEC: dsp_point(schema, ctx, MVAU_DSP_SOFTVEC),
        MVAU_DSP_PACKED: dsp_point(schema, ctx, MVAU_DSP_PACKED),
        MVAU_HLS: hls_point(schema, ctx),
    }
    for backend, p in points.items():
        arts = emit_point(MvauDataflowOp.pool, p, ctx)
        assert isinstance(arts, Artifacts), backend
        assert len(arts.generated) == 1, backend


def test_emit_needs_only_point_and_dict_context():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    for p in (dsp_point(schema, ctx, MVAU_DSP_SOFTVEC), hls_point(schema, ctx)):
        assert emit_point(MvauDataflowOp.pool, p, ctx).generated[0].content()
    assert not hasattr(ctx, "graph") and not hasattr(ctx, "model")


# --- thresholds bake -------------------------------------------------------


def _thresh_context(mw=6, mh=8, steps=7, tdt="INT16"):
    weights = np.random.RandomState(0).randint(-7, 7, size=(mw, mh)).astype(np.float32)
    thr = np.sort(np.random.RandomState(1).randint(0, 100, size=(mh, steps)).astype(np.float32), axis=1)
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
    schema, ctx = MvauDataflowOp.compile(), _thresh_context(steps=7)
    arts = emit_point(MvauDataflowOp.pool, hls_point(schema, ctx, PE=2, SIMD=2, resType="lut"), ctx)
    assert "thresh.h" in {d.filename for d in arts.data_files}
    thresh = next(d for d in arts.data_files if d.filename == "thresh.h").content
    assert "ThresholdsActivation<" in thresh
    cpp = arts.generated[0].content()
    assert '#include "thresh.h"' in cpp
    assert "PassThroughActivation" not in cpp
    assert not _UNFILLED.search(cpp)


def test_hls_no_threshold_node_has_no_thresh_h():
    schema, ctx = MvauDataflowOp.compile(), make_context()
    arts = emit_point(MvauDataflowOp.pool, hls_point(schema, ctx, PE=2, SIMD=2, resType="lut"), ctx)
    assert "thresh.h" not in {d.filename for d in arts.data_files}
    cpp = arts.generated[0].content()
    assert "PassThroughActivation" in cpp
    assert '#include "thresh.h"' not in cpp
