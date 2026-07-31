############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding emit — hermetic RTL/HLS codegen (structural).

The Point→emit seam produces correct typed Artifacts: the RTL top has all slots bound and
the right .dat/.sv set; the HLS backend bakes the single ThresholdsActivation ROM into
thresh.h (the ONE home for that assertion — the old duplicate in test_thresholding_kernel
is dropped); the narrow-quant path adjusts steps. Emit is hermetic by construction — it
needs only a resolved Point + a dict-built Context, never a graph.
"""

import re

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.model.artifacts import Artifacts
from finn.kernels.engine.context import Context
from finn.kernels.engine.resolve import resolve
from finn.kernels.model.backend import EmitError, emit_point
from finn.kernels.compute.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_pool,
    thresholding_space,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"
_UNFILLED = re.compile(r"\$[A-Z][A-Z0-9_]*\$")


def sorted_thresholds(channels=4, steps=7):
    rng = np.random.RandomState(0)
    return np.sort(rng.randint(0, 50, size=(channels, steps)).astype(np.float32), axis=-1)


def make_context(channels=4, steps=7, idt="UINT8", tdt="UINT8", odt="UINT3", thresholds=None):
    if thresholds is None:
        thresholds = sorted_thresholds(channels, steps)
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": DataType[tdt], "inp": DataType[idt], "out": DataType[odt]},
        initializers={"thresholds": thresholds},
        fpgapart=VERSAL, clk_ns=5.0,
    )


def rtl_point(schema, ctx, **overrides):
    a = {"backend": THRESHOLDING_RTL, "PE": 2}
    a.update(overrides)
    return resolve(schema, ctx, a)


# --- RTL artifact shape ----------------------------------------------------


def test_emit_produces_expected_artifacts():
    schema = thresholding_space()
    ctx = make_context(channels=4, steps=7, odt="UINT3")  # o_bits=3 -> 7 steps
    arts = emit_point(thresholding_pool(), rtl_point(schema, ctx, PE=2, depth_trigger_bram=1024), ctx)
    assert isinstance(arts, Artifacts)
    assert len(arts.generated) == 1
    assert arts.generated[0].filename == "thresholding_top.v"
    # PE * o_bits = 2 * 3 = 6 memory-init files, correctly named
    assert len(arts.data_files) == 2 * 3
    assert {d.filename for d in arts.data_files} == {
        f"threshs_{pe}_{st}.dat" for pe in range(2) for st in range(3)
    }
    assert tuple(s.resource.split("/")[-1] for s in arts.static_files) == (
        "thresholding.sv", "thresholding_axi.sv", "axilite.sv",
    )
    assert arts.ipi.commands


def test_rendered_top_has_bound_values_and_no_unfilled_slots():
    schema = thresholding_space()
    ctx = make_context(channels=4, steps=7, idt="UINT8", odt="UINT3")
    content = emit_point(
        thresholding_pool(), rtl_point(schema, ctx, PE=2, depth_trigger_bram=1024, depth_trigger_uram=0), ctx
    ).generated[0].content()
    assert "parameter  PE = 2" in content
    assert "parameter  WI = 8" in content  # input bitwidth
    assert "parameter  N = 7" in content   # 2^o_bits - 1
    assert "parameter  DEPTH_TRIGGER_BRAM = 1024" in content
    assert "parameter  SIGNED = 0" in content  # UINT8 input
    assert not _UNFILLED.search(content)  # no unfilled $UPPERCASE$
    assert "$clog2(" in content  # Verilog $clog2 left intact


def test_dat_content_is_valid_hex():
    schema = thresholding_space()
    thr = np.array([[2, 5, 9]], dtype=np.float32)  # 1 channel, 3 sorted steps
    ctx = make_context(channels=1, steps=3, idt="UINT8", tdt="UINT8", odt="UINT2", thresholds=thr)
    arts = emit_point(thresholding_pool(), rtl_point(schema, ctx, PE=1), ctx)
    assert len(arts.data_files) == 2  # PE=1 * o_bits=2
    for d in arts.data_files:
        for line in d.content.splitlines():
            int(line, 16)  # raises if not valid hex


def test_narrow_quant_adjusts_steps():
    schema = thresholding_space()
    thr = np.sort(np.random.RandomState(1).randint(0, 50, size=(4, 6)).astype(np.float32), axis=-1)
    ctx = make_context(channels=4, steps=6, idt="UINT8", tdt="UINT8", odt="UINT3", thresholds=thr)
    content = emit_point(
        thresholding_pool(), rtl_point(schema, ctx, PE=2, numSteps=6), ctx
    ).generated[0].content()
    assert "parameter  N = 7" in content  # unsigned narrow-quant fix keeps N=7


# --- HLS baked ROM (the ONE home for this assertion) -----------------------


def test_hls_bakes_thresholds_activation_rom():
    schema = thresholding_space()
    ctx = make_context(channels=4, steps=7, odt="UINT3")
    arts = emit_point(thresholding_pool(), resolve(schema, ctx, {"backend": THRESHOLDING_HLS, "PE": 2}), ctx)
    assert arts.generated[0].filename == "top_thresholding_top.cpp"
    assert {d.filename for d in arts.data_files} == {"thresh.h"}
    body = arts.data_files[0].content
    body = body() if callable(body) else body
    assert body.startswith("static ThresholdsActivation<")
    cpp = arts.generated[0].content()
    assert "Thresholding_Batch" in cpp
    assert not _UNFILLED.findall(cpp)


def test_emit_point_unknown_impl_raises():
    schema = thresholding_space()
    ctx = make_context()
    with pytest.raises(EmitError, match="not in the pool"):
        emit_point((), rtl_point(schema, ctx, PE=2), ctx)


# --- hermeticity -----------------------------------------------------------


def test_emit_needs_only_point_and_dict_context():
    schema = thresholding_space()
    ctx = make_context()
    arts = emit_point(thresholding_pool(), rtl_point(schema, ctx, PE=2), ctx)
    assert arts.generated[0].content()
    assert not hasattr(ctx, "graph") and not hasattr(ctx, "model")
