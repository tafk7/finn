############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""VVAU fixture tests — the model generalizes to a 2nd op with a SINGLETON RTL pool.

VVAU folds PE over Channels and SIMD over the kernel window K=k_h*k_w, and its only
RTL implementation is the Versal/DSP58 packed core (no softvec/DSP48/LUT).
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import AbsentAxisError, Context, Illegal, Point, resolve
from finn.kernels.ops.vvau import VVAU_HLS, VVAU_RTL, vvau_schema

SEVEN_SERIES = "xc7z020clg400-1"  # DSP48E1, not Versal
ULTRASCALE = "xcku040-ffva1156-2-e"  # DSP48E2, not Versal
VERSAL = "xcvc1902-vsva2197-2MP-e-S"  # DSP58, Versal


@pytest.fixture
def schema():
    return vvau_schema()


def make_context(fpgapart=VERSAL, channels=8, k_h=3, k_w=3, wdt="INT8", idt="INT8", weights=None):
    # Depthwise weight layout (Channels, 1, k_h, k_w).
    if weights is None:
        rng = np.random.RandomState(0)
        weights = rng.randint(-7, 7, size=(channels, 1, k_h, k_w)).astype(np.float32)
    return Context(
        shapes={
            "weights": weights.shape,
            "inp": (1, 1, 1, k_h * k_w * channels),
            "out": (1, 1, 1, channels),
        },
        datatypes={"weights": DataType[wdt], "inp": DataType[idt], "out": DataType["INT16"]},
        initializers={"weights": weights},
        fpgapart=fpgapart,
        clk_ns=5.0,
    )


def base_assignment(**overrides):
    a = {
        "implementation": VVAU_HLS,
        "PE": 2,
        "SIMD": 3,
        "mem_mode": "internal_decoupled",
        "noActivation": 1,
    }
    a.update(overrides)
    return a


# --- pool shape ------------------------------------------------------------


def test_pool_is_hls_and_single_rtl(schema):
    from finn.kernels.ops.vvau import vvau_pool

    names = [b.name for b in vvau_pool()]
    assert names == [VVAU_HLS, VVAU_RTL]  # exactly two, RTL is a singleton


# --- folding geometry (PE folds Channels, SIMD folds K) --------------------


def test_folding_wmem(schema):
    # Channels=8, K=9, PE=2, SIMD=3 -> WMEM = (9*8/2)/3 = 12.
    r = resolve(schema, make_context(), base_assignment(PE=2, SIMD=3))
    assert isinstance(r, Point)
    assert r.WMEM == (9 * 8 // 2) // 3
    assert r.instream_width == DataType["INT8"].bitwidth() * 3 * 2  # i_bits*SIMD*PE
    assert r.outstream_width == r.outputDataType.bitwidth() * 2  # o_bits*PE


def test_pe_must_divide_channels(schema):
    # PE=3 is not a divisor of Channels=8 -> rejected at the divisor-domain check.
    r = resolve(schema, make_context(channels=8), base_assignment(PE=3))
    assert isinstance(r, Illegal)
    assert any("PE" in reason for reason in r.reasons)


def test_simd_must_divide_kernel_window(schema):
    # K = 3*3 = 9; SIMD=2 is not a divisor of 9 -> rejected at the divisor-domain check.
    r = resolve(schema, make_context(k_h=3, k_w=3), base_assignment(SIMD=2))
    assert isinstance(r, Illegal)
    assert any("SIMD" in reason for reason in r.reasons)


# --- device pruning: RTL is Versal/DSP58-only ------------------------------


def test_rtl_illegal_on_seven_series(schema):
    r = resolve(
        schema,
        make_context(SEVEN_SERIES),
        base_assignment(implementation=VVAU_RTL, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("Versal" in reason for reason in r.reasons)


def test_rtl_illegal_on_ultrascale(schema):
    r = resolve(
        schema,
        make_context(ULTRASCALE),
        base_assignment(implementation=VVAU_RTL, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Illegal)
    assert any("Versal" in reason for reason in r.reasons)


def test_rtl_legal_on_versal(schema):
    r = resolve(
        schema,
        make_context(VERSAL),
        base_assignment(implementation=VVAU_RTL, resType="dsp", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Point)
    assert r.dsp_version == 3  # forced DSP58
    assert r.language == "rtl"


def test_hls_builds_anywhere(schema):
    for part in (SEVEN_SERIES, ULTRASCALE, VERSAL):
        r = resolve(schema, make_context(part), base_assignment(mem_mode="internal_embedded"))
        assert isinstance(r, Point), part
        assert r.language == "hls"


# --- guarded axis: ram_style exists only under decoupled -------------------


def test_ram_style_absent_when_not_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_embedded"))
    assert isinstance(r, Point)
    assert "ram_style" not in r
    with pytest.raises(AbsentAxisError):
        _ = r.ram_style


def test_ram_style_present_when_decoupled(schema):
    r = resolve(schema, make_context(), base_assignment(mem_mode="internal_decoupled"))
    assert isinstance(r, Point)
    assert r.ram_style == "auto"


# --- forced-derived + legal/illegal ----------------------------------------


def test_dsp_version_and_language_not_axes(schema):
    axis_names = schema.axis_names
    for name in ("dsp_version", "dsp_primitive", "language", "SEGMENTLEN", "accDataType", "WMEM"):
        assert name not in axis_names, f"{name} must be Derived, not an Axis"


def test_rtl_rejects_no_activation_zero(schema):
    # RTL-VVU requires noActivation=1 (embedded thresholds unsupported).
    r = resolve(
        schema,
        make_context(VERSAL),
        base_assignment(
            implementation=VVAU_RTL, resType="dsp", mem_mode="internal_embedded", noActivation=0
        ),
    )
    assert isinstance(r, Illegal)
    assert any("noActivation" in reason for reason in r.reasons)


# --- composability: a 3rd bundle composes additively -----------------------


def test_third_implementation_composes_additively():
    from finn.kernels.space import Derived, Implementation, pool_schema
    from finn.kernels.ops.vvau import vvau_pool, vvau_shared

    def aie_feasible(p, ctx):
        return None  # hypothetical AIE backend, always feasible here

    aie = Implementation(
        name="vvau_aie",
        feasible=aie_feasible,
        derived=(Derived("language", lambda p, ctx: "aie"),),
        sources=("vvau_aie.cpp",),
    )
    axes, derived, predicates = vvau_shared()
    schema3 = pool_schema("implementation", axes, derived, predicates, vvau_pool() + (aie,))
    r = resolve(
        schema3,
        make_context(SEVEN_SERIES),
        base_assignment(implementation="vvau_aie", mem_mode="internal_embedded"),
    )
    assert isinstance(r, Point)
    assert r.language == "aie"
    assert r.sources == ("vvau_aie.cpp",)
