############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding fixture tests — the model-stressing op.

Proves "op-level shared is per-op": the RTL bundle contributes impl-local axes
(depth_trigger_*, uniform_thres, deep_pipeline) that are ABSENT under the HLS sibling,
while both bundles share the SAME integer dtype envelope (no fabricated per-bundle
dtype gate).
"""

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import AbsentAxisError, Context, Illegal, Point, resolve
from finn.kernels.ops.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_pool,
    thresholding_schema,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _language_of(point, pool=None):
    """The selected backend's static ``language`` field (F5: no longer projected onto the
    point) — looked up in the pool by the resolved ``implementation`` name."""
    by_name = {b.name: b for b in (pool if pool is not None else thresholding_pool())}
    return by_name[point["backend"]].language


@pytest.fixture
def schema():
    return thresholding_schema()


def sorted_thresholds(channels=8, steps=7, signed=False):
    rng = np.random.RandomState(0)
    lo = -50 if signed else 0
    raw = rng.randint(lo, 50, size=(channels, steps)).astype(np.float32)
    return np.sort(raw, axis=-1)


def make_context(channels=8, steps=7, idt="UINT8", tdt="UINT8", odt="UINT3", thresholds=None):
    if thresholds is None:
        thresholds = sorted_thresholds(channels, steps)
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": DataType[tdt], "inp": DataType[idt], "out": DataType[odt]},
        initializers={"thresholds": thresholds},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def base_assignment(**overrides):
    a = {"backend": THRESHOLDING_HLS, "PE": 2}
    a.update(overrides)
    return a


# --- pool shape ------------------------------------------------------------


def test_pool_is_hls_and_rtl(schema):
    from finn.kernels.ops.thresholding import thresholding_pool

    assert [b.name for b in thresholding_pool()] == [THRESHOLDING_HLS, THRESHOLDING_RTL]


# --- folding + derived -----------------------------------------------------


def test_tmem_and_widths(schema):
    r = resolve(schema, make_context(channels=8), base_assignment(PE=2))
    assert isinstance(r, Point)
    assert r.TMEM == 8 // 2
    assert r.instream_width == DataType["UINT8"].bitwidth() * 2  # i_bits * PE
    assert r.outstream_width == DataType["UINT3"].bitwidth() * 2  # o_bits * PE


def test_pe_must_divide_channels(schema):
    r = resolve(schema, make_context(channels=8), base_assignment(PE=3))  # 3 ∤ 8
    assert isinstance(r, Illegal)
    assert any("PE" in reason for reason in r.reasons)


def test_num_steps_must_match_threshold_shape(schema):
    # threshold tensor has 7 steps; asserting numSteps=5 is illegal.
    r = resolve(schema, make_context(steps=7), base_assignment(numSteps=5))
    assert isinstance(r, Illegal)
    assert any("numSteps" in reason or "steps" in reason for reason in r.reasons)


# --- THE model-stress test: RTL-local axes disjoint from HLS ---------------


def test_rtl_local_axes_absent_under_hls(schema):
    r = resolve(schema, make_context(), base_assignment(backend=THRESHOLDING_HLS))
    assert isinstance(r, Point)
    for name in ("depth_trigger_bram", "depth_trigger_uram", "uniform_thres", "deep_pipeline"):
        assert name not in r, f"{name} must NOT exist under the HLS bundle"
        with pytest.raises(AbsentAxisError):
            _ = r[name]


def test_rtl_local_axes_present_under_rtl(schema):
    r = resolve(
        schema,
        make_context(),
        base_assignment(backend=THRESHOLDING_RTL, depth_trigger_bram=1024, uniform_thres=1),
    )
    assert isinstance(r, Point)
    assert r.depth_trigger_bram == 1024
    assert r.uniform_thres == 1
    assert r.deep_pipeline == 1  # default
    assert _language_of(r) == "rtl"


def test_assigning_rtl_axis_under_hls_is_illegal(schema):
    # An HLS point has no depth_trigger_bram axis; pinning it is a caller error.
    r = resolve(
        schema,
        make_context(),
        base_assignment(backend=THRESHOLDING_HLS, depth_trigger_bram=1024),
    )
    assert isinstance(r, Illegal)
    assert any("depth_trigger_bram" in reason for reason in r.reasons)


# --- no fabricated per-bundle dtype gate -----------------------------------


def test_same_integer_dtype_resolves_under_both_bundles(schema):
    # Both backends share the SAME integer dtype envelope: an identical context must
    # resolve to a Point under HLS AND under RTL (no per-bundle dtype feasibility gate).
    ctx = make_context(idt="INT8", tdt="INT8", odt="INT4", thresholds=sorted_thresholds(signed=True))
    r_hls = resolve(schema, ctx, base_assignment(backend=THRESHOLDING_HLS))
    r_rtl = resolve(schema, ctx, base_assignment(backend=THRESHOLDING_RTL))
    assert isinstance(r_hls, Point)
    assert isinstance(r_rtl, Point)


# --- data-dependent predicates ---------------------------------------------


def test_rtl_rejects_unsorted_thresholds(schema):
    # Descending thresholds violate the binary-search requirement.
    thr = sorted_thresholds()[:, ::-1].copy()  # reverse each row -> descending
    r = resolve(
        schema,
        make_context(thresholds=thr),
        base_assignment(backend=THRESHOLDING_RTL),
    )
    assert isinstance(r, Illegal)
    assert any("sorted" in reason for reason in r.reasons)


def test_hls_allows_unsorted_thresholds(schema):
    # The sorted gate is RTL-only; HLS does not impose it.
    thr = sorted_thresholds()[:, ::-1].copy()
    r = resolve(
        schema,
        make_context(thresholds=thr),
        base_assignment(backend=THRESHOLDING_HLS),
    )
    assert isinstance(r, Point)


def test_unsigned_input_requires_nonneg_thresholds(schema):
    # Unsigned input with a negative threshold value is illegal (data + config).
    thr = sorted_thresholds(signed=True)  # contains negatives
    r = resolve(
        schema,
        make_context(idt="UINT8", thresholds=thr),
        base_assignment(backend=THRESHOLDING_HLS),
    )
    assert isinstance(r, Illegal)
    assert any("thresholds >= 0" in reason or "non-negative" in reason.lower() for reason in r.reasons)


# --- composability ---------------------------------------------------------


def test_third_implementation_composes_additively():
    from finn.kernels.space import Backend, pool_schema
    from finn.kernels.ops.thresholding import thresholding_pool, thresholding_shared

    stub = Backend(
        name="thresholding_stub",
        language="stub",
        sources=("stub.sv",),
    )
    axes, derived, predicates = thresholding_shared()
    pool3 = thresholding_pool() + (stub,)
    schema3 = pool_schema("backend", axes, derived, predicates, pool3)
    r = resolve(schema3, make_context(), base_assignment(backend="thresholding_stub"))
    assert isinstance(r, Point)
    assert _language_of(r, pool3) == "stub"
    assert r.sources == ("stub.sv",)
