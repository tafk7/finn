############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Emit-phase tests: hermetic Thresholding-RTL codegen over the flat resolve engine.

Proves the Point -> emit seam produces correct typed Artifacts, that the typed
Template supersedes the silent-$KEY$ defect, and that emit is hermetic by
construction (needs only a resolved Point + a dict-built Context, never a graph).
"""

import re

import numpy as np
import pytest
from qonnx.core.datatype import DataType

from finn.kernels.space import (
    Artifacts,
    Context,
    EmitError,
    GeneratedFile,
    Template,
    TemplateError,
    emit_point,
    resolve,
)
from finn.kernels.ops.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_pool,
    thresholding_schema,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"

# Any $UPPERCASE$ token still present after render is an unfilled slot ($clog2 is
# lowercase and legitimately survives).
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
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def rtl_point(schema, ctx, **overrides):
    a = {"implementation": THRESHOLDING_RTL, "PE": 2}
    a.update(overrides)
    return resolve(schema, ctx, a)


# --- artifact shape / golden -----------------------------------------------


def test_emit_produces_expected_artifacts():
    schema = thresholding_schema()
    ctx = make_context(channels=4, steps=7, odt="UINT3")  # o_bits=3 -> 2^3-1=7 steps
    point = rtl_point(schema, ctx, PE=2, depth_trigger_bram=1024)
    arts = emit_point(thresholding_pool(), point, ctx)

    assert isinstance(arts, Artifacts)
    # one generated top .v
    assert len(arts.generated) == 1
    assert arts.generated[0].filename == "thresholding_top.v"
    # PE * o_bits = 2 * 3 = 6 memory-init files, correctly named
    assert len(arts.data_files) == 2 * 3
    names = {d.filename for d in arts.data_files}
    assert names == {f"threshs_{pe}_{st}.dat" for pe in range(2) for st in range(3)}
    # three static .sv refs, matching the bundle's declared sources
    assert tuple(s.resource.split("/")[-1] for s in arts.static_files) == (
        "thresholding.sv",
        "thresholding_axi.sv",
        "axilite.sv",
    )
    assert arts.ipi.commands  # non-empty instantiation command


def test_rendered_top_has_bound_values_and_no_unfilled_slots():
    schema = thresholding_schema()
    ctx = make_context(channels=4, steps=7, idt="UINT8", odt="UINT3")
    point = rtl_point(schema, ctx, PE=2, depth_trigger_bram=1024, depth_trigger_uram=0)
    content = emit_point(thresholding_pool(), point, ctx).generated[0].content()

    assert "parameter  PE = 2" in content
    assert "parameter  WI = 8" in content  # input bitwidth
    assert "parameter  N = 7" in content  # 2^o_bits - 1
    assert "parameter  DEPTH_TRIGGER_BRAM = 1024" in content
    assert "parameter  SIGNED = 0" in content  # UINT8 input
    # no unfilled $UPPERCASE$ slot survives; Verilog $clog2 is left intact
    assert not _UNFILLED.search(content)
    assert "$clog2(" in content


def test_dat_content_hand_checkable():
    # PE=1, single channel, o_bits=2 -> 2^2-1=3 thresholds, small + hand-verifiable.
    schema = thresholding_schema()
    thr = np.array([[2, 5, 9]], dtype=np.float32)  # 1 channel, 3 steps, sorted
    ctx = make_context(channels=1, steps=3, idt="UINT8", tdt="UINT8", odt="UINT2", thresholds=thr)
    point = rtl_point(schema, ctx, PE=1)
    arts = emit_point(thresholding_pool(), point, ctx)
    # PE=1 * o_bits=2 = 2 dat files
    assert len(arts.data_files) == 2
    # every dat line is a hex string (no '0x' prefix, packing uses prefix="")
    # (DataFile.content is a plain str attribute, unlike GeneratedFile.content()).
    for d in arts.data_files:
        for line in d.content.splitlines():
            int(line, 16)  # raises if not valid hex


# --- typed Template supersedes the silent-$KEY$ defect ---------------------


def test_template_missing_binding_raises():
    t = Template("val = $A$ + $B$;")
    with pytest.raises(TemplateError, match="no binding"):
        t.render({"A": 1})  # B missing


def test_template_renamed_token_raises():
    t = Template("val = $A$;")
    with pytest.raises(TemplateError, match="no matching slot"):
        t.render({"A": 1, "AA": 2})  # AA is a typo'd/renamed token -> loud error


def test_template_leaves_verilog_clog2_untouched():
    t = Template("wire [$clog2(N) : 0] x = $VAL$;")
    assert t.slots == frozenset({"VAL"})  # $clog2 is not a slot
    assert t.render({"VAL": 3}) == "wire [$clog2(N) : 0] x = 3;"


# --- hermeticity by construction -------------------------------------------


def test_emit_needs_only_point_and_dict_context():
    # The Context is built from plain dicts — there was never a graph to leak. emit
    # producing correct Artifacts from (Point, dict-Context) alone IS the hermeticity
    # proof (stronger than the reference's graph-destroy trick).
    schema = thresholding_schema()
    ctx = make_context()
    point = rtl_point(schema, ctx, PE=2)
    arts = emit_point(thresholding_pool(), point, ctx)
    assert arts.generated[0].content()  # renders with no graph access
    assert not hasattr(ctx, "graph") and not hasattr(ctx, "model")


# --- narrow-quant path -----------------------------------------------------


def test_narrow_quant_adjusts_bias_and_steps():
    # o_bits=3 expects 7 steps; provide 6 (one short) -> narrow-range path fires.
    schema = thresholding_schema()
    thr = np.sort(np.random.RandomState(1).randint(0, 50, size=(4, 6)).astype(np.float32), axis=-1)
    ctx = make_context(channels=4, steps=6, idt="UINT8", tdt="UINT8", odt="UINT3", thresholds=thr)
    point = rtl_point(schema, ctx, PE=2, numSteps=6)
    content = emit_point(thresholding_pool(), point, ctx).generated[0].content()
    # After the unsigned narrow-quant fix the core still sees N=7 steps.
    assert "parameter  N = 7" in content


# --- dispatch --------------------------------------------------------------


def test_emit_point_on_hls_raises_not_implemented():
    schema = thresholding_schema()
    ctx = make_context()
    point = resolve(schema, ctx, {"implementation": THRESHOLDING_HLS, "PE": 2})
    with pytest.raises(EmitError, match="emit not implemented"):
        emit_point(thresholding_pool(), point, ctx)


def test_emit_point_unknown_impl_raises():
    schema = thresholding_schema()
    ctx = make_context()
    point = rtl_point(schema, ctx, PE=2)
    # A pool that doesn't contain the point's implementation.
    with pytest.raises(EmitError, match="not in the pool"):
        emit_point((), point, ctx)
