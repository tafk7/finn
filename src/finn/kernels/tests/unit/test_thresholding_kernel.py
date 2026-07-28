############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding ported onto the kernel system (Arc 2 roll-out).

Proves the Kernel assembly resolves (adding the ``parameters.thresholds.*`` namespace),
that BOTH backends resolve their threshold delivery to the ``embedded`` (constant)
topology — no memstream cell — and that the HLS backend emits a baked ``thresh.h`` ROM
through the shared serializer while the RTL backend's data-dependent ``.dat`` scatter stays
FUSED (never routes through ``layout``): the static-vs-data-dependent boundary, demonstrated
structurally.
"""

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.space import Context, emit_point, resolve
from finn.kernels.ops.thresholding import (
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    thresholding_kernel,
    thresholding_kernel_schema,
    thresholding_pool,
)

VERSAL = "xcvc1902-vsva2197-2MP-e-S"


def _ctx(channels=4, steps=7, idt="UINT8", tdt="UINT8", odt="UINT3"):
    thr = np.sort(
        np.random.RandomState(0).randint(0, 50, size=(channels, steps)).astype(np.float32),
        axis=-1,
    )
    return Context(
        shapes={"thresholds": (channels, steps), "inp": (1, channels), "out": (1, channels)},
        datatypes={"thresholds": DataType[tdt], "inp": DataType[idt], "out": DataType[odt]},
        initializers={"thresholds": thr},
        fpgapart=VERSAL,
        clk_ns=5.0,
    )


def test_kernel_declares_one_threshold_delivered_parameter():
    k = thresholding_kernel()
    ifaces = [dp.iface for dp in k.delivered_parameters]
    assert ifaces == ["thresholds"]


def test_both_backends_resolve_thresholds_to_embedded_topology():
    sch = thresholding_kernel_schema()
    ctx = _ctx()
    for impl in (THRESHOLDING_HLS, THRESHOLDING_RTL):
        p = resolve(sch, ctx, {"backend": impl, "PE": 2})
        assert p["parameters.thresholds.topology"] == "embedded"


def test_hls_backend_emits_baked_thresh_h_rom():
    ctx = _ctx(channels=4, steps=7, odt="UINT3")
    p = resolve(thresholding_kernel_schema(), ctx, {"backend": THRESHOLDING_HLS, "PE": 2})
    arts = emit_point(thresholding_pool(), p, ctx)
    names = {d.filename for d in arts.data_files}
    assert names == {"thresh.h"}
    body = arts.data_files[0].content
    body = body() if callable(body) else body
    assert body.startswith("static ThresholdsActivation<")


def test_rtl_threshold_dat_path_is_fused_not_routed_through_layout():
    # The boundary demonstration: the RTL binary-search .dat serializer is data-dependent
    # (self-addressed) → it must NOT depend on the shared static-schedule `layout` serializer.
    # Check the module's actual imported names, not doc-comment mentions.
    from finn.kernels.ops.thresholding import emit_rtl

    imported = set(vars(emit_rtl))
    assert "layout" not in imported, "RTL threshold emit must not import the layout serializer"
    assert "weight_constraint" not in imported
    assert "threshold_constraint" not in imported
    # The HLS threshold emit, by contrast, DOES route through layout (the separable ROM).
    from finn.kernels.ops.thresholding import emit_hls

    assert "layout" in set(vars(emit_hls))
