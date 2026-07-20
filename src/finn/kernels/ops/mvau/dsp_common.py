############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Shared CODE for the two MVAU DSP RTL bundles (soft-vec and packed).

softvec and packed are flat PEERS in the pool; they merely share declarations, which
we dedup by CODE here rather than by introducing a "substrate" node in the engine.
The op-agnostic DSP-RTL helpers (dsp_primitive/dsp_version/segmentlen/num_lanes and
the generic RTL predicates) live in ``fixtures/_dsp_rtl.py`` and are reused across
ops; this module holds only the MVAU-SPECIFIC pieces (``narrow_weights``, the
``_rtl_mvu_feasible`` gate, the MVAU source-file list).
"""

from __future__ import annotations

import numpy as np

from finn.kernels.ops._dsp_rtl import (
    dsp_primitive,
    dsp_version,
    num_lanes,  # noqa: F401  (re-exported for impl_dsp_packed)
    pumped_compute_needs_simd,
    rtl_no_lut,
    segmentlen,
    segmentlen_feasible,
)
from finn.kernels.space import Derived, discrete_axis, predicate
from finn.util.basic import get_dsp_block

from .names import INPUT, WEIGHTS


def _narrow_weights(p, ctx):
    # rtl:279-288 — data-dependent packing eligibility (MVAU-specific: reads MVAU's
    # weight tensor + dynamic_input/mlo axes).
    weights = ctx.initializer(WEIGHTS)
    if weights is None:
        return 0
    wdt = ctx.tensor_datatype(WEIGHTS)
    if np.min(weights) == wdt.min() or p.get("dynamic_input", 0) or p.get("mlo_max_iter", 0) > 1:
        return 0
    return 1


@predicate("RTL-MVU feasibility (_mvu_rtl_possible)")
def _rtl_mvu_feasible(p, ctx):
    # The real gate deciding whether an RTL-MVU can be used at all
    # (specialize_layers.py:235 `_mvu_rtl_possible`). Reads config + device + data.
    if p.noActivation != 1:
        return "RTL-MVU requires noActivation=1 (embedded thresholds unsupported)"
    if p.binaryXnorMode != 0:
        return "RTL-MVU does not support binaryXnorMode"
    wdt = ctx.tensor_datatype(WEIGHTS)
    idt = ctx.tensor_datatype(INPUT)
    if not wdt.signed():
        return "RTL-MVU requires signed weights"
    if idt.bitwidth() < 2 or wdt.bitwidth() < 2:
        return "RTL-MVU requires input and weight bitwidth >= 2"
    # DSP48E1 supports only narrow weights (device + data). narrow_weights is a
    # Derived already on the point.
    if get_dsp_block(ctx.fpgapart) == "DSP48E1" and not p.narrow_weights:
        return "RTL-MVU on DSP48E1 requires narrow weights (specialize_layers:268)"
    return None


# The AXI wrapper + shared plumbing both MVAU DSP cores compile against. Both DSP
# bundles list `mvu_vvu_axi.sv` — the OVERLAP that surfaces the non-separation (2b)
# and motivates the 2c split (documented in the package __init__).
SHARED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "add_multi.sv",
)


def dsp_rtl_common():
    """Axes/derived/predicates common to the two MVAU DSP RTL bundles (shared CODE)."""
    axes = (
        # Double-pumped DSP compute — a real lever on the DSP RTL cores (rtl:53).
        discrete_axis("pumpedCompute", {0, 1}, 0),
        # RTL exposes no LUT path today: resType is DSP-only (rtl:256).
        discrete_axis("resType", {"dsp"}, "dsp"),
    )
    derived = (
        Derived("language", lambda p, ctx: "rtl"),
        Derived("dsp_primitive", dsp_primitive),
        Derived("dsp_version", dsp_version),
        Derived("SEGMENTLEN", segmentlen),
        Derived("narrow_weights", _narrow_weights),
    )
    predicates = (
        rtl_no_lut,
        _rtl_mvu_feasible,
        segmentlen_feasible,
        pumped_compute_needs_simd,
    )
    return axes, derived, predicates
