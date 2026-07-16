############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Shared CODE for the two DSP RTL bundles (soft-vec and packed).

softvec and packed are flat PEERS in the pool; they merely share declarations, which
we dedup by CODE here rather than by introducing a "substrate" node in the engine.
The genuinely-shared *plumbing* (replay/pump/output-queue) lives at the op level
(``shared.py``), not here — this module is only the DSP-RTL-specific
axes/derived/predicates and the shared source-file list.
"""

from __future__ import annotations

import math

import numpy as np

from finn.design_space.space import Derived, discrete_axis, predicate
from finn.util.basic import get_dsp_block

from .names import VERSION, INPUT, WEIGHTS


def _dsp_primitive(p, ctx):
    # FORCED from the device: the part physically has exactly one DSP block
    # (util:391). No alternative design ⇒ Derived, never a pool axis (§1.2.2).
    return get_dsp_block(ctx.fpgapart)


def _dsp_version(p, ctx):
    return VERSION[get_dsp_block(ctx.fpgapart)]


def _segmentlen(p, ctx):
    # rtl:230-251 — derived from clk target, SIMD, pumpedCompute.
    clk = ctx.clk if ctx.clk is not None else 5.0
    if p.get("pumpedCompute", 0):
        ref_clk = clk / 2
        simd_factor = 6
    else:
        ref_clk = clk
        simd_factor = 3
    critical_path_dsps = math.floor((ref_clk - 0.741) / 0.605 + 1)
    max_chain_len = math.ceil(p.SIMD / simd_factor)
    return min(critical_path_dsps, max_chain_len)


def _narrow_weights(p, ctx):
    # rtl:279-288 — data-dependent packing eligibility.
    weights = ctx.initializer(WEIGHTS)
    if weights is None:
        return 0
    wdt = ctx.tensor_datatype(WEIGHTS)
    if np.min(weights) == wdt.min() or p.get("dynamic_input", 0) or p.get("mlo_max_iter", 0) > 1:
        return 0
    return 1


@predicate("RTL rejects resType=lut")
def _rtl_no_lut(p, ctx):
    # Defensive: the DSP resType domain is {dsp}, so lut is already rejected at the
    # domain check; kept for an explicit message (rtl:256).
    if p.resType == "lut":
        return "LUT-based RTL MVU is not supported (rtl:256)"
    return None


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


@predicate("SEGMENTLEN feasible only if ref_clk > 0.741ns")
def _segmentlen_feasible(p, ctx):
    clk = ctx.clk if ctx.clk is not None else 5.0
    ref_clk = clk / 2 if p.get("pumpedCompute", 0) else clk
    if ref_clk <= 0.741:
        return f"Infeasible clk target: ref_clk={ref_clk}ns <= 0.741ns (rtl:242)"
    return None


@predicate("pumpedCompute => SIMD != 1")
def _pumped_compute_needs_simd(p, ctx):
    if p.get("pumpedCompute", 0) and p.SIMD == 1:
        return "pumpedCompute with SIMD=1 is not meaningful (rtl:334)"
    return None


# The AXI wrapper + shared plumbing both DSP cores compile against. Both DSP bundles
# list `mvu_vvu_axi.sv` — the OVERLAP that surfaces the non-separation (2b) and
# motivates the 2c split (documented in the package __init__).
SHARED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "add_multi.sv",
)


def dsp_rtl_common():
    """Axes/derived/predicates common to the two DSP RTL bundles (shared CODE)."""
    axes = (
        # Double-pumped DSP compute — a real lever on the DSP RTL cores (rtl:53).
        discrete_axis("pumpedCompute", {0, 1}, 0),
        # RTL exposes no LUT path today: resType is DSP-only (rtl:256).
        discrete_axis("resType", {"dsp"}, "dsp"),
    )
    derived = (
        Derived("language", lambda p, ctx: "rtl"),
        Derived("dsp_primitive", _dsp_primitive),
        Derived("dsp_version", _dsp_version),
        Derived("SEGMENTLEN", _segmentlen),
        Derived("narrow_weights", _narrow_weights),
    )
    predicates = (
        _rtl_no_lut,
        _rtl_mvu_feasible,
        _segmentlen_feasible,
        _pumped_compute_needs_simd,
    )
    return axes, derived, predicates


def num_lanes(w, a, version, narrow) -> int:
    # Faithful port of the elaboration math (mvu_vvu_axi.sv:308-311, mvu.sv:113-115).
    a_width = 25 + 2 * (1 if version > 1 else 0)
    if a_width == w:
        return 1
    min_lane_width = w + a - 1
    return 1 + (a_width - (0 if narrow else 1) - w) // min_lane_width
