############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Op-agnostic DSP-RTL helpers shared by every op whose RTL backend targets the
finn-rtllib MVU/VVU compute cores (MVAU, VVAU, …).

These read only the device (``ctx.fpgapart``) and generic point fields (``SIMD``,
``pumpedCompute`` via ``.get``), never an op-specific tensor — so they belong here,
not in any one op's package. Op-SPECIFIC DSP logic (MVAU's ``_rtl_mvu_feasible`` /
``narrow_weights``, each op's source-file list) stays in that op's package.
"""

from __future__ import annotations

import math

from finn.kernels.engine.predicate import predicate
from finn.util.basic import get_dsp_block

# DSP block -> $VERSION$ (silicon consequence, FORCED from fpgapart; §1.2.2).
VERSION = {"DSP48E1": 1, "DSP48E2": 2, "DSP58": 3}


def _dsp_block(ctx):
    # FORCED from the device (util:391). A part-less trial/probe context (``fpgapart == ""``)
    # cannot resolve a DSP block — raise a NARROW ``ValueError`` (the "unresolvable for this
    # probe" signal T0.1 narrowed on) rather than letting ``get_dsp_block("")`` surface an
    # IndexError read as a kernel bug. A real device probe resolves cleanly.
    if not ctx.fpgapart:
        raise ValueError("DSP block needs a non-empty fpgapart (part-less probe context)")
    return get_dsp_block(ctx.fpgapart)


def dsp_primitive(p, ctx):
    # No alternative design ⇒ Derived, never a pool axis (§1.2.2).
    return _dsp_block(ctx)


def dsp_version(p, ctx):
    return VERSION[_dsp_block(ctx)]


def segmentlen(p, ctx):
    # rtl:230-251 — derived from clk target, SIMD, pumpedCompute. pumpedCompute is
    # read via .get so ops without that axis (VVAU) resolve to the unpumped path.
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


def num_lanes(w, a, version, narrow) -> int:
    # Faithful port of the elaboration math (mvu_vvu_axi.sv:308-311, mvu.sv:113-115).
    a_width = 25 + 2 * (1 if version > 1 else 0)
    if a_width == w:
        return 1
    min_lane_width = w + a - 1
    return 1 + (a_width - (0 if narrow else 1) - w) // min_lane_width


@predicate("RTL rejects resType=lut", deps={"resType"})
def rtl_no_lut(p, ctx):
    # Defensive: DSP RTL cores expose no LUT path, so the resType domain is {dsp} and
    # lut is already rejected at the domain check; kept for an explicit message
    # (rtl:256).
    if p.resType == "lut":
        return "LUT-based RTL compute core is not supported (rtl:256)"
    return None


@predicate(
    "SEGMENTLEN feasible only if ref_clk > 0.741ns", optional_deps={"pumpedCompute"}
)
def segmentlen_feasible(p, ctx):
    clk = ctx.clk if ctx.clk is not None else 5.0
    ref_clk = clk / 2 if p.get("pumpedCompute", 0) else clk
    if ref_clk <= 0.741:
        return f"Infeasible clk target: ref_clk={ref_clk}ns <= 0.741ns (rtl:242)"
    return None


@predicate("pumpedCompute => SIMD != 1", deps={"SIMD"}, optional_deps={"pumpedCompute"})
def pumped_compute_needs_simd(p, ctx):
    if p.get("pumpedCompute", 0) and p.SIMD == 1:
        return "pumpedCompute with SIMD=1 is not meaningful (rtl:334)"
    return None
