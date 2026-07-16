############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Implementation bundle: ``vvau_rtl`` — the singleton RTL compute core.

Unlike MVAU (which has TWO RTL pool members, softvec + packed), VVU has exactly ONE
reachable RTL core: the DSP58 INT8-packed core. In the shared wrapper
``mvu_vvu_axi.sv:313`` the ``!IS_MVU`` term forces the ``genINT8`` branch
unconditionally, so ``mvu.sv`` (softvec) is compiled-in but never elaborated for VVU
(``vectorvectoractivation_rtl.py:272`` sets ``$IS_MVU$=0``; ``:252-266``
``_resolve_dsp_version`` always returns 3 and asserts Versal + non-lut). VVU is
therefore Versal/DSP58-only — that is this bundle's OWN feasibility.

The wrapper source list is shared with MVAU (``mvu.sv`` appears but is dead for VVU) —
the same ``.sources`` overlap that motivates the 2c split, here made explicit.
"""

from __future__ import annotations

from finn.design_space.fixtures._dsp_rtl import (
    dsp_primitive,
    dsp_version,
    pumped_compute_needs_simd,
    rtl_no_lut,
    segmentlen,
    segmentlen_feasible,
)
from finn.design_space.space import Derived, Implementation, discrete_axis
from finn.util.basic import is_versal

from .names import INPUT, VVAU_RTL, WEIGHTS
from .registry import register


def _vvu_rtl_feasible(p, ctx):
    # The real gate (specialize_layers.py:281 `_vvu_rtl_possible`): DSP58/Versal only,
    # noActivation, signed weights <=8b, activations <=8b (or 9b signed).
    if not is_versal(ctx.fpgapart):
        return f"{VVAU_RTL} requires a Versal (DSP58) device; {ctx.fpgapart} is not Versal"
    if p.noActivation != 1:
        return "RTL-VVU requires noActivation=1 (embedded thresholds unsupported)"
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    in_ok = idt.bitwidth() <= 8 or (idt.bitwidth() == 9 and idt.min() < 0)
    if not in_ok:
        return f"RTL-VVU activation width out of range (got {idt.bitwidth()})"
    if wdt.bitwidth() > 8:
        return f"RTL-VVU weight width must be <=8 (got {wdt.bitwidth()})"
    if wdt.min() >= 0:
        return "RTL-VVU requires signed weights"
    return None


@register
def rtl_bundle() -> Implementation:
    return Implementation(
        name=VVAU_RTL,
        feasible=_vvu_rtl_feasible,
        axes=(
            # RTL exposes no LUT path: resType is DSP-only.
            discrete_axis("resType", {"dsp"}, "dsp"),
        ),
        derived=(
            Derived("language", lambda p, ctx: "rtl"),
            Derived("dsp_primitive", dsp_primitive),
            Derived("dsp_version", dsp_version),  # always 3 (Versal) once feasible
            Derived("SEGMENTLEN", segmentlen),
        ),
        predicates=(rtl_no_lut, segmentlen_feasible, pumped_compute_needs_simd),
        # mvu.sv is listed but dead for VVU (only genINT8 is reachable) — documented
        # overlap, same 2c motivation as MVAU.
        sources=(
            "mvu_pkg.sv",
            "mvu_vvu_axi.sv",
            "replay_buffer.sv",
            "mvu.sv",
            "mvu_vvu_8sx9_dsp58.sv",
            "add_multi.sv",
        ),
    )
