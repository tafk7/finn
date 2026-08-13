############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Shared CODE for the two MVAU DSP RTL backends (soft-vec and packed).

softvec and packed are flat PEERS in the pool; they merely share declarations, which
we dedup by CODE here rather than by introducing a "substrate" node in the engine.
The op-agnostic DSP-RTL helpers (dsp_primitive/dsp_version/segmentlen/num_lanes and
the generic RTL predicates) live in ``compute/mvau/_dsp_rtl.py``; this module holds only
the MVAU-SPECIFIC pieces (``narrow_weights``, the
``_rtl_mvu_feasible`` gate, the MVAU source-file list).
"""

from __future__ import annotations

import numpy as np

from finn.kernels.compute.mvau._dsp_rtl import (
    dsp_primitive,
    dsp_version,
    num_lanes,  # noqa: F401  (re-exported for impl_rtl_packed)
    pumped_compute_needs_simd,
    rtl_no_lut,
    segmentlen,
    segmentlen_feasible,
)
from finn.kernels.engine.axis import discrete_axis
from finn.kernels.engine.datatype_support import DatatypeKind, DatatypeSupport
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.predicate import predicate
from finn.util.basic import get_dsp_block

from .names import INPUT, THRESHOLDS, WEIGHTS

# The RTL/DSP MVU cores are integer matmuls (the signed/bitwidth gates in _rtl_mvu_feasible
# assume it). Declared as datatype support per port and shared by both DSP backends, so the
# integer requirement has one home the pool unions for the frontend claim.
RTL_MVU_SUPPORT = {
    INPUT: DatatypeSupport(kind=DatatypeKind.INTEGER),
    WEIGHTS: DatatypeSupport(kind=DatatypeKind.INTEGER),
}


def _narrow_weights(p, ctx):
    # rtl:279-288 — data-dependent packing eligibility. NARROW_WEIGHTS is a RANGE COMMITMENT
    # to the DSP packing logic, not a dtype width: `mvu.sv:44` defines it as "weights in
    # (-W:W) rather than [-W:W)", i.e. a promise that no weight equals wdt.min(). The RTL
    # spends that promise on LANE DENSITY — the non-narrow path reserves a guard bit
    # (`mvu.sv:115,135`), so narrow weights fit more SIMD lanes per DSP. The commitment is
    # checked ONLY in simulation (`mvu.sv:281-288` `$warning`); in silicon a violation is
    # silently wrong arithmetic.
    #
    # The unifying rule is VISIBILITY: the promise is only sound if the build can SEE the
    # values it is promising about.
    #
    #   * runtime-writable — the driver overwrites the values after the bitstream ships. A
    #     phase-0 Context MANDATE, so this term is a given rather than a delivery choice,
    #     which is what lets `_rtl_mvu_feasible` consult this derived without asking about a
    #     post-specialization value.
    #   * MLO — weights are fetched per iteration from external memory through
    #     `fetch_weights_wrapper`, so no single static set exists to narrow.
    #
    # WE DELIBERATELY DIVERGE FROM FINN ON THE RUNTIME-WRITABLE TERM. Baseline
    # (`matrixvectoractivation_rtl.py:305-315`) gates only on
    # `mem_mode in ["dynamic", "external_mem"]` and emits NARROW_WEIGHTS=1 for a
    # runtime-writable node, deriving the promise from an initializer the driver may replace.
    # That is unsound, and FINN's own siblings in the same file disagree with it:
    # `minimize_weight_bit_width` (:562) and `minimize_accumulator_width` (:514) both guard on
    # `runtime_writeable_weights OR mem_mode in ["external", "external_mem", "dynamic"]` — a
    # STRICT SUPERSET of the guard here. Three value-derived quantities, one missing the
    # check, no comment anywhere acknowledging it: an oversight, not a considered exemption.
    # Sharpest case: UltraScale + ram_style=ultra FORCES runtime_writeable_weights=1
    # (`matrixvectoractivation.py:1004`), the memstream ships INIT_FILE="" — no weights in the
    # bitstream at all — and the wrapper still claims NARROW_WEIGHTS=1.
    #
    # Observable consequence: on DSP48E1 at runtime_writeable=1 our `_rtl_mvu_feasible`
    # rejects the RTL cores where FINN admits them, so such a node routes to HLS. Pinned by
    # `tests/compute/mvau/test_resolve.py::test_runtime_writeable_suppresses_narrow_weights`.
    #
    # The MLO term is NOT a divergence: FINN reaches the same 0 by another route, since
    # mlo_max_iter>0 forces mem_mode="external_mem" (`matrixvectoractivation_rtl.py:63-73`),
    # which its mem_mode guard already catches.
    #
    # MLO is TRUTHY here, not `> 1`. Every baseline reader gates on truthiness
    # (`hwcustomop.py:374` `en_mlo = "EN_MLO" if mlo_max_iter else "NO_MLO"`;
    # `matrixvectoractivation_rtl.py:72`, `thresholding.py:133`), and it is the right test
    # for THIS question: `mlo_max_iter == 1` still routes weights through the external fetch
    # path, so the values are just as invisible as at N > 1. `> 1` conflated this with the
    # separate GEOMETRY question — memstream's `SETS < 2` really does collapse to the
    # single-set hardware (`memstream.sv:75` genSingleSet) — and answering both with one
    # comparison narrowed weights the build cannot see.
    weights = ctx.initializer(WEIGHTS)
    if weights is None:
        return 0
    wdt = ctx.tensor_datatype(WEIGHTS)
    if (
        np.min(weights) == wdt.min()
        or ctx.is_runtime_writeable(WEIGHTS)
        or p.get("mlo_max_iter", 0)
    ):
        return 0
    return 1


@predicate(
    "RTL-MVU feasibility (_mvu_rtl_possible)",
    # The DSP48E1 arm reads `narrow_weights`. Declared now that it is honest to do so: with
    # runtime-writability a phase-0 Context mandate, narrow_weights no longer depends on any
    # POST-SPECIALIZATION value, so consulting it during a capability query is well-founded.
    # Optional because the derived exists only in the DSP RTL backends.
    optional_deps={"narrow_weights"},
)
def _rtl_mvu_feasible(p, ctx):
    # The real gate deciding whether an RTL-MVU can be used at all
    # (specialize_layers.py:235 `_mvu_rtl_possible`). Reads config + device + data.
    # Emergent: the RTL/DSP core has no activation/threshold logic, so it is feasible only
    # on a node WITHOUT thresholds — the emergent form of the old `noActivation==1` gate,
    # and equivalently "the DSP core cannot consume thresholds" (there is no threshold port).
    if ctx.has_tensor(THRESHOLDS):
        return "RTL-MVU cannot consume thresholds (embedded thresholding unsupported; specialize_layers:241)"
    wdt = ctx.tensor_datatype(WEIGHTS)
    idt = ctx.tensor_datatype(INPUT)
    # Integer i/w is declared as datatype support (RTL_MVU_SUPPORT, compiled by pool_space
    # into a guarded predicate). It runs alongside this gate — the signed/bitwidth checks
    # below assume integer types, which the support gate guarantees.
    if not wdt.signed():
        return "RTL-MVU requires signed weights"
    if idt.bitwidth() < 2 or wdt.bitwidth() < 2:
        return "RTL-MVU requires input and weight bitwidth >= 2"
    # DSP48E1 supports only narrow weights (device + data). narrow_weights is a
    # Derived already on the point.
    if get_dsp_block(ctx.fpgapart) == "DSP48E1" and not p.narrow_weights:
        return "RTL-MVU on DSP48E1 requires narrow weights (specialize_layers:268)"
    return None


# The core-agnostic AXI wrapper body + shared plumbing both MVAU DSP cores compile
# against. Post-2c-split, the fused `mvu_vvu_axi.sv` is retired from our source lists:
# the shared body lives in the two `.svh` fragments (included by the per-core `.sv`
# wrappers), so this set is genuinely shared — each backend appends its OWN per-core
# wrapper + core on top. This removes the internal genINT8/genSoftVec `generate` fork
# whose two-core closure made the fused source list incomplete/ambiguous.
SHARED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi_base_head.svh",
    "mvu_vvu_axi_base_tail.svh",
    "replay_buffer.sv",
    "add_multi.sv",
)


def dsp_rtl_common():
    """Axes/derived/predicates common to the two MVAU DSP RTL backends (shared CODE)."""
    axes = (
        # Double-pumped DSP compute — a real lever on the DSP RTL cores (rtl:53).
        discrete_axis("pumpedCompute", {0, 1}, 0),
        # RTL exposes no LUT path today: resType is DSP-only (rtl:256).
        discrete_axis("resType", {"dsp"}, "dsp"),
    )
    derived = (
        Derived("dsp_primitive", dsp_primitive),
        Derived("dsp_version", dsp_version),
        # SEGMENTLEN reads the fold (SIMD) and the double-pump knob; narrow_weights reads
        # the weight-storage runtime-writable bit and the MLO iteration count. Declared so
        # each one's stratum — what must be pinned before it is knowable — is inferable.
        Derived("SEGMENTLEN", segmentlen, deps={"SIMD", "pumpedCompute"}),
        Derived(
            "narrow_weights",
            _narrow_weights,
            deps={"mlo_max_iter"},
        ),
    )
    predicates = (
        rtl_no_lut,
        _rtl_mvu_feasible,
        segmentlen_feasible,
        pumped_compute_needs_simd,
    )
    return axes, derived, predicates
