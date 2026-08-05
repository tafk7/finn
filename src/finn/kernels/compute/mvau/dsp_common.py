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

from finn.kernels.model.param_names import runtime_writeable_key

from .op import INPUT, THRESHOLDS, WEIGHTS

# The RTL/DSP MVU cores are integer matmuls (the signed/bitwidth gates in _rtl_mvu_feasible
# assume it). Declared as datatype support per port and shared by both DSP bundles, so the
# integer requirement has one home the pool unions for the frontend claim.
RTL_MVU_SUPPORT = {
    INPUT: DatatypeSupport(kind=DatatypeKind.INTEGER),
    WEIGHTS: DatatypeSupport(kind=DatatypeKind.INTEGER),
}


def _narrow_weights(p, ctx):
    # rtl:279-288 — data-dependent packing eligibility (MVAU-specific: reads MVAU's
    # weight tensor + mlo axis). Runtime-writable weights cannot be value-narrow-packed:
    # you cannot value-narrow weights you cannot see statically. Reads the composed delivery
    # axis directly (.get so it is safe on a param-free point — absent ⇒ static). The
    # mlo_max_iter term is packing eligibility (coord B), out of the dtype-authority scope.
    weights = ctx.initializer(WEIGHTS)
    if weights is None:
        return 0
    wdt = ctx.tensor_datatype(WEIGHTS)
    runtime_writeable = bool(p.get(runtime_writeable_key(WEIGHTS), 0))
    if np.min(weights) == wdt.min() or runtime_writeable or p.get("mlo_max_iter", 0) > 1:
        return 0
    return 1


@predicate("RTL-MVU feasibility (_mvu_rtl_possible)")
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
# wrappers), so this set is genuinely shared — each bundle appends its OWN per-core
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
    """Axes/derived/predicates common to the two MVAU DSP RTL bundles (shared CODE)."""
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
            optional_deps={runtime_writeable_key(WEIGHTS)},
        ),
    )
    predicates = (
        rtl_no_lut,
        _rtl_mvu_feasible,
        segmentlen_feasible,
        pumped_compute_needs_simd,
    )
    return axes, derived, predicates
