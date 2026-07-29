############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend bundle: ``mvau_dsp_softvec`` — the unified soft-vectorized DSP
core (``mvu.sv``).

Spans ALL DSP primitives (DSP48E1/E2/DSP58 via ``case(VERSION)``, mvu.sv:337-710);
``dsp_primitive`` is FORCED from the device (no overlap ⇒ Derived, stays inside this
bundle). Shares DSP-RTL declarations with the packed bundle via ``dsp_rtl_common``
(shared CODE, not a modeled node) — the two remain flat pool peers.
"""

from __future__ import annotations

from finn.kernels.model.backend import Backend
from finn.kernels.model.param_names import STREAM

from .dsp_common import SHARED_SOURCES, dsp_rtl_common
from .emit_rtl import _V_WRAPPER_SCHEMA, emit_mvau_rtl
from .op import COMPUTE_STREAM, MVAU_DSP_SOFTVEC, WEIGHTS
from .registry import register


# softvec has NO extra feasibility gate: the soft-vectorized core builds on any DSP part,
# and the shared RTL-MVU gate (_rtl_mvu_feasible, in dsp_rtl_common) carries the
# config/dtype requirements.
@register
def softvec_bundle() -> Backend:
    axes, derived, predicates = dsp_rtl_common()
    return Backend(
        name=MVAU_DSP_SOFTVEC,
        language="rtl",
        # rtl_core_module names the per-core wrapper the emitted top instantiates
        # (2c split): softvec owns mvu_vvu_axi_softvec.sv + mvu.sv, disjoint from packed.
        rtl_core_module="mvu_vvu_axi_softvec",
        axes=axes,
        derived=derived,
        predicates=predicates,
        sources=SHARED_SOURCES + ("mvu_vvu_axi_softvec.sv", "mvu.sv"),
        emit=emit_mvau_rtl,
        # The typed contract for the emitted top. softvec + packed share this ONE
        # `_V_WRAPPER_SCHEMA` object (N:1 by reference) — the per-core wrapper difference
        # rides the `rtl_core_module`/`MODULE_NAME_COMPUTE_CORE` slot, not a fork of the schema.
        schema=_V_WRAPPER_SCHEMA,
        stream=COMPUTE_STREAM,
        # The RTL/DSP core is a streamed-weight core: it has NO embedded-weight path (base
        # FINN: internal_embedded is HLS-only), so it consumes weights in STREAM mode only —
        # this makes the `embedded` topology illegal for it (a correctness fix, not a new
        # restriction). It also has no activation logic; the _rtl_mvu_feasible gate rejects
        # any node WITH thresholds, so thresholds never reaches delivery here (no consumes
        # entry needed — an absent interface is permissive, and the gate is the real rejecter).
        consumes={WEIGHTS: {STREAM}},
    )
