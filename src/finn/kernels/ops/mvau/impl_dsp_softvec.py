############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Implementation bundle: ``mvau_dsp_softvec`` — the unified soft-vectorized DSP
core (``mvu.sv``).

Spans ALL DSP primitives (DSP48E1/E2/DSP58 via ``case(VERSION)``, mvu.sv:337-710);
``dsp_primitive`` is FORCED from the device (no overlap ⇒ Derived, stays inside this
bundle). Shares DSP-RTL declarations with the packed bundle via ``dsp_rtl_common``
(shared CODE, not a modeled node) — the two remain flat pool peers.
"""

from __future__ import annotations

from finn.kernels.space import Implementation

from .dsp_common import SHARED_SOURCES, dsp_rtl_common
from .emit_rtl import emit_mvau_rtl
from .names import MVAU_DSP_SOFTVEC
from .registry import register


def _softvec_feasible(p, ctx):
    # The soft-vectorized core builds on any DSP part; the RTL-MVU gate
    # (_rtl_mvu_feasible, in dsp_rtl_common) carries the config/dtype requirements.
    # get_dsp_block always returns a block for a real part, so no extra gate here.
    return None


@register
def softvec_bundle() -> Implementation:
    axes, derived, predicates = dsp_rtl_common()
    return Implementation(
        name=MVAU_DSP_SOFTVEC,
        feasible=_softvec_feasible,
        axes=axes,
        derived=derived,
        predicates=predicates,
        sources=SHARED_SOURCES + ("mvu.sv",),
        emit=emit_mvau_rtl,
    )
