############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Backend bundle: ``mvau_dsp_packed`` — the DSP58 INT8-packed core
(``mvu_vvu_8sx9_dsp58.sv``).

It OVERLAPS softvec on DSP58+small-widths (the wrapper's ``else: genSoftVec`` branch
proves softvec also builds there) ⇒ a genuine CHOICE, its own pool member. Its
feasibility is its OWN — it computes NUM_LANES for real (audit F1) rather than
replicating the shared wrapper's ``generate`` fork.
"""

from __future__ import annotations

from finn.kernels.engine.predicate import predicate
from finn.kernels.model.backend import Backend
from finn.kernels.model.param_names import STREAM
from finn.util.basic import get_dsp_block

from .dsp_common import SHARED_SOURCES, dsp_rtl_common, num_lanes
from .emit_rtl import _V_WRAPPER_SCHEMA, emit_mvau_rtl
from .op import COMPUTE_STREAM, INPUT, MVAU_DSP_PACKED, VERSION, WEIGHTS
from .registry import register


@predicate("mvau_dsp_packed feasibility (DSP58 ∧ w<=8 ∧ a<=9 ∧ NUM_LANES<=3)")
def _packed_feasible(p, ctx):
    # F1 — the DSP58 INT8-packed core (mvu_vvu_8sx9_dsp58.sv) is feasible only under
    # the FULL generate condition (mvu_vvu_axi.sv:313): DSP58 AND w<=8 AND a<=9 AND
    # NUM_LANES<=3. The old predicate dropped the NUM_LANES<=3 term on a false lemma
    # ("lanes<=3 follows from w<=8 & a<=9") — false: small widths yield MORE lanes
    # (W=2,A=2 on DSP58 -> 9 lanes -> FINN routes to softvec). Compute NUM_LANES for
    # real. This is packed's OWN feasibility, not a fork-replication.
    dsp = get_dsp_block(ctx.fpgapart)
    if dsp != "DSP58":
        return f"{MVAU_DSP_PACKED} requires DSP58; {ctx.fpgapart} has {dsp}"
    w = ctx.tensor_datatype(WEIGHTS).bitwidth()
    a = ctx.tensor_datatype(INPUT).bitwidth()
    if w > 8 or a > 9:
        return (
            f"{MVAU_DSP_PACKED} requires weight_width<=8 (got {w}) and "
            f"activation_width<=9 (got {a}) (mvu_vvu_axi.sv:313)"
        )
    lanes = num_lanes(w, a, VERSION[dsp], p.narrow_weights)
    if lanes > 3:
        return (
            f"{MVAU_DSP_PACKED} requires NUM_LANES<=3 (got {lanes} for "
            f"w={w}, a={a}, narrow={p.narrow_weights}); FINN routes this to softvec "
            f"(mvu_vvu_axi.sv:311-313)"
        )
    return None


@register
def packed_bundle() -> Backend:
    axes, derived, predicates = dsp_rtl_common()
    return Backend(
        name=MVAU_DSP_PACKED,
        language="rtl",
        # rtl_core_module names the per-core wrapper the emitted top instantiates
        # (2c split): packed owns mvu_vvu_axi_packed.sv + mvu_vvu_8sx9_dsp58.sv,
        # disjoint from softvec.
        rtl_core_module="mvu_vvu_axi_packed",
        axes=axes,
        derived=derived,
        # packed's OWN feasibility gate, appended to the shared RTL predicates.
        predicates=predicates + (_packed_feasible,),
        sources=SHARED_SOURCES + ("mvu_vvu_axi_packed.sv", "mvu_vvu_8sx9_dsp58.sv"),
        emit=emit_mvau_rtl,
        # Shares the ONE `_V_WRAPPER_SCHEMA` with softvec (N:1 by reference); see softvec.
        schema=_V_WRAPPER_SCHEMA,
        stream=COMPUTE_STREAM,
        # Streamed-weight DSP core (see softvec) — weights STREAM-only (embedded illegal),
        # thresholds rejected by the _rtl_mvu_feasible gate, so no thresholds consumes entry.
        consumes={WEIGHTS: {STREAM}},
    )
