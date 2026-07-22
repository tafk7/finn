############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The MVAU kernel component — one op definition + one file per backend.

    op.py               THE OP — read this to understand WHAT an MVAU is. Interfaces,
                        design space, datatype rules, cost, assembly, + the FINN wrapper.
    impl_hls.py         A BACKEND — the HLS compute core (copy to add a backend).
    impl_rtl_softvec.py A BACKEND — the unified soft-vectorized DSP RTL core (mvu.sv).
    impl_rtl_packed.py  A BACKEND — the DSP58 INT8-packed RTL core (mvu_vvu_8sx9_dsp58.sv).
    dsp_common.py       Shared RTL declarations for the two DSP backends (not a modeled
                        node — shared CODE only).
    emit_*.py           Tier-4 codegen (hermetic emit(point,context)->Artifacts).

This package is the acid test for the composability thesis (design-space-model.md
§1.2.1/§1.2.2): MVAU is one op (op.py) plus a **pool** of self-contained
:class:`Implementation` bundles, each in its own ``impl_*.py`` that **registers itself**
(``registry.py``). Adding a backend is purely additive — drop in one ``impl_*.py``, import
it here (or let discovery find it), edit nothing else. No bundle imports a sibling; the
pool is assembled from the registry, so a new backend cannot perturb an existing one.
Source of truth for each axis/derived/predicate: ``kernel-design/kernel-final-design/
mvau-design-space.md``.

Two structural relationships, kept distinct (model §1.2.2/§5):
  * SELECTION (sum) — HLS *or* RTL soft-vec *or* RTL DSP58-packed. The pool.
  * COMPOSITION (product) — a decoupled MVAU co-exists with a weight-delivery sub-kernel
    (the ``parameters`` pool, composed into the schema in ``op.py`` §5).

------------------------------------------------------------------------------------
2c FORWARD REQUIREMENT — physical RTL split (codegen-phase work, NOT done here)
------------------------------------------------------------------------------------
FINN today fuses softvec+packed in one wrapper (``mvu_vvu_axi.sv``) and forks between
them with a ``generate`` block (mvu_vvu_axi.sv:313). That fork duplicates ``mvu.sv``'s
NUM_LANES math (the source's own ``@todo``, axi:305-307) and caused audit finding F1 —
the physical manifestation of the non-separation surfaced by the two DSP bundles'
overlapping ``.sources`` (both list ``mvu_vvu_axi.sv``). Splitting the shared wrapper is
synthesizable-RTL surgery needing Vivado; it belongs to the emit phase. No ``.sv`` is
touched. Required end state, captured so that phase needs no re-derivation:
  * Factor the shared plumbing (replay buffer, double-pump machinery axi:177-303, output
    queue axi:344-396, AXI I/O, SEGMENTLEN/NARROW_WEIGHTS param plumbing) into a
    core-agnostic ``mvu_vvu_axi_base.sv`` that instantiates the compute core as a
    parameterized sub-module — i.e. op-level plumbing.
  * ``mvu_vvu_axi_softvec.sv`` = base + ``mvu`` core only.
  * ``mvu_vvu_axi_packed.sv``  = base + ``mvu_vvu_8sx9_dsp58`` core only.
  * Delete the ``generate genINT8/genSoftVec`` fork and the duplicated NUM_LANES
    interception. The NUM_LANES≤3 routing becomes purely the Python-side
    ``mvau_dsp_packed`` feasibility (impl_rtl_packed.py) + preference (packed>softvec).
  * Update ``MVAU_rtl.instantiate_ip``/``get_rtl_file_list`` (rtl:168-175, 365-372) to
    emit the per-core file list selected by the resolved ``implementation``.
  Acceptance: softvec and packed bundles have DISJOINT ``.sources``; Vivado elaboration of
  each per-core wrapper matches pre-split behaviour; FINN RTL-MVU tests pass. Blocked on:
  the codegen/emit phase + Vivado.
------------------------------------------------------------------------------------
"""

from __future__ import annotations

from .op import (  # noqa: F401  (re-exported public surface)
    INPUT,
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    OUTPUT,
    WEIGHTS,
    MvauKernelOp,
    mvau_kernel,
    mvau_pool,
    mvau_schema,
    mvau_shared,
)

# Import the built-in bundle modules for their registration side effect. A third-party
# backend adds one such import (or is discovered) and nothing else.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl_softvec  # noqa: E402,F401
from . import impl_rtl_packed  # noqa: E402,F401

__all__ = [
    "mvau_kernel",
    "mvau_schema",
    "mvau_pool",
    "mvau_shared",
    "MvauKernelOp",
    "MVAU_HLS",
    "MVAU_DSP_SOFTVEC",
    "MVAU_DSP_PACKED",
    "INPUT",
    "OUTPUT",
    "WEIGHTS",
]
