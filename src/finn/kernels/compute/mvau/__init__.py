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
:class:`Backend` bundles, each in its own ``impl_*.py`` that **registers itself**
(``registry.py``). Adding a backend is purely additive — drop in one ``impl_*.py``, import
it here (or let discovery find it), edit nothing else. No bundle imports a sibling; the
pool is assembled from the registry, so a new backend cannot perturb an existing one.
Source of truth for each axis/derived/predicate:
``scratchpad/reference/mvau-design-space.md``.

Two structural relationships, kept distinct (model §1.2.2/§5):
  * SELECTION (sum) — HLS *or* RTL soft-vec *or* RTL DSP58-packed. The pool.
  * COMPOSITION (product) — a decoupled MVAU co-exists with a weight-delivery sub-kernel
    (the ``parameters`` pool, composed into the schema in ``op.py`` §5).

------------------------------------------------------------------------------------
2c — physical RTL split (DONE; ported from feature/mvu-wrapper-split, rtlsim-verified)
------------------------------------------------------------------------------------
FINN's fused wrapper (``mvu_vvu_axi.sv``) forked softvec/packed with a ``generate``
block (mvu_vvu_axi.sv:313) that duplicated ``mvu.sv``'s NUM_LANES math (the source's own
``@todo``, axi:305-307; audit finding F1). Because that one file references BOTH cores,
any bundle shipping it had an incomplete/ambiguous transitive closure — the physical
non-separation surfaced by the two DSP bundles' overlapping ``.sources``. The split
resolves it at the source:
  * Shared core-agnostic plumbing (params, AXI I/O, replay buffer, double-pump machinery,
    input unflatten/VVU interleave, flow-control, output queue) lives in
    ``mvu_vvu_axi_base_head.svh`` + ``mvu_vvu_axi_base_tail.svh`` — ``include`` fragments,
    NO ``generate`` fork, NO NUM_LANES math (core selection is now purely Python-side).
  * ``mvu_vvu_axi_softvec.sv`` = head + ``mvu`` core + tail.
  * ``mvu_vvu_axi_packed.sv``  = head + ``mvu_vvu_8sx9_dsp58`` core + tail.
  * NUM_LANES≤3 routing is purely ``mvau_dsp_packed`` feasibility (impl_rtl_packed.py) +
    preference (packed>softvec). Emit selects the per-core wrapper via the selected
    backend's ``rtl_core_module`` field (emit_rtl.py ``$MODULE_NAME_COMPUTE_CORE$`` slot); each
    bundle's ``.sources`` are now DISJOINT on the core/wrapper (share only the base
    ``.svh``). The fused ``mvu_vvu_axi.sv`` is retired from our emit path (kept in-tree
    only as the golden for the rtlsim bit-equivalence oracle).
  Verified: ``rtlsim_split_equiv_mvau.py`` proves each per-core wrapper is BIT-IDENTICAL
  to the fused wrapper's matching fork branch (softvec on DSP48E2, packed on DSP58/INT8);
  ``elaborate_mvau_emit.py`` elaborates each bundle against its own disjoint source set.
  Because this forks vendored HDL, the byte-``run_diff`` oracle no longer applies to the
  instantiation line; the rtlsim-equivalence check is its behavioural replacement.
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
    mvau_space,
)

# Import the built-in bundle modules for their registration side effect. A third-party
# backend adds one such import (or is discovered) and nothing else.
from . import impl_hls  # noqa: E402,F401
from . import impl_rtl_softvec  # noqa: E402,F401
from . import impl_rtl_packed  # noqa: E402,F401

__all__ = [
    "mvau_kernel",
    "mvau_space",
    "mvau_pool",
    "MvauKernelOp",
    "MVAU_HLS",
    "MVAU_DSP_SOFTVEC",
    "MVAU_DSP_PACKED",
    "INPUT",
    "OUTPUT",
    "WEIGHTS",
]
