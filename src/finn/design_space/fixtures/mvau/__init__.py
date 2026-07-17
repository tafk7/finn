############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""The MVAU design space as a pool of composable implementation bundles.

This package is the acid test for the composability thesis (design-space-model.md
§1.2.1/§1.2.2): MVAU is decomposed into **op-level shared** elements (``shared.py``)
plus a **pool** of self-contained :class:`Implementation` bundles, one per distinct
buildable compute design, each in its own ``impl_*.py`` module that **registers
itself** (``registry.py``). Adding a backend is purely additive — drop in one
``impl_*.py`` file, import it here (or let discovery find it), edit nothing else.
This is the thesis realised *structurally*, not merely asserted: no bundle imports a
sibling, and the pool is assembled from the registry, so a new backend cannot perturb
an existing one. Source of truth for each axis/derived/predicate (file:line into real
FINN): ``kernel-design/kernel-final-design/mvau-design-space.md``.

Two structural relationships, kept distinct (see the model §1.2.2/§5):
  * SELECTION (sum) — HLS *or* RTL soft-vec *or* RTL DSP58-packed. The pool. Built
    via :func:`~finn.design_space.space.pool_schema`.
  * COMPOSITION (product) — a decoupled MVAU co-exists with a weight-delivery
    sub-kernel (memstream/dynload/fetch-weights). Deferred: it rides the ``Derived``
    primitive (§5) and needs the MLO analysis. The delivery axes in ``shared.py``
    stay op-level and are *labelled* as borrowed from that future sub-kernel.

The three built-in pool members, one per distinct compute SOURCE TEMPLATE:
  * ``mvau_hls``         (``impl_hls.py``)         — HLS core; own resType∈{lut,dsp}.
  * ``mvau_dsp_softvec`` (``impl_dsp_softvec.py``) — the unified soft-vectorized DSP
    core, ``mvu.sv``; spans DSP48E1/E2/DSP58; dsp_primitive FORCED from device.
  * ``mvau_dsp_packed``  (``impl_dsp_packed.py``)  — the DSP58 INT8-packed core,
    ``mvu_vvu_8sx9_dsp58.sv``; overlaps softvec on DSP58+small-widths ⇒ own member.

FINN today fuses softvec+packed in one wrapper (``mvu_vvu_axi.sv``) and forks between
them with a ``generate`` block (mvu_vvu_axi.sv:313). That fork — which duplicates
``mvu.sv``'s NUM_LANES math (the source's own ``@todo``, axi:305-307) and caused audit
finding F1 — is the physical manifestation of the non-separation surfaced by the two
DSP bundles' overlapping ``.sources`` (both list ``mvu_vvu_axi.sv``).

------------------------------------------------------------------------------------
2c FORWARD REQUIREMENT — physical RTL split (codegen-phase work, NOT done here)
------------------------------------------------------------------------------------
Splitting the shared wrapper is synthesizable-RTL surgery needing Vivado; it belongs
to the emit phase, which does not exist yet. No ``.sv`` is touched. Required end state,
captured so that phase needs no re-derivation:
  * Factor the shared plumbing (replay buffer, double-pump machinery axi:177-303,
    output queue axi:344-396, AXI I/O, SEGMENTLEN/NARROW_WEIGHTS param plumbing) into
    a core-agnostic ``mvu_vvu_axi_base.sv`` that instantiates the compute core as a
    parameterized sub-module — i.e. op-level plumbing, matching ``shared.py``.
  * ``mvu_vvu_axi_softvec.sv`` = base + ``mvu`` core only.
  * ``mvu_vvu_axi_packed.sv``  = base + ``mvu_vvu_8sx9_dsp58`` core only.
  * Delete the ``generate genINT8/genSoftVec`` fork and the duplicated NUM_LANES
    interception. The NUM_LANES≤3 routing becomes purely the Python-side
    ``mvau_dsp_packed`` feasibility (impl_dsp_packed.py) + preference (packed>softvec).
  * Update ``MVAU_rtl.instantiate_ip``/``get_rtl_file_list`` (rtl:168-175, 365-372)
    to emit the per-core file list selected by the resolved ``implementation``.
  Acceptance: softvec and packed bundles have DISJOINT ``.sources``; Vivado
  elaboration of each per-core wrapper matches pre-split behaviour; FINN RTL-MVU
  tests pass. Blocked on: the codegen/emit phase + Vivado.
------------------------------------------------------------------------------------

Tensor-name convention for the Context this schema resolves against:
    "inp"      the activation input tensor   (inputDataType, dynamic)
    "weights"  the weight tensor             (weightDataType + initializer VALUES)
    "out"      the output tensor             (outputDataType, unless derived)
"""

from __future__ import annotations

from finn.design_space.space import Schema, compose, pool_schema
from finn.design_space.fixtures.parameters import parameters_schema

from .names import (  # noqa: F401  (re-exported for callers/tests)
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
)
from .parameters_coupling import coupling_derived, coupling_predicates
from .registry import build_pool
from .shared import op_axes, op_derived, op_predicates

# Import the built-in bundle modules for their registration side effect. A
# third-party backend adds one such import (or is discovered) and nothing else.
from . import impl_hls  # noqa: E402,F401
from . import impl_dsp_softvec  # noqa: E402,F401
from . import impl_dsp_packed  # noqa: E402,F401


def mvau_shared():
    """The op-level shared (axes, derived, predicates) — everything every MVU has."""
    return op_axes(), op_derived(), op_predicates()


def mvau_pool():
    """The registered MVAU implementations (flat peers), in registration order."""
    return build_pool()


def mvau_schema() -> Schema:
    """The full MVAU design space as a resolve ``Schema``.

    Two selection pools composed into one space: the COMPUTE pool (``implementation``:
    HLS / DSP-softvec / DSP-packed) and the PARAMETERS pool (``parameters.topology``:
    embedded / decoupled memstream). The cross-coordinate couplings that read both the
    compute fold and the chosen topology (``weight_stream_width``, the pumpedMemory/
    fold gate) are appended to the compute op schema before compose, where both
    surfaces are in scope. Composition is a plain schema union — no new engine
    primitive ([[param-delivery-space]], ``tests/test_composition_mapping.py``)."""
    axes, derived, predicates = mvau_shared()
    op = pool_schema(
        "implementation",
        axes,
        derived + coupling_derived(),
        predicates + coupling_predicates(),
        mvau_pool(),
    )
    return compose(op, parameters_schema())
