############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the BACKEND-SCOPED contract shared by the MVAU compute cores.

``kernel.py`` owns the op IDENTITY (what an MVAU is, realization-invariant). This module
owns the facts that are shared across the compute backends but are REALIZATION choices, not
identity — parked here rather than in ``kernel.py`` so the identity file stays pure:

- ``COMPUTE_STREAM`` — the default BLOCK→STREAM fold (SIMD/PE) every core adopts. Folding is
  backend-owned by construction (``Backend.ports[iface].stream``); the identity only declares
  the BLOCK. The three built-in cores fold identically, so the map lives once here; a core
  that tiled differently would compose its own.
- ``mvau_out_dtype`` / ``mvau_register_dtypes`` — the value-dependent datatype derivations
  (outputDataType on the out PORT; accDataType as an internal REGISTER). Which datatypes
  exist and how they are value-optimized is a realization fact (design-space-model §6 /
  MOTIVATION §2.2), so each core declares these on its port ``derived_dtype`` +
  ``derived_dtypes`` rather than the op declaring them — all one ``DatatypeSpec`` vocabulary.
  Identical across the three cores today; a future float core diverges by supplying different
  specs. The narrowed WEIGHT dtype is NOT here: it belongs to the storage OWNER, which
  publishes it as ``parameters.<iface>.datatype`` (a ``ParamDatatype``).

Depends only on the identity's tensor-name constants (imported from ``kernel.py``) — a
one-directional edge, no cycle. The backend modules import these via ``op.py``.
"""

from __future__ import annotations

import numpy as np
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.engine.datatype_spec import DependentSpec
from finn.kernels.engine.spec_helpers import smallest_datatype_for_range
from finn.kernels.model.param_names import param_datatype_key

from .names import INPUT, OUTPUT, THRESHOLDS, WEIGHTS


# =============================================================================
# COMPUTE TILING — the default BLOCK->STREAM lowering the pool members adopt. Positional over
# each interface's `block`: SIMD folds the reduction dim MW (inp pos 1, weights pos 0), PE
# folds the output dim MH (out pos 1, weights pos 1). The engine derives the SIMD/PE dials,
# divisibility, and widths from this — none hand-written. A tiled backend overrides its own.
# =============================================================================

COMPUTE_STREAM = {
    INPUT: [1, "SIMD"],
    OUTPUT: [1, "PE"],
    WEIGHTS: ["SIMD", "PE"],
}


# =============================================================================
# DATATYPE CONTRACT — accDataType/outputDataType (base:469-527). Data-dependent: actual weight
# VALUES when the storage owner has visibility, worst-case dtype envelope otherwise. The
# static-vs-worst-case CHOICE is no longer re-derived here — it is READ off the storage owner's
# published pd (parameters.<iface>.datatype.values_visible). The accumulator is
# compute-core-owned storage; it depends on delivery only THROUGH that published bit, never by
# peeking at storage it does not own (design-space-model §2). The ParamDatatype is a parameters-
# pool derived, so accDataType declares a dep on it (DependentSpec below) — the unified topo-sort
# orders acc after storage.
# =============================================================================


def _acc_datatype(p, ctx):
    # base:469-527. The two algorithms are TODAY'S two branches, now selected by the owner's
    # published authority instead of a re-derived op-side predicate:
    #   visible (owner sees values) -> per-column value-eval over the real matrix (FINN-tight)
    #   blind   (runtime-writable/…) -> worst-case dtype envelope over an (MW,MH) bounds matrix
    # values_visible mirrors the pre-seam static-vs-runtime-writable gate on the composed point
    # by construction, so this is bit-identical to the old sizing; the HW golden is the guard.
    idt = ctx.tensor_datatype(INPUT)
    pd = p[param_datatype_key(WEIGHTS)]
    weights = ctx.initializer(WEIGHTS)
    # values_visible authorizes narrowing; weights-present is the orthogonal value-availability
    # safety check (a visible owner with no materialized initializer still falls to envelope).
    if pd is not None and pd.values_visible and weights is not None:
        acc_min, acc_max = calculate_matvec_accumulator_range(weights, idt)
    else:
        wdt = ctx.tensor_datatype(WEIGHTS)
        mw, mh = ctx.tensor_shape(WEIGHTS)
        lower = wdt.min() * np.ones((mw, mh))
        upper = wdt.max() * np.ones((mw, mh))
        lo_r = calculate_matvec_accumulator_range(lower, idt)
        hi_r = calculate_matvec_accumulator_range(upper, idt)
        acc_min = min(min(lo_r), min(hi_r))
        acc_max = max(max(lo_r), max(hi_r))
    return smallest_datatype_for_range(float(acc_min), float(acc_max))


def _output_datatype(p, ctx):
    # base:517 — outputDataType = accDataType when there is NO activation (output IS the
    # accumulator), else the graph output dtype (the thresholds map the accumulator down).
    # Emergent: no threshold operand ⇒ no activation.
    if not ctx.has_tensor(THRESHOLDS):
        return _acc_datatype(p, ctx)
    return ctx.tensor_datatype(OUTPUT)


def mvau_out_dtype():
    """The out port's produced-dtype :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec`
    — the ``_output_datatype`` callable (accumulator under ``noActivation``, else the graph
    output dtype). Declared on each backend's ``out`` port ``derived_dtype`` so the stream-
    width fold reads the realized output type. Backend-scoped: a future float core supplies a
    different rule.

    Wrapped in a :class:`~finn.kernels.engine.datatype_spec.DependentSpec` with the WEIGHTS
    storage dep: under ``noActivation`` the output IS the accumulator, so the out-port stream
    width transitively reads the ParamDatatype and must order after it. The tiling engine's
    ``stream_width.out`` derived inherits these deps (:func:`~finn.kernels.model.tiling._width_derived`)."""
    return DependentSpec(_output_datatype, deps={param_datatype_key(WEIGHTS)})


def mvau_register_dtypes():
    """The BACKEND-SCOPED internal-register dtype specs (base:469-527) — ``{accDataType}``, a
    :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` callable. Declared on each
    compute backend's :attr:`~finn.kernels.model.backend.Backend.derived_dtypes` (no port —
    an internal register), so ``pool_space`` merges it onto the point under its name (emit
    reads ``point.accDataType`` unchanged). All three MVAU cores narrow IDENTICALLY today; a
    future core diverges by supplying different specs. The out-port's ``outputDataType`` is
    the sibling ``derived_dtype`` on the port (:func:`mvau_out_dtype`).

    ``accDataType`` reads the WEIGHTS storage owner's published pd, a parameters-pool
    derived — so it is wrapped in a :class:`~finn.kernels.engine.datatype_spec.DependentSpec`
    declaring a dep on ``param_datatype_key(WEIGHTS)``. The unified topo-sort (R2) orders the
    accumulator after that pd, across the compute/parameters pool boundary.

    The narrowed WEIGHT dtype is no longer a compute register: the storage OWNER publishes it
    as ``parameters.<iface>.datatype`` (:class:`~finn.kernels.engine.param_datatype.ParamDatatype`),
    since datatype authority belongs to whoever owns the values at rest. Weight serialization
    reads the graph dtype directly (HLS ``context.tensor_datatype(WEIGHTS)``, RTL
    ``point.narrow_weights``), so there was never a runtime reader of the old register."""
    return {
        "accDataType": DependentSpec(_acc_datatype, deps={param_datatype_key(WEIGHTS)}),
    }
