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
  publishes it as ``parameters.<iface>.storageDataType`` (a ``StorageDescriptor``).

Depends only on the identity's tensor-name constants + ``weights_may_change`` (imported from
``kernel.py``) — a one-directional edge, no cycle. The impl bundles import these via ``op.py``.
"""

from __future__ import annotations

import numpy as np
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.engine.spec_helpers import smallest_datatype_for_range

from .kernel import INPUT, OUTPUT, THRESHOLDS, WEIGHTS, weights_may_change


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
# DATATYPE CONTRACT — accDataType/outputDataType (base:469-527). Data-dependent:
# actual weight VALUES when static, worst-case bounds otherwise, gated by weights_may_change.
# =============================================================================


def _acc_datatype(p, ctx):
    # base:469-527 — worst-case type bounds when weights may change, actual weight
    # VALUES when static. The canonical data-dependent Derived.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if weights_may_change(p) or weights is None:
        mw, mh = ctx.tensor_shape(WEIGHTS)
        lower = wdt.min() * np.ones((mw, mh))
        upper = wdt.max() * np.ones((mw, mh))
        lo_r = calculate_matvec_accumulator_range(lower, idt)
        hi_r = calculate_matvec_accumulator_range(upper, idt)
        acc_min = min(min(lo_r), min(hi_r))
        acc_max = max(max(lo_r), max(hi_r))
    else:
        acc_min, acc_max = calculate_matvec_accumulator_range(weights, idt)
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
    different rule."""
    return _output_datatype


def mvau_register_dtypes():
    """The BACKEND-SCOPED internal-register dtype specs (base:469-527) — ``{accDataType}``, a
    :class:`~finn.kernels.engine.datatype_spec.DatatypeSpec` callable. Declared on each
    compute backend's :attr:`~finn.kernels.model.backend.Backend.derived_dtypes` (no port —
    an internal register), so ``pool_space`` merges it onto the point under its name (emit
    reads ``point.accDataType`` unchanged). All three MVAU cores narrow IDENTICALLY today; a
    future core diverges by supplying different specs. The out-port's ``outputDataType`` is
    the sibling ``derived_dtype`` on the port (:func:`mvau_out_dtype`).

    The narrowed WEIGHT dtype is no longer a compute register: the storage OWNER publishes it
    as ``parameters.<iface>.storageDataType`` (:class:`~finn.kernels.engine.storage_descriptor.StorageDescriptor`),
    since datatype authority belongs to whoever owns the values at rest. Weight serialization
    reads the graph dtype directly (HLS ``context.tensor_datatype(WEIGHTS)``, RTL
    ``point.narrow_weights``), so there was never a runtime reader of the old register."""
    return {
        "accDataType": _acc_datatype,
    }
