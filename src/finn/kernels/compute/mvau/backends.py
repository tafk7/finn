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
- ``mvau_dtype_backend`` — the value-dependent datatype derivations
  (accDataType/weightDataType/outputDataType). Which datatypes exist and how they are
  value-optimized is a realization fact (design-space-model §6 / MOTIVATION §2.2), so each
  core composes these into its own ``derived`` rather than the op declaring them. Identical
  across the three cores today; a future float core diverges by composing a different helper.

Depends only on the identity's tensor-name constants + ``weights_may_change`` (imported from
``kernel.py``) — a one-directional edge, no cycle. The impl bundles import these via ``op.py``.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.engine.derived import Derived
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
# DATATYPE CONTRACT — accDataType/weightDataType/outputDataType (base:469-549). Data-dependent:
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


def _weight_datatype(p, ctx):
    # base:529-549 — VALUE_OPTIMIZED narrow, only when weights are statically known.
    weights = ctx.initializer(WEIGHTS)
    if weights is None or weights_may_change(p):
        return ctx.tensor_datatype(WEIGHTS)
    w_min = float(weights.min())
    w_max = float(weights.max())
    if w_min < 0:
        extreme = w_min if abs(w_min) > w_max else -w_max - 1
        return DataType.get_smallest_possible(extreme)
    return DataType.get_smallest_possible(w_max)


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


def mvau_dtype_backend():
    """The BACKEND-SCOPED datatype derivations (base:469-549) —
    accDataType/weightDataType/outputDataType. Composed by each backend rather than declared
    on the op identity: all three MVAU cores narrow IDENTICALLY today, but a future backend
    (e.g. a float core) diverges by composing a different helper — or none. ``outputDataType``
    reads ``accDataType`` as a plain function call, so the tuple order here is irrelevant."""
    return (
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
    )
