############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU **op-level shared** elements — everything every MVU has, regardless of the
chosen compute core. This is the op "contract" that all implementation bundles
resolve against; a bundle never edits this file.

Includes the AXI/replay/output plumbing that only *looks* softvec/packed-shared
(every MVU needs it), the PE×SIMD combinatorial mass, the threshold cluster, and the
weight-delivery cluster (a RESERVED COMPOSITION SEAM — see below). Source of truth
with file:line into real FINN: ``kernel-design/kernel-final-design/mvau-design-space.md``.
"""

from __future__ import annotations

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.primitives.spec_helpers import smallest_datatype_for_range
from finn.kernels.space import (
    Derived,
    discrete_axis,
    divisor_axis,
    fixed_axis,
    predicate,
    predicate_axis,
)
from finn.kernels.ops.parameters.names import RUNTIME_WRITEABLE

from .names import INPUT, OUTPUT, WEIGHTS


# =============================================================================
# Small guard/helper functions (named, not lambdas, for legible tracebacks).
# =============================================================================


def _has_activation(p) -> bool:
    return p.noActivation == 0


def weights_may_change(p) -> bool:
    """Weights are not statically known — accDataType/weightDataType must use
    worst-case bounds rather than actual values (base:482-498). This is the one
    CROSS-COORDINATE coupling from the parameters subsystem back into the compute
    dtype derivations: staticness (coordinate C) is decided by the composed
    ``parameters.*`` fields. Reads with ``.get`` so it is safe on a point where the
    parameters pool is absent (a future param-free op) — absent ⇒ statically known.

    Increment-1 topologies are ``embedded`` (static) and ``decoupled`` (static unless
    runtime-writable). The external / dynamic / MLO staticness sources return when
    those topologies land (each will be its own ``parameters.topology`` value)."""
    return bool(p.get(RUNTIME_WRITEABLE, 0))


def _matrix_dim(idx):
    def default(p, ctx):
        return ctx.tensor_shape(WEIGHTS)[idx]

    return default


def _is_int_list(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v)


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


# =============================================================================
# Op-level SHARED axes — present under every implementation.
# =============================================================================


def op_axes():
    return (
        # --- context-fixed matrix dims (addressed like axes) -----------------
        fixed_axis("MW", _matrix_dim(0)),
        fixed_axis("MH", _matrix_dim(1)),
        # --- the real PE x SIMD combinatorial mass ---------------------------
        divisor_axis("PE", "MH", 1, deps={"MH"}),
        divisor_axis("SIMD", "MW", 1, deps={"MW"}),
        # --- activation / threshold cluster ----------------------------------
        discrete_axis("noActivation", {0, 1}, 0),
        predicate_axis(
            "ActVal",
            "int",
            lambda v: isinstance(v, int),
            0,
            guard=_has_activation,
            deps={"noActivation"},
        ),
        discrete_axis(
            "ram_style_thresholds",
            {"auto", "block", "distributed"},
            "auto",
            guard=_has_activation,
            deps={"noActivation"},
        ),
        discrete_axis("binaryXnorMode", {0, 1}, 0),
        predicate_axis("numInputVectors", "list[int]", _is_int_list, [1]),
        # F4: mlo_max_iter is a per-node iteration count, unbounded non-neg int.
        # The `64` in the old {0..64} domain was n_max_layers (a fabric-wide MLO
        # table size, hwcustomop.py:378) — a different entity that does not bound
        # this axis (hwcustomop.py:317-319, mlo_max_iter unbounded).
        predicate_axis("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
        # --- weight-delivery cluster: MOVED OUT to the `parameters` pool ------
        # mem_mode/ram_style/runtime_writeable_weights/pumpedMemory/dynamic_input used
        # to live here as the "reserved composition seam". They are now the
        # `parameters` subsystem (fixtures/parameters/), composed into the MVAU schema
        # via `compose(...)` under the `parameters.*` namespace. mem_mode is gone:
        # being the `decoupled` topology IS "internal_decoupled". The cross-coordinate
        # couplings (memstream geometry, the pumpedMemory/fold gate) are contributed at
        # compose time by mvau/parameters_coupling.py. See
        # kernel-design/kernel-final-design/param-delivery-design-space.md.
    )


# =============================================================================
# Op-level SHARED derived — computed for every implementation.
# =============================================================================


def _wmem(p, ctx):
    return p.MW * p.MH // (p.PE * p.SIMD)


def _tmem(p, ctx):
    return p.MH // p.PE if p.noActivation == 0 else 0


def _acc_datatype(p, ctx):
    # base:469-527 — worst-case type bounds when weights may change, actual weight
    # VALUES when static. The canonical data-dependent Derived.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if p.binaryXnorMode == 1 and weights is not None:
        weights = 2 * weights - 1
    if weights_may_change(p) or weights is None:
        lower = wdt.min() * np.ones((p.MW, p.MH))
        upper = wdt.max() * np.ones((p.MW, p.MH))
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
    # base:517 — outputDataType = accDataType when noActivation, else the graph dtype.
    if p.noActivation == 1:
        return _acc_datatype(p, ctx)
    return ctx.tensor_datatype(OUTPUT)


def _instream_width(p, ctx):
    return ctx.tensor_datatype(INPUT).bitwidth() * p.SIMD


def _outstream_width(p, ctx):
    return _output_datatype(p, ctx).bitwidth() * p.PE


def op_derived():
    # NOTE: `weight_stream_width` (0 for embedded, PE*SIMD*wbits otherwise) is
    # CROSS-COORDINATE — it reads both the compute fold AND the parameters topology —
    # so it is contributed at compose time by mvau/parameters_coupling.py, not here.
    return (
        Derived("WMEM", _wmem),
        Derived("TMEM", _tmem),
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
        Derived("instream_width", _instream_width),
        Derived("outstream_width", _outstream_width),
    )


# =============================================================================
# Op-level SHARED predicates — one kind; provenance is what each reads.
# =============================================================================


@predicate("MH % PE == 0")
def _pe_divides_mh(p, ctx):
    return None if p.MH % p.PE == 0 else f"MH={p.MH} not divisible by PE={p.PE}"


@predicate("MW % SIMD == 0")
def _simd_divides_mw(p, ctx):
    return None if p.MW % p.SIMD == 0 else f"MW={p.MW} not divisible by SIMD={p.SIMD}"


# The `pumpedMemory => not(PE==SIMD==1)` gate and the `ram_style=ultra & not versal
# => runtime_writeable=1` URAM gate MOVED to the parameters subsystem: the URAM gate
# is self-contained in the decoupled topology bundle; the pumpedMemory/fold gate is
# cross-coordinate (reads the compute fold) and is contributed at compose time by
# mvau/parameters_coupling.py.


@predicate("weight initializer must exist unless params are not statically known")
def _weights_present(p, ctx):
    # Weights must exist as an initializer unless the parameters subsystem says they
    # are not statically known (runtime-writable / external / dynamic / MLO). Reads
    # the composed staticness via weights_may_change (parameters.*), with .get safety
    # for a future param-free op (no parameters pool ⇒ still requires an initializer).
    if ctx.initializer(WEIGHTS) is None:
        if not weights_may_change(p):
            return "weight initializer required unless params are not static (base:782)"
    return None


# NOTE (audit F2, DROPPED): a `bipolar x bipolar => nonneg thresholds` predicate used
# to live here, but it checked the scalar `ActVal` (the threshold activation's bias,
# base:156) whereas FINN's assertion is over the THRESHOLD TENSOR VALUES
# (`orig_thres_matrix >= 0`, base:578) — a different object. The fixture's Context
# models inp/weights/out; thresholds are NOT a first-class Context tensor, so the
# correct action is to drop it. Reinstate when thresholds become a Context tensor.


def op_predicates():
    return (
        _pe_divides_mh,
        _simd_divides_mw,
        _weights_present,
    )
