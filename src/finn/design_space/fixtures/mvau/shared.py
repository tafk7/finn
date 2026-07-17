############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
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

from finn.design_space.primitives.spec_helpers import smallest_datatype_for_range
from finn.design_space.space import (
    Derived,
    discrete_axis,
    divisor_axis,
    fixed_axis,
    predicate,
    predicate_axis,
)
from finn.util.basic import is_versal

from .names import DECOUPLED, EMBEDDED, EXTERNAL, INPUT, OUTPUT, WEIGHTS


# =============================================================================
# Small guard/helper functions (named, not lambdas, for legible tracebacks).
# =============================================================================


def _is_decoupled(p) -> bool:
    return p.mem_mode == DECOUPLED


def _is_decoupled_or_external(p) -> bool:
    return p.mem_mode in (DECOUPLED, EXTERNAL)


def _has_activation(p) -> bool:
    return p.noActivation == 0


def weights_may_change(p) -> bool:
    """Weights are not statically known (runtime-writeable / external / dynamic /
    multi-layer-offload) — accDataType/weightDataType must use worst-case bounds
    rather than actual values (base:482-498). Exposed for bundle derived that share
    the same data-staticness test (e.g. narrow_weights)."""
    return bool(
        p.get("runtime_writeable_weights", 0)
        or p.mem_mode == EXTERNAL
        or p.get("mlo_max_iter", 0)
        or p.get("dynamic_input", 0)
    )


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
        # --- weight-delivery cluster -----------------------------------------
        # RESERVED COMPOSITION SEAM (design-space-model.md §5): mem_mode is really a
        # *selector* over which weight-delivery SUB-KERNEL is composed in
        # (internal_decoupled -> memstream, dynamic_input -> dynload, mlo -> fetch-
        # weights; rtl:312-322). ram_style/runtime_writeable/pumpedMemory/sip_depth
        # are that sub-kernel's OWN axes/derived, surfaced here at the op level only
        # because composition is deferred. When the memstream Kernel lands, this
        # whole cluster MOVES OUT through a `weight_delivery` Derived (a Derived
        # returning a resolved sub-Point) rather than being re-modelled.
        discrete_axis("mem_mode", {EMBEDDED, DECOUPLED, EXTERNAL}, DECOUPLED),
        discrete_axis(
            "ram_style",
            {"auto", "block", "distributed", "ultra"},
            "auto",
            guard=_is_decoupled,
            deps={"mem_mode"},
        ),
        discrete_axis(
            "runtime_writeable_weights", {0, 1}, 0, guard=_is_decoupled, deps={"mem_mode"}
        ),
        discrete_axis("pumpedMemory", {0, 1}, 0, guard=_is_decoupled, deps={"mem_mode"}),
        discrete_axis(
            "dynamic_input",
            {0, 1},
            0,
            guard=_is_decoupled_or_external,
            deps={"mem_mode"},
        ),
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


def _weight_stream_width(p, ctx):
    # base:256-275 — 0 for embedded (no port); PE*SIMD*wbits otherwise.
    if p.mem_mode == EMBEDDED:
        return 0
    return p.PE * p.SIMD * ctx.tensor_datatype(WEIGHTS).bitwidth()


def _outstream_width(p, ctx):
    return _output_datatype(p, ctx).bitwidth() * p.PE


def op_derived():
    return (
        Derived("WMEM", _wmem),
        Derived("TMEM", _tmem),
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
        Derived("instream_width", _instream_width),
        Derived("weight_stream_width", _weight_stream_width),
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


@predicate("pumpedMemory => not (PE == SIMD == 1)")
def _pumped_memory_not_1x1(p, ctx):
    if p.get("pumpedMemory", 0) and p.PE == 1 and p.SIMD == 1:
        return "pumpedMemory with PE=SIMD=1 is a known-bad configuration (base:717)"
    return None


@predicate("ram_style=ultra & not versal => runtime_writeable=1")
def _uram_requires_ultrascale(p, ctx):
    # THE combination gate — reads point AND device in one condition (hls:147).
    if "ram_style" not in p:
        return None
    if (
        p.ram_style == "ultra"
        and not p.get("mlo_max_iter", 0)
        and not is_versal(ctx.fpgapart)
        and p.get("runtime_writeable_weights", 0) != 1
    ):
        return (
            "URAM weights on a non-Versal (UltraScale) device require "
            "runtime_writeable_weights=1 (hls:147)"
        )
    return None


@predicate("weight initializer must exist unless external/dynamic/mlo")
def _weights_present(p, ctx):
    if ctx.initializer(WEIGHTS) is None:
        if not (p.mem_mode == EXTERNAL or p.get("dynamic_input", 0) or p.get("mlo_max_iter", 0)):
            return "weight initializer required unless external/dynamic/mlo (base:782)"
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
        _pumped_memory_not_1x1,
        _uram_requires_ultrascale,
        _weights_present,
    )
