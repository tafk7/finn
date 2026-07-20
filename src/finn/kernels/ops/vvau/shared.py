############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""VVAU (VectorVectorActivation) **op-level shared** elements — everything every VVU
implementation has, regardless of the chosen compute core.

VVAU differs from MVAU in its FOLDING geometry (vectorvectoractivation.py:237-288):
the depthwise inner-product dim is the kernel window ``K = k_h*k_w`` and the output
dim is ``Channels``. **PE folds Channels** (nf = Channels//PE); **SIMD folds the
kernel window** (sf = K//SIMD). Contrast MVAU where PE/SIMD fold matrix rows/cols.
The weight-delivery cluster (mem_mode/ram_style/…) is shared by both impls, same as
MVAU. Source of truth: ``kernel-design/kernel-final-design/mvau-design-space.md`` +
``vectorvectoractivation.py``.

Context convention: the weight tensor ``"weights"`` shape is ``(Channels, 1, k_h,
k_w)`` (FINN depthwise layout), so ``Channels`` and ``K`` are read from it.
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
from finn.util.basic import is_versal

from .names import DECOUPLED, EMBEDDED, EXTERNAL, INPUT, OUTPUT, WEIGHTS


# =============================================================================
# Small guard/helper functions.
# =============================================================================


def _is_decoupled(p) -> bool:
    return p.mem_mode == DECOUPLED


def _is_decoupled_or_external(p) -> bool:
    return p.mem_mode in (DECOUPLED, EXTERNAL)


def _has_activation(p) -> bool:
    return p.noActivation == 0


def weights_may_change(p) -> bool:
    """Weights not statically known (runtime-writeable / external / dynamic) — accum
    and weight dtype must use worst-case bounds rather than actual values."""
    return bool(
        p.get("runtime_writeable_weights", 0)
        or p.mem_mode == EXTERNAL
        or p.get("dynamic_input", 0)
    )


def _channels(p, ctx):
    # weights shape (Channels, 1, k_h, k_w) -> Channels.
    return ctx.tensor_shape(WEIGHTS)[0]


def _kernel_window(p, ctx):
    # weights shape (Channels, 1, k_h, k_w) -> k_h * k_w.
    shp = ctx.tensor_shape(WEIGHTS)
    return int(shp[2] * shp[3])


def _is_int_list(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v)


# =============================================================================
# Op-level SHARED axes.
# =============================================================================


def op_axes():
    return (
        # --- context-fixed folding dims (addressed like axes) ----------------
        fixed_axis("Channels", _channels),
        fixed_axis("K", _kernel_window),  # kernel window k_h*k_w
        # --- VVAU folding: PE folds Channels, SIMD folds the kernel window ----
        divisor_axis("PE", "Channels", 1, deps={"Channels"}),
        divisor_axis("SIMD", "K", 1, deps={"K"}),
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
        # --- weight-delivery cluster (shared by both impls, same as MVAU) ----
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
        discrete_axis(
            "dynamic_input", {0, 1}, 0, guard=_is_decoupled_or_external, deps={"mem_mode"}
        ),
    )


# =============================================================================
# Op-level SHARED derived.
# =============================================================================


def _wmem(p, ctx):
    # calc_wmem: (k_h*k_w*ch // PE) // SIMD  (vectorvectoractivation.py:281-288).
    return (p.K * p.Channels // p.PE) // p.SIMD


def _tmem(p, ctx):
    return p.Channels // p.PE if p.noActivation == 0 else 0


def _acc_datatype(p, ctx):
    # minimize_accumulator_width (vectorvectoractivation.py:395-452): worst-case type
    # bounds when weights may change, actual weight VALUES when static. The inner
    # product spans the kernel window K.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if weights_may_change(p) or weights is None:
        lower = wdt.min() * np.ones((p.K, 1))
        upper = wdt.max() * np.ones((p.K, 1))
        lo_r = calculate_matvec_accumulator_range(lower, idt)
        hi_r = calculate_matvec_accumulator_range(upper, idt)
        acc_min = min(min(lo_r), min(hi_r))
        acc_max = max(max(lo_r), max(hi_r))
    else:
        flat = weights.reshape(-1, 1)
        acc_min, acc_max = calculate_matvec_accumulator_range(flat, idt)
    return smallest_datatype_for_range(float(acc_min), float(acc_max))


def _output_datatype(p, ctx):
    if p.noActivation == 1:
        return _acc_datatype(p, ctx)
    return ctx.tensor_datatype(OUTPUT)


def _instream_width(p, ctx):
    # in0 = i_bits * SIMD * PE  (vectorvectoractivation.py:201-206).
    return ctx.tensor_datatype(INPUT).bitwidth() * p.SIMD * p.PE


def _weight_stream_width(p, ctx):
    # weights = SIMD * PE * wp for decoupled/external, else 0 (:207-217).
    if p.mem_mode == EMBEDDED:
        return 0
    return p.SIMD * p.PE * ctx.tensor_datatype(WEIGHTS).bitwidth()


def _outstream_width(p, ctx):
    # out = o_bits * PE  (:232-235).
    return _output_datatype(p, ctx).bitwidth() * p.PE


def op_derived():
    return (
        Derived("WMEM", _wmem),
        Derived("TMEM", _tmem),
        Derived("accDataType", _acc_datatype),
        Derived("outputDataType", _output_datatype),
        Derived("instream_width", _instream_width),
        Derived("weight_stream_width", _weight_stream_width),
        Derived("outstream_width", _outstream_width),
    )


# =============================================================================
# Op-level SHARED predicates.
# =============================================================================


@predicate("Channels % PE == 0")
def _pe_divides_channels(p, ctx):
    return None if p.Channels % p.PE == 0 else f"Channels={p.Channels} not divisible by PE={p.PE}"


@predicate("K (k_h*k_w) % SIMD == 0")
def _simd_divides_k(p, ctx):
    return None if p.K % p.SIMD == 0 else f"K={p.K} not divisible by SIMD={p.SIMD}"


@predicate("ram_style=ultra & not versal => runtime_writeable=1")
def _uram_requires_ultrascale(p, ctx):
    if "ram_style" not in p:
        return None
    if (
        p.ram_style == "ultra"
        and not is_versal(ctx.fpgapart)
        and p.get("runtime_writeable_weights", 0) != 1
    ):
        return (
            "URAM weights on a non-Versal (UltraScale) device require "
            "runtime_writeable_weights=1"
        )
    return None


@predicate("weight initializer must exist unless external/dynamic")
def _weights_present(p, ctx):
    if ctx.initializer(WEIGHTS) is None:
        if not (p.mem_mode == EXTERNAL or p.get("dynamic_input", 0)):
            return "weight initializer required unless external/dynamic"
    return None


def op_predicates():
    return (
        _pe_divides_channels,
        _simd_divides_k,
        _uram_requires_ultrascale,
        _weights_present,
    )
