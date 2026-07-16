############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""The MVAU decision tree encoded as a resolve ``Schema``.

This is the acid test for the generic design-space model: every construct in
``design-space-model.md`` §1 earns its place in the worked MVAU tree of §8. The
source of truth for each axis/derived/predicate (with file:line into real FINN)
is ``kernel-final-design/mvau-design-space.md``. Numbers are cross-checked against
``finn/custom_op/fpgadataflow/matrixvectoractivation.py`` (base), its ``hls/`` and
``rtl/`` subclasses, and ``finn/util/basic.py``.

Tensor-name convention for the Context this schema resolves against:
    "inp"      the activation input tensor   (inputDataType, dynamic)
    "weights"  the weight tensor             (weightDataType + initializer VALUES)
    "out"      the output tensor             (outputDataType, unless derived)

The schema is built by :func:`mvau_schema`. It reads matrix dims MW/MH from the
weight tensor shape, so one schema serves any Context.
"""

from __future__ import annotations

import math

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.design_space.primitives.spec_helpers import smallest_datatype_for_range
from finn.design_space.space import (
    Axis,
    Derived,
    Predicate,
    Schema,
    discrete_axis,
    divisor_axis,
    fixed_axis,
    predicate,
    predicate_axis,
)
from finn.util.basic import get_dsp_block, is_versal

# =============================================================================
# The implementation realization pool — keyed on COMPUTE DESIGN, not language.
# =============================================================================
#
# A pool member is one distinct compute SOURCE TEMPLATE — the test for pool
# membership (design-space-model.md §1.2.2). `language` (hls/rtl) and
# `dsp_primitive` (DSP48E1/E2/DSP58) are DERIVED properties of the chosen design,
# never independent choices; {hls, rtl} is an arbitrary language grouping, not the
# axis. Three buildable designs, one per template:
#
#   mvau_hls          HLS compute core (language=hls).
#   mvau_dsp_softvec  the generic soft-vectorized DSP core, mvu.sv — any DSP part
#                     (DSP48E1/E2/DSP58, VERSION-parameterized). language=rtl.
#   mvau_dsp_packed   the DSP58 INT8-packed core, mvu_vvu_8sx9_dsp58.sv — 3 MACs
#                     per DSP58, w<=8 x a<=9. language=rtl.
#
# These are peers in the pool, NOT a divergence from FINN. FINN selects between
# them with a forced `generate` block (mvu_vvu_axi.sv:313, keyed on device+widths).
# That is FINN hardcoding a PREFERENCE ("prefer packed when it fits") into
# silicon-generation code — it is not evidence that packed is the only design. Both
# cores build on a DSP58+INT8 config (the `else` branch proves softvec is valid
# there too); packed is merely better. We decompose FINN's buried `generate` into
# its true parts: three pool members, a device+dtype feasibility PREDICATE that
# narrows the pool, and a PREFERENCE (packed > softvec when both feasible). The
# pool is defined by buildable designs, never contracted by current-device
# feasibility — future-proof by construction (§1.2.2).

MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

REALIZATION_POOL = frozenset({MVAU_HLS, MVAU_DSP_SOFTVEC, MVAU_DSP_PACKED})

# Designs whose compute core is a DSP RTL core (dsp_primitive is FORCED from the
# device here, and pumpedCompute exists only here — rtl:53).
_DSP_REALIZATIONS = frozenset({MVAU_DSP_SOFTVEC, MVAU_DSP_PACKED})

# language is a DERIVED property of the chosen design, never a free choice.
_LANGUAGE = {
    MVAU_HLS: "hls",
    MVAU_DSP_SOFTVEC: "rtl",
    MVAU_DSP_PACKED: "rtl",
}

DECOUPLED = "internal_decoupled"
EMBEDDED = "internal_embedded"
EXTERNAL = "external"

WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"


# =============================================================================
# Small guard helpers (named, not lambdas, for legible tracebacks).
# =============================================================================


def _is_decoupled(p) -> bool:
    return p.mem_mode == DECOUPLED


def _is_decoupled_or_external(p) -> bool:
    return p.mem_mode in (DECOUPLED, EXTERNAL)


def _has_activation(p) -> bool:
    return p.noActivation == 0


def _is_dsp_impl(p) -> bool:
    return p.implementation in _DSP_REALIZATIONS


def _weights_may_change(p) -> bool:
    """Weights are not statically known (runtime-writeable / external / dynamic /
    multi-layer-offload) — accDataType/weightDataType must use worst-case bounds
    rather than actual values (base:482-498)."""
    return bool(
        p.get("runtime_writeable_weights", 0)
        or p.mem_mode == EXTERNAL
        or p.get("mlo_max_iter", 0)
        or p.get("dynamic_input", 0)
    )


# =============================================================================
# Axes
# =============================================================================


def _matrix_dim(idx):
    def default(p, ctx):
        return ctx.tensor_shape(WEIGHTS)[idx]

    return default


def mvau_axes():
    return (
        # --- root choice: the realization pool -------------------------------
        discrete_axis("implementation", REALIZATION_POOL, MVAU_HLS),
        # --- context-fixed matrix dims (addressed like axes) -----------------
        fixed_axis("MW", _matrix_dim(0)),
        fixed_axis("MH", _matrix_dim(1)),
        # --- the real PE x SIMD combinatorial mass ---------------------------
        divisor_axis("PE", "MH", 1, deps={"MH"}),
        divisor_axis("SIMD", "MW", 1, deps={"MW"}),
        # --- resType: domain is a property of the chosen realization ---------
        Axis(
            name="resType",
            domain=_restype_domain,
            default=_restype_default,
            deps=frozenset({"implementation"}),
        ),
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
        predicate_axis(
            "numInputVectors",
            "list[int]",
            _is_int_list,
            [1],
        ),
        discrete_axis("mlo_max_iter", set(range(0, 65)), 0),
        # --- weight-delivery cluster -----------------------------------------
        discrete_axis("mem_mode", {EMBEDDED, DECOUPLED, EXTERNAL}, DECOUPLED),
        discrete_axis(
            "ram_style",
            {"auto", "block", "distributed", "ultra"},
            "auto",
            guard=_is_decoupled,
            deps={"mem_mode"},
        ),
        discrete_axis(
            "runtime_writeable_weights",
            {0, 1},
            0,
            guard=_is_decoupled,
            deps={"mem_mode"},
        ),
        discrete_axis(
            "pumpedMemory", {0, 1}, 0, guard=_is_decoupled, deps={"mem_mode"}
        ),
        discrete_axis(
            "dynamic_input",
            {0, 1},
            0,
            guard=_is_decoupled_or_external,
            deps={"mem_mode"},
        ),
        # --- double-pumped DSP compute: exists only on a DSP RTL core --------
        discrete_axis(
            "pumpedCompute", {0, 1}, 0, guard=_is_dsp_impl, deps={"implementation"}
        ),
    )


def _restype_domain(p, ctx):
    impl = p.implementation
    if impl == MVAU_HLS:
        return frozenset({"lut", "dsp"})  # hls:58 default lut, dsp available
    # DSP compute designs are DSP-only: the RTL lut path is not exposed (rtl:256).
    return frozenset({"dsp"})


def _restype_default(p, ctx):
    if p.implementation in _DSP_REALIZATIONS:
        return "dsp"
    return "lut"  # hls default (hls:58)


def _is_int_list(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v)


# =============================================================================
# Derived quantities (computed after axes are fixed; never dimensions)
# =============================================================================


def _wmem(p, ctx):
    return p.MW * p.MH // (p.PE * p.SIMD)


def _tmem(p, ctx):
    return p.MH // p.PE if p.noActivation == 0 else 0


def _language(p, ctx):
    return _LANGUAGE[p.implementation]


def _dsp_primitive(p, ctx):
    # FORCED from the device inside a DSP realization; the part physically has one
    # (util:391). None for non-DSP realizations.
    if p.implementation in _DSP_REALIZATIONS:
        return get_dsp_block(ctx.fpgapart)
    return None


_VERSION = {"DSP48E1": 1, "DSP48E2": 2, "DSP58": 3}


def _dsp_version(p, ctx):
    prim = _dsp_primitive(p, ctx)
    return _VERSION[prim] if prim is not None else None


def _segmentlen(p, ctx):
    # rtl:230-251 — derived from clk target, SIMD, pumpedCompute.
    if p.implementation not in _DSP_REALIZATIONS:
        return None
    clk = ctx.clk if ctx.clk is not None else 5.0
    if p.get("pumpedCompute", 0):
        ref_clk = clk / 2
        simd_factor = 6
    else:
        ref_clk = clk
        simd_factor = 3
    critical_path_dsps = math.floor((ref_clk - 0.741) / 0.605 + 1)
    max_chain_len = math.ceil(p.SIMD / simd_factor)
    return min(critical_path_dsps, max_chain_len)


def _narrow_weights(p, ctx):
    # rtl:279-288 — data-dependent packing eligibility.
    weights = ctx.initializer(WEIGHTS)
    if weights is None:
        return 0
    wdt = ctx.tensor_datatype(WEIGHTS)
    if np.min(weights) == wdt.min() or p.get("dynamic_input", 0) or p.get("mlo_max_iter", 0) > 1:
        return 0
    return 1


def _acc_datatype(p, ctx):
    # base:469-527 — worst-case type bounds when weights may change, actual weight
    # VALUES when static. The canonical data-dependent Derived.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if p.binaryXnorMode == 1 and weights is not None:
        weights = 2 * weights - 1
    if _weights_may_change(p) or weights is None:
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
    if weights is None or _weights_may_change(p):
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


def mvau_derived():
    return (
        Derived("WMEM", _wmem),
        Derived("TMEM", _tmem),
        Derived("language", _language),
        Derived("dsp_primitive", _dsp_primitive),
        Derived("dsp_version", _dsp_version),
        Derived("SEGMENTLEN", _segmentlen),
        Derived("narrow_weights", _narrow_weights),
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
        Derived("instream_width", _instream_width),
        Derived("weight_stream_width", _weight_stream_width),
        Derived("outstream_width", _outstream_width),
    )


# =============================================================================
# Predicates — one kind; provenance is what each reads, not a declared phase.
# =============================================================================


@predicate("MH % PE == 0")
def _pe_divides_mh(p, ctx):
    return None if p.MH % p.PE == 0 else f"MH={p.MH} not divisible by PE={p.PE}"


@predicate("MW % SIMD == 0")
def _simd_divides_mw(p, ctx):
    return None if p.MW % p.SIMD == 0 else f"MW={p.MW} not divisible by SIMD={p.SIMD}"


@predicate("pumpedCompute => SIMD != 1")
def _pumped_compute_needs_simd(p, ctx):
    if p.get("pumpedCompute", 0) and p.SIMD == 1:
        return "pumpedCompute with SIMD=1 is not meaningful (rtl:334)"
    return None


@predicate("pumpedMemory => not (PE == SIMD == 1)")
def _pumped_memory_not_1x1(p, ctx):
    if p.get("pumpedMemory", 0) and p.PE == 1 and p.SIMD == 1:
        return "pumpedMemory with PE=SIMD=1 is a known-bad configuration (base:717)"
    return None


@predicate("HLS: SIMD >= MW/1024")
def _hls_simd_lower_bound(p, ctx):
    if p.language == "hls" and p.SIMD < p.MW / 1024:
        return f"HLS array-partition limit: SIMD={p.SIMD} < MW/1024={p.MW / 1024} (hls:216)"
    return None


@predicate("RTL rejects resType=lut")
def _rtl_no_lut(p, ctx):
    if p.language == "rtl" and p.resType == "lut":
        return "LUT-based RTL MVU is not supported (rtl:256)"
    return None


@predicate("mvau_dsp_packed feasibility (DSP58 & w<=8 & a<=9)")
def _packed_feasible(p, ctx):
    # The DSP58 INT8-packed core (mvu_vvu_8sx9_dsp58.sv) is feasible only under the
    # full generate condition (mvu_vvu_axi.sv:313): a DSP58 part AND weight width <=8
    # AND activation width <=9. On DSP58+INT8 BOTH packed and softvec are feasible
    # (this predicate lets both through — the choice between them is a PREFERENCE,
    # not a feasibility gate). Off any of these terms, packed is illegal and
    # softvec/hls remain. (lanes<=3 follows from w<=8 & a<=9 on DSP58.)
    if p.implementation != MVAU_DSP_PACKED:
        return None
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
    return None


@predicate("RTL-MVU feasibility (_mvu_rtl_possible)")
def _rtl_mvu_feasible(p, ctx):
    # The real gate that decides whether an RTL-MVU can be used at all
    # (specialize_layers.py:235 `_mvu_rtl_possible`). Reads config + device + data.
    if p.implementation not in _DSP_REALIZATIONS:
        return None
    if p.noActivation != 1:
        return "RTL-MVU requires noActivation=1 (embedded thresholds unsupported)"
    if p.binaryXnorMode != 0:
        return "RTL-MVU does not support binaryXnorMode"
    wdt = ctx.tensor_datatype(WEIGHTS)
    idt = ctx.tensor_datatype(INPUT)
    if not wdt.signed():
        return "RTL-MVU requires signed weights"
    if idt.bitwidth() < 2 or wdt.bitwidth() < 2:
        return "RTL-MVU requires input and weight bitwidth >= 2"
    # DSP48E1 supports only narrow weights: non-narrow weights on a DSP48E1 part
    # cannot use the RTL soft-vectorized core (device + data). narrow_weights is a
    # Derived on the point (computed above).
    if get_dsp_block(ctx.fpgapart) == "DSP48E1" and not p.narrow_weights:
        return "RTL-MVU on DSP48E1 requires narrow weights (specialize_layers:268)"
    return None


@predicate("SEGMENTLEN feasible only if ref_clk > 0.741ns")
def _segmentlen_feasible(p, ctx):
    if p.implementation not in _DSP_REALIZATIONS:
        return None
    clk = ctx.clk if ctx.clk is not None else 5.0
    ref_clk = clk / 2 if p.get("pumpedCompute", 0) else clk
    if ref_clk <= 0.741:
        return f"Infeasible clk target: ref_clk={ref_clk}ns <= 0.741ns (rtl:242)"
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


@predicate("bipolar x bipolar => nonneg int thresholds")
def _bipolar_thresholds_nonneg(p, ctx):
    if p.noActivation == 0 and "ActVal" in p:
        idt = ctx.tensor_datatype(INPUT)
        wdt = ctx.tensor_datatype(WEIGHTS)
        if idt == DataType["BIPOLAR"] and wdt == DataType["BIPOLAR"]:
            if not (isinstance(p.ActVal, int) and p.ActVal >= 0):
                return "bipolar x bipolar requires non-negative integer thresholds (base:576)"
    return None


@predicate("true-binary (non-bipolar) inputs unsupported")
def _no_true_binary(p, ctx):
    idt = ctx.tensor_datatype(INPUT)
    if idt == DataType["BINARY"] and p.language == "hls":
        return "true binary (non-bipolar) inputs are not supported (hls:167)"
    return None


def mvau_predicates():
    return (
        _pe_divides_mh,
        _simd_divides_mw,
        _pumped_compute_needs_simd,
        _pumped_memory_not_1x1,
        _hls_simd_lower_bound,
        _rtl_no_lut,
        _packed_feasible,
        _rtl_mvu_feasible,
        _segmentlen_feasible,
        _uram_requires_ultrascale,
        _weights_present,
        _bipolar_thresholds_nonneg,
        _no_true_binary,
    )


# =============================================================================
# The schema
# =============================================================================


def mvau_schema() -> Schema:
    """The full MVAU design space as a resolve ``Schema``."""
    return Schema(
        axes=mvau_axes(),
        derived=mvau_derived(),
        predicates=mvau_predicates(),
    )
