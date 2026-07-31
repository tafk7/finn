############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding **op-level shared** elements — deliberately SMALL.

Thresholding is the model-stressing op: its memory-delivery cluster is NOT op-level
(it is asymmetric and impl-local — HLS has mem_mode/ram_style, RTL has depth-triggers
— and is DEFERRED entirely this task). What genuinely belongs to every implementation
is only: the ``PE`` fold over ``NumChannels``, the threshold-step count ``numSteps``,
the activation bias ``ActVal``, ``numInputVectors``, and the shared *intent*
``runtime_writeable_weights`` (its realization is per-bundle). Derived: TMEM geometry,
output/threshold datatypes, and the PE-scaled stream widths.

Context convention: the ``thresholds`` tensor shape is ``(NumChannels, numSteps)``.
Both backends have IDENTICAL integer dtype envelopes — there is intentionally NO
per-bundle dtype feasibility gate (a fabricated one was falsified; see the package
__init__).
"""

from __future__ import annotations

from qonnx.core.datatype import DataType

from finn.kernels.engine.axis import divisor_axis, fixed_axis, predicate_axis
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.predicate import predicate

from .names import INPUT, OUTPUT, THRESHOLDS


def _require_ctx(ctx, what):
    # A context-fixed axis needs the Context. A context-less probe (the nodeattr registry
    # enumerating static axes with ``context=None``) legitimately cannot resolve it — raise a
    # NARROW ``ValueError`` (the "unprobeable for this probe" signal T0.1 narrowed on) rather
    # than letting ``None.tensor_shape`` surface as an ``AttributeError`` read as a kernel bug.
    if ctx is None:
        raise ValueError(f"{what} needs a Context (context-dependent axis; probe gave none)")
    return ctx


def _num_channels(p, ctx):
    # thresholds shape (NumChannels, numSteps) -> NumChannels.
    return _require_ctx(ctx, "NumChannels").tensor_shape(THRESHOLDS)[0]


def _num_steps_default(p, ctx):
    # numSteps matches the threshold tensor's step dimension.
    return int(_require_ctx(ctx, "numSteps").tensor_shape(THRESHOLDS)[1])


def _is_int_list(v) -> bool:
    return isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v)


def _is_pos_int(v) -> bool:
    return isinstance(v, int) and v >= 1


# =============================================================================
# Op-level SHARED axes — everything every Thresholding impl has.
# =============================================================================


def op_axes():
    return (
        # context-fixed channel count (from the threshold tensor).
        fixed_axis("NumChannels", _num_channels),
        # PE folds NumChannels.
        divisor_axis("PE", "NumChannels", 1, deps={"NumChannels"}),
        # threshold-step count (matches threshold tensor's step dim).
        predicate_axis("numSteps", "pos int", _is_pos_int, _num_steps_default),
        # activation accumulator bias (ActVal).
        predicate_axis("ActVal", "int", lambda v: isinstance(v, int), 0),
        predicate_axis("numInputVectors", "list[int]", _is_int_list, [1]),
        # Shared INTENT to be runtime-writable; the delivery mechanism is per-bundle.
        # (mem_mode/ram_style deferred entirely this task.)
        predicate_axis(
            "runtime_writeable_weights", "{0,1}", lambda v: v in (0, 1), 0
        ),
    )


# =============================================================================
# Op-level SHARED derived.
# =============================================================================


def _tmem(p, ctx):
    return p.NumChannels // p.PE


def _threshold_datatype(p, ctx):
    # Threshold (weight) dtype narrowed from actual values when statically known,
    # else the graph dtype. Mirrors the mvau weight-dtype derive.
    thr = ctx.initializer(THRESHOLDS)
    if thr is None or p.get("runtime_writeable_weights", 0):
        return ctx.tensor_datatype(THRESHOLDS)
    t_min = float(thr.min())
    t_max = float(thr.max())
    if t_min < 0:
        extreme = t_min if abs(t_min) > t_max else -t_max - 1
        return DataType.get_smallest_possible(extreme)
    return DataType.get_smallest_possible(t_max)


def _instream_width(p, ctx):
    return ctx.tensor_datatype(INPUT).bitwidth() * p.PE


def _outstream_width(p, ctx):
    return ctx.tensor_datatype(OUTPUT).bitwidth() * p.PE


def op_derived():
    return (
        Derived("TMEM", _tmem),
        # No outputDataType derived: the output dtype IS the graph output dtype (the trivial
        # DatatypeSpec — None → graph fallback), resolved uniformly like every other output's
        # derived_dtype. The out port declares no derived_dtype; no second mechanism here.
        Derived("thresholdDataType", _threshold_datatype),
        Derived("instream_width", _instream_width),
        Derived("outstream_width", _outstream_width),
    )


# =============================================================================
# Op-level SHARED predicates.
# =============================================================================


@predicate("NumChannels % PE == 0")
def _pe_divides_channels(p, ctx):
    if p.NumChannels % p.PE != 0:
        return f"NumChannels={p.NumChannels} not divisible by PE={p.PE}"
    return None


@predicate("threshold tensor is 2D with shape[1] == numSteps")
def _threshold_shape_matches_steps(p, ctx):
    shp = ctx.tensor_shape(THRESHOLDS)
    if len(shp) != 2:
        return f"threshold tensor must be 2D (got shape {shp})"
    if shp[1] != p.numSteps:
        return f"threshold steps {shp[1]} != numSteps={p.numSteps}"
    return None


@predicate("unsigned input => thresholds >= 0")
def _unsigned_input_nonneg_thresholds(p, ctx):
    # Data-dependent (thresholding.py:241-243): unsigned activations require all
    # threshold values non-negative.
    if ctx.tensor_datatype(INPUT).signed():
        return None
    thr = ctx.initializer(THRESHOLDS)
    if thr is not None and (thr < 0).any():
        return "unsigned input requires all thresholds >= 0 (thresholding.py:243)"
    return None


def op_predicates():
    return (
        _pe_divides_channels,
        _threshold_shape_matches_steps,
        _unsigned_input_nonneg_thresholds,
    )
