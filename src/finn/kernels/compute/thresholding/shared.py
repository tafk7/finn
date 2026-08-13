############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding's shared derivation HELPERS — the closures its design space is built from.

**The design space itself lives on the op class**, in ``op.py``: `ThresholdingDataflowOp`'s
class body declares the pool, axes, deriveds, predicates and attrs, because those are
op-CLASS facts (F6 — the op IS the kernel). Read that class to see what a Thresholding is.

What stays here is the closure bodies those declarations reference, plus the ONE genuinely
cross-op function: :func:`_threshold_datatype`, which **MVAU also imports** so the fused
(MVAU-with-thresholds) and standalone paths cannot drift on threshold dtype. That shared
consumer is why this module exists at all rather than folding into ``op.py`` — it would
otherwise be a leaf with a single importer.

Notes that outlived the move, because they explain absences a reader will wonder about:

* ``op_axes`` is EMPTY, and that is the point: this op declares no hand-authored choice.
  Every scalar it used to carry as an Axis was either a Context fact (``numSteps``), inert
  (``numInputVectors``, deleted), or a constant (``ActVal``). The ``PE`` fold is
  tiling-generated from the declared stream.
* The memory-delivery cluster is NOT op-level — it is asymmetric and backend-local (HLS has
  mem_mode/ram_style, RTL has depth-triggers).
* The threshold DTYPE is the storage owner's published ``ParamDatatype.dtype`` (thresholds
  compose the parameters pool in embedded mode) — read, not re-derived. Runtime-writability
  is a delivery-topology concern owned by the parameters pool, not a self-declared op axis;
  the former ``runtime_writeable_weights`` op-axis was DEAD and is deleted.
* Both backends have IDENTICAL integer dtype envelopes — there is intentionally NO
  per-backend dtype feasibility gate (a fabricated one was falsified; see the package
  docstring).

Context convention: the ``thresholds`` tensor shape is ``(NumChannels, numSteps)``.
"""

from __future__ import annotations

from finn.kernels.engine.attr import attr
from finn.kernels.model.fold_depth import threshold_fold_depth
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


def _threshold_shape(ctx, what):
    """The ``(NumChannels, numSteps)`` threshold shape, rank-checked.

    Both geometry deriveds index this positionally, and a DERIVED runs before any predicate —
    so on a non-2D tensor the raw index raises ``IndexError``, which is neither legible nor in
    the ``(ValueError, KeyError, AbsentAxisError)`` set the feasibility trials treat as "not
    resolvable for this context" (``DataflowOp.first_feasible_backend``). It would propagate as a
    kernel bug (INV5) on a node that is merely ineligible. Raising the narrow ValueError here
    keeps the diagnosis at the shape read, where the real reason is in hand.

    ``ShapeRank(THRESHOLDS, 2)`` on the interface states the same requirement DECLARATIVELY,
    for the frontend claim; this is the guard for the path that reaches the shape first."""
    shp = tuple(_require_ctx(ctx, what).tensor_shape(THRESHOLDS))
    if len(shp) != 2:
        raise ValueError(
            f"{what}: threshold tensor must be 2D (NumChannels, numSteps), got shape {shp}"
        )
    return shp


def _num_channels(p, ctx):
    # thresholds shape (NumChannels, numSteps) -> NumChannels.
    return _threshold_shape(ctx, "NumChannels")[0]


def _num_steps(p, ctx):
    # numSteps IS the threshold tensor's step dimension — not "defaults to it". It was an
    # Axis defaulting to this value with a predicate forbidding any other, which is a
    # derivation stated twice.
    return int(_threshold_shape(ctx, "numSteps")[1])


# =============================================================================
# Op-level SHARED axes — everything every Thresholding backend has.
# =============================================================================


# =============================================================================
# Op-level SHARED derived.
# =============================================================================


def _tmem(p, ctx):
    # ONE implementation: the free function in model/fold_depth.py is the same quantity
    # (probe-confirmed identical), reached from the memstream side. Emit reads point.TMEM,
    # so the key stays — but it now computes via the shared function rather than restating
    # NumChannels // PE, which is how the two drifted into being two homes for one fact.
    return threshold_fold_depth(p, ctx, iface=THRESHOLDS)


def _threshold_datatype(p, ctx):
    # The DECLARED graph dtype of the threshold tensor — matching baseline FINN, which reads
    # ``self.get_input_datatype(1)`` (hls/thresholding_hls.py:210) and never value-narrows it.
    #
    # TODO: Determine validity of threshold memory datawidth optimization.
    #
    # The storage owner publishes a value-NARROWED dtype under
    # ``param_datatype_key(THRESHOLDS)`` (``value_optimized`` → ``get_smallest_possible`` over
    # the actual values), and this derived used to return it. That narrowing is very likely a
    # real resource win — a UINT8-declared table whose values fit in 5 bits needs a 5-bit ROM —
    # but it CHANGES THE EMITTED HARDWARE relative to FINN: the ``ThresholdsActivation<>`` type
    # parameter, and hence the threshold memory width, differ whenever the declared dtype is
    # wider than the values require. Nothing has validated that narrower table functionally,
    # and PARITY WITH FINN OUTRANKS THE OPTIMIZATION until something does.
    #
    # The existing byte-diff test did not catch the divergence because its fixture's thresholds
    # saturate UINT8, making the narrowing a no-op there. Re-enabling this needs a
    # non-saturating differential case plus an rtlsim check, not just a green byte gate.
    #
    # Read by BOTH threshold consumers — standalone Thresholding and MVAU's fused thresh.h
    # (``compute/mvau/emit_hls.py:223``) — so flipping it back is a one-line change here.
    return ctx.tensor_datatype(THRESHOLDS)


# =============================================================================
# Op-level SHARED predicates.
# =============================================================================


# NOTE — no hand-written "NumChannels % PE == 0" predicate: the tiling engine generates a
# divisibility rule per folded block dim from the declared stream. It was written by hand
# only because this op did not use the generator.


# NOTE — the "threshold tensor is 2D with shape[1] == numSteps" predicate is GONE, both
# halves for different reasons. The ``== numSteps`` half died with the axis: numSteps is now
# DERIVED from shape[1], so the comparison could only compare a value to itself — it read as
# legality while really being a consistency check on a duplicate. The RANK half is genuine and
# MOVED: it is declared as ``ShapeRank(THRESHOLDS, 2)`` on the interface (op.py) and enforced
# at the shape read (``_threshold_shape``), because a predicate runs AFTER the deriveds that
# index the shape and so could never fire first.


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


