############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding **op-level shared** elements — deliberately SMALL.

Thresholding is the model-stressing op: its memory-delivery cluster is NOT op-level
(it is asymmetric and backend-local — HLS has mem_mode/ram_style, RTL has depth-triggers
— and is DEFERRED entirely this task). What genuinely belongs to every implementation
is only the activation bias ``ActVal`` -- a node CONSTANT (``kernel_attrs``), not a dial.
Derived: the ``NumChannels``/``numSteps`` threshold geometry, TMEM, output/threshold
datatypes, and the PE-scaled stream widths. The ``PE`` fold itself is tiling-generated.

``op_axes`` is EMPTY, and that is the point: this op declares no hand-authored choice. Every
scalar it used to carry as an Axis was either a Context fact (``numSteps``), inert
(``numInputVectors``, deleted), or a constant (``ActVal``).

The threshold DTYPE is the storage owner's published ``ParamDatatype.dtype`` (thresholds
compose the parameters pool in embedded mode) — read, not re-derived. Runtime-writability
is a delivery-topology concern owned by the parameters pool (``runtime_writeable_key``), not
a self-declared op axis; the former ``runtime_writeable_weights`` op-axis was DEAD (no backend
consumed it) and is deleted.

Context convention: the ``thresholds`` tensor shape is ``(NumChannels, numSteps)``.
Both backends have IDENTICAL integer dtype envelopes — there is intentionally NO
per-backend dtype feasibility gate (a fabricated one was falsified; see the package
__init__).
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


def op_axes():
    return ()
    # EMPTY, and every removal was measured:
    #
    # ``numSteps`` -> Derived (below). Its only LEGAL value was the Context one: its default
    # already read ``tensor_shape(THRESHOLDS)[1]``, and a predicate rejected anything else. An
    # axis whose domain is a singleton determined by Context is a derivation wearing an Axis
    # costume — the same correction ``NumChannels`` already had, for the same reason. The
    # predicate went with it: it was not a legality rule, it was a consistency check on a
    # duplicate.
    #
    # ``numInputVectors`` -> DELETED. It was INERT: nothing in this package read it, and
    # pinning it changed no resolved value (folded shapes track the TENSOR). Baseline FINN
    # needs it because it RECONSTRUCTS shapes from nodeattrs (``tuple(vecs + [ich])``,
    # thresholding.py:211); we source shapes from the live graph, so the reconstruction input
    # is dead weight. MVAU already treats it as Context (``geometry.py``: ``nvec`` from the
    # input tensor's leading dims) and publishes no such nodeattr — the two ops disagreed
    # about one quantity and MVAU was right.
    #
    # ``ActVal`` -> kernel_attrs (a node CONSTANT; it is the one value here the graph really
    # does not hold once infer absorbs the MultiThreshold).
    #
    # NOT here either, and deliberately: the ``PE`` fold dial and the ``NumChannels``
    # divisibility rule are GENERATED by the tiling engine from each backend's declared
    # ``stream=COMPUTE_STREAM``. They were hand-written while the stream was declared but
    # never wired, which also forced ``NumChannels`` to be carried as a pseudo-axis purely
    # to feed ``divisor_axis``. It is a Context fact, so it is now a Derived (below).


def kernel_attrs():
    # The absorbed MultiThreshold's out_bias — infer bakes it (op.py) and emit reads it off
    # the Point (emit_hls.py, emit_rtl.py). No graph home once the frontend node is gone,
    # which is what makes it an Attr rather than a Context fact. See engine/attr.py.
    return (attr("ActVal", "int", lambda v: isinstance(v, int), 0),)


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


def op_derived():
    return (
        # NumChannels is a CONTEXT fact (the threshold tensor's channel extent), not a
        # choice. It was an Axis only because divisor_axis("PE", "NumChannels") needed to
        # read it off the point; with PE generated from the declared stream, it can be what
        # it always was. Emit reads point.NumChannels, so the key stays.
        Derived("NumChannels", _num_channels),
        # numSteps is the threshold tensor's STEP extent — the sibling fact to NumChannels,
        # and it took the same route: an Axis whose default read Context, whose domain was a
        # singleton, and whose "legality" predicate only checked it had not been overridden
        # to something inconsistent. Emit reads point.numSteps, so the key stays.
        Derived("numSteps", _num_steps),
        Derived("TMEM", _tmem, deps={"NumChannels", "PE"}),
        # No outputDataType derived: the output dtype IS the graph output dtype (the trivial
        # DatatypeSpec — None → graph fallback), resolved uniformly like every other output's
        # derived_dtype. The out port declares no derived_dtype; no second mechanism here.
        # thresholdDataType is Context-only while the narrowing is disabled for FINN parity
        # (see _threshold_datatype's TODO), so it declares NO dep. Re-enabling the narrowing
        # means reading the storage owner's published ParamDatatype again, which is a
        # parameters-pool derived — restore ``deps={param_datatype_key(THRESHOLDS)}`` with it
        # so the unified topo-sort orders this after the publisher.
        Derived("thresholdDataType", _threshold_datatype),
    )
    # instream_width/outstream_width are GONE: they are the singular-stream pair
    # resolution-phases.md §4 dissolved, hand-reintroduced here because this op bypassed the
    # tiling engine. The generated per-interface ``stream_width.<iface>`` replaces them.


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


def op_predicates():
    return (_unsigned_input_nonneg_thresholds,)
