############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the matrix-vector activation kernel DEFINITION: WHAT it is.

The declarative op definition (mirrors FINN's ``matrixvectoractivation.py``, but as a
design space). A backend author reads it top to bottom; each ``impl_*.py`` bundle in this
package declares only the HOW for one compute core. The FINN-facing
:class:`~finn.kernels.compute.mvau.op.MvauKernelOp` wrapper lives beside this in ``op.py``.
Source of truth for each axis/derived/predicate (file:line into real FINN):
``kernel-design/kernel-final-design/mvau-design-space.md``.

Sections (what each replaces in the classic FINN MVAU):
    1. CONSTANTS         tensor names + pool-member identities
    2. INTERFACES        the ONNX-facing arity + direction  (FINN: node in/out wiring)
    3. OP DESIGN SPACE   op_axes/op_derived/op_predicates — the shared BLOCK structure
                         (FINN: get_nodeattr_types + the shape/dtype getters' math)
    4. COMPUTE TILING    the BLOCK->STREAM lowering shared by the pool
    5. DEMAND            the compute->memory demand stage of the supply waterfall
    6. COST              rough op-level get_exp_cycles  (FINN: get_exp_cycles)
    7. ASSEMBLY          mvau_kernel() / mvau_schema()  (FINN: the class itself)

Tensor-name convention for the Context this schema resolves against:
    "inp"      the activation input tensor   (inputDataType, dynamic)
    "weights"  the weight tensor             (weightDataType + initializer VALUES)
    "out"      the output tensor             (outputDataType, unless derived)
"""

from __future__ import annotations

import numpy as np

from finn.kernels.engine.axis import discrete_axis, predicate_axis
from finn.kernels.engine.derived import Derived
from finn.kernels.engine.predicate import predicate
from finn.kernels.model.kernel import InterfaceSchema, Kernel, KernelSchema
from finn.kernels.model.param_contract import DeliveredParam
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL
from finn.kernels.model.param_names import runtime_writeable_key
from finn.kernels.dataflow.memory import parameters_pool
from finn.kernels.compute.thresholding.shared import (
    _threshold_datatype,
    _unsigned_input_nonneg_thresholds,
)

from .registry import build_pool


# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Pool-member identities (values of the root `implementation` axis). A new backend
# defines its own name in its own bundle file; these three are the built-ins.
MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"
# The optional activation-threshold operand. Shares the spelling with the standalone
# Thresholding op, so ``ctx.initializer(THRESHOLDS)`` reads the same node slot. Present iff
# the fused node has a threshold initializer (a 3-input node) — the emergent existence that
# supersedes the declared ``noActivation`` flag.
THRESHOLDS = "thresholds"


# =============================================================================
# 2. INTERFACES — the ONNX-facing arity + direction (no tiling; tiling is impl-owned)
# =============================================================================


def mvau_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure (the math). No
    semantic role: whether ``weights`` is a stored parameter or a live activation emerges
    from graph context (initializer?) at resolve time. Stream folding (SIMD/PE) is impl-owned.

    The block reads as the matmul: ``inp`` iterates its vector count (``1``) and holds the
    reduction dim MW in-block (``FULL``); ``weights`` is the whole matrix ``(MW, MH)`` in one
    block; ``out`` iterates vectors and holds MH."""
    return (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),        # (n_vecs, MW)
        InterfaceSchema("weights", Direction.IN, block=[FULL, FULL]),  # (MW, MH)
        # thresholds — the OPTIONAL activation operand, (NumChannels, numSteps). Present iff a
        # threshold initializer is attached (a 3-input node); absent nodes skip it in every
        # Context-reading loop. Constant-only in the fused core, so no impl folds it.
        InterfaceSchema("thresholds", Direction.IN, block=[FULL, FULL], optional=True),
        # dtype_source: the out stream width uses the derived output type (= accDataType with
        # no activation), not the raw graph dtype — so stream_width.out matches emit.
        InterfaceSchema("out", Direction.OUT, block=[1, FULL], dtype_source="outputDataType"),
    )


# =============================================================================
# 3. OP DESIGN SPACE — the shared BLOCK structure (axes / derived / predicates).
#    Everything every MVU has, regardless of the chosen compute core. An impl
#    bundle never edits this; it resolves against it.
# =============================================================================

# -- small guard/helper functions (named, not lambdas, for legible tracebacks) --


def weights_may_change(p) -> bool:
    """Whether weights are not statically known — accDataType/weightDataType then use
    worst-case bounds rather than actual values (base:482-498). The staticness fact is
    decided by the composed ``parameters.*`` fields; read with ``.get`` so it is safe on a
    point where the parameters pool is absent (a param-free op) — absent ⇒ static."""
    return bool(p.get(runtime_writeable_key(WEIGHTS), 0))


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


# -- op-level SHARED axes — present under every implementation ------------------


def op_axes():
    # MW/MH/numInputVectors are NOT axes — they are BLOCK extents / the input's leading dims,
    # read off the Context by emit + the cadence closures. Whether the MVU has an activation is
    # EMERGENT (``ctx.initializer(THRESHOLDS) is not None``), not a declared flag. Weight
    # delivery (mem_mode/ram_style/pumpedMemory/…) lives in the composed ``parameters`` pool.
    return (
        # ActVal — activation bias; unused on a no-threshold node.
        predicate_axis("ActVal", "int", lambda v: isinstance(v, int), 0),
        discrete_axis("binaryXnorMode", {0, 1}, 0),
        # mlo_max_iter — per-node iteration count, unbounded non-neg (hwcustomop.py:317-319).
        predicate_axis("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
    )


# -- op-level SHARED derived — the identity's threshold-operand datatype. The
#    accumulator/weight/output datatype contract is BACKEND-SCOPED and lives in
#    ``backends.py`` (composed per compute core), not here.


def _threshold_dtype(p, ctx):
    # thresholdDataType = value-narrowed threshold dtype (standalone _threshold_datatype),
    # or None when absent. Mirrors the weight-dtype derive.
    return _threshold_datatype(p, ctx) if ctx.has_tensor(THRESHOLDS) else None


def _identity_dtype_derived():
    """The op-identity datatype facts: the THRESHOLD operand's value-narrowed dtype, which is
    realization-invariant (present iff the threshold operand is wired, independent of the
    compute core). The accumulator/weight/output datatypes are backend-scoped — see
    :func:`mvau_dtype_backend`."""
    return (
        # threshold identity — live value iff the operand is present, else None.
        Derived("thresholdDataType", _threshold_dtype),
    )


def op_derived():
    # PURE IDENTITY — only the realization-invariant threshold-operand derivations. The
    # accumulator/weight/output datatype contract is backend-scoped (mvau_dtype_backend),
    # composed per compute core. Stream widths + demand are generated by the tiling engine /
    # param_contract; geometry (MW/MH/fold depth) is read off Context by emit.
    return _identity_dtype_derived()


# -- op-level SHARED predicates. Divisibility (MH%PE, MW%SIMD) is engine-generated from the
#    tiling, not here; the pumpedMemory / URAM gates live in the parameters subsystem.


@predicate("weight initializer must exist unless params are not statically known")
def _weights_present(p, ctx):
    if ctx.initializer(WEIGHTS) is None:
        if not weights_may_change(p):
            return "weight initializer required unless params are not static (base:782)"
    return None


# Threshold legality — a 2-D shape check (the step count is the tensor's own shape[1], with
# no independent numSteps axis to validate against — unlike standalone Thresholding), plus
# the shared unsigned-input ⇒ thresholds >= 0 rule. Both no-op when the operand is absent.


@predicate("threshold tensor is 2D (NumChannels, numSteps) when present")
def _mvau_threshold_shape(p, ctx):
    if not ctx.has_tensor(THRESHOLDS):
        return None
    shp = ctx.tensor_shape(THRESHOLDS)
    if len(shp) != 2:
        return f"threshold tensor must be 2D (got shape {shp})"
    return None


@predicate("unsigned input => thresholds >= 0 (when present)")
def _mvau_threshold_nonneg(p, ctx):
    if not ctx.has_tensor(THRESHOLDS):
        return None
    return _unsigned_input_nonneg_thresholds.check(p, ctx)


def op_predicates():
    return (_weights_present, _mvau_threshold_shape, _mvau_threshold_nonneg)


# =============================================================================
# 4. COMPUTE TILING — the default BLOCK->STREAM lowering (``COMPUTE_STREAM``) is a
#    BACKEND-SCOPED fact (folding is realization, not identity), so it lives in ``backends.py``
#    beside the datatype contract, composed by each compute core. The op declares only the
#    BLOCK (see ``mvau_interfaces``); the engine derives SIMD/PE dials + widths from the fold.
# =============================================================================


# =============================================================================
# 5. DELIVERY / 6. COST — both generic, no op-level authoring.
#    Delivery: the op DECLARES which interfaces it delivers + each one's cadence (see
#    _delivered_parameters); the Kernel synthesizes the COMPUTE→DEMAND→MEMORY waterfall
#    generically (model/param_contract.py). Cost: MVAU's nf·sf·n_vecs falls out of the
#    generic max-over-interfaces floor now that weights is a 2-D block streamed SIMD·PE,
#    so it declares NO cost_model.
# =============================================================================


# =============================================================================
# 7. ASSEMBLY — the full MVAU design space as a Kernel (and as a bare Schema).
# =============================================================================


def mvau_shared():
    """The op-level shared (axes, derived, predicates) — everything every MVU has,
    independent of the composed parameters couplings. Used by tests that assemble a
    bare ``pool_schema`` directly."""
    return op_axes(), op_derived(), op_predicates()


def mvau_pool():
    """The registered MVAU implementations (flat peers), in registration order."""
    return build_pool()


def _weight_cadence(p, ctx) -> int:
    # Weights are consumed once per layer.
    return 1


def _threshold_cadence(p, ctx) -> int:
    # Thresholds are re-traversed once per output vector: cadence = prod(numInputVectors) =
    # the input tensor's non-reduction dims (param-delivery-design-space.md §3.1).
    return int(np.prod(ctx.tensor_shape(INPUT)[:-1]))


# The parameter interfaces this op delivers (WHAT + cadence); the generic Kernel wiring
# (model/param_contract.py) owns the HOW.
def _delivered_parameters():
    return (
        DeliveredParam("weights", _weight_cadence, pool=parameters_pool(WEIGHTS)),
        DeliveredParam(THRESHOLDS, _threshold_cadence, pool=parameters_pool(THRESHOLDS)),
    )


def mvau_kernel() -> Kernel:
    """The full MVAU design space as a :class:`Kernel` — the WHAT-owning op node.

    The compute pool (HLS / DSP-softvec / DSP-packed) with impl-owned tiling, plus declared
    delivered parameters (weights + thresholds). The Kernel synthesizes the
    COMPUTE→DEMAND→MEMORY supply waterfall per interface generically
    (model/param_contract.py); the getters project from a resolved point via the impl
    ``stream``."""
    return Kernel(
        identity=KernelSchema(
            name="MVAU",
            interfaces=mvau_interfaces(),
            op_axes=op_axes(),
            op_derived=op_derived(),
            op_predicates=op_predicates(),
        ),
        pool=mvau_pool(),
        delivered_parameters=_delivered_parameters(),
    )


def mvau_schema():
    """The full MVAU design space as a resolve ``Schema`` — delegates to
    :func:`mvau_kernel` (identical assembly). Kept as the name emit/composition tests
    resolve against."""
    return mvau_kernel().schema()
