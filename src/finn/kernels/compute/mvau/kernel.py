############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the matrix-vector activation kernel DEFINITION: WHAT it is.

The declarative op definition (mirrors FINN's ``matrixvectoractivation.py``, but as a
design space). A backend author reads it top to bottom; each ``impl_*.py`` backend in this
package declares only the HOW for one compute core. The FINN-facing
:class:`~finn.kernels.compute.mvau.op.MvauDataflowOp` wrapper lives beside this in ``op.py``.
Source of truth for each axis/derived/predicate (file:line into real FINN):
``scratchpad/reference/mvau-design-space.md``.

Sections (what each replaces in the classic FINN MVAU):
    1. CONSTANTS         tensor names + pool-member identities
    2. INTERFACES        the ONNX-facing arity + direction  (FINN: node in/out wiring)
    3. OP DESIGN SPACE   op_axes/op_derived/op_predicates — the shared BLOCK structure
                         (FINN: get_nodeattr_types + the shape/dtype getters' math)
    4. COMPUTE TILING    the BLOCK->STREAM lowering shared by the pool
    5. DELIVERY / COST   both fully generic — no op-level authoring (see the note below)
    6. ASSEMBLY          mvau_kernel() / mvau_space()  (FINN: the class itself)

Tensor-name convention for the Context this schema resolves against:
    "inp"        the activation input tensor   (dynamic)
    "weights"    the weight tensor             (static initializer VALUES)
    "out"        the output tensor             (dtype derived when there is no activation)
    "thresholds" the OPTIONAL activation operand — present iff a 3-input fused node
"""

from __future__ import annotations

from finn.kernels.engine.attr import attr
from finn.kernels.engine.constraints import IsStatic, ShapeRank, SparsityFree, ValueNonNeg
from finn.kernels.engine.derived import Derived
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL
from finn.kernels.compute.thresholding.shared import _threshold_datatype




# =============================================================================
# 1. CONSTANTS
# =============================================================================

# Pool-member identities (values of the root `implementation` axis). A new backend
# defines its own name in its own backend file; these three are the built-ins.
MVAU_HLS = "mvau_hls"
MVAU_DSP_SOFTVEC = "mvau_dsp_softvec"
MVAU_DSP_PACKED = "mvau_dsp_packed"

# Context tensor names this schema resolves against.
WEIGHTS = "weights"
INPUT = "inp"
OUTPUT = "out"
# Presence is EMERGENT (initializer attached?), superseding the classic ``noActivation`` flag.
THRESHOLDS = "thresholds"


# =============================================================================
# 2. INTERFACES — the ONNX-facing arity + direction (no tiling; tiling is backend-owned)
# =============================================================================


def mvau_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure (the math). No
    semantic role: whether ``weights`` is a stored parameter or a live activation emerges
    from graph context (initializer?) at resolve time. Stream folding (SIMD/PE) is backend-owned.

    The block reads as the matmul: ``inp`` iterates its vector count (``1``) and holds the
    reduction dim MW in-block (``FULL``); ``weights`` is the whole matrix ``(MW, MH)`` in one
    block; ``out`` iterates vectors and holds MH."""
    return (
        InterfaceSchema("inp", Direction.IN, block=[1, FULL]),        # (n_vecs, MW)
        # weights — the STATIC + DENSE requirements are declared here, where the pool can
        # widen them. Both were hand-written escapes in the frontend claim, which meant the
        # vocabulary member (IsStatic) sat dead while the fact it encodes lived in Python
        # and could drift. A backend that can consume dynamic or sparse weights now widens
        # what infer accepts by declaring so, with no frontend edit.
        InterfaceSchema(
            "weights", Direction.IN, block=[FULL, FULL],  # (MW, MH)
            constraints=(IsStatic(WEIGHTS), SparsityFree(WEIGHTS)),
        ),
        # thresholds — optional (NumChannels, numSteps); ShapeRank auto-skips when absent.
        InterfaceSchema(
            "thresholds", Direction.IN, block=[FULL, FULL], optional=True,
            constraints=(ShapeRank(THRESHOLDS, 2),),
        ),
        # out — dtype is backend-derived (accDataType under no-activation), declared as the
        # out port's derived_dtype (mvau_out_dtype, backends.py); here only arity/block.
        InterfaceSchema("out", Direction.OUT, block=[1, FULL]),
    )


# =============================================================================
# 3. OP DESIGN SPACE — the shared BLOCK structure (axes / derived / predicates).
#    Everything every MVU has, regardless of the chosen compute core. An backend
#    backend never edits this; it resolves against it.
# =============================================================================

# -- small guard/helper functions (named, not lambdas, for legible tracebacks) --


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


# -- op-level SHARED axes — present under every implementation ------------------


def op_axes():
    # EMPTY: MVAU has no hand-authored DSE dials. SIMD/PE are tiling-engine-generated from the
    # interface BLOCKs; ram_style/… are parameters-pool-generated; MW/MH are Context extents.
    return ()


def kernel_attrs():
    # Node CONSTANTS: on the Point, read by emit, never explored. See engine/attr.py for why
    # these are a primitive rather than Context fields — both are named in deps
    # (`narrow_weights` reads mlo_max_iter), and a Context field can never be.
    return (
        # The absorbed MultiThreshold's out_bias. THE case proving the category is real:
        # infer REMOVES that node (op.py), so the graph no longer holds this value.
        attr("ActVal", "int", lambda v: isinstance(v, int), 0),
        # MLO parameter-set cardinality, written by loop_rolling.py. Still misnamed (the RTL
        # calls it SETS) and still carrying two concepts — mlo-cardinality.md owns that
        # question. It is an Attr here purely so the strata are honest meanwhile; nothing
        # about this forecloses where that pass decides the value should live.
        attr("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
    )


# -- op-level SHARED derived — ONLY the threshold-operand dtype (realization-invariant). The
#    accumulator/weight/output dtypes are backend-scoped (mvau_register_dtypes / mvau_out_dtype
#    in backends.py); stream widths + geometry are tiling/emit-generated.


def _threshold_dtype(p, ctx):
    # The threshold operand's DECLARED dtype, or None on a 2-input node. Delegates to the
    # SHARED Thresholding rule so the fused and standalone paths cannot drift — see
    # ``compute/thresholding/shared.py::_threshold_datatype`` for the parity TODO on
    # re-enabling value-narrowing.
    return _threshold_datatype(p, ctx) if ctx.has_tensor(THRESHOLDS) else None


def op_derived():
    # No dep: while the narrowing is off for FINN parity the shared rule is Context-only, so
    # there is nothing to order against. Re-enabling it restores the read of the thresholds
    # ParamDatatype (a parameters-pool derived) AND this dep together — the topo-sort needs the
    # dep to place this after the publisher, across the compute/parameters pool boundary.
    return (Derived("thresholdDataType", _threshold_dtype),)


# -- op-level SHARED legality. Divisibility and the URAM/pumped gates are engine- and
#    parameters-generated; the only op-authored rule is the relational one below.


def _unsigned_input(p, ctx) -> bool:
    return not ctx.tensor_datatype(INPUT).signed()


def _mvau_constraints():
    # unsigned input ⇒ all thresholds >= 0 (thresholding.py:243). Reads INPUT to gate a rule on
    # THRESHOLDS, so it is kernel-level (relational), not a per-port constraint.
    return (ValueNonNeg(THRESHOLDS, when=_unsigned_input),)


def op_predicates():
    return ()  # all legality is now declarative constraints (per-port + the relational one)


# 4. COMPUTE TILING — backend-scoped (COMPUTE_STREAM in backends.py); the op declares only the
#    interface BLOCK, and the engine derives SIMD/PE + widths from it.
# 5. DELIVERY / 6. COST — fully generic, no op-level authoring: delivery is derived from the
#    pool's mem_modes (model/param_contract.py); the cost floor falls out of the tiling.


# =============================================================================
# 7. ASSEMBLY — the full MVAU design space as a DataflowKernel.
# =============================================================================


def mvau_pool():
    """The MVAU backends, in declaration order (= selection precedence)."""
    from .op import MvauDataflowOp

    return MvauDataflowOp.pool


def mvau_kernel():
    """The MVAU design space — now simply the op class.

    Kept as a one-line shim: `MvauDataflowOp` IS the kernel (the container collapsed into it,
    F6), so there is no separate object to build. Callers that just want the class should say
    so; this exists so the ~40 sites naming `mvau_kernel()` keep reading naturally."""
    from .op import MvauDataflowOp

    return MvauDataflowOp


def mvau_space():
    """The full MVAU design space as a resolve ``DesignSpace`` — delegates to
    :func:`mvau_kernel` (identical assembly). Kept as the name emit/composition tests
    resolve against."""
    return mvau_kernel().compile()
