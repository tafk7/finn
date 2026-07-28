############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the matrix-vector activation kernel: WHAT it is, and how FINN sees it.

This is the one op-definition file (mirrors FINN's ``matrixvectoractivation.py``, but
declarative). A backend author reads it top to bottom; each ``impl_*.py`` bundle in this
package declares only the HOW for one compute core. Source of truth for each
axis/derived/predicate (file:line into real FINN):
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
    8. FINN WRAPPER      MvauKernelOp(KernelOp)         (FINN: MVAU(HWCustomOp))

Tensor-name convention for the Context this schema resolves against:
    "inp"      the activation input tensor   (inputDataType, dynamic)
    "weights"  the weight tensor             (weightDataType + initializer VALUES)
    "out"      the output tensor             (outputDataType, unless derived)
"""

from __future__ import annotations

import logging

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import calculate_matvec_accumulator_range

from finn.kernels.adapter import KernelOp, PortSpec, TransformationResult
from finn.kernels.primitives.spec_helpers import smallest_datatype_for_range
from finn.kernels.space import (
    FULL,
    Context,
    DeliveredParam,
    Derived,
    Direction,
    Illegal,  # noqa: F401  (kept available for callers/tests)
    Interface,
    Kernel,
    KernelSchema,
    Role,
    discrete_axis,
    fixed_axis,
    predicate,
    predicate_axis,
)
from finn.kernels.space.param_names import runtime_writeable_key
from finn.kernels.ops._dsp_rtl import VERSION  # noqa: F401  (re-exported for bundles)
from finn.kernels.ops.parameters import parameters_pool
from finn.kernels.ops.thresholding.shared import (
    _num_steps_default,
    _threshold_datatype,
    _threshold_shape_matches_steps,
    _unsigned_input_nonneg_thresholds,
)

from .registry import build_pool

logger = logging.getLogger(__name__)


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
    block; ``out`` iterates vectors and holds MH. ``weights`` is an ORDINARY interface — no
    WidthOnly, no special flag; its PE·SIMD stream is just a 2-D fold of a 2-D block."""
    return (
        Interface("inp", Direction.IN, block=[1, FULL]),        # (n_vecs, MW)
        Interface("weights", Direction.IN, block=[FULL, FULL]),  # (MW, MH)
        # thresholds — the OPTIONAL activation operand, (NumChannels, numSteps). Present iff
        # a threshold initializer is attached (a 3-input node); absent nodes skip it in every
        # Context-reading loop. ALWAYS constant in the fused core (baked into thresh.h — no
        # port, no stream), so no impl declares a `stream` fold for it: it carries only
        # identity (block + the Context tensor binding) for the threshold deriveds/predicates.
        Interface("thresholds", Direction.IN, block=[FULL, FULL], optional=True),
        # dtype_source="outputDataType": the stream width uses the derived output type
        # (= accDataType under noActivation), not the raw graph dtype — so the generated
        # stream_width.out matches the emit-side value.
        Interface("out", Direction.OUT, block=[1, FULL], dtype_source="outputDataType"),
    )


# =============================================================================
# 3. OP DESIGN SPACE — the shared BLOCK structure (axes / derived / predicates).
#    Everything every MVU has, regardless of the chosen compute core. An impl
#    bundle never edits this; it resolves against it.
# =============================================================================

# -- small guard/helper functions (named, not lambdas, for legible tracebacks) --


def weights_may_change(p) -> bool:
    """Weights are not statically known — accDataType/weightDataType must use
    worst-case bounds rather than actual values (base:482-498). This is the one
    CROSS-COORDINATE coupling from the parameters subsystem back into the compute
    dtype derivations: staticness (coordinate C) is decided by the composed
    ``parameters.*`` fields. Reads with ``.get`` so it is safe on a point where the
    parameters pool is absent (a future param-free op) — absent ⇒ statically known.

    This is a FORK, not a cycle (interface-supply-waterfall.md §3): one given fact
    (staticness) fans out to two consumers — compute dtype (up) and the memory reload port
    (down). A future M2 increment relocates staticness to the requirement tier both READ,
    at which point this helper reads that given instead of the parameters axis.

    Increment-1 topologies are ``embedded`` (static) and ``decoupled`` (static unless
    runtime-writable). The external / dynamic / MLO staticness sources return when
    those topologies land (each will be its own ``parameters.<iface>.topology`` value)."""
    return bool(p.get(runtime_writeable_key(WEIGHTS), 0))


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


# -- op-level SHARED axes — present under every implementation ------------------


def op_axes():
    return (
        # MW/MH are NOT axes and NO LONGER point aliases — they are BLOCK extents (the
        # matmul's reduction + output dims) declared on the interfaces' `block`, read
        # straight off the Context by emit (the shared mvau_geometry accessor). The PE/SIMD
        # fold DIALS the engine derives from each impl's `stream`.
        # --- activation / threshold cluster ----------------------------------
        # noActivation is GONE: whether the fused MVU has an activation is EMERGENT —
        # ``ctx.initializer(THRESHOLDS) is not None`` (a 3-input node). The old declared flag
        # + its guards dissolve into that Context read (the same declared-slot / emergent-
        # existence motion as role emergence). ActVal is now an always-present bias axis (as
        # in the standalone Thresholding op); it is simply unused on a no-threshold node.
        # ram_style_thresholds is dropped: thresholds are constant-only in the fused core
        # (baked into thresh.h, no threshold RAM), so there is no ram-style choice to make.
        predicate_axis("ActVal", "int", lambda v: isinstance(v, int), 0),
        discrete_axis("binaryXnorMode", {0, 1}, 0),
        # numInputVectors is NOT an axis and NOT a point alias — it IS the input tensor's
        # leading (non-reduction) dims (FINN: get_normal_input_shape = numInputVectors + [MW]),
        # read off the Context directly by emit + the cadence closures.
        # F4: mlo_max_iter is a per-node iteration count, unbounded non-neg int.
        # The `64` in the old {0..64} domain was n_max_layers (a fabric-wide MLO
        # table size, hwcustomop.py:378) — a different entity that does not bound
        # this axis (hwcustomop.py:317-319, mlo_max_iter unbounded).
        predicate_axis("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
        # --- weight-delivery cluster: MOVED OUT to the `parameters` pool ------
        # mem_mode/ram_style/runtime_writeable_weights/pumpedMemory/dynamic_input used
        # to live here as the "reserved composition seam". They are now the
        # `parameters` subsystem (ops/parameters/), folded into the MVAU schema via a
        # `BackendInterface` per delivered interface under the `parameters.*` namespace.
        # mem_mode is gone: being the `decoupled` topology IS "internal_decoupled". The
        # cross-coordinate couplings (memstream geometry, the pumpedMemory/fold gate) are
        # owned by that BackendInterface's guarded delivery sub-schema.
    )


# -- op-level SHARED derived — computed for every implementation ----------------


def _acc_datatype(p, ctx):
    # base:469-527 — worst-case type bounds when weights may change, actual weight
    # VALUES when static. The canonical data-dependent Derived.
    idt = ctx.tensor_datatype(INPUT)
    wdt = ctx.tensor_datatype(WEIGHTS)
    weights = ctx.initializer(WEIGHTS)
    if p.binaryXnorMode == 1 and weights is not None:
        weights = 2 * weights - 1
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
    # Emergent: no threshold initializer ⇒ no activation.
    if not _has_thresholds(ctx):
        return _acc_datatype(p, ctx)
    return ctx.tensor_datatype(OUTPUT)


# -- threshold identity — present iff a threshold initializer is attached ---------
#
# The fused MVU's thresholds operand is OPTIONAL (a 3-input node). Existence is emergent:
# ``ctx.initializer(THRESHOLDS) is not None``. These reuse the STANDALONE Thresholding op's
# helpers (imported), wrapped so they no-op (return None / pass) on a no-threshold node
# rather than KeyError-ing on the absent tensor — the identity carries a live value only
# when the operand is present.


def _has_thresholds(ctx) -> bool:
    return ctx.initializer(THRESHOLDS) is not None


def _num_steps(p, ctx):
    # numSteps = the threshold tensor's step dim (standalone _num_steps_default), or None
    # when the node has no thresholds. Defensive on a malformed (non-2-D) tensor: return
    # None so the shape PREDICATE emits the clean legality error rather than this derived
    # crashing first (deriveds run before predicates).
    if not _has_thresholds(ctx):
        return None
    if len(ctx.tensor_shape(THRESHOLDS)) != 2:
        return None
    return _num_steps_default(p, ctx)


def _threshold_dtype(p, ctx):
    # thresholdDataType = value-narrowed threshold dtype (standalone _threshold_datatype),
    # or None when absent. Mirrors the weight-dtype derive.
    return _threshold_datatype(p, ctx) if _has_thresholds(ctx) else None


def _identity_dtype_derived():
    """THE REAL DATATYPE CONTRACT (base:469-549) — accDataType/weightDataType/outputDataType.
    Data-dependent derivations: they read the actual weight VALUES when static, worst-case
    bounds otherwise. ``weights_may_change`` is the one static→dtype coupling that gates
    them (a FORK, not a cycle — see its docstring)."""
    return (
        Derived("accDataType", _acc_datatype),
        Derived("weightDataType", _weight_datatype),
        Derived("outputDataType", _output_datatype),
        # threshold identity — live values iff the operand is present, else None.
        Derived("numSteps", _num_steps),
        Derived("thresholdDataType", _threshold_dtype),
    )


def op_derived():
    # PURE IDENTITY — the datatype contract only. The geometry aliases (MW/MH/WMEM/TMEM/
    # numInputVectors) are GONE: emit reads block extents + fold depth directly (the shared
    # mvau_geometry accessor over Context + space/folding). NOT here, by design:
    #   * `stream_width.<iface>` — the tiling engine generates one per interface from each
    #     impl's `stream` folds (the `out` interface's dtype_source="outputDataType" gives it
    #     the accumulator type when the node has no activation).
    #   * `parameters.<iface>.stream_width` — a per-TOPOLOGY fact (0 embedded / demand
    #     bit_rate decoupled) owned by the delivery pool, namespaced per interface.
    #   * `parameters.<iface>.demand` — the DEMAND stage of the supply waterfall, synthesized
    #     generically by the Kernel from the declared `delivered_parameters` (space/delivery.py),
    #     between the compute pool and the delivery pool (it reads the compute pool's resolved
    #     `stream_width.<iface>`, only present AFTER compute tiling).
    return _identity_dtype_derived()


# -- op-level SHARED predicates — one kind; provenance is what each reads --------


# The divisibility predicates (MH%PE==0, MW%SIMD==0) are NOT hand-written here — the
# tiling engine generates one per Fold spec from COMPUTE_TILING. The op declares only
# the block-structural math facts below.


# The `pumpedMemory => not(PE==SIMD==1)` gate and the `ram_style=ultra & not versal
# => runtime_writeable=1` URAM gate live in the parameters subsystem: the URAM gate
# is self-contained in the decoupled topology bundle; the pumpedMemory/fold gate is
# cross-coordinate (reads the compute fold) and is contributed in section 5.


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


# Threshold legality — reinstated now that thresholds ARE a first-class Context tensor
# (the F2 drop-note below was explicit: "Reinstate when thresholds become a Context tensor").
# Both reuse the standalone Thresholding op's predicates, wrapped to no-op on a no-threshold
# node (the operand is optional). FINN's assertion is over the THRESHOLD TENSOR VALUES
# (`orig_thres_matrix >= 0`, base:578) / the 2-D (NumChannels, numSteps) shape — exactly what
# the standalone predicates check, now that the tensor exists.


@predicate("threshold tensor is 2D with shape[1] == numSteps (when present)")
def _mvau_threshold_shape(p, ctx):
    if not _has_thresholds(ctx):
        return None
    return _threshold_shape_matches_steps.check(p, ctx)


@predicate("unsigned input => thresholds >= 0 (when present)")
def _mvau_threshold_nonneg(p, ctx):
    if not _has_thresholds(ctx):
        return None
    return _unsigned_input_nonneg_thresholds.check(p, ctx)


def op_predicates():
    return (_weights_present, _mvau_threshold_shape, _mvau_threshold_nonneg)


# =============================================================================
# 4. COMPUTE TILING — the BLOCK->STREAM lowering shared by all three compute impls.
# =============================================================================

# They fold identically — SIMD folds the reduction dim MW on the activation (last axis),
# The STREAM folding shared by all three compute impls: SIMD folds the reduction dim MW
# (inp position 1, weights position 0), PE folds the output dim MH (out position 1, weights
# position 1). weights is a 2-D fold of its 2-D block → PE·SIMD tile/cycle. Positional over
# each interface's `block`. Declared once here; a tiled backend overrides only the weights
# entry (deliver PE/TH along MH). The engine DERIVES the SIMD/PE dials (divisor domains),
# divisibility, and widths from these — none hand-written.
COMPUTE_STREAM = {
    INPUT: [1, "SIMD"],
    OUTPUT: [1, "PE"],
    WEIGHTS: ["SIMD", "PE"],
}


# =============================================================================
# 5. DELIVERY — the compute→memory supply waterfall is now GENERIC (space/delivery.py).
#    The op no longer hand-wires the DEMAND stage or the topology-mode guard: it DECLARES
#    which interfaces it delivers + each one's CADENCE (see mvau_kernel's
#    delivered_parameters), and the Kernel synthesizes the (demand, guarded delivery
#    sub-schema) pair generically — reading each compute backend's `consumes` and each
#    topology's `mode`. The memory backend still owns its own realization (memstream
#    width/depth/sets/init_file + the pumped/URAM gates), sized from the published demand.
# =============================================================================


# =============================================================================
# 6. COST — no op-level override needed.
# =============================================================================
#
# MVAU's cost IS the reduction product nf·sf·n_vecs, and it falls out of the generic
# max-over-interfaces floor for FREE now that `weights` is a proper 2-D block streamed
# SIMD·PE: its stream-cycle count is MW·MH/(SIMD·PE) = sf·nf — the largest interface term,
# times the input's n_vecs leading dims. So MVAU declares NO cost_model (the old override
# only existed because weights was modelled as a width-only port skipped by the floor).


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
    # Thresholds are consumed once per ACTIVATION output beat: cadence = prod(folded_in[:-1])
    # = prod(numInputVectors) — the TAP_REP (param-delivery-design-space.md §3.1). folded_in is
    # numInputVectors + [MW/SIMD]; its leading dims [:-1] are exactly numInputVectors (the input
    # tensor's non-reduction dims), so the threshold memory is re-traversed once per output
    # vector. Sized from resolved geometry (numInputVectors is an op derived), same waterfall
    # stage as demand. In the fused core thresholds are constant (demand=None), so this does not
    # yet size a streamer — but it is the real quantity a decoupled/MLO threshold variant needs.
    return int(np.prod(ctx.tensor_shape(INPUT)[:-1]))


# The parameter interfaces this op delivers, as DeliveredParam declarations (WHAT + cadence);
# the generic Kernel wiring (space/delivery.py) owns the HOW. ``weights`` is always live;
# ``thresholds`` is the optional activation operand (present iff its initializer is attached,
# always constant-mode in the fused core → demand None → baked into thresh.h).
def _delivered_parameters():
    return (
        DeliveredParam("weights", _weight_cadence, pool=parameters_pool(WEIGHTS)),
        DeliveredParam(THRESHOLDS, _threshold_cadence, pool=parameters_pool(THRESHOLDS)),
    )


def mvau_kernel() -> Kernel:
    """The full MVAU design space as a :class:`Kernel` — the WHAT-owning op node.

    The compute pool (``implementation``: HLS / DSP-softvec / DSP-packed) with impl-owned
    tiling, plus DECLARED delivered parameters (weights + thresholds). The Kernel synthesizes
    the supply waterfall COMPUTE→DEMAND→MEMORY per interface generically (space/delivery.py):
    the demand stage reads the compute pool's resolved ``stream_width.<iface>`` and publishes
    ``parameters.<iface>.demand``; the selected delivery topology (its domain guarded to the
    backend's consumable modes) then sizes its own memstream geometry from that demand — the
    op no longer brokers memstream realization or the topology guard. The getters project from
    a resolved point via the impl ``stream``."""
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


# =============================================================================
# 8. FINN WRAPPER — MvauKernelOp(KernelOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# Binds the kernel to a FINN node: three ports (activation in, weights in, activation
# out) mapping the graph's tensor slots to the kernel's inp/weights/out interfaces.
# Registered as ``MVAUKernel_hls`` in the ``finn.custom_op.fpgadataflow.hls`` domain
# (see the hls custom_op dict) so ``is_hls_node`` sees it and the estimate analyses run
# — while the real compute impl is chosen by the ``implementation`` nodeattr, not the
# domain (consumer-surface-model.md R11).

_PORTS = (
    PortSpec(iface="inp", direction="in", index=0, role=Role.DATA_IN),
    PortSpec(iface="weights", direction="in", index=1, role=Role.WEIGHT_SINK),
    # thresholds — the OPTIONAL 3rd input (a 3-input fused MVU). WEIGHT_SINK: a parameter the
    # kernel consumes internally (always baked/constant here). Skipped by the adapter when the
    # node omits the slot (a 2-input node), so its Context tensor is absent — the emergent
    # existence that supersedes noActivation.
    PortSpec(iface="thresholds", direction="in", index=2, role=Role.WEIGHT_SINK, optional=True),
    PortSpec(iface="out", direction="out", index=0, role=Role.DATA_OUT),
)


class MvauKernelOp(KernelOp):
    """MVAU (matrix-vector activation) as a Kernel-backed FINN op."""

    # -- Seam A: frontend claim (mirror of InferQuantizedMatrixVectorActivation) -------

    @staticmethod
    def _operand_map(node: NodeProto) -> dict:
        """The FRONTEND ``MatMul`` tensor → kernel interface-name mapping: ``input[0]`` is
        the activation (``inp``), ``input[1]`` the weight (``weights``), ``output[0]`` the
        result (``out``). The ONE place this mapping lives — both the feasibility trial
        (:meth:`_trial_context`) and the build (:meth:`infer_from`) read it, so a mis-mapped
        operand fails both identically instead of letting the claim and the build diverge."""
        return {INPUT: node.input[0], WEIGHTS: node.input[1], OUTPUT: node.output[0]}

    @classmethod
    def _trial_context(cls, node: NodeProto, model: ModelWrapper) -> Context:
        """The trial :class:`Context` for a FRONTEND ``MatMul`` node, mapping its operands to
        this kernel's interface names via the shared :meth:`_operand_map`. Reads
        shapes/dtypes/initializers off the model — the ``out`` shape is needed too (the tiling
        fold-dial domains read every interface's block extent)."""
        graph_ctx = Context.from_model(model, "")
        operands = cls._operand_map(node)
        shapes, datatypes, inits = {}, {}, {}
        for iface, tname in operands.items():
            if tname in graph_ctx.shapes:
                shapes[iface] = graph_ctx.shapes[tname]
            if tname in graph_ctx.datatypes:
                datatypes[iface] = graph_ctx.datatypes[tname]
            init = graph_ctx.initializer(tname)
            if init is not None:
                inits[iface] = init
        return Context(
            shapes=shapes, datatypes=datatypes, initializers=inits, fpgapart=graph_ctx.fpgapart
        )

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether ``node`` is a ``MatMul`` this kernel can claim (optionally with a following
        ``MultiThreshold``). The claim is STRUCTURAL PATTERN (op-owned) ∧ ∃ a feasible backend
        (pool-delegated): the op owns the shape of the pattern, but WHICH datatypes are
        buildable is a backend fact, so it delegates to :meth:`Kernel.has_feasible_point`
        rather than encoding an integer literal here (F2/D-R5). A future float backend widens
        what infer accepts with ZERO edits here; today an all-integer pool rejects a float
        MatMul FOR THE RIGHT REASON (no feasible backend). Mirrors
        ``InferQuantizedMatrixVectorActivation``'s match (convert_to_hw_layers.py:1493) WITHOUT
        its bakes; the binary/sparse/dynamic cases are out of the vertical slice.
        """
        # --- structural pattern (op-owned) ---
        if node.op_type != "MatMul":
            return False  # a plain structural no-match — legitimately "not mine", stays silent
        # Sparse weights route to VVAU in the classic flow — not our pattern.
        if model.get_tensor_sparsity(node.input[1]) is not None:
            return False
        # The slice claims the STATIC-weight case (a weight initializer must be present);
        # the dynamic-weight branch is out of scope.
        if model.get_initializer(node.input[1]) is None:
            return False

        # --- feasibility (pool-delegated): ∃ a backend with a legal point? ---
        if not cls.kernel().has_feasible_point(cls._trial_context(node, model)):
            # A node that MATCHES the structural pattern but has NO feasible backend is
            # "should be a kernel, but unbuildable by the current pool" — it correctly rides
            # FINN's classic path, but that is a SILENT loss of a structurally-valid kernel
            # (INV5). Log it, distinct from the plain structural no-match above.
            logger.info(
                "MVAU: %s matches the MatMul pattern but no backend has a feasible point "
                "(e.g. non-integer datatypes) — leaving it on FINN's classic path.",
                node.name,
            )
            return False
        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int
    ) -> TransformationResult:
        """Build the unresolved ``finn.kernels`` MVAU node that replaces this ``MatMul``
        (absorbing a following ``MultiThreshold`` when present). Thin per F2′: it re-points
        the SAME input/weight/threshold tensors and bakes ONLY ``ActVal`` — the one residual
        op-owned param with no graph home once the MultiThreshold is absorbed (its
        ``out_bias``). MW/MH/SIMD/PE/mem_mode/numInputVectors and all dtypes stay derived
        live from Context; the folding axes are unset until resolve (Seam B).
        """
        operands = cls._operand_map(node)
        mm_input = operands[INPUT]
        mm_weight = operands[WEIGHTS]
        mm_output = operands[OUTPUT]

        consumer = model.find_consumer(mm_output)
        has_activation = consumer is not None and consumer.op_type == "MultiThreshold"

        if has_activation:
            mt_thres = consumer.input[1]
            mt_output = consumer.output[0]
            actval = int(getCustomOp(consumer).get_nodeattr("out_bias"))
            kernel_node = helper.make_node(
                "MVAU",
                [mm_input, mm_weight, mt_thres],
                [mt_output],
                domain="finn.kernels",
                backend="fpgadataflow",
                name="MVAU_" + node.name,
                ActVal=actval,
            )
            return TransformationResult(
                nodes_to_insert=[kernel_node], nodes_to_remove=[node, consumer]
            )

        kernel_node = helper.make_node(
            "MVAU",
            [mm_input, mm_weight],
            [mm_output],
            domain="finn.kernels",
            backend="fpgadataflow",
            name="MVAU_" + node.name,
            ActVal=0,
        )
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=[node])

    @classmethod
    def kernel(cls):
        return mvau_kernel()

    def ports(self) -> tuple[PortSpec, ...]:
        return _PORTS

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        # MVAU's outputDataType is a resolved derived: the graph dtype when the node has
        # thresholds (they map the accumulator down), or the weight-derived accumulator type
        # when it has none. Read it off the point so infer propagates the exact (possibly
        # narrowed) type.
        if index == 0 and "outputDataType" in point:
            return point["outputDataType"]
        return super()._output_datatype_from_point(kernel, ctx, point, index)

    def get_folding_axes(self):
        """The folding dials this op exposes, each mapped to its resolved max value —
        the capability SetFolding queries instead of op_type prefix-matching
        (consumer-surface-model.md R1). SIMD folds the reduction dim MW, PE the output
        dim MH; both are the weight block's extents, read straight off the Context
        (``tensor_shape(weights) == (MW, MH)``), not a stored nodeattr."""
        _, ctx, _ = self._point()
        mw, mh = ctx.tensor_shape(WEIGHTS)
        return {"SIMD": int(mw), "PE": int(mh)}
