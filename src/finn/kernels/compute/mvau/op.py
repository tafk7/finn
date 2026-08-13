############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — the matrix-vector activation op. Its class body declares the whole design space.

`MvauDataflowOp` COMPOSES three cells — the compute kernel (HLS / DSP-softvec / DSP-packed)
plus one parameter-delivery kernel per delivered interface (``weights``, ``thresholds``) —
into a single design space with three selection roots. The op is the PRODUCT; each cell is
a SUM over its pool. What F6 merged into this class was the CONTAINER (`DataflowKernel`),
not a kernel: there is no separate container object and no ``.kernel()`` hop, but the op is
still not one kernel. See ``model/cell.py`` for the tier table.

The parameter cells are DERIVED, not declared — `__init_subclass__` reads the pool's
``mem_modes`` — so this file names only the compute pool and the op-level design space.

Also here: the Seam-A frontend claim (``can_infer_from``/``infer_from``, mirroring
``InferQuantizedMatrixVectorActivation``), which is op-owned pattern knowledge.

Tensor names live in ``names.py``, a LEAF, because the backend modules key Context lookups
by them and this module imports the backends. Each backend's IDENTITY string lives in that
backend's own ``impl_*.py`` (a pool member names itself). The backend-scoped contract (fold
map, datatype derivations) is in ``backends.py``. All are re-exported here so
``from .op import X`` keeps resolving for the tests and helpers that read them through this
module.
"""

from __future__ import annotations

import logging

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.engine.attr import attr
from finn.kernels.engine.constraints import IsStatic, ShapeRank, SparsityFree, ValueNonNeg
from finn.kernels.engine.derived import Derived
from finn.kernels.model.kernel import InterfaceSchema
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL
from finn.kernels.compute.thresholding.shared import _threshold_datatype
from finn.kernels.ir import DataflowOp, TransformationResult
from finn.kernels.engine.datatype_spec import resolve_datatype_spec
from finn.kernels.engine.point import Illegal  # noqa: F401  (kept available for callers/tests)
from finn.kernels.compute.mvau._dsp_rtl import VERSION  # noqa: F401  (re-exported for backends)

# The kernel DEFINITION — re-exported so `from .op import X` keeps working for the backend
# backends, the composition helper, and the tests that resolve against this module.
from .impl_hls import MVAU_HLS, hls_bundle as hls_backend  # noqa: F401  (name re-exported)
from .impl_rtl_packed import (  # noqa: F401  (name re-exported)
    MVAU_DSP_PACKED,
    packed_bundle as packed_backend,
)
from .impl_rtl_softvec import (  # noqa: F401  (name re-exported)
    MVAU_DSP_SOFTVEC,
    softvec_bundle as softvec_backend,
)
from .names import INPUT, OUTPUT, THRESHOLDS, WEIGHTS  # noqa: F401  (re-exported)

# The BACKEND-SCOPED shared contract (fold map + datatype derivations) — re-exported so the
# backend modules read `from .op import COMPUTE_STREAM, mvau_out_dtype, mvau_register_dtypes`.
from .backends import (  # noqa: F401  (re-exported public surface)
    COMPUTE_STREAM,
    mvau_out_dtype,
    mvau_register_dtypes,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OP DESIGN SPACE helpers — the shared BLOCK structure's small closures.
# Named functions (not lambdas) so a traceback reads.
# =============================================================================


def _is_nonneg_int(v) -> bool:
    return isinstance(v, int) and v >= 0


def _threshold_dtype(p, ctx):
    # The threshold operand's DECLARED dtype, or None on a 2-input node. Delegates to the
    # SHARED Thresholding rule so the fused and standalone paths cannot drift — see
    # ``compute/thresholding/shared.py::_threshold_datatype`` for the parity TODO on
    # re-enabling value-narrowing.
    return _threshold_datatype(p, ctx) if ctx.has_tensor(THRESHOLDS) else None


def _unsigned_input(p, ctx) -> bool:
    return not ctx.tensor_datatype(INPUT).signed()


# =============================================================================
# INTERFACES — the ONNX-facing arity + direction (no tiling; tiling is backend-owned)
# =============================================================================


def mvau_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure (the math). No
    semantic role: whether ``weights`` is a stored parameter or a live activation emerges
    from graph context (initializer?) at resolve time. Stream folding (SIMD/PE) is backend-owned.

    The block reads as the matmul: ``inp`` iterates its vector count (``1``) and holds the
    reduction dim MW in-block (``FULL``); ``weights`` is the whole matrix ``(MW, MH)`` in one
    block; ``out`` iterates vectors and holds MH."""
    return (
        InterfaceSchema(INPUT, Direction.IN, block=[1, FULL]),          # (n_vecs, MW)
        # weights — the STATIC + DENSE requirements are declared here, where the pool can
        # widen them. Both were hand-written escapes in the frontend claim, which meant the
        # vocabulary member (IsStatic) sat dead while the fact it encodes lived in Python
        # and could drift. A backend that can consume dynamic or sparse weights now widens
        # what infer accepts by declaring so, with no frontend edit.
        InterfaceSchema(
            WEIGHTS, Direction.IN, block=[FULL, FULL],  # (MW, MH)
            constraints=(IsStatic(WEIGHTS), SparsityFree(WEIGHTS)),
        ),
        # thresholds — optional (NumChannels, numSteps); ShapeRank auto-skips when absent.
        InterfaceSchema(
            THRESHOLDS, Direction.IN, block=[FULL, FULL], optional=True,
            constraints=(ShapeRank(THRESHOLDS, 2),),
        ),
        # out — dtype is backend-derived (accDataType under no-activation), declared as the
        # out port's derived_dtype (mvau_out_dtype, backends.py); here only arity/block.
        InterfaceSchema(OUTPUT, Direction.OUT, block=[1, FULL]),
    )


# =============================================================================
# FINN WRAPPER — MvauDataflowOp(DataflowOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# The interface↔node-slot binding is the op's own interface list (inp=0, weights=1,
# optional thresholds=2, out=0 — declaration order). The real compute backend is chosen by the
# ``backend`` nodeattr, not the domain (consumer-surface-model.md R11).


class MvauDataflowOp(DataflowOp):
    """MVAU (matrix-vector activation) — an op composing three cells.

    The class body below is the design space that used to be a separate `DataflowKernel`
    value reached through a `.kernel()` classmethod. Every field here is op-CLASS identity —
    true of every MVAU node, not of any one — so a class body is its home (F6).

    Only the COMPUTE cell's pool (HLS / DSP-softvec / DSP-packed) is written here. The other
    two cells are derived: weights + thresholds become delivered parameters because some
    backend names them in its ``mem_modes``, and `DataflowOp.__init_subclass__` turns each
    into a storage-topology pool of its own. So the compiled space has three selection roots
    (``backend``, ``parameters.weights.topology``, ``parameters.thresholds.topology``), and
    a design point picks one member from each.

    ``__init_subclass__`` also resolves the interface slot indices and validates per-port
    direction — at class definition, so an authoring mistake is an import error.
    """

    # -- the design space ---------------------------------------------------
    name = "MVAU"

    # The ONNX-facing arity + BLOCK structure (the math). Stream folding is backend-owned.
    interfaces = mvau_interfaces()

    # The compute pool: HLS / DSP-softvec / DSP-packed. Declaration order IS selection
    # precedence. Adding a backend = write one impl_*.py and name it here.
    pool = (hls_backend(), softvec_backend(), packed_backend())

    # EMPTY: MVAU has no hand-authored DSE dials. SIMD/PE are tiling-engine-generated from
    # the interface BLOCKs; ram_style/… are parameters-pool-generated; MW/MH are Context
    # extents.
    op_axes = ()

    # All legality is declarative constraints (per-port + the relational one below).
    op_predicates = ()

    # ONLY the threshold-operand dtype is realization-invariant. The accumulator / weight /
    # output dtypes are backend-scoped (`mvau_register_dtypes` / `mvau_out_dtype` in
    # backends.py); stream widths + geometry are tiling/emit-generated.
    #
    # No dep: while narrowing is off for FINN parity the shared rule is Context-only, so
    # there is nothing to order against. Re-enabling it restores the read of the thresholds
    # ParamDatatype AND this dep together — the topo-sort needs the dep to place this after
    # the publisher, across the compute/parameters pool boundary.
    op_derived = (Derived("thresholdDataType", _threshold_dtype),)

    # Node CONSTANTS: on the Point, read by emit, never explored. See engine/attr.py for why
    # these are a primitive rather than Context fields — both are named in deps
    # (`narrow_weights` reads mlo_max_iter), and a Context field can never be.
    kernel_attrs = (
        # The absorbed MultiThreshold's out_bias. THE case proving the category is real:
        # infer REMOVES that node, so the graph no longer holds this value.
        attr("ActVal", "int", lambda v: isinstance(v, int), 0),
        # MLO parameter-set cardinality, written by loop_rolling.py. Still misnamed (the RTL
        # calls it SETS) and still carrying two concepts — mlo-cardinality.md owns that
        # question. It is an Attr here purely so the strata are honest meanwhile.
        attr("mlo_max_iter", "nonneg int", _is_nonneg_int, 0),
    )

    # unsigned input ⇒ all thresholds >= 0 (thresholding.py:243). Reads INPUT to gate a rule
    # on THRESHOLDS, so it is kernel-level (relational), not a per-port constraint.
    constraints = (ValueNonNeg(THRESHOLDS, when=_unsigned_input),)

    # -- Seam A: frontend claim (mirror of InferQuantizedMatrixVectorActivation) -------

    @classmethod
    def _candidate_slots(cls, node: NodeProto, model: ModelWrapper) -> tuple[list, list]:
        """The node slots the kernel node WOULD carry if it claimed this ``MatMul`` — the
        activation and weight inputs, plus the absorbed ``MultiThreshold``'s thresholds and
        output when one follows.

        The ONE description of the frontend→kernel wiring, read by both the claim
        (:meth:`can_infer_from`, via :meth:`candidate_op`) and the build
        (:meth:`infer_from`). It returns SLOTS, positionally, not an
        ``{interface: tensor}`` map: the interface binding is
        :attr:`InterfaceSchema.index`, and restating it here is what let the old
        ``_operand_map`` drift from the build path (F9)."""
        inputs = [node.input[0], node.input[1]]
        outputs = [node.output[0]]
        consumer = model.find_consumer(node.output[0])
        if consumer is not None and consumer.op_type == "MultiThreshold":
            inputs.append(consumer.input[1])
            outputs = [consumer.output[0]]
        return inputs, outputs

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether ``node`` is a ``MatMul`` this kernel can claim (optionally with a following
        ``MultiThreshold``). The claim is STRUCTURAL PATTERN (op-owned) ∧ ∃ a feasible backend
        (pool-delegated): the op owns the shape of the pattern, but WHICH datatypes are
        buildable is a backend fact, so it delegates to :meth:`DataflowOp.has_feasible_point`
        rather than encoding an integer literal here (F2/D-R5). A future float backend widens
        what infer accepts with ZERO edits here; today an all-integer pool rejects a float
        MatMul FOR THE RIGHT REASON (no feasible backend). Mirrors
        ``InferQuantizedMatrixVectorActivation``'s match (convert_to_hw_layers.py:1493) WITHOUT
        its bakes; the binary/sparse/dynamic cases are out of the vertical slice.
        """
        # --- structural pattern (op-owned) ---
        if node.op_type != "MatMul":
            return False  # a plain structural no-match — legitimately "not mine", stays silent

        # NO escape hatches here. The static-weight and dense-weight requirements used to be
        # hand-written below this line; they are now IsStatic/SparsityFree constraints on the
        # weights interface, so the claim is exactly "the pattern matches AND some backend
        # can build it" — and a backend that widens either requirement needs no edit here.

        # --- feasibility (pool-delegated): ∃ a backend with a legal point? ---
        # Asked of a CANDIDATE kernel node built from the same slots infer_from would use, so
        # the claim and the build cannot disagree about which tensor is which (F9). The
        # candidate is never inserted; the graph is unmodified.
        inputs, outputs = cls._candidate_slots(node, model)
        candidate = cls.candidate_op(model, inputs, outputs)
        if not cls.has_feasible_point(candidate._context()):
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
        # The SAME slots the claim interrogated (F9) — one description of the wiring, so a
        # node that was claimed is built over exactly the tensors it was claimed on.
        inputs, outputs = cls._candidate_slots(node, model)

        consumer = model.find_consumer(node.output[0])
        has_activation = consumer is not None and consumer.op_type == "MultiThreshold"
        actval = (
            int(getCustomOp(consumer).get_nodeattr("out_bias")) if has_activation else 0
        )

        kernel_node = helper.make_node(
            "MVAU",
            inputs,
            outputs,
            domain="finn.kernels",
            name="MVAU_" + node.name,
            ActVal=actval,
        )
        removed = [node, consumer] if has_activation else [node]
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=removed)

    @classmethod
    def kernel(cls):
        return mvau_kernel()

    def _output_datatype_from_point(self, ctx, point, index):
        # MVAU's output dtype is the out port's derived_dtype spec: the graph dtype when the
        # node has thresholds (they map the accumulator down), or the weight-derived
        # accumulator type when it has none. Resolve it so infer propagates the exact
        # (possibly narrowed) type — the SAME rule the stream-width fold and emit read. The
        # spec is a DependentSpec (it carries the ParamDatatype dep); resolve_datatype_spec
        # unwraps it, so this reads authority off the ParamDatatype like every other consumer.
        if index == 0:
            return resolve_datatype_spec(mvau_out_dtype(), iface=OUTPUT, point=point, context=ctx)
        return super()._output_datatype_from_point(ctx, point, index)

    def get_folding_axes(self):
        """The folding dials this op exposes, each mapped to its resolved max value —
        the capability SetFolding queries instead of op_type prefix-matching
        (consumer-surface-model.md R1). SIMD folds the reduction dim MW, PE the output
        dim MH; both are the weight block's extents, read straight off the Context
        (``tensor_shape(weights) == (MW, MH)``), not a stored nodeattr."""
        ctx = self._context()
        mw, mh = ctx.tensor_shape(WEIGHTS)
        return {"SIMD": int(mw), "PE": int(mh)}


# =============================================================================
# COMPATIBILITY SHIMS — there is no container object; these are spellings, not objects.
# =============================================================================


def mvau_kernel():
    """The MVAU design space — which is simply the op class.

    A one-line alias kept so the ~40 sites naming ``mvau_kernel()`` keep reading naturally.
    There is no object to build: `MvauDataflowOp` holds the space in its class body (F6)."""
    return MvauDataflowOp


def mvau_pool():
    """The MVAU backends, in declaration order (= selection precedence)."""
    return MvauDataflowOp.pool


def mvau_space():
    """The full MVAU design space as a resolve ``DesignSpace``."""
    return MvauDataflowOp.compile()
