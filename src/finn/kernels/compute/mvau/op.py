############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — how FINN's build flow sees the kernel: ``MvauDataflowOp(DataflowOp)``.

The kernel DEFINITION (interfaces, design space, datatype rules, cost, assembly) lives in
``kernel.py`` — read that to understand WHAT an MVAU is. This file holds only the FINN
wrapper: the Seam-A frontend claim (``can_infer_from``/``infer_from``, mirror of
``InferQuantizedMatrixVectorActivation``) and the port binding that maps the graph's tensor
slots to the kernel's inp/weights/thresholds/out interfaces.

The kernel's public surface (constants, ``mvau_kernel``/``mvau_space``/``mvau_pool``/…) is
re-exported here so ``from finn.kernels.compute.mvau.op import X`` keeps resolving — the backend
backends and tests read constants/assembly through this module.
"""

from __future__ import annotations

import logging

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.ir import DataflowOp, TransformationResult
from finn.kernels.engine.datatype_spec import resolve_datatype_spec
from finn.kernels.engine.point import Illegal  # noqa: F401  (kept available for callers/tests)
from finn.kernels.compute.mvau._dsp_rtl import VERSION  # noqa: F401  (re-exported for backends)

# The kernel DEFINITION — re-exported so `from .op import X` keeps working for the backend
# backends, the composition helper, and the tests that resolve against this module.
from .impl_hls import hls_bundle as hls_backend
from .impl_rtl_packed import packed_bundle as packed_backend
from .impl_rtl_softvec import softvec_bundle as softvec_backend
from .kernel import _mvau_constraints
from .kernel import (  # noqa: F401  (re-exported public surface)
    INPUT,
    MVAU_DSP_PACKED,
    MVAU_DSP_SOFTVEC,
    MVAU_HLS,
    OUTPUT,
    THRESHOLDS,
    WEIGHTS,
    mvau_interfaces,
    mvau_kernel,
    mvau_pool,
    mvau_space,
    kernel_attrs,
    op_axes,
    op_derived,
    op_predicates,
)

# The BACKEND-SCOPED shared contract (fold map + datatype derivations) — re-exported so the
# backend modules read `from .op import COMPUTE_STREAM, mvau_out_dtype, mvau_register_dtypes`.
from .backends import (  # noqa: F401  (re-exported public surface)
    COMPUTE_STREAM,
    mvau_out_dtype,
    mvau_register_dtypes,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FINN WRAPPER — MvauDataflowOp(DataflowOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# The interface↔node-slot binding is the kernel's own interface list (inp=0, weights=1,
# optional thresholds=2, out=0 — declaration order). The real compute backend is chosen by the
# ``backend`` nodeattr, not the domain (consumer-surface-model.md R11).


class MvauDataflowOp(DataflowOp):
    """MVAU (matrix-vector activation) — the op class IS the kernel.

    The class body below is the design space that used to be a separate `DataflowKernel`
    value reached through a `.kernel()` classmethod. Every field here is op-CLASS identity —
    true of every MVAU node, not of any one — so a class body is its home (F6).

    The compute pool (HLS / DSP-softvec / DSP-packed) carries backend-owned tiling. Weights +
    thresholds are DERIVED as delivered parameters from the pool's ``mem_modes`` (a backend
    declares which param ports it consumes) by `DataflowOp.__init_subclass__`, which also
    resolves the interface slot indices and validates per-port direction — all at class
    definition, so an authoring mistake is an import error.
    """

    # -- the design space (see kernel.py for what each piece means) ---------
    name = "MVAU"
    interfaces = mvau_interfaces()
    pool = (hls_backend(), softvec_backend(), packed_backend())
    op_axes = op_axes()
    op_derived = op_derived()
    op_predicates = op_predicates()
    kernel_attrs = kernel_attrs()
    constraints = _mvau_constraints()

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
