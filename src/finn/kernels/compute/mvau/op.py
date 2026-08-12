############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""MVAU — how FINN's build flow sees the kernel: ``MvauKernelOp(KernelOp)``.

The kernel DEFINITION (interfaces, design space, datatype rules, cost, assembly) lives in
``kernel.py`` — read that to understand WHAT an MVAU is. This file holds only the FINN
wrapper: the Seam-A frontend claim (``can_infer_from``/``infer_from``, mirror of
``InferQuantizedMatrixVectorActivation``) and the port binding that maps the graph's tensor
slots to the kernel's inp/weights/thresholds/out interfaces.

The kernel's public surface (constants, ``mvau_kernel``/``mvau_space``/``mvau_pool``/…) is
re-exported here so ``from finn.kernels.compute.mvau.op import X`` keeps resolving — the impl
bundles and tests read constants/assembly through this module.
"""

from __future__ import annotations

import logging

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.ir import KernelOp, TransformationResult
from finn.kernels.engine.datatype_spec import resolve_datatype_spec
from finn.kernels.engine.point import Illegal  # noqa: F401  (kept available for callers/tests)
from finn.kernels.compute.mvau._dsp_rtl import VERSION  # noqa: F401  (re-exported for bundles)

# The kernel DEFINITION — re-exported so `from .op import X` keeps working for the impl
# bundles, the composition helper, and the tests that resolve against this module.
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
# impl bundles read `from .op import COMPUTE_STREAM, mvau_out_dtype, mvau_register_dtypes`.
from .backends import (  # noqa: F401  (re-exported public surface)
    COMPUTE_STREAM,
    mvau_out_dtype,
    mvau_register_dtypes,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FINN WRAPPER — MvauKernelOp(KernelOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# The interface↔node-slot binding is the kernel's own interface list (inp=0, weights=1,
# optional thresholds=2, out=0 — declaration order). The real compute impl is chosen by the
# ``backend`` nodeattr, not the domain (consumer-surface-model.md R11).


class MvauKernelOp(KernelOp):
    """MVAU (matrix-vector activation) as a _LegacyKernel-backed FINN op."""

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
    def _trial_context(cls, node: NodeProto, model: ModelWrapper) -> "Context":
        """The trial :class:`Context` for a FRONTEND ``MatMul`` node, mapping its operands to
        this kernel's interface names via the shared :meth:`_operand_map`. Reads
        shapes/dtypes/initializers off the model — the ``out`` shape is needed too (the tiling
        fold-dial domains read every interface's block extent)."""
        from finn.kernels.engine.context import Context

        graph_ctx = Context.from_model(model, "")
        operands = cls._operand_map(node)
        shapes, datatypes, inits, sparsity = {}, {}, {}, {}
        for iface, tname in operands.items():
            if tname in graph_ctx.shapes:
                shapes[iface] = graph_ctx.shapes[tname]
            if tname in graph_ctx.datatypes:
                datatypes[iface] = graph_ctx.datatypes[tname]
            init = graph_ctx.initializer(tname)
            if init is not None:
                inits[iface] = init
            # Sparsity must be carried, not just shapes/dtypes: the weights port declares a
            # SparsityFree constraint, and a given the trial context drops is a rule that
            # silently cannot fire.
            sp = graph_ctx.tensor_sparsity(tname)
            if sp is not None:
                sparsity[iface] = sp
        return Context(
            shapes=shapes,
            datatypes=datatypes,
            initializers=inits,
            fpgapart=graph_ctx.fpgapart,
            sparsity=sparsity,
        )

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether ``node`` is a ``MatMul`` this kernel can claim (optionally with a following
        ``MultiThreshold``). The claim is STRUCTURAL PATTERN (op-owned) ∧ ∃ a feasible backend
        (pool-delegated): the op owns the shape of the pattern, but WHICH datatypes are
        buildable is a backend fact, so it delegates to :meth:`_LegacyKernel.has_feasible_point`
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
            name="MVAU_" + node.name,
            ActVal=0,
        )
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=[node])

    @classmethod
    def kernel(cls):
        return mvau_kernel()

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        # MVAU's output dtype is the out port's derived_dtype spec: the graph dtype when the
        # node has thresholds (they map the accumulator down), or the weight-derived
        # accumulator type when it has none. Resolve it so infer propagates the exact
        # (possibly narrowed) type — the SAME rule the stream-width fold and emit read. The
        # spec is a DependentSpec (it carries the ParamDatatype dep); resolve_datatype_spec
        # unwraps it, so this reads authority off the ParamDatatype like every other consumer.
        if index == 0:
            return resolve_datatype_spec(mvau_out_dtype(), iface=OUTPUT, point=point, context=ctx)
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
