############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Thresholding — the multi-threshold activation kernel: WHAT it is, and how FINN sees it.

The op-definition file (mirrors FINN's ``thresholding.py``, declarative). The shared
design space (axes/derived/predicates) lives in ``shared.py``; each ``impl_*.py`` bundle
declares the HOW for one compute core (HLS baked-ROM, RTL binary-search). This file adds
the two assemblies the bundles could not: the ``Kernel`` (identity + pool + delivered
parameters) and the FINN ``KernelOp`` wrapper.

Tensor-name convention for the Context this schema resolves against:
    "inp"         the activation input tensor   (inputDataType, dynamic)
    "thresholds"  the threshold tensor          (thresholdDataType + initializer VALUES)
    "out"         the output tensor             (graph output dtype)

The ``thresholds`` interface is the op's ONE parameter interface. It is a SEPARABLE,
static-schedule memory in the HLS backend (a baked ``thresh.h`` ROM read by the
output-channel loop → the ``embedded``/constant topology) and a FUSED, data-dependent
memory in the RTL backend (the binary-search ``.dat`` scatter, addressed by the runtime
comparison outcome). Both resolve to the ``embedded`` (constant) topology → demand None,
no memstream cell; the HLS/RTL difference is purely each compute emit's baked artifact.
"""

from __future__ import annotations

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.ir import KernelOp, TransformationResult
from finn.kernels.model.kernel import InterfaceSchema, Kernel
from finn.kernels.model.ports import Direction
from finn.kernels.model.tiling import FULL

from .names import (  # noqa: F401
    COMPUTE_STREAM,
    INPUT,
    OUTPUT,
    THRESHOLDING_HLS,
    THRESHOLDING_RTL,
    THRESHOLDS,
)
from finn.kernels.dataflow.parameters.registry import generation as parameters_generation
from finn.kernels.model.registry import registry_cached

from .registry import build_pool, generation
from .shared import op_axes, op_derived, op_predicates


# =============================================================================
# INTERFACES — the ONNX-facing arity + direction.
# =============================================================================


def thresholding_interfaces():
    """The op-side interface list — identity + DIRECTION + BLOCK structure. ``inp`` and
    ``out`` iterate the spatial/vector count (``1``) and hold the channel dim in-block
    (``FULL``); ``thresholds`` is the whole ``(NumChannels, numSteps)`` matrix in one
    block. PE folds the channel dim (out position 1); the threshold block folds with it."""
    return (
        InterfaceSchema(INPUT, Direction.IN, block=[1, FULL]),          # (n_vecs, NumChannels)
        InterfaceSchema(THRESHOLDS, Direction.IN, block=[FULL, FULL]),  # (NumChannels, numSteps)
        InterfaceSchema(OUTPUT, Direction.OUT, block=[1, FULL]),
    )


# =============================================================================
# ASSEMBLY — the full Thresholding design space as a Kernel (and as a DesignSpace).
# =============================================================================


def thresholding_pool():
    """The registered Thresholding implementations (flat peers), in registration order."""
    return build_pool()


@registry_cached(generation, parameters_generation)
def thresholding_kernel() -> Kernel:
    """The full Thresholding design space as a :class:`Kernel` — the WHAT-owning op node.

    The compute pool (``implementation``: HLS / RTL) with impl-owned tiling. The threshold
    interface is DERIVED as a delivered parameter from the pool's ``mem_modes``; the Kernel
    builds its DeliveredParam and synthesizes the supply waterfall generically; both backends
    consume thresholds in embedded mode → the delivery resolves to the ``embedded`` topology
    (no memstream cell).

    CACHED on the compute + parameters registry generations — see :func:`mvau_kernel`."""
    return Kernel(
        name="Thresholding",
        interfaces=thresholding_interfaces(),
        op_axes=op_axes(),
        op_derived=op_derived(),
        op_predicates=op_predicates(),
        pool=thresholding_pool(),
    )


# =============================================================================
# FINN WRAPPER — ThresholdingKernelOp(KernelOp): how FINN's build flow sees this kernel.
# =============================================================================
#
# The interface↔node-slot binding is the kernel's own interface list (inp=0, thresholds=1,
# out=0 — declaration order); no separate PortSpec (F9).


class ThresholdingKernelOp(KernelOp):
    """Thresholding (multi-threshold activation) as a Kernel-backed FINN op."""

    # -- Seam A: frontend claim (mirror of InferThresholdingLayer) ---------------------

    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether ``node`` is a STANDALONE ``MultiThreshold`` this kernel can claim.
        Mirrors ``InferThresholdingLayer`` (convert_to_hw_layers.py:159) as explicit
        preconditions (return False, not assert).

        Precedence: a ``MultiThreshold`` fed by a ``MatMul`` is the FUSED activation the
        MVAU kernel absorbs — it must NOT be claimed standalone. Since MVAU precedes this op
        in the pool and removes the consumer during its own inference, that node is normally
        gone before we reach it; the explicit producer check makes the ordering robust even
        against a stale node-list snapshot.

        WHY THESE CHECKS STAY IN PYTHON (unlike MVAU's, which became IsStatic/SparsityFree
        constraints): every one below is a FRONTEND-GRAPH fact, not a design-space fact. The
        producer identity, the tensor layout and ``out_scale`` describe the shape of the
        pattern this op claims — the op's own business — not what a backend can build. A
        constraint is the right home only for the latter, because its whole purpose is to let
        a NEW BACKEND widen the claim; no backend will ever make an NCHW layout claimable.

        The dtype check is the interesting case, and it deliberately did NOT migrate to
        ``DatatypeSupport``. Two reasons, both checked rather than assumed:
          1. It is a UNION of kinds (integer OR fixed OR float32/16). ``DatatypeSupport`` is
             one kind plus a bitwidth range, so expressing it would need a custom callable —
             a closure on the port, which is no more declarative than the closure here.
          2. Both backends have IDENTICAL envelopes and there is no verified per-bundle gate
             (a fabricated one was previously falsified — see the package ``__init__`` and
             ``scratchpad/reference/toy-vs-brainsmith-thresholding.md`` A1). Empirically this
             check rejects only ``SCALEDINT`` among FINN's dtypes, so declaring a backend
             gate would be inventing a distinction no backend actually makes.
        Migrating it would move a frontend-pattern fact into the design space AND fabricate
        backend knowledge to do it. When a Thresholding backend appears whose dtype envelope
        genuinely differs, THAT is the moment to declare a gate — with a real case behind it.
        """
        if node.op_type != "MultiThreshold":
            return False
        producer = model.find_producer(node.input[0])
        if producer is not None and producer.op_type == "MatMul":
            return False

        # Pattern-shape dtype admissibility (mirrors InferThresholdingLayer), NOT a
        # buildability gate — see the docstring.
        idt = model.get_tensor_datatype(node.input[0])
        tdt = model.get_tensor_datatype(node.input[1])
        idt_ok = idt.is_integer() or idt.is_fixed_point() or idt in ["FLOAT32", "FLOAT16"]
        tdt_ok = tdt.is_integer() or tdt.is_fixed_point() or tdt in ["FLOAT32", "FLOAT16"]
        if not (idt_ok and tdt_ok):
            return False

        # The slice claims NHWC/2-D layouts only; NCHW would need a layout-conversion node
        # (FINN's :194-206), deferred. A None layout (plain 2-D matmul activations) is fine.
        from qonnx.core.data_layout import NCHW

        if model.get_tensor_layout(node.input[0]) == NCHW:
            return False

        # out_scale must be 1 for HW conversion (FINN :215).
        if getCustomOp(node).get_nodeattr("out_scale") != 1.0:
            return False
        return True

    @classmethod
    def infer_from(
        cls, node: NodeProto, model: ModelWrapper, insert_index: int
    ) -> TransformationResult:
        """Build the unresolved ``finn.kernels`` Thresholding node replacing this
        ``MultiThreshold``. Thin per F2′: it re-points the SAME input/threshold tensors and
        bakes ONLY ``ActVal`` (the ``out_bias`` residual). NumChannels/numSteps/PE/
        numInputVectors and all dtypes stay derived live from Context.
        """
        actval = int(getCustomOp(node).get_nodeattr("out_bias"))
        kernel_node = helper.make_node(
            "Thresholding",
            [node.input[0], node.input[1]],
            [node.output[0]],
            domain="finn.kernels",
            name="Thresholding_" + node.name,
            ActVal=actval,
        )
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=[node])

    @classmethod
    def kernel(cls):
        return thresholding_kernel()

    def get_folding_axes(self):
        """PE folds the channel dim NumChannels (the threshold tensor's leading extent),
        read straight off the Context."""
        _, ctx, _ = self._point()
        channels = ctx.tensor_shape(THRESHOLDS)[0]
        return {"PE": int(channels)}
