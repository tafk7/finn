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
    "out"         the output tensor             (outputDataType)

The ``thresholds`` interface is the op's ONE parameter interface. It is a SEPARABLE,
static-schedule memory in the HLS backend (a baked ``thresh.h`` ROM read by the
output-channel loop → the ``embedded``/constant topology) and a FUSED, data-dependent
memory in the RTL backend (the binary-search ``.dat`` scatter, addressed by the runtime
comparison outcome). Both resolve to the ``embedded`` (constant) topology → demand None,
no memstream cell; the HLS/RTL difference is purely each compute emit's baked artifact.
"""

from __future__ import annotations

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.kernels.adapter import KernelOp, PortSpec, TransformationResult
from finn.kernels.space import (
    FULL,
    DeliveredParam,
    Direction,
    Interface,
    Kernel,
    KernelSchema,
    Role,
)
from finn.kernels.ops.parameters import parameters_pool

from .names import INPUT, OUTPUT, THRESHOLDING_HLS, THRESHOLDING_RTL, THRESHOLDS  # noqa: F401
from .registry import build_pool
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
        Interface(INPUT, Direction.IN, block=[1, FULL]),          # (n_vecs, NumChannels)
        Interface(THRESHOLDS, Direction.IN, block=[FULL, FULL]),  # (NumChannels, numSteps)
        Interface(OUTPUT, Direction.OUT, block=[1, FULL], dtype_source="outputDataType"),
    )


# =============================================================================
# COMPUTE TILING — the BLOCK->STREAM lowering shared by both compute impls.
# =============================================================================
#
# PE folds the channel dim (NumChannels) on the input, output, and the threshold block's
# leading (channel) extent. The threshold's step dim is unfolded (whole row per beat).
COMPUTE_STREAM = {
    INPUT: [1, "PE"],
    OUTPUT: [1, "PE"],
    THRESHOLDS: ["PE", 1],
}


# =============================================================================
# DELIVERY — the threshold parameter interface + its cadence.
# =============================================================================


def _threshold_cadence(p, ctx) -> int:
    # Thresholds are re-traversed once per output beat: cadence = prod(numInputVectors)
    # = the input's non-channel leading dims (the TAP_REP). In the baked ROM the thresholds
    # are constant (demand=None), so this does not size a streamer yet — but it is the real
    # quantity a decoupled/MLO threshold variant would need.
    return int(np.prod(ctx.tensor_shape(INPUT)[:-1]))


def _delivered_parameters():
    return (DeliveredParam(THRESHOLDS, _threshold_cadence, pool=parameters_pool(THRESHOLDS)),)


# =============================================================================
# ASSEMBLY — the full Thresholding design space as a Kernel (and as a Schema).
# =============================================================================


def thresholding_pool():
    """The registered Thresholding implementations (flat peers), in registration order."""
    return build_pool()


def thresholding_kernel() -> Kernel:
    """The full Thresholding design space as a :class:`Kernel` — the WHAT-owning op node.

    The compute pool (``implementation``: HLS / RTL) with impl-owned tiling, plus the
    DECLARED delivered threshold parameter. The Kernel synthesizes the supply waterfall
    generically; both backends consume thresholds in constant mode → the delivery resolves
    to the ``embedded`` topology (no memstream cell)."""
    return Kernel(
        identity=KernelSchema(
            name="Thresholding",
            interfaces=thresholding_interfaces(),
            op_axes=op_axes(),
            op_derived=op_derived(),
            op_predicates=op_predicates(),
        ),
        pool=thresholding_pool(),
        delivered_parameters=_delivered_parameters(),
    )


def thresholding_kernel_schema():
    """The full Thresholding design space as a resolve ``Schema`` — delegates to
    :func:`thresholding_kernel`."""
    return thresholding_kernel().schema()


# =============================================================================
# FINN WRAPPER — ThresholdingKernelOp(KernelOp): how FINN's build flow sees this kernel.
# =============================================================================

_PORTS = (
    PortSpec(iface=INPUT, direction="in", index=0, role=Role.DATA_IN),
    # thresholds — the parameter the kernel consumes internally (baked/constant here).
    PortSpec(iface=THRESHOLDS, direction="in", index=1, role=Role.WEIGHT_SINK),
    PortSpec(iface=OUTPUT, direction="out", index=0, role=Role.DATA_OUT),
)


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
        """
        if node.op_type != "MultiThreshold":
            return False
        producer = model.find_producer(node.input[0])
        if producer is not None and producer.op_type == "MatMul":
            return False

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
            backend="fpgadataflow",
            name="Thresholding_" + node.name,
            ActVal=actval,
        )
        return TransformationResult(nodes_to_insert=[kernel_node], nodes_to_remove=[node])

    @classmethod
    def kernel(cls):
        return thresholding_kernel()

    def ports(self) -> tuple[PortSpec, ...]:
        return _PORTS

    def _output_datatype_from_point(self, kernel, ctx, point, index):
        if index == 0 and "outputDataType" in point:
            return point["outputDataType"]
        return super()._output_datatype_from_point(kernel, ctx, point, index)

    def get_folding_axes(self):
        """PE folds the channel dim NumChannels (the threshold tensor's leading extent),
        read straight off the Context."""
        _, ctx, _ = self._point()
        channels = ctx.tensor_shape(THRESHOLDS)[0]
        return {"PE": int(channels)}
