############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The FINN adapter layer: ``KernelOp(HWCustomOp)`` — the op-agnostic base that lets a
model-free ``Kernel`` back a real ONNX node and answer FINN's build-flow contract.

``kernel_op.py``     the base adapter (nodeattr↔Context bridge, the HWCustomOp getters).
``nodeattr_registry.py`` schema-axes → FINN nodeattr types (the R12 dissolution).

Concrete per-op wrappers live WITH their kernel definition (e.g.
``finn.kernels.compute.mvau.MvauKernelOp``), not here — this package is the shared
infrastructure only.

See ``kernel-design/finn-hw-backend-analysis/consumer-surface-model.md`` for the
consumer surface this satisfies, and the design doc
``kernel-design/kernel-final-design/kernelop-tensor-block-stream.md`` for the model.
"""

from finn.kernels.ir import KernelOp, PortSpec, TransformationResult
from .infer import InferKernels

__all__ = [
    "KernelOp",
    "PortSpec",
    "InferKernels",
    "TransformationResult",
]
