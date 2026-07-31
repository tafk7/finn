############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""The qonnx-IR presence of a kernel: ``KernelOp(HWCustomOp)`` and its supporting
machinery.

``kernel_op.py``          the base op — the nodeattr↔Context bridge + HWCustomOp getters,
                          plus the ``TransformationResult`` infer contract.
``nodeattr_registry.py``  schema-axes → FINN nodeattr types (the R12 dissolution).
``routing.py``            taxonomy routing (``is_specialized``/``kernel_hw_language``) — a
                          FINN-integration classifier that stays IN the kernel package.

The concrete per-op wrappers (``MvauKernelOp`` …) live WITH their kernel definition in
``compute/``; this package is the shared IR infrastructure only.
"""

from .kernel_op import KernelOp, TransformationResult
from .routing import is_specialized, kernel_hw_language

__all__ = [
    "KernelOp",
    "TransformationResult",
    "is_specialized",
    "kernel_hw_language",
]
