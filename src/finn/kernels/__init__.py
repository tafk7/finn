############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""Generic design-space model (resolve core).

`primitives/` holds proven domain value objects — ordered parameters, datatype
range-builders, template resolution, interface shape/width — reused by the engine.
`space/` holds the engine: Context, Axis, Derived, Predicate, and resolve.

This package IS the ``finn.kernels`` qonnx DOMAIN (handoff Seam C): the domain string
equals the module path (a sibling of ``finn.custom_op``, mirroring brainsmith's
``brainsmith.kernels``). The ``custom_op`` dict below is how qonnx's ``getCustomOp``
resolves a ``(domain="finn.kernels", op_type=...)`` node to its KernelOp class — the ONE
coupling the infer seam needs so the post-infer InferShapes/InferDataTypes passes can
instantiate kernel nodes. Keyed by op_type (decoupled from class name), one op per kernel;
NO per-language classes (backend is the ``implementation`` axis, DATA not identity).
"""

# Populated at bottom-of-module import time; keeps the heavy op imports off any early
# import path and avoids import-order surprises with the adapter/engine packages.
custom_op: dict = {}


def _register_kernel_ops() -> None:
    from finn.kernels.ops.mvau.op import MvauKernelOp
    from finn.kernels.ops.thresholding.op import ThresholdingKernelOp

    custom_op["MVAU"] = MvauKernelOp
    custom_op["Thresholding"] = ThresholdingKernelOp


_register_kernel_ops()
