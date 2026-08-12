############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
"""Generic design-space model (resolve core).

``engine/`` holds the pure resolve core: Context, Axis, Derived, Predicate, and resolve
(plus the vendored value objects — ordered parameters, datatype range-builders).
``model/`` holds the op-model framework (DataflowOp/Backend/Interface, tiling, ports, the
parameter-source contract); ``emit/``, ``ir/``, ``compute/`` and ``dataflow/`` build on it.

This package IS the ``finn.kernels`` qonnx DOMAIN (handoff Seam C): the domain string
equals the module path (a sibling of ``finn.custom_op``, mirroring brainsmith's
``brainsmith.kernels``). The ``custom_op`` dict below is how qonnx's ``getCustomOp``
resolves a ``(domain="finn.kernels", op_type=...)`` node to its DataflowOp class — the ONE
coupling the infer seam needs so the post-infer InferShapes/InferDataTypes passes can
instantiate kernel nodes. Keyed by op_type (decoupled from class name), one op per kernel;
NO per-language classes (backend is the ``backend`` axis, DATA not identity).
"""

# Populated at bottom-of-module import time; keeps the heavy op imports off any early
# import path and avoids import-order surprises with the adapter/engine packages.
custom_op: dict = {}


def _register_kernel_ops() -> None:
    from finn.kernels.compute.mvau.op import MvauDataflowOp
    from finn.kernels.compute.thresholding.op import ThresholdingDataflowOp

    custom_op["MVAU"] = MvauDataflowOp
    custom_op["Thresholding"] = ThresholdingDataflowOp


_register_kernel_ops()
