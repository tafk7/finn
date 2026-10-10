# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One ONNX entry spec per KernelOp (``finn.harness.reference.OpSpec``): its positive
graphs and its negative graphs with their codes. ``tests/kernel_ops/test_reference.py``
checks them offline, in the fast gate.

They live here, not beside the kernels' specs (``tests/kernels/specs``): they are ONNX
graphs, and the kernel tests read no graph (``tests/layering.py``).
"""

from __future__ import annotations

from finn.harness.reference import OpSpec
from kernel_ops.specs import matmul, thresholding, windowed_matmul

SPECS: tuple[OpSpec, ...] = (matmul.SPEC, thresholding.SPEC, windowed_matmul.SPEC)

__all__ = ["SPECS"]
