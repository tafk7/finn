# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.kernels``: the KernelOps.

A KernelOp is an ONNX node tied to a design space of kernels (``base.KernelOp``). qonnx
loads a domain by importing it and takes its op classes from ``__all__``, so this module
exports only op classes. ``opset_version`` is the domain version they are written for.
"""

from finn.custom_op.kernels.conv import Conv
from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding

opset_version = 1

__all__ = ["Conv", "MatMul", "Thresholding"]
