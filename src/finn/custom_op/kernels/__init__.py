# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.kernels``: KernelOps, each binding one kernel point.

A KernelOp is a qonnx ``CustomOp`` that reads its facts from the attached model,
binds a node root (``roots``: generated from the placement the op states, its
kernel on boundary channels) through a process-wide cache (``cache``), replays
the sparse choices its node holds (its kernel's) and those of the parameter channels
it owns, which their initializers' tensors state, and answers the compiler's queries
from the configured point (``base``); a shell root (``shell``) places the same
placement on channels its nodes share, each channel's choices stated on its tensor
(``base.CHANNEL``). qonnx resolves this
domain by importing it and reads its op classes from ``__all__``, so this module
exports op classes only; their mechanics live in its submodules, the graph
transformations in ``finn.transformation.kernels``.

Each op class states its ``op_type`` and ``op_version`` (its kernel's version)
in its own body, and is named by its op type at version 1. A model imports the
domain at ``opset_version``, the version its op classes are written for.
"""

from finn.custom_op.kernels.matmul import MatMul
from finn.custom_op.kernels.thresholding import Thresholding

opset_version = 1

__all__ = ["MatMul", "Thresholding"]
