# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX domain ``finn.custom_op.kernels``: KernelOps, each binding one kernel point.

A KernelOp is a qonnx ``CustomOp`` that reads its facts from the attached model,
binds a node root (``roots``: its kernel placed on boundary streams) through a
process-wide cache (``cache``), replays the sparse choices its node holds, and
answers the compiler's queries from the configured point (``base``). qonnx
resolves this domain by importing it and reads its op classes from ``__all__``,
so this module exports op classes only; their mechanics live in its submodules,
the graph transformations in ``finn.transformation.kernels``.
"""

__all__: list[str] = []
