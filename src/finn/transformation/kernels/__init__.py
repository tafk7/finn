# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` rewrites the nodes a KernelOp binds and states the build
target; ``InferKernelTensors`` infers every tensor in graph order, the KernelOps
answering from their node roots; ``kernel_choices_config`` exports the nodes'
choices, sparse, for ``ApplyConfig``.
"""

from finn.transformation.kernels.config import kernel_choices_config
from finn.transformation.kernels.convert import ToKernelOps
from finn.transformation.kernels.infer import InferKernelTensors

__all__ = ["InferKernelTensors", "ToKernelOps", "kernel_choices_config"]
