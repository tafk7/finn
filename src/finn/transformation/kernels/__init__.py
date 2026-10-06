# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` rewrites the nodes a KernelOp binds and states the build
target (``resolve_target``: a part's and a shell's capabilities);
``InferKernelTensors`` infers every tensor in graph order, the KernelOps
answering from their kernels; ``CommitKernelChoices`` commits their open
choices by a policy (``PlaceholderPolicy``, the DSE seam's placeholder);
``kernel_choices_config`` exports the nodes' choices, sparse, for
``ApplyConfig``; ``PackagePartition`` packages a partition of KernelOps as the
IP the shells read (the stitched-IP contract).
"""

from finn.transformation.kernels.choose import CommitKernelChoices, PlaceholderPolicy
from finn.transformation.kernels.config import kernel_choices_config
from finn.transformation.kernels.convert import ToKernelOps, resolve_target
from finn.transformation.kernels.infer import InferKernelTensors
from finn.transformation.kernels.package import PackagePartition

__all__ = [
    "CommitKernelChoices",
    "InferKernelTensors",
    "PackagePartition",
    "PlaceholderPolicy",
    "ToKernelOps",
    "kernel_choices_config",
    "resolve_target",
]
