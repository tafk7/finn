# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` rewrites the nodes a KernelOp binds and states the build
target (``resolve_target``: a part's and a shell's capabilities), which a shell
build reads back (``shell_target``);
``InferKernelTensors`` infers every tensor in graph order, the KernelOps
answering from their kernels; ``ExploreKernelChoices`` explores their open
choices through the DSE seam (``finn.kernels.explore``) by a list of strategies
(``strategy``: one from its spec) and saves them;
``kernel_choices_config`` exports the nodes' choices, sparse, for
``ApplyConfig``; ``PackagePartition`` packages a partition of KernelOps as the
IP the shells read (the stitched-IP contract), and ``ElaboratePartition``
compiles and elaborates its RTL in XSim, a check before a shell builds it.
"""

from finn.transformation.kernels.choose import (
    KERNEL_STRATEGIES,
    Explored,
    ExploreKernelChoices,
    explore_kernel_choices,
    strategy,
)
from finn.transformation.kernels.config import kernel_choices_config
from finn.transformation.kernels.convert import ToKernelOps, resolve_target, shell_target
from finn.transformation.kernels.infer import InferKernelTensors
from finn.transformation.kernels.package import ElaboratePartition, PackagePartition

__all__ = [
    "KERNEL_STRATEGIES",
    "ElaboratePartition",
    "ExploreKernelChoices",
    "Explored",
    "InferKernelTensors",
    "PackagePartition",
    "ToKernelOps",
    "explore_kernel_choices",
    "kernel_choices_config",
    "resolve_target",
    "shell_target",
    "strategy",
]
