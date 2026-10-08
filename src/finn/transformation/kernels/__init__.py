# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` rewrites the nodes a KernelOp binds and states the build
target it is given (``finn.platform.resolve_target``);
``InferKernelTensors`` infers every tensor in graph order, the KernelOps
answering from their kernels; ``ExploreKernelChoices`` explores their open
choices through the DSE seam (``finn.kernels.explore``) by a list of strategies
(``strategy``: one from its spec) and saves them, and a completion policy
(``completion``: one by its name) completes what they leave open wherever a
partition is costed or built, never saved;
``partition_bottleneck`` reads a partition's slowest members from its saved
choices, completed; ``kernel_choices_config`` exports the nodes' choices, sparse, for
``ApplyConfig``; ``PackagePartition`` packages a partition of KernelOps as the
IP the shells read, and ``ElaboratePartition`` compiles and elaborates its RTL
in XSim, a check before a shell builds it; ``integration.integration`` exports
what a shell's integration builds around the partition, from its ends.
"""

from finn.transformation.kernels.choose import (
    KERNEL_COMPLETIONS,
    KERNEL_STRATEGIES,
    Explored,
    ExploreKernelChoices,
    completion,
    explore_kernel_choices,
    partition_bottleneck,
    strategy,
)
from finn.transformation.kernels.config import kernel_choices_config
from finn.transformation.kernels.convert import ToKernelOps
from finn.transformation.kernels.infer import InferKernelTensors
from finn.transformation.kernels.package import ElaboratePartition, PackagePartition

__all__ = [
    "KERNEL_COMPLETIONS",
    "KERNEL_STRATEGIES",
    "ElaboratePartition",
    "ExploreKernelChoices",
    "Explored",
    "InferKernelTensors",
    "PackagePartition",
    "ToKernelOps",
    "completion",
    "explore_kernel_choices",
    "kernel_choices_config",
    "partition_bottleneck",
    "strategy",
]
