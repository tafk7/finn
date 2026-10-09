# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps (``finn.custom_op.kernels``).

``ToKernelOps`` rewrites the nodes a KernelOp's pattern matches and the kernels
admit (the domain's KernelOps by their anchor, ``kernel_ops_by_anchor``),
inferring every tensor as it goes, and states the build target it is given
(``finn.platform.resolve_target``), and keeps an ``Outcome`` for every node it
visits, with the findings that say why a node stays on the host
(``kernel_ops_report``, ``kernel_ops_summary``); ``refuse_host_between`` refuses
host nodes between KernelOps (``between_kernel_ops``), which no partition can
leave out;
``InferKernelTensors`` infers every tensor in graph order, the KernelOps
answering from their kernels (``infer_node``, the step on one node, which
``ToKernelOps`` shares); ``cut.CutKernelPartition`` cuts the KernelOps once into
the partition, the parent graph's one ``StreamingDataflowPartition``
(``finn.custom_op.partition``); ``ExploreKernelChoices`` explores the open
choices of a partition's body (the KernelOps the cut put together) through the
DSE seam (``finn.kernels.explore``) by a list of strategies
(``strategy``: one from its spec) and saves them, and a completion policy
(``completion``: one by its name) completes what they leave open wherever a
partition is costed or built, never saved;
``shell_bottleneck`` reads the slowest members of a partition's shell root, ends
included, from its saved choices, completed; ``kernel_choices_config`` exports a
body's choices, sparse, by graph name (the nodes' and the tensors' channel choices),
which ``Pinned`` reads back; ``PackagePartition`` packages a
partition of KernelOps as the IP the shells read, and ``ElaboratePartition`` compiles
and elaborates its RTL in XSim, a check before a shell builds it;
``integration.integration`` exports what a shell's integration builds around the
partition, from its ends.
"""

from finn.transformation.kernels.choose import (
    KERNEL_COMPLETIONS,
    KERNEL_STRATEGIES,
    Explored,
    ExploreKernelChoices,
    completion,
    explore_kernel_choices,
    shell_bottleneck,
    strategy,
)
from finn.transformation.kernels.config import kernel_choices_config
from finn.transformation.kernels.convert import (
    Outcome,
    ToKernelOps,
    between_kernel_ops,
    kernel_ops_by_anchor,
    kernel_ops_report,
    kernel_ops_summary,
    refuse_host_between,
)
from finn.transformation.kernels.infer import InferKernelTensors, infer_node
from finn.transformation.kernels.package import ElaboratePartition, PackagePartition

__all__ = [
    "KERNEL_COMPLETIONS",
    "KERNEL_STRATEGIES",
    "ElaboratePartition",
    "ExploreKernelChoices",
    "Explored",
    "InferKernelTensors",
    "Outcome",
    "PackagePartition",
    "ToKernelOps",
    "between_kernel_ops",
    "completion",
    "explore_kernel_choices",
    "infer_node",
    "kernel_choices_config",
    "kernel_ops_by_anchor",
    "kernel_ops_report",
    "kernel_ops_summary",
    "refuse_host_between",
    "shell_bottleneck",
    "strategy",
]
