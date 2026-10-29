# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Graph transformations of the KernelOps: conversion (``convert``), inference
(``infer``), the cut (``cut``), exploration of the choices (``choose``, ``config``), and
packaging and integration (``package``, ``hls``, ``integration``).
"""

from finn.transformation.kernels.choose import (
    Explored,
    exploration_summary,
    explore_kernel_choices,
    shell_bottleneck,
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
    "ElaboratePartition",
    "Explored",
    "InferKernelTensors",
    "Outcome",
    "PackagePartition",
    "ToKernelOps",
    "between_kernel_ops",
    "explore_kernel_choices",
    "exploration_summary",
    "infer_node",
    "kernel_choices_config",
    "kernel_ops_by_anchor",
    "kernel_ops_report",
    "kernel_ops_summary",
    "refuse_host_between",
    "shell_bottleneck",
]
