# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public FINN dataflow authoring surface."""

from finn.dataflow.kernel import (
    BindingDefinition,
    KernelDefinition,
    KernelInstance,
    KernelPlacement,
    RegionDeclaration,
    assemble_kernel_specs,
    build_kernel_semantic_declarations,
    gate_design_space_spec,
)
from finn.dataflow.kernels import (
    NO_KERNEL,
    SELECTED_KERNEL_SEMANTICS,
    Kernel,
    KernelDemand,
    KernelProvider,
    KernelSelection,
    KernelSelectionPaths,
    SelectedKernel,
    selected_kernel,
)
from finn.dataflow.op import (
    AssignmentMapping,
    DataflowAssignmentCommit,
    DataflowBuildConfigView,
    DataflowOp,
    DataflowOpError,
    NodeAttrCodec,
    NodeAttributeType,
    dataflow_problem_fingerprint,
)
from finn.dataflow.resolution import DataflowOpResult, NetworkRef, RegionRef, ResolvedDataflowOp
from finn.dataflow.selection import FiniteSelectionResult, enumerate_feasible_points
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    gate_spec,
)

__all__ = [
    "AssignmentMapping",
    "BindingDefinition",
    "DataflowAssignmentCommit",
    "DataflowBuildConfigView",
    "DataflowOp",
    "DataflowOpError",
    "DataflowOpResult",
    "FiniteSelectionResult",
    "Kernel",
    "KernelDefinition",
    "KernelDemand",
    "KernelInstance",
    "KernelPlacement",
    "KernelProvider",
    "KernelSelection",
    "KernelSelectionPaths",
    "NO_KERNEL",
    "NetworkRef",
    "NodeAttrCodec",
    "NodeAttributeType",
    "RegionDeclaration",
    "RegionRef",
    "ResolvedDataflowOp",
    "SELECTED_KERNEL_SEMANTICS",
    "SelectedKernel",
    "SpecAuthoringError",
    "SpecAuthoringIssue",
    "assemble_kernel_specs",
    "assemble_specs",
    "build_kernel_semantic_declarations",
    "dataflow_problem_fingerprint",
    "enumerate_feasible_points",
    "gate_design_space_spec",
    "gate_spec",
    "selected_kernel",
]
