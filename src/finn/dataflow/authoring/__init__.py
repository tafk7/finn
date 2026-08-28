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

__all__ = [
    "BindingDefinition",
    "AssignmentMapping",
    "DataflowAssignmentCommit",
    "DataflowBuildConfigView",
    "DataflowOp",
    "DataflowOpError",
    "DataflowOpResult",
    "FiniteSelectionResult",
    "KernelDefinition",
    "KernelInstance",
    "KernelPlacement",
    "NetworkRef",
    "NodeAttrCodec",
    "NodeAttributeType",
    "RegionDeclaration",
    "RegionRef",
    "ResolvedDataflowOp",
    "assemble_kernel_specs",
    "build_kernel_semantic_declarations",
    "dataflow_problem_fingerprint",
    "enumerate_feasible_points",
    "gate_design_space_spec",
]
