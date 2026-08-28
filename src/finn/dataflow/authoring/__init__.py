# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public FINN dataflow authoring surface."""

from finn.dataflow.authoring.op_design import (
    BUILD_OWNED,
    GRAPH_OWNED,
    OpDesign,
    ProblemProvenance,
    Provenance,
)
from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Ref,
    Scope,
    divisors_of,
    domain,
    finite,
    reject,
    unresolved,
)
from finn.dataflow.kernel import RegionDeclaration, build_kernel_semantic_declarations
from finn.dataflow.kernels import (
    NO_KERNEL,
    SELECTED_KERNEL_SEMANTICS,
    Kernel,
    KernelDemand,
    KernelExport,
    KernelProvider,
    KernelSelection,
    KernelSelectionPaths,
    SelectedKernel,
    admissible_kernels,
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
    "AuthoringError",
    "BUILD_OWNED",
    "ConstraintRef",
    "DataflowAssignmentCommit",
    "DataflowBuildConfigView",
    "DataflowOp",
    "DataflowOpError",
    "DataflowOpResult",
    "FiniteSelectionResult",
    "GRAPH_OWNED",
    "Kernel",
    "KernelDemand",
    "KernelExport",
    "KernelProvider",
    "KernelSelection",
    "KernelSelectionPaths",
    "NO_KERNEL",
    "NetworkRef",
    "NodeAttrCodec",
    "NodeAttributeType",
    "OpDesign",
    "ProblemProvenance",
    "Provenance",
    "Ref",
    "RegionDeclaration",
    "RegionRef",
    "ResolvedDataflowOp",
    "SELECTED_KERNEL_SEMANTICS",
    "Scope",
    "SelectedKernel",
    "SpecAuthoringError",
    "SpecAuthoringIssue",
    "admissible_kernels",
    "assemble_specs",
    "build_kernel_semantic_declarations",
    "dataflow_problem_fingerprint",
    "divisors_of",
    "domain",
    "enumerate_feasible_points",
    "finite",
    "gate_spec",
    "reject",
    "selected_kernel",
    "unresolved",
]
