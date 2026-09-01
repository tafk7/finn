# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public FINN dataflow authoring surface."""

from importlib import import_module
from typing import TYPE_CHECKING

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
from finn.dataflow.resolution import DataflowOpResult, NetworkRef, RegionRef, ResolvedDataflowOp
from finn.dataflow.selection import FiniteSelectionResult, enumerate_feasible_points
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    gate_spec,
)

if TYPE_CHECKING:
    from finn.dataflow.authoring.design import (
        DataflowDesign,
        DataflowDesignScope,
        InputSupplyAlternative,
        InputSupplyDeclaration,
        declare_dataflow_design_inventory,
    )
    from finn.dataflow.authoring.kernel_design import (
        FEASIBILITY,
        SOURCE_ADMISSION,
        KernelDesign,
        declare_kernel,
        declare_kernel_design,
        kernel_namespace,
    )
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
        bind_kernel,
        selected_kernel,
    )
    from finn.dataflow.op import (
        AssignmentMapping,
        DataflowAssignmentCommit,
        DataflowBuildConfigView,
        DataflowOp,
        dataflow_problem_fingerprint,
    )
    from finn.dataflow.op_contracts import DataflowOpError, NodeAttrCodec, NodeAttributeType

_LAZY_EXPORTS = {
    name: ("finn.dataflow.authoring.kernel_design", name)
    for name in (
        "FEASIBILITY",
        "SOURCE_ADMISSION",
        "KernelDesign",
        "declare_kernel",
        "declare_kernel_design",
        "kernel_namespace",
    )
}
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.authoring.design", name)
        for name in (
            "DataflowDesign",
            "DataflowDesignScope",
            "InputSupplyAlternative",
            "InputSupplyDeclaration",
            "declare_dataflow_design_inventory",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.kernels", name)
        for name in (
            "NO_KERNEL",
            "SELECTED_KERNEL_SEMANTICS",
            "Kernel",
            "KernelDemand",
            "KernelExport",
            "KernelProvider",
            "KernelSelection",
            "KernelSelectionPaths",
            "SelectedKernel",
            "admissible_kernels",
            "bind_kernel",
            "selected_kernel",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.op", name)
        for name in (
            "AssignmentMapping",
            "DataflowAssignmentCommit",
            "DataflowBuildConfigView",
            "DataflowOp",
            "dataflow_problem_fingerprint",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.op_contracts", name)
        for name in ("DataflowOpError", "NodeAttrCodec", "NodeAttributeType")
    }
)


def __getattr__(name: str) -> object:
    """Load transitional authoring exports only when an old caller requests one."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AssignmentMapping",
    "AuthoringError",
    "BUILD_OWNED",
    "ConstraintRef",
    "DataflowAssignmentCommit",
    "DataflowBuildConfigView",
    "DataflowDesign",
    "DataflowDesignScope",
    "DataflowOp",
    "DataflowOpError",
    "DataflowOpResult",
    "FEASIBILITY",
    "FiniteSelectionResult",
    "GRAPH_OWNED",
    "Kernel",
    "KernelDemand",
    "KernelDesign",
    "KernelExport",
    "KernelProvider",
    "KernelSelection",
    "KernelSelectionPaths",
    "InputSupplyAlternative",
    "InputSupplyDeclaration",
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
    "SOURCE_ADMISSION",
    "Scope",
    "SelectedKernel",
    "SpecAuthoringError",
    "SpecAuthoringIssue",
    "admissible_kernels",
    "assemble_specs",
    "bind_kernel",
    "build_kernel_semantic_declarations",
    "dataflow_problem_fingerprint",
    "declare_kernel",
    "declare_kernel_design",
    "declare_dataflow_design_inventory",
    "divisors_of",
    "domain",
    "enumerate_feasible_points",
    "finite",
    "gate_spec",
    "kernel_namespace",
    "reject",
    "selected_kernel",
    "unresolved",
]
