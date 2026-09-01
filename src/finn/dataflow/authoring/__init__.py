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
    )
    from finn.dataflow.authoring.input_supply import (
        InputSupplyAlternative,
        InputSupplyDeclaration,
    )
    from finn.dataflow.authoring.inventory import (
        DataflowOpAuthoring,
        DataflowDesignEntry,
        DataflowDesignInventory,
        DesignSelectionMetadata,
        declare_dataflow_design_inventory,
        declare_dataflow_op_authoring,
        selected_design_metadata,
    )
    from finn.dataflow.op import (
        AssignmentMapping,
        DataflowAssignmentCommit,
        DataflowBuildConfigView,
        DataflowOp,
        dataflow_problem_fingerprint,
    )
    from finn.dataflow.op_contracts import DataflowOpError, NodeAttrCodec, NodeAttributeType

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {}
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.authoring.design", name)
        for name in (
            "DataflowDesign",
            "DataflowDesignScope",
        )
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.authoring.input_supply", name)
        for name in ("InputSupplyAlternative", "InputSupplyDeclaration")
    }
)
_LAZY_EXPORTS.update(
    {
        name: ("finn.dataflow.authoring.inventory", name)
        for name in (
            "DataflowOpAuthoring",
            "DataflowDesignEntry",
            "DataflowDesignInventory",
            "DesignSelectionMetadata",
            "declare_dataflow_design_inventory",
            "declare_dataflow_op_authoring",
            "selected_design_metadata",
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
    "DataflowDesignEntry",
    "DataflowDesignInventory",
    "DataflowDesignScope",
    "DataflowOp",
    "DataflowOpAuthoring",
    "DataflowOpError",
    "FiniteSelectionResult",
    "GRAPH_OWNED",
    "InputSupplyAlternative",
    "InputSupplyDeclaration",
    "DesignSelectionMetadata",
    "NodeAttrCodec",
    "NodeAttributeType",
    "OpDesign",
    "ProblemProvenance",
    "Provenance",
    "Ref",
    "Scope",
    "SpecAuthoringError",
    "SpecAuthoringIssue",
    "assemble_specs",
    "dataflow_problem_fingerprint",
    "declare_dataflow_design_inventory",
    "declare_dataflow_op_authoring",
    "divisors_of",
    "domain",
    "enumerate_feasible_points",
    "finite",
    "gate_spec",
    "reject",
    "selected_design_metadata",
    "unresolved",
]
