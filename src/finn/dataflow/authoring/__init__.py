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
        DataflowDesignEntry,
        DataflowDesignInventory,
        declare_dataflow_design_inventory,
        declare_dataflow_op_authoring,
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
            "DataflowDesignEntry",
            "DataflowDesignInventory",
            "declare_dataflow_design_inventory",
            "declare_dataflow_op_authoring",
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
    """Load dependency-heavy authoring entry points on first use."""

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
    "DataflowOpError",
    "GRAPH_OWNED",
    "InputSupplyAlternative",
    "InputSupplyDeclaration",
    "NodeAttrCodec",
    "NodeAttributeType",
    "OpDesign",
    "ProblemProvenance",
    "Provenance",
    "Ref",
    "Scope",
    "dataflow_problem_fingerprint",
    "declare_dataflow_design_inventory",
    "declare_dataflow_op_authoring",
    "divisors_of",
    "domain",
    "finite",
    "reject",
    "unresolved",
]
