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
from finn.dataflow.authoring.persistence import Persist, PortableCodec
from finn.dataflow.authoring.projection import (
    Attribute,
    BuildFact,
    BuildFlag,
    BuildString,
    DatatypeAttribute,
    InitializerAnalysis,
    InputTensor,
    NoInitializer,
    OptionalInitializer,
    OutputTensor,
    RequiredInitializer,
    SourceScope,
    TargetClockPeriod,
    TargetFpgaPart,
    TensorShape,
)
from finn.dataflow.authoring.declarations import (
    Choice,
    Condition,
    DependentDomain,
    Imported,
    Problem,
    Readiness,
    constraint,
    derived,
    divisors_of as class_divisors_of,
    finite_values,
    not_,
    present,
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
    from finn.dataflow.authoring.compiler import ClosedDesigns, UsesDesign
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
    {name: ("finn.dataflow.authoring.compiler", name) for name in ("ClosedDesigns", "UsesDesign")}
)
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
    "Attribute",
    "BUILD_OWNED",
    "BuildFact",
    "BuildFlag",
    "BuildString",
    "Choice",
    "ClosedDesigns",
    "Condition",
    "ConstraintRef",
    "DataflowAssignmentCommit",
    "DataflowBuildConfigView",
    "DataflowDesign",
    "DataflowDesignEntry",
    "DataflowDesignInventory",
    "DataflowDesignScope",
    "DataflowOp",
    "DataflowOpError",
    "DatatypeAttribute",
    "DependentDomain",
    "GRAPH_OWNED",
    "InputSupplyAlternative",
    "InputSupplyDeclaration",
    "InputTensor",
    "InitializerAnalysis",
    "Imported",
    "NodeAttrCodec",
    "NodeAttributeType",
    "NoInitializer",
    "OpDesign",
    "OptionalInitializer",
    "OutputTensor",
    "Persist",
    "PortableCodec",
    "ProblemProvenance",
    "Problem",
    "Provenance",
    "Ref",
    "Readiness",
    "RequiredInitializer",
    "Scope",
    "SourceScope",
    "TargetClockPeriod",
    "TargetFpgaPart",
    "TensorShape",
    "UsesDesign",
    "dataflow_problem_fingerprint",
    "class_divisors_of",
    "constraint",
    "declare_dataflow_design_inventory",
    "declare_dataflow_op_authoring",
    "divisors_of",
    "derived",
    "domain",
    "finite",
    "finite_values",
    "not_",
    "present",
    "reject",
    "unresolved",
]
