# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Supported evaluation API for FINN dataflow design spaces."""

from importlib import import_module
from typing import TYPE_CHECKING

from finn.dataflow._engine import (
    ABSENT,
    AbsenceMode,
    Absent,
    Answer,
    CommitResult,
    Constraint,
    ConstraintAssessment,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DecisionState,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    DesignSpace,
    DesignSpaceSpec,
    Engine,
    EvaluationError,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ItemOutcome,
    ProblemField,
    ProblemSchema,
    ProposalAdoptionMode,
    ProposalAdoptionResult,
    QualifiedPath,
    ReadinessAssessment,
    ReadinessProfile,
    RequestError,
    Unresolved,
    ValidationError,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.design.region import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    QONNX_DATATYPE_SEMANTICS,
    QONNX_DATATYPE_VALUE_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.network_validation import (
    NetworkValidationIssue,
    NetworkValidationReport,
    validate_network,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import (
    RegionValidationIssue,
    RegionValidationReport,
    validate_region,
)

if TYPE_CHECKING:
    from finn.dataflow.resolution import NetworkRef, ResolvedDataflowOp

_LAZY_EXPORTS = {
    name: ("finn.dataflow.resolution", name) for name in ("NetworkRef", "ResolvedDataflowOp")
}


def __getattr__(name: str) -> object:
    """Load resolved-operation values without creating an import cycle."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ABSENT",
    "DATAFLOW_REGION_SEMANTICS",
    "QONNX_DATATYPE_SEMANTICS",
    "QONNX_DATATYPE_VALUE_SEMANTICS",
    "DATAFLOW_NETWORK_SEMANTICS",
    "NETWORK_VALIDATION_REPORT_SEMANTICS",
    "REGION_VALIDATION_REPORT_SEMANTICS",
    "AbsenceMode",
    "Absent",
    "Answer",
    "CommitResult",
    "Constraint",
    "ConstraintAssessment",
    "ConstraintSet",
    "DataflowRegion",
    "DataflowNetwork",
    "Decided",
    "Decision",
    "DecisionDomain",
    "DecisionState",
    "DependencyKind",
    "DependencyRef",
    "DependencyView",
    "DerivedProperty",
    "DesignPoint",
    "DesignSpace",
    "DesignSpaceSpec",
    "Engine",
    "EvaluationError",
    "EvaluatorSpec",
    "Finding",
    "FindingKind",
    "ItemOutcome",
    "NetworkValidationIssue",
    "NetworkValidationReport",
    "NetworkRef",
    "ProblemField",
    "ProblemSchema",
    "ProposalAdoptionMode",
    "ProposalAdoptionResult",
    "QualifiedPath",
    "ReadinessAssessment",
    "ReadinessProfile",
    "RegionValidationIssue",
    "RegionValidationReport",
    "RequestError",
    "ResolvedDataflowOp",
    "Unresolved",
    "ValidationError",
    "ValueSemantics",
    "as_object_semantics",
    "validate_region",
    "validate_network",
]
