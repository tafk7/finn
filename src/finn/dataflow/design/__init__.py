# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Supported evaluation API for FINN dataflow design spaces."""

from importlib import import_module
from typing import TYPE_CHECKING

from finn.dataflow._engine import (  # noqa: F401 - private compatibility for internal leaves
    ABSENT,
    AbsenceMode as AbsenceMode,
    Absent,
    Answer,
    CommitResult,
    Constraint as Constraint,
    ConstraintAssessment,
    ConstraintSet as ConstraintSet,
    Decided,
    Decision as Decision,
    DecisionDomain as DecisionDomain,
    DecisionState,
    DependencyKind as DependencyKind,
    DependencyRef as DependencyRef,
    DependencyView as DependencyView,
    DerivedProperty as DerivedProperty,
    DesignPoint,
    DesignSpace,
    DesignSpaceSpec as DesignSpaceSpec,
    Engine,
    EvaluationError,
    EvaluatorSpec as EvaluatorSpec,
    Finding,
    FindingKind,
    ItemOutcome,
    ProblemField as ProblemField,
    ProblemSchema as ProblemSchema,
    ProposalAdoptionMode,
    ProposalAdoptionResult,
    QualifiedPath,
    ReadinessAssessment,
    ReadinessProfile as ReadinessProfile,
    RequestError,
    Unresolved,
    ValidationError,
    ValueSemantics as ValueSemantics,
    as_object_semantics as as_object_semantics,
)
from finn.dataflow.design.region import (  # noqa: F401 - internal declaration semantics
    DATAFLOW_NETWORK_SEMANTICS as DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS as DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS as NETWORK_VALIDATION_REPORT_SEMANTICS,
    QONNX_DATATYPE_SEMANTICS as QONNX_DATATYPE_SEMANTICS,
    QONNX_DATATYPE_VALUE_SEMANTICS as QONNX_DATATYPE_VALUE_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS as REGION_VALIDATION_REPORT_SEMANTICS,
)
from finn.dataflow.network import DataflowNetwork as DataflowNetwork
from finn.dataflow.network_validation import (  # noqa: F401 - internal compatibility
    NetworkValidationIssue as NetworkValidationIssue,
    NetworkValidationReport as NetworkValidationReport,
    validate_network as validate_network,
)
from finn.dataflow.region import DataflowRegion as DataflowRegion
from finn.dataflow.region_validation import (  # noqa: F401 - internal compatibility
    RegionValidationIssue as RegionValidationIssue,
    RegionValidationReport as RegionValidationReport,
    validate_region as validate_region,
)

if TYPE_CHECKING:
    from finn.dataflow.resolution import DataflowOpResult, NetworkRef, ResolvedDataflowOp

_LAZY_EXPORTS = {
    name: ("finn.dataflow.resolution", name)
    for name in ("DataflowOpResult", "NetworkRef", "ResolvedDataflowOp")
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
    "Absent",
    "Answer",
    "CommitResult",
    "ConstraintAssessment",
    "DataflowOpResult",
    "Decided",
    "DecisionState",
    "DesignPoint",
    "DesignSpace",
    "Engine",
    "EvaluationError",
    "Finding",
    "FindingKind",
    "ItemOutcome",
    "NetworkRef",
    "ProposalAdoptionMode",
    "ProposalAdoptionResult",
    "QualifiedPath",
    "ReadinessAssessment",
    "RequestError",
    "ResolvedDataflowOp",
    "Unresolved",
    "ValidationError",
]
