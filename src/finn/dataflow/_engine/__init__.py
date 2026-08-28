# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Deliberate primary API for the design-space engine."""

from .declarations import (
    ABSENT,
    AbsenceMode,
    Constraint,
    ConstraintSet,
    Decision,
    DecisionDomain,
    DependencyKind,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    ProblemField,
    ProblemSchema,
    ReadinessProfile,
    as_object_semantics,
)
from .engine import Engine
from .errors import EvaluationError, RequestError, ValidationError
from .points import CommitResult, DesignPoint, ProposalAdoptionResult
from .primitives import QualifiedPath, ValueSemantics
from .results import (
    Absent,
    Answer,
    ConstraintAssessment,
    Decided,
    DecisionState,
    Finding,
    FindingKind,
    ItemOutcome,
    ProposalAdoptionMode,
    ReadinessAssessment,
    Unresolved,
)
from .validation import DesignSpace

__all__ = [
    "ABSENT",
    "AbsenceMode",
    "Absent",
    "Answer",
    "CommitResult",
    "Constraint",
    "ConstraintAssessment",
    "ConstraintSet",
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
    "ProblemField",
    "ProblemSchema",
    "ProposalAdoptionMode",
    "ProposalAdoptionResult",
    "QualifiedPath",
    "ReadinessAssessment",
    "ReadinessProfile",
    "RequestError",
    "Unresolved",
    "ValidationError",
    "ValueSemantics",
    "as_object_semantics",
]
