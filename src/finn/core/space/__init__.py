# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed design-space authoring, immutable specialization and public services."""

from . import extensions, forcing, graph, inspection, selections
from ._configuration import BoundDecision, BoundValue, Space
from ._execution import (
    CleanupFailure,
    NativeCancellationDetails,
    NativeEvaluationError,
    cancellation_details,
)
from .compiler import Model, design_space
from .declarations import (
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    Derived,
    LocatedParam,
    Members,
    Param,
    Present,
    Users,
    ValueRef,
    View,
    ViewKey,
    constraint,
    derived,
    required,
    selected,
    view,
)
from .domains import Domain, Requirement, divisors_of, domain, finite, requires, requiring
from .edits import Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import (
    ConfigurationError,
    DefinitionError,
    EvaluationError,
    ReferenceUseError,
    RequestError,
    ValueUnavailableError,
)
from .expressions import Expr
from .extensions import composite
from .graph import Located
from .references import DecisionHandle, ValueHandle
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Finding,
    FindingKind,
    Inapplicable,
    QueryResult,
    ReadinessAssessment,
    Rejected,
    Unresolved,
    ViewAssessment,
    reject,
    require_value,
)
from .selections import Selection, SelectionEntry
from .semantics import ValueSemantics, default_semantics

__all__ = [
    # Families, node declarations and the compile step
    "Space",
    "design_space",
    "Model",
    "composite",
    "Param",
    "LocatedParam",
    "Const",
    "Decision",
    "required",
    "selected",
    "Derived",
    "Constraint",
    "ConstraintGroup",
    "View",
    # Graph primitives
    "Present",
    "Members",
    "Users",
    "Located",
    "derived",
    "constraint",
    "view",
    # Typed references and bound accessors
    "ValueRef",
    "ViewKey",
    "ValueHandle",
    "DecisionHandle",
    "BoundValue",
    "BoundDecision",
    "Expr",
    # Domains and value semantics
    "Domain",
    "domain",
    "finite",
    "divisors_of",
    "Requirement",
    "requires",
    "requiring",
    "ValueSemantics",
    "default_semantics",
    # Results and assessments
    "QueryResult",
    "Available",
    "Inapplicable",
    "Rejected",
    "Unresolved",
    "Finding",
    "FindingKind",
    "DecisionState",
    "ConstraintAssessment",
    "ReadinessAssessment",
    "ViewAssessment",
    "reject",
    "require_value",
    # Configuration revisions and selections
    "Change",
    "ChangeRequest",
    "ChangeOutcome",
    "ConfigurationResult",
    "Selection",
    "SelectionEntry",
    # Errors and native cleanup diagnostics
    "DefinitionError",
    "RequestError",
    "EvaluationError",
    "ReferenceUseError",
    "ConfigurationError",
    "ValueUnavailableError",
    "CleanupFailure",
    "NativeEvaluationError",
    "NativeCancellationDetails",
    "cancellation_details",
    # Public services
    "extensions",
    "forcing",
    "graph",
    "inspection",
    "selections",
]
