# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed design-space authoring, immutable specialization and public services."""

from . import inspection
from ._configuration import BoundDecision, BoundValue, Space
from ._failure import NativeEvaluationError
from ._nodes import space_formals
from ._scopes import model_modes
from ._value_types import value_type
from .collection import RESERVED_NAMES
from .compiler import Model, design_space
from .declarations import (
    MISSING,
    Constraint,
    Decision,
    Derived,
    Members,
    Param,
    Select,
    View,
    constraint,
    declared_annotation,
    derived,
    required,
    view,
)
from .domains import Domain, divisors_of, domain
from .edits import BrokenRule, Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import (
    ConfigurationError,
    DefinitionError,
    EvaluationError,
    ReferenceUseError,
    RequestError,
    ValueUnavailableError,
)
from .extensions import composite
from .graph import Located
from .occurrence import bind
from .references import DecisionHandle, ValueHandle
from .results import (
    Available,
    ConstraintAssessment,
    DecisionState,
    Inapplicable,
    QueryResult,
    ReadinessAssessment,
    Rejected,
    Unresolved,
    ViewAssessment,
    reject,
)

__all__ = [
    # Space classes, node declarations and the compile step
    "Space",
    "design_space",
    "bind",
    "Model",
    "composite",
    "Param",
    "Decision",
    "Select",
    "required",
    "Derived",
    "Constraint",
    "View",
    # A class's formals and names, as a composer declares Params for them
    "space_formals",
    "declared_annotation",
    "MISSING",
    "RESERVED_NAMES",
    # The test modes a model is made in
    "model_modes",
    # Graph primitives
    "Members",
    "Located",
    "derived",
    "constraint",
    "view",
    # Typed references and bound accessors
    "ValueHandle",
    "DecisionHandle",
    "BoundValue",
    "BoundDecision",
    # Domains and value semantics
    "Domain",
    "domain",
    "divisors_of",
    "value_type",
    # Results and assessments
    "QueryResult",
    "Available",
    "Inapplicable",
    "Rejected",
    "Unresolved",
    "DecisionState",
    "ConstraintAssessment",
    "ReadinessAssessment",
    "ViewAssessment",
    "reject",
    # Configuration revisions
    "Change",
    "ChangeRequest",
    "ChangeOutcome",
    "ConfigurationResult",
    "BrokenRule",
    # Errors and native cleanup diagnostics
    "DefinitionError",
    "RequestError",
    "EvaluationError",
    "ReferenceUseError",
    "ConfigurationError",
    "ValueUnavailableError",
    "NativeEvaluationError",
    # Public services
    "inspection",
]
