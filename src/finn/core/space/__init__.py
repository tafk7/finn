# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed design-space authoring, immutable specialization and public services."""

from . import codecs, extensions, graph, inspection, selections
from ._configuration import BoundDecision, BoundValue, BoundView, ChoiceView, Space
from ._execution import (
    CleanupFailure,
    NativeCancellationDetails,
    NativeEvaluationError,
    cancellation_details,
)
from .codecs import CodecBinding, JSONValue, SelectionSchema, ValueCodec, codec_for
from .compiler import SpaceModel, compile_space
from .declarations import (
    AcceptedViewRef,
    Carried,
    Const,
    Constraint,
    ConstraintGroup,
    Decision,
    DecisionRef,
    Derived,
    Ends,
    Fold,
    Interface,
    Net,
    Param,
    Port,
    Subspace,
    SubspaceChoice,
    ValueKey,
    ValueRef,
    View,
    ViewKey,
    constraint,
    derived,
    view,
)
from .domains import Domain, divisors_of, domain, finite
from .edits import Change, ChangeOutcome, ChangeRequest, ConfigurationResult
from .errors import (
    ConfigurationError,
    DefinitionError,
    EvaluationError,
    RequestError,
    ValueUnavailableError,
)
from .expressions import Expr
from .extensions import ScopeBuilder
from .graph import End, EndRef, Interpretation, Link, NetEntry, PortEntry, Topology
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
    # Definitions and composition
    "Space",
    "SpaceModel",
    "compile_space",
    "Param",
    "Const",
    "Decision",
    "Derived",
    "Constraint",
    "ConstraintGroup",
    "View",
    "Subspace",
    "SubspaceChoice",
    "ScopeBuilder",
    # Graph composition
    "Interface",
    "Port",
    "Net",
    "Link",
    "Carried",
    "Ends",
    "Fold",
    "Interpretation",
    "Topology",
    "NetEntry",
    "PortEntry",
    "End",
    "EndRef",
    "derived",
    "constraint",
    "view",
    # Typed references and bound accessors
    "ValueRef",
    "DecisionRef",
    "AcceptedViewRef",
    "ValueKey",
    "ViewKey",
    "ValueHandle",
    "DecisionHandle",
    "BoundValue",
    "BoundDecision",
    "BoundView",
    "ChoiceView",
    "Expr",
    # Domains and value semantics
    "Domain",
    "domain",
    "finite",
    "divisors_of",
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
    "CodecBinding",
    "JSONValue",
    "SelectionSchema",
    "ValueCodec",
    "codec_for",
    # Errors and native cleanup diagnostics
    "DefinitionError",
    "RequestError",
    "EvaluationError",
    "ConfigurationError",
    "ValueUnavailableError",
    "CleanupFailure",
    "NativeEvaluationError",
    "NativeCancellationDetails",
    "cancellation_details",
    # Public services
    "codecs",
    "extensions",
    "graph",
    "inspection",
    "selections",
]
