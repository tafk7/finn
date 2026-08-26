# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import inspect

from finn.dataflow import _engine as design_space
from finn.dataflow._engine import Decision, DesignSpace, Engine


def test_package_root_is_a_deliberate_primary_api() -> None:
    expected = {
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
    }
    assert set(design_space.__all__) == expected


def test_advanced_conformance_and_graph_types_live_in_named_submodules() -> None:
    assert "MonotonicityHarness" not in design_space.__all__
    assert "GraphAnalysis" not in design_space.__all__
    assert "DesignSpaceIdentity" not in design_space.__all__
    assert "Classification" not in design_space.__all__


def test_engine_exposes_unambiguous_primary_operations() -> None:
    public_methods = {
        name
        for name, member in inspect.getmembers(Engine, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == {
        "adopt_profile_proposals",
        "adopt_proposals",
        "check_readiness",
        "commit_assignments",
        "decision_state",
        "enumerate_candidates",
        "evaluate_constraint_set",
        "evaluate_constraints",
        "inspect",
        "query_property",
        "start",
        "try_commit_assignments",
        "try_start",
        "try_validate",
        "validate",
    }


def test_runtime_space_retains_one_declaration_view_and_no_validation_graph() -> None:
    decision_fields = {field.name for field in dataclasses.fields(Decision)}
    space_fields = {field.name for field in dataclasses.fields(DesignSpace)}
    assert decision_fields == {"path", "value_semantics", "domain", "applies_if", "proposal"}
    assert "specification" not in space_fields
    assert "compiled_facts" not in space_fields
    assert "identity" not in space_fields
    assert "_plan" in space_fields
