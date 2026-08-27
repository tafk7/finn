# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import finn.dataflow.design as design


def _run_import_check(source: str) -> None:
    subprocess.run([sys.executable, "-c", source], check=True)


def test_public_design_api_is_deliberate_and_pinned() -> None:
    assert set(design.__all__) == {
        "ABSENT",
        "DATAFLOW_NETWORK_SEMANTICS",
        "DATAFLOW_REGION_SEMANTICS",
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
        "Unresolved",
        "ValidationError",
        "ValueSemantics",
        "as_object_semantics",
        "validate_region",
        "validate_network",
    }


def test_model_import_does_not_load_engine_or_design() -> None:
    _run_import_check(
        "import sys; import finn.dataflow.region; "
        "assert 'finn.dataflow._engine' not in sys.modules; "
        "assert 'finn.dataflow.design' not in sys.modules"
    )


def test_private_engine_import_does_not_load_design() -> None:
    _run_import_check(
        "import sys; import finn.dataflow._engine; assert 'finn.dataflow.design' not in sys.modules"
    )


def test_design_import_loads_engine_and_region_by_design() -> None:
    _run_import_check(
        "import sys; import finn.dataflow.design; "
        "assert 'finn.dataflow._engine' in sys.modules; "
        "assert 'finn.dataflow.region' in sys.modules"
    )


def test_private_engine_has_no_finn_or_region_imports() -> None:
    package = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "_engine"
    forbidden: set[str] = set()
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.update(
                    alias.name
                    for alias in node.names
                    if alias.name == "finn" or alias.name.startswith("finn.")
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "finn" or node.module.startswith("finn."):
                    forbidden.add(node.module)
    assert forbidden == set()


def test_mvau_specification_does_not_import_the_private_engine() -> None:
    source_path = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "mvau_design.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("._engine" in module for module in imported_modules)
