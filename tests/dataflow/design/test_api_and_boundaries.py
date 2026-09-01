# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from importlib import import_module
import os
import shutil
import subprocess
import sys
from pathlib import Path

import finn.dataflow.design as design
import pytest
from finn.dataflow.testing import (
    assert_fresh_import_avoids,
    assert_no_raw_declaration_construction,
    assert_public_surface,
)


def _run_import_check(source: str) -> None:
    subprocess.run([sys.executable, "-c", source], check=True)


def test_public_design_api_is_deliberate_and_pinned() -> None:
    assert set(design.__all__) == {
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


def test_resolved_operation_values_have_an_evaluation_time_canonical_import() -> None:
    resolution = import_module("finn.dataflow.resolution")

    assert design.DataflowOpResult is resolution.NetworkRef
    assert design.NetworkRef is resolution.NetworkRef
    assert design.ResolvedDataflowOp is resolution.ResolvedDataflowOp
    assert not hasattr(resolution, "RegionRef")


def test_authoring_facade_exposes_design_declaration_entry_points() -> None:
    authoring = import_module("finn.dataflow.authoring")
    design_implementation = import_module("finn.dataflow.authoring.design")
    supply_implementation = import_module("finn.dataflow.authoring.input_supply")
    inventory_implementation = import_module("finn.dataflow.authoring.inventory")

    assert authoring.DataflowDesign is design_implementation.DataflowDesign
    assert authoring.DataflowDesignScope is design_implementation.DataflowDesignScope
    assert authoring.InputSupplyAlternative is supply_implementation.InputSupplyAlternative
    assert authoring.InputSupplyDeclaration is supply_implementation.InputSupplyDeclaration
    assert (
        authoring.declare_dataflow_design_inventory
        is inventory_implementation.declare_dataflow_design_inventory
    )


def test_public_authoring_api_is_declaration_only_and_pinned() -> None:
    assert_public_surface(
        "finn.dataflow.authoring",
        (
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
        ),
    )


def test_operation_namespaces_have_one_narrow_public_entry() -> None:
    assert_public_surface("finn.dataflow.ops", ())
    assert_public_surface(
        "finn.dataflow.ops.mvau",
        ("MVAUDataflowBuildContext", "MvauDataflowOp"),
    )


def test_final_facades_do_not_eagerly_load_operations() -> None:
    for module in (
        "finn.dataflow.design",
        "finn.dataflow.authoring",
        "finn.dataflow.artifacts",
        "finn.dataflow.kernels",
        "finn.dataflow.testing",
        "finn.dataflow.ops",
        "finn.dataflow.ops.mvau",
    ):
        assert_fresh_import_avoids(
            module,
            (
                "finn.dataflow.ops.mvau.op",
                "finn.dataflow.ops.mvau.inventory",
            )
            if module != "finn.dataflow.ops.mvau.op"
            else (),
        )


def test_runtime_modules_do_not_construct_domain_declarations() -> None:
    root = Path(__file__).parents[3]
    assert_no_raw_declaration_construction(
        (
            root / "src" / "finn" / "dataflow" / "op.py",
            root / "src" / "finn" / "dataflow" / "resolution.py",
            root / "src" / "finn" / "dataflow" / "selection.py",
            root / "src" / "finn" / "transformation" / "fpgadataflow" / "select_dataflow_design.py",
        )
    )


def test_authoring_implementation_responsibilities_are_physically_split() -> None:
    authoring = import_module("finn.dataflow.authoring")
    assert authoring.DataflowDesign.__module__ == "finn.dataflow.authoring.design"
    assert authoring.InputSupplyDeclaration.__module__ == "finn.dataflow.authoring.input_supply"
    assert authoring.DataflowDesignInventory.__module__ == "finn.dataflow.authoring.inventory"


def test_region_result_is_rejected_by_static_operation_authoring_type() -> None:
    mypy = shutil.which("mypy")
    if mypy is None:
        pytest.skip("mypy is not installed in this test environment")
    root = Path(__file__).parents[3]
    fixture = root / "tests" / "dataflow" / "typing" / "invalid_region_result.py"
    environment = dict(os.environ)
    environment["MYPYPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            mypy,
            "--no-incremental",
            "--strict",
            "--explicit-package-bases",
            str(fixture),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert 'Argument "result"' in completed.stdout
    assert "Ref[DataflowRegion]" in completed.stdout
    assert "Ref[NetworkRef]" in completed.stdout


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
    source_path = (
        Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "ops" / "mvau" / "__init__.py"
    )
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("._engine" in module for module in imported_modules)
