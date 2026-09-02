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
        "Decided",
        "DecisionState",
        "DesignPoint",
        "DesignSpace",
        "Engine",
        "EvaluationError",
        "Finding",
        "FindingKind",
        "ItemOutcome",
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

    assert design.ResolvedDataflowOp is resolution.ResolvedDataflowOp
    assert not hasattr(resolution, "DataflowOpResult")
    assert not hasattr(resolution, "NetworkRef")
    assert not hasattr(resolution, "RegionRef")


def test_authoring_facade_exposes_design_declaration_entry_points() -> None:
    authoring = import_module("finn.dataflow.authoring")
    design_implementation = import_module("finn.dataflow.authoring.design")

    assert authoring.DataflowDesign is design_implementation.DataflowDesign
    assert authoring.PhysicalComposition is design_implementation.PhysicalComposition
    for private_name in (
        "DataflowDesignEntry",
        "DataflowDesignInventory",
        "DataflowDesignScope",
        "DataflowOpAuthoring",
        "InputSupplyAlternative",
        "InputSupplyDeclaration",
        "OpDesign",
        "Ref",
        "Scope",
        "declare_dataflow_design_inventory",
        "declare_dataflow_op_authoring",
    ):
        assert not hasattr(authoring, private_name)


def test_public_authoring_api_is_declaration_only_and_pinned() -> None:
    assert_public_surface(
        "finn.dataflow.authoring",
        (
            "AssignmentMapping",
            "AuthoringError",
            "Attribute",
            "BuildFact",
            "BuildFlag",
            "BuildString",
            "Choice",
            "ClosedDesigns",
            "Condition",
            "Connection",
            "Constant",
            "Covers",
            "DataflowAssignmentCommit",
            "DataflowBuildConfigView",
            "DataflowDesign",
            "DataflowOp",
            "DataflowOpError",
            "DatatypeAttribute",
            "DependentDomain",
            "EdgeClaim",
            "InputTensor",
            "InitializerAnalysis",
            "Imported",
            "KernelInput",
            "NodeAttrCodec",
            "NodeAttributeType",
            "Network",
            "NoInitializer",
            "OptionalInitializer",
            "OutputTensor",
            "Parameter",
            "PhysicalComposition",
            "Persist",
            "PortableCodec",
            "Problem",
            "Provenance",
            "Region",
            "RegionClaim",
            "Readiness",
            "RequiredInitializer",
            "SourceScope",
            "SourceInput",
            "Sources",
            "TargetClockPeriod",
            "TargetFpgaPart",
            "TensorShape",
            "UsesDesign",
            "UsesInputSupply",
            "Kernels",
            "dataflow_problem_fingerprint",
            "class_divisors_of",
            "constraint",
            "derived",
            "finite_values",
            "not_",
            "present",
            "reject",
            "unresolved",
        ),
    )


def test_public_kernel_api_is_declaration_only_and_pinned() -> None:
    assert_public_surface(
        "finn.dataflow.kernels",
        ("Kernel", "PhysicalComponent", "scalar_parameters"),
    )
    kernels = import_module("finn.dataflow.kernels")
    authoring = import_module("finn.dataflow.authoring")
    assert "define" not in kernels.Kernel.__dict__
    assert "define_design" not in kernels.Kernel.__dict__
    assert "define" not in authoring.DataflowDesign.__dict__


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
    supply = import_module("finn.dataflow.authoring.input_supply")
    inventory = import_module("finn.dataflow.authoring.inventory")
    assert authoring.DataflowDesign.__module__ == "finn.dataflow.authoring.design"
    assert supply.InputSupplyDeclaration.__module__ == "finn.dataflow.authoring.input_supply"
    assert inventory.DataflowDesignInventory.__module__ == "finn.dataflow.authoring.inventory"
    assert not hasattr(authoring, "InputSupplyDeclaration")
    assert not hasattr(authoring, "DataflowDesignInventory")


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
    assert "Ref[DataflowNetwork]" in completed.stdout


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
