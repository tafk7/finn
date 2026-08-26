# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path


def test_core_package_imports_only_the_standard_library_and_itself() -> None:
    package = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "_engine"
    allowed = {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "re",
        "types",
        "typing",
        "weakref",
    }
    external: set[str] = set()
    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])
    assert external <= allowed


def test_no_production_module_mentions_removed_defensive_infrastructure() -> None:
    package = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "_engine"
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    for removed in (
        "AdapterObligations",
        "ConstructionToken",
        "DesignSpaceIdentity",
        "ProblemIdentity",
        "IdentityAllocator",
        "OptionalService",
        "BlockedAnalysis",
        "Defective",
        "RequestRejected",
        "Classification",
    ):
        assert removed not in source


def test_point_ownership_is_not_split_across_overlapping_modules() -> None:
    package = Path(__file__).parents[3] / "src" / "finn" / "dataflow" / "_engine"
    assert (package / "points.py").is_file()
    assert not (package / "model.py").exists()
    assert not (package / "outcomes.py").exists()
