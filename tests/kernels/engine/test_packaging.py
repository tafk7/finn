# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[3] / "src" / "finn" / "kernels" / "_engine"


def test_core_package_imports_only_the_standard_library_and_itself() -> None:
    assert PACKAGE.is_dir()
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
    for source_path in PACKAGE.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 1, source_path
    assert external <= allowed


def test_no_production_module_mentions_removed_defensive_infrastructure() -> None:
    assert PACKAGE.is_dir()
    source = "\n".join(path.read_text() for path in PACKAGE.glob("*.py"))
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
    assert (PACKAGE / "points.py").is_file()
    assert not (PACKAGE / "model.py").exists()
    assert not (PACKAGE / "outcomes.py").exists()
