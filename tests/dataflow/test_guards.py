# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One canonical value owner and the exact private isl import boundary.

Canonical logical dataflow imports the standard library, NumPy, QONNX's datatype
and range values, findings and itself. Only _isl.py imports the native backend.
The retired production namespace supplies no aliases or compatibility imports.
"""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "finn.dataflow"
HOME = (ROOT / "src" / "finn" / "dataflow", ROOT / "tests" / "dataflow")
ALLOWED = (
    PACKAGE,
    "numpy",
    "qonnx.core.datatype",
    "qonnx.core.ranges",
    "finn.core.reporting.findings",
    "finn.dataflow.datatypes",
)


def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def allowed_import(path: Path, name: str) -> bool:
    return (
        name.partition(".")[0] in sys.stdlib_module_names
        or any(within(name, allowed) for allowed in ALLOWED)
        or (path == HOME[0] / "_isl.py" and within(name, "islpy"))
    )


def imports(source: str) -> list[tuple[int, str]]:
    """Every absolute name ``source`` imports, wherever the import sits, as ``(line, name)``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return found


def test_no_source_or_test_imports_the_retired_production_namespace() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for root in (ROOT / "src", ROOT / "tests")
        for path in sorted(root.rglob("*.py"))
        if ".venv" not in path.parts
        for line, name in imports(path.read_text())
        if within(name, "finn.dataflow_next")
    ]
    assert found == []
    assert not (ROOT / "src" / "finn" / "dataflow_next" / "__init__.py").exists()


def test_the_revision_imports_only_its_approved_dependencies() -> None:
    sources = sorted(HOME[0].rglob("*.py"))
    assert sources
    found = [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for path in sources
        for line, name in imports(path.read_text())
        if not allowed_import(path, name)
    ]
    assert found == []


def test_private_isl_permission_does_not_broaden_layering() -> None:
    assert allowed_import(HOME[0] / "_isl.py", "islpy.Map")
    assert not allowed_import(HOME[0] / "logical.py", "islpy.Map")
    assert not allowed_import(HOME[0] / "_isl.py", "finn.core.space.Space")
    assert not allowed_import(HOME[0] / "_isl.py", "finn.kernels.Kernel")


def test_the_walker_finds_every_form_of_the_import() -> None:
    """A check that has never rejected anything is a check nobody has tested."""
    for line in (
        "import finn.dataflow",
        "import finn.dataflow.steps as steps",
        "from finn.dataflow.steps import steps",
        "from finn.dataflow import traversal",
        "def f():\n    from finn.dataflow.steps import steps",
    ):
        assert any(within(name, PACKAGE) for _, name in imports(line)), line
    for line in ("import finn.dataflow_next", "from finn.dataflow_next.steps import steps"):
        assert not any(within(name, PACKAGE) for _, name in imports(line)), line


def test_importing_the_revision_loads_no_engine_kernels_or_graph() -> None:
    script = "\n".join(
        (
            "import importlib, pkgutil, sys",
            "import finn.dataflow as package",
            "for info in pkgutil.walk_packages(package.__path__, 'finn.dataflow.'):",
            "    importlib.import_module(info.name)",
            "bad = sorted(",
            "    name for name in sys.modules",
            "    if name.startswith(('finn.core.', 'finn.kernels', 'finn.custom_op', 'onnx'))",
            "    and not name.startswith('finn.core.reporting')",
            "    or (",
            "        name.startswith('qonnx.')",
            "        and name not in ('qonnx.core', 'qonnx.core.datatype', 'qonnx.core.ranges')",
            "    )",
            ")",
            "raise SystemExit('loaded: ' + ', '.join(bad) if bad else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_root_re_exports_nothing_and_every_module_imports() -> None:
    root = import_module(PACKAGE)
    assert not hasattr(root, "__all__")
    assert not hasattr(root, "__getattr__")
    names = [info.name for info in pkgutil.walk_packages(root.__path__, f"{PACKAGE}.")]
    assert {f"{PACKAGE}.tensor", f"{PACKAGE}.traversal", f"{PACKAGE}.steps"} <= set(names)
    for name in names:
        import_module(name)
