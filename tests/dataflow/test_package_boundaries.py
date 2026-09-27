# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The dependency direction between the Space engine, dataflow, kernels and parked code.

```text
finn.core.space  <-  finn.dataflow  <-  finn.kernels  <-  finn.parked / graph integration
```

``finn.dataflow`` holds canonical logical values and imports only the engine,
QONNX's datatype module and the standard library. ``finn.kernels`` builds on it
and never reaches into parked code. ``finn.parked`` is the retired
implementation: it may import anything live, and nothing live imports it.
"""

from __future__ import annotations

import ast
import pkgutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src"
FINN = SOURCE / "finn"
DATAFLOW = FINN / "dataflow"
KERNELS = FINN / "kernels"
PARKED = FINN / "parked"
TESTS = ROOT / "tests"


def _module_name(path: Path) -> str:
    parts = path.relative_to(SOURCE).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported_modules(path: Path, package: str | None = None) -> set[str]:
    """Absolute names of every module an import statement in ``path`` names.

    Relative imports are resolved against ``package``.  Every level of a dotted
    name is included, so ``from a.b import c`` also reports ``a.b.c``.
    """

    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert package is not None, f"relative import outside a package: {path}"
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _package_of(path: Path) -> str:
    name = _module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _sources(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.rglob("*.py")))
    assert paths, directory
    return paths


def test_dataflow_imports_only_the_engine_qonnx_datatypes_and_the_standard_library() -> None:
    allowed_packages = ("finn.dataflow", "finn.core.space", "qonnx.core.datatype")
    for path in _sources(DATAFLOW):
        invalid = {
            name
            for name in _imported_modules(path, _package_of(path))
            if not any(_within(name, prefix) for prefix in allowed_packages)
            and name.split(".")[0] not in sys.stdlib_module_names
            and name != "__future__"
            # ``from qonnx.core.datatype import X`` also names its parents.
            and name not in ("qonnx", "qonnx.core")
        }
        assert not invalid, (path, invalid)


def test_dataflow_loads_neither_kernels_nor_parked_code_at_runtime() -> None:
    """Importing every canonical module pulls in nothing above the value layer."""

    script = "\n".join(
        (
            "import importlib, pkgutil, sys",
            "import finn.dataflow as package",
            "for info in pkgutil.walk_packages(package.__path__, 'finn.dataflow.'):",
            "    importlib.import_module(info.name)",
            "bad = sorted(",
            "    name for name in sys.modules",
            "    if name.startswith(('finn.kernels', 'finn.parked', 'finn.custom_op', 'onnx'))",
            "    or (",
            "        name.startswith('qonnx.')",
            "        and name not in ('qonnx.core', 'qonnx.core.datatype')",
            "    )",
            ")",
            "raise SystemExit('loaded: ' + ', '.join(bad) if bad else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_kernels_never_import_parked_code() -> None:
    for path in _sources(KERNELS):
        named = _imported_modules(path, _package_of(path))
        assert not any(_within(name, "finn.parked") for name in named), path


def test_nothing_outside_parked_imports_parked_code() -> None:
    for path in _sources(FINN):
        if path.is_relative_to(PARKED):
            continue
        named = _imported_modules(path, _package_of(path))
        assert not any(_within(name, "finn.parked") for name in named), path

    for path in _sources(TESTS):
        named = _imported_modules(path, "tests")
        assert not any(_within(name, "finn.parked") for name in named), path


def test_the_logical_facade_loads_no_model_until_a_public_name_is_requested() -> None:
    script = "\n".join(
        (
            "import sys",
            "import finn.dataflow.model.logical",
            "loaded = [name for name in (",
            "    'finn.dataflow.model.logical.region',",
            "    'finn.dataflow.model.logical.network',",
            "    'finn.dataflow.model.logical.composition',",
            "    'finn.dataflow.datatypes',",
            ") if name in sys.modules]",
            "raise SystemExit('loaded: ' + ', '.join(loaded) if loaded else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_package_roots_re_export_nothing() -> None:
    """One import path per concept: owning modules, not package roots."""

    for name in ("finn.dataflow", "finn.dataflow.model"):
        root = import_module(name)
        assert not hasattr(root, "__all__"), name
        assert not hasattr(root, "__getattr__"), name
        for value in ("DataflowRegion", "DataflowNetwork", "Kernel", "Space", "LogicalView"):
            assert not hasattr(root, value), (name, value)


def test_every_canonical_module_is_importable() -> None:
    package = import_module("finn.dataflow")
    names = [info.name for info in pkgutil.walk_packages(package.__path__, "finn.dataflow.")]
    assert "finn.dataflow.model.logical.semantics" in names
    for name in names:
        import_module(name)
