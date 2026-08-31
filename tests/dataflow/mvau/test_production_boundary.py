# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D9 dependency boundary between production MVAU and Provider-era compatibility."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from finn.dataflow.mvau.designs.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.mvau.providers import __all__ as production_provider_exports

ROOT = Path(__file__).parents[3] / "src" / "finn" / "dataflow"
PRODUCTION_FILES = (
    ROOT / "ops" / "mvau.py",
    ROOT / "ops" / "mvau_op.py",
    ROOT / "mvau" / "source.py",
    ROOT / "mvau" / "physical.py",
    ROOT / "mvau" / "input_supply.py",
    ROOT / "mvau" / "semantics.py",
    ROOT / "mvau" / "providers.py",
    *(ROOT / "mvau" / "designs").glob("*.py"),
    *(ROOT / "mvau" / "hardware").glob("*.py"),
)
FORBIDDEN_MODULES = {
    "finn.dataflow.kernels",
    "finn.dataflow.mvau.compat",
    "finn.dataflow.mvau.compute_kernels",
    "finn.dataflow.mvau.decomposed",
}
FORBIDDEN_NAMES = {
    "DecomposedBindings",
    "KernelBinding",
    "KernelProvider",
    "provider_of",
    "elaborate_mvau_rtl_softvec",
    "SOFT_VECTOR_PROVIDER_ID",
    "MEMSTREAM_PROVIDER_ID",
}


def _imports(path: Path) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.extend((node.module, item.name) for item in node.names)
        elif isinstance(node, ast.Import):
            imported.extend((item.name, "") for item in node.names)
    return tuple(imported)


def test_production_dot_product_imports_no_provider_era_api() -> None:
    for path in PRODUCTION_FILES:
        for module, name in _imports(path):
            assert not any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_MODULES
            ), (path, module)
            assert name not in FORBIDDEN_NAMES, (path, name)


def test_removed_binding_wrappers_do_not_reappear_in_production() -> None:
    for path in PRODUCTION_FILES:
        names = {
            node.id for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Name)
        }
        assert not names & {"KernelBinding", "DecomposedBindings"}, path


def test_production_elaboration_dispatch_exports_no_provider_registry() -> None:
    assert production_provider_exports == ["elaborate_mvau"]


def test_production_elaboration_import_does_not_load_provider_era_modules() -> None:
    forbidden = (
        "finn.dataflow.kernels",
        "finn.dataflow.mvau.compute_kernels",
        "finn.dataflow.mvau.decomposed",
        "finn.dataflow.mvau.compat.operation",
        "finn.dataflow.parameters.supply_kernels",
    )
    script = "\n".join(
        (
            "import sys",
            "import finn.dataflow.mvau.providers",
            f"forbidden = {forbidden!r}",
            "loaded = [name for name in forbidden if name in sys.modules]",
            "raise SystemExit('loaded legacy modules: ' + ', '.join(loaded) if loaded else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_fused_kernel_is_absent_from_every_production_design_candidate() -> None:
    candidates = {
        candidate.id
        for declaration in MVAU_DESIGN_INVENTORY.inventory.declarations
        for placement in declaration.placements
        for candidate in placement.candidates
    }
    assert "mvu_vvu_axi" not in candidates
