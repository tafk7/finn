# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""D9 dependency boundary between production MVAU and Provider-era compatibility."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from finn.dataflow.ops.mvau.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.ops.mvau.elaboration import __all__ as production_provider_exports
from finn.dataflow.ops.mvau.artifacts.ipxact import PreparedIpPackage
from finn.dataflow.ops.mvau.artifacts.package import PackagedDecomposedArtifact
from finn.dataflow.ops.mvau.artifacts.render import render_decomposed_wrapper
from finn.dataflow.ops.mvau.artifacts.source import MVAUDecomposedArtifactRequirements
from finn.dataflow.ops.mvau.artifacts.synthesis import PreparedDecomposedSynthesis
from finn.dataflow.testing import (
    assert_fresh_import_avoids,
    assert_no_raw_declaration_construction,
)

ROOT = Path(__file__).parents[3] / "src" / "finn" / "dataflow"
PRODUCTION_FILES = (
    ROOT / "ops" / "mvau" / "__init__.py",
    ROOT / "ops" / "mvau" / "op.py",
    ROOT / "ops" / "mvau" / "source.py",
    ROOT / "ops" / "mvau" / "projection.py",
    ROOT / "ops" / "mvau" / "persistence.py",
    ROOT / "ops" / "mvau" / "origin.py",
    ROOT / "ops" / "mvau" / "assignments.py",
    ROOT / "ops" / "mvau" / "associations.py",
    ROOT / "ops" / "mvau" / "binding.py",
    ROOT / "ops" / "mvau" / "inventory.py",
    ROOT / "ops" / "mvau" / "input_supply.py",
    ROOT / "ops" / "mvau" / "regions.py",
    ROOT / "ops" / "mvau" / "physical.py",
    ROOT / "ops" / "mvau" / "semantics.py",
    ROOT / "ops" / "mvau" / "elaboration.py",
    *(ROOT / "ops" / "mvau" / "designs").glob("*.py"),
    *(ROOT / "ops" / "mvau" / "artifacts").glob("*.py"),
)
FORBIDDEN_MODULES = {
    "finn.dataflow.mvau.compat",
    "finn.dataflow.mvau.compute_kernels",
    "finn.dataflow.mvau.decomposed",
}
FORBIDDEN_RUNTIME_MODULES = (
    "finn.dataflow.authoring.kernel_design",
    "finn.dataflow.mvau.compute_kernels",
    "finn.dataflow.mvau.decomposed",
    "finn.dataflow.mvau.compat.operation",
    "finn.dataflow.parameters.supply_kernels",
)
FORBIDDEN_NAMES = {
    "DecomposedBindings",
    "KernelBinding",
    "KernelProvider",
    "provider_of",
    "elaborate_mvau_rtl_softvec",
    "SOFT_VECTOR_PROVIDER_ID",
    "MEMSTREAM_PROVIDER_ID",
}
DELETED_MVAU_MODULES = (
    "finn.dataflow.kernel",
    "finn.dataflow.authoring.kernel_design",
    "finn.dataflow.mvau.compat.operation",
    "finn.dataflow.mvau.compute_kernels",
    "finn.dataflow.mvau.compute_pool",
    "finn.dataflow.mvau.decomposed",
    "finn.dataflow.mvau.legacy_design",
    "finn.dataflow.mvau.weight_adapter",
    "finn.dataflow.mvau.weight_adapter_kernel",
    "finn.dataflow.mvau_design",
    "finn.dataflow.parameters.supply_kernels",
)


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


def test_production_mvau_import_graph_is_acyclic() -> None:
    modules = {
        "finn.dataflow." + ".".join(path.relative_to(ROOT).with_suffix("").parts): path
        for path in PRODUCTION_FILES
    }
    edges: dict[str, set[str]] = {name: set() for name in modules}
    for source, path in modules.items():
        for imported, _name in _imports(path):
            edges[source].update(
                target
                for target in modules
                if imported == target or imported.startswith(f"{target}.")
            )

    def reaches(source: str, target: str) -> bool:
        pending = list(edges[source])
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(edges[current] - visited)
        return False

    cycles = {
        tuple(sorted((left, right)))
        for left in modules
        for right in modules
        if left < right and reaches(left, right) and reaches(right, left)
    }
    assert cycles == set()


def test_operation_specific_authoring_does_not_reconstruct_compiled_declarations() -> None:
    operation_files = (
        ROOT / "ops" / "mvau" / "__init__.py",
        ROOT / "ops" / "mvau" / "op.py",
        ROOT / "ops" / "mvau" / "designs" / "dot_product.py",
        ROOT / "ops" / "mvau" / "designs" / "batch_interleaved.py",
        ROOT / "ops" / "mvau" / "inventory.py",
    )
    assert_no_raw_declaration_construction(operation_files)


def test_production_elaboration_dispatch_exports_no_provider_registry() -> None:
    assert production_provider_exports == ["elaborate_mvau"]


def test_production_elaboration_import_does_not_load_provider_era_modules() -> None:
    assert_fresh_import_avoids("finn.dataflow.ops.mvau.elaboration", FORBIDDEN_RUNTIME_MODULES)


def test_production_mvau_op_import_does_not_load_provider_era_modules() -> None:
    assert_fresh_import_avoids("finn.dataflow.ops.mvau.op", FORBIDDEN_RUNTIME_MODULES)


def test_operation_namespace_does_not_eagerly_load_mvau() -> None:
    _run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import finn.dataflow.ops; "
            "assert 'finn.dataflow.ops.mvau' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert _run.returncode == 0, _run.stderr


def test_relocated_production_module_paths_are_absent() -> None:
    script = "\n".join(
        (
            "import importlib.util",
            "old = ('finn.dataflow.ops.mvau_op', 'finn.dataflow.mvau.source', "
            "'finn.dataflow.mvau.designs.inventory', 'finn.dataflow.mvau.hardware.composition', "
            "'finn.dataflow.ops.mvau.designs.inventory', "
            "'finn.dataflow.ops.mvau.artifacts._implementation', "
            "'finn.dataflow.ops.mvau.composition')",
            "present = []",
            "for name in old:",
            "    try:",
            "        found = importlib.util.find_spec(name)",
            "    except ModuleNotFoundError:",
            "        found = None",
            "    if found is not None:",
            "        present.append(name)",
            "message = 'old production modules remain: ' + ', '.join(present)",
            "raise SystemExit(message if present else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_provider_era_mvau_modules_are_not_importable() -> None:
    script = "\n".join(
        (
            "import importlib.util",
            f"deleted = {DELETED_MVAU_MODULES!r}",
            "present = []",
            "for name in deleted:",
            "    try:",
            "        found = importlib.util.find_spec(name)",
            "    except ModuleNotFoundError:",
            "        found = None",
            "    if found is not None:",
            "        present.append(name)",
            "raise SystemExit('deleted modules remain: ' + ', '.join(present) if present else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_physical_kernel_vocabulary_has_one_canonical_package() -> None:
    script = "\n".join(
        (
            "import importlib.util",
            "import finn.dataflow.kernels as kernels",
            "assert kernels.Kernel.__name__ == 'Kernel'",
            "assert kernels.KernelScope.__name__ == 'KernelScope'",
            "assert not hasattr(kernels, 'HardwareKernel')",
            "assert not hasattr(kernels, 'HardwareDesign')",
            "assert not hasattr(kernels, 'HardwareKernelSelection')",
            "assert importlib.util.find_spec('finn.dataflow.hardware') is None",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_artifact_stage_implementations_have_one_owner_each() -> None:
    assert render_decomposed_wrapper.__module__.endswith(".artifacts.render")
    assert MVAUDecomposedArtifactRequirements.__module__.endswith(".artifacts.source")
    assert PackagedDecomposedArtifact.__module__.endswith(".artifacts.package")
    assert PreparedDecomposedSynthesis.__module__.endswith(".artifacts.synthesis")
    assert PreparedIpPackage.__module__.endswith(".artifacts.ipxact")


def test_weight_adapter_forcing_code_is_test_only() -> None:
    assert not (ROOT / "mvau" / "weight_adapter.py").exists()
    assert Path(__file__).with_name("weight_adapter_region.py").is_file()


def test_fused_kernel_is_absent_from_every_production_design_candidate() -> None:
    candidates = {
        candidate.id
        for declaration in MVAU_DESIGN_INVENTORY.inventory.declarations
        for placement in declaration.placements
        for candidate in placement.candidates
    }
    assert "mvu_vvu_axi" not in candidates
