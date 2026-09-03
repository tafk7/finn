# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The package boundaries the U1.5 reset established, and that they stay clean.

Two kinds of claim live here.  The *retirement* claims say that the superseded
experimental stacks are gone and did not leave a forwarding alias behind --
they are what stops the deleted API creeping back in as a convenience import
while U2 to U4 are being written.  The *direction* claims say which package may
import which, and they are the older, durable ones: the canonical Region and
Network values stay importable without the engine, and artifacts never reach
back up into the layers that project into it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src" / "finn"
DATAFLOW = SOURCE / "dataflow"

#: Every module path the reset retired.  A retired name must not be importable
#: and must not appear in an import statement anywhere in the tree.
RETIRED_MODULES = (
    "finn.custom_op.dataflow",
    "finn.dataflow.authoring",
    "finn.dataflow.design",
    "finn.dataflow.op",
    "finn.dataflow.op_contracts",
    "finn.dataflow.ops.mvau.associations",
    "finn.dataflow.ops.mvau.binding",
    "finn.dataflow.ops.mvau.elaboration",
    "finn.dataflow.ops.mvau.inventory",
    "finn.dataflow.ops.mvau.op",
    "finn.dataflow.ops.mvau.problem",
    "finn.dataflow.ops.mvau.semantics",
    "finn.dataflow.ops.mvau.source",
    "finn.dataflow.resolution",
    "finn.dataflow.selection",
    "finn.dataflow.spec_algebra",
    "finn.dataflow.testing",
    "finn.transformation.fpgadataflow.infer_mvau_dataflow",
    "finn.transformation.fpgadataflow.select_dataflow_design",
)


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names any import statement in ``path`` names."""

    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _assert_fresh_import_avoids(module: str, forbidden: tuple[str, ...]) -> None:
    """Import ``module`` in a fresh process and reject forbidden transitive imports."""

    script = "\n".join(
        (
            "from importlib import import_module",
            "import sys",
            f"import_module({module!r})",
            f"forbidden = {forbidden!r}",
            "loaded = [name for name in forbidden if name in sys.modules]",
            "raise SystemExit('loaded: ' + ', '.join(loaded) if loaded else 0)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_every_retired_module_is_gone_and_unreferenced() -> None:
    for module in RETIRED_MODULES:
        try:
            import_module(module)
        except ImportError:
            pass
        else:  # pragma: no cover - the assertion below is the report
            raise AssertionError(f"{module} is still importable")

    offenders = {
        str(path.relative_to(ROOT)): sorted(
            name
            for name in _imported_modules(path)
            if any(_within(name, retired) for retired in RETIRED_MODULES)
        )
        for path in SOURCE.rglob("*.py")
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_no_forwarding_alias_survives_the_reset() -> None:
    """A retired package must not come back as a one-line re-export module."""

    for module in RETIRED_MODULES:
        relative = Path(*module.split(".")[1:])
        assert not (SOURCE / relative.with_suffix(".py")).exists(), module
        assert not (SOURCE / relative / "__init__.py").exists(), module


def test_the_final_package_boundaries_are_the_approved_ones() -> None:
    assert tuple(import_module("finn.dataflow.ops").__all__) == ()
    assert tuple(import_module("finn.dataflow.ops.mvau").__all__) == ()
    assert tuple(import_module("finn.dataflow.ops.mvau.designs").__all__) == ()
    assert set(import_module("finn.dataflow.kernels").__all__) == {
        "Kernel",
        "KernelPhysicalResult",
        "Parameter",
        "PhysicallyUnsupported",
        "Region",
        "RegionRefused",
        "kernel_dataflow",
        "kernel_physical",
        "kernel_source_derivation",
        "portable_kernel_component",
        "resolve_kernel_contributions",
        "DotpAxiKernel",
        "DspBlock",
        "ReplayBufferKernel",
    }
    assert set(import_module("finn.dataflow.designs").__all__) == {
        "DATAFLOW_PROJECTION",
        "Boundary",
        "Connection",
        "DataflowDesign",
        "Kernels",
        "Sink",
        "design_dataflow",
    }


def test_the_generic_substrate_does_not_import_a_layer() -> None:
    """``model`` is layer-neutral: it names no Kernel, Design or operation."""

    layers = ("finn.dataflow.kernels", "finn.dataflow.designs", "finn.dataflow.ops")
    for path in (DATAFLOW / "model").rglob("*.py"):
        named = _imported_modules(path)
        assert not any(_within(name, layer) for name in named for layer in layers), path


def test_artifact_projection_stays_one_way() -> None:
    """Layers project into artifacts; artifacts never reach back up."""

    upstream = (
        "finn.dataflow.model",
        "finn.dataflow.kernels",
        "finn.dataflow.designs",
        "finn.dataflow.ops",
        "finn.dataflow._engine",
    )
    for path in (DATAFLOW / "artifacts").rglob("*.py"):
        named = _imported_modules(path)
        assert not any(_within(name, package) for name in named for package in upstream), path


def test_canonical_values_stay_importable_without_the_engine() -> None:
    _assert_fresh_import_avoids("finn.dataflow.region", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids("finn.dataflow.network", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids("finn.dataflow.ops.mvau.regions", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids("finn.dataflow.ops.mvau.networks", ("finn.dataflow._engine",))


def test_layer_facades_do_not_eagerly_load_their_implementations() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.kernels",
        (
            "finn.dataflow.kernels.dotp_axi",
            "finn.dataflow.kernels.replay_buffer",
            "finn.dataflow.kernels.artifacts",
        ),
    )
    _assert_fresh_import_avoids("finn.dataflow.designs", ("finn.dataflow.designs.design",))
    _assert_fresh_import_avoids(
        "finn.dataflow.ops.mvau", ("finn.dataflow.ops.mvau.designs.dot_product",)
    )


def test_the_private_engine_imports_no_finn_module() -> None:
    forbidden: set[str] = set()
    for path in (DATAFLOW / "_engine").glob("*.py"):
        forbidden.update(
            name for name in _imported_modules(path) if name == "finn" or name.startswith("finn.")
        )
    assert forbidden == set()


def test_the_traditional_custom_op_oracle_survived_the_reset() -> None:
    """The reset removed experiments, not FINN's established implementation."""

    assert (SOURCE / "custom_op" / "fpgadataflow" / "rtlbackend.py").is_file()
    assert (
        SOURCE / "custom_op" / "fpgadataflow" / "rtl" / "matrixvectoractivation_rtl.py"
    ).is_file()
