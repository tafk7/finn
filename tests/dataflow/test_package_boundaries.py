# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Current package ownership and dependency boundaries.

Canonical model values remain usable without the engine. Generic Space code
stays independent of dataflow domains, and artifact processing does not import
its compiler consumers. Public facades load implementations only when requested.
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


def test_the_final_package_boundaries_are_the_approved_ones() -> None:
    assert tuple(import_module("finn.dataflow.ops").__all__) == ()
    assert tuple(import_module("finn.dataflow.ops.mvau").__all__) == ()
    assert set(import_module("finn.dataflow.kernels").__all__) == {
        "DotpAxiKernel",
        "DspBlock",
        "MVAU",
        "MVAUAssembly",
        "WeightDelivery",
        "mvau_assembly",
        "replay_buffer_requirements",
        "cyclic_stream_requirements",
    }
    for framework_name in (
        "Kernel",
        "KernelChoice",
        "LogicalView",
        "ModuleBuildRequirements",
        "ModuleParameter",
        "NetworkBoundary",
        "NetworkEdge",
        "PhysicalView",
        "PhysicallyUnsupported",
        "RegionDeclaration",
        "RelationView",
        "EdgeSink",
        "kernel_dataflow",
        "kernel_physical",
    ):
        assert not hasattr(import_module("finn.dataflow.kernels"), framework_name)


def test_the_generic_substrate_does_not_import_a_layer() -> None:
    """``space`` is layer-neutral: it names no Kernel, Kernel or operation.

    Domain value semantics now live with the domain, so there is no exception
    beneath ``space``.
    """

    layers = ("finn.dataflow.kernels", "finn.dataflow.ops")
    for path in (DATAFLOW / "space").rglob("*.py"):
        named = _imported_modules(path)
        assert not any(_within(name, layer) for name in named for layer in layers), path
        assert not any(_within(name, "finn.dataflow.model") for name in named), path


def test_pure_logical_values_import_nothing_above_or_beside_them() -> None:
    """``model`` is the bottom of the dataflow stack, and depends on none of it.

    The reverse direction is what makes the model a *value* layer: a Region can
    be constructed, compared and validated with no compiler, no engine, no
    Kernel and no graph in the process.  ``space`` may import ``model``; this is
    the claim that it never runs the other way.
    """

    forbidden = (
        "finn.dataflow.space",
        "finn.dataflow._engine",
        "finn.dataflow.kernels",
        "finn.dataflow.ops",
        "finn.dataflow.parameters",
        "finn.dataflow.artifacts",
        "onnx",
    )
    adapters = {
        "authoring.py",
        "interface_authoring.py",
        "semantics.py",
        "datatype_semantics.py",
        "result_semantics.py",
        "datatype_domains.py",
        "view.py",
        "view_authoring.py",
        "contract_authoring.py",
        "contract_expressions.py",
        "_contract_support.py",
    }
    for path in (DATAFLOW / "model" / "logical").rglob("*.py"):
        if path.name in adapters:
            continue
        named = _imported_modules(path)
        assert not any(_within(name, package) for name in named for package in forbidden), path


def test_pure_physical_and_public_interface_values_keep_one_way_dependencies() -> None:
    forbidden = (
        "finn.dataflow.space",
        "finn.dataflow._engine",
        "finn.dataflow.kernels",
        "finn.dataflow.ops",
    )
    pure = (
        DATAFLOW / "model" / "physical" / "layout.py",
        DATAFLOW / "model" / "physical" / "structure.py",
        DATAFLOW / "model" / "physical" / "validation.py",
        DATAFLOW / "model" / "physical" / "lowering.py",
        DATAFLOW / "model" / "physical" / "interface.py",
        DATAFLOW / "model" / "logical" / "interface.py",
    )
    for path in pure:
        named = _imported_modules(path)
        assert not any(_within(name, package) for name in named for package in forbidden), path


def test_domain_facades_and_pure_value_modules_are_lazy() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.model",
        (
            "finn.dataflow.model.kernel",
            "finn.dataflow.model.logical",
            "finn.dataflow.model.physical",
            "finn.dataflow.model.relations",
            "finn.dataflow.space",
            "finn.dataflow._engine",
        ),
    )
    _assert_fresh_import_avoids(
        "finn.dataflow.model.physical.structure",
        ("finn.dataflow.space", "finn.dataflow._engine", "finn.dataflow.kernels"),
    )
    _assert_fresh_import_avoids(
        "finn.dataflow.model.physical.interface",
        ("finn.dataflow.space", "finn.dataflow._engine", "finn.dataflow.kernels"),
    )


def test_minimal_dotp_does_not_load_historical_authoring_or_build_processing() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.kernels.dotp_axi_minimal",
        (
            "finn.dataflow.kernels.matmul",
            "finn.dataflow.parameters",
            "finn.dataflow.model.kernel",
            "finn.dataflow.model.children",
            "finn.dataflow.model.logical.authoring",
            "finn.dataflow.model.logical.view_authoring",
            "finn.dataflow.model.logical.composition",
            "finn.dataflow.model.logical.presentation",
            "finn.dataflow.model.physical.axi_stream_binding",
            "finn.dataflow.model.physical.interface",
            "finn.dataflow.model.physical.authoring",
            "finn.dataflow.artifacts.build",
            "finn.dataflow.artifacts.contributions",
            "finn.dataflow.artifacts.render",
            "finn.dataflow.artifacts.packaging",
            "finn.dataflow.artifacts.store",
        ),
    )


def test_axi_declarations_do_not_load_regions_or_composition_adapters() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.model.physical.axi_stream",
        (
            "finn.dataflow.model.logical.region",
            "finn.dataflow.model.logical.composition",
            "finn.dataflow.model.physical.interface",
            "finn.dataflow.model.physical.axi_stream_binding",
            "finn.dataflow.artifacts.build",
            "finn.dataflow.kernels",
        ),
    )


def test_logical_facade_loads_no_model_until_a_public_name_is_requested() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.model.logical",
        (
            "finn.dataflow.model.logical.region",
            "finn.dataflow.model.logical.network",
            "finn.dataflow.model.logical.composition",
            "finn.dataflow.model.logical.datatypes",
        ),
    )


def test_module_requirements_are_independent_of_build_processing() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.artifacts.requirements",
        (
            "finn.dataflow.artifacts.build",
            "finn.dataflow.artifacts.contributions",
            "finn.dataflow.artifacts.render",
            "finn.dataflow.artifacts.store",
            "finn.dataflow.artifacts.packaging",
            "finn.dataflow.model",
            "finn.dataflow.space",
            "finn.dataflow._engine",
        ),
    )


def test_neither_package_is_re_exported_from_the_dataflow_root() -> None:
    """One import path per concept: ``finn.dataflow`` itself exports nothing.

    A value reachable as both ``finn.dataflow.X`` and ``finn.dataflow.model.X``
    reads as a value with two owners, which is exactly what the model/space
    split exists to end.
    """

    root = import_module("finn.dataflow")
    assert not hasattr(root, "__all__")
    for name in (
        "DataflowRegion",
        "DataflowNetwork",
        "Space",
        "Problem",
        "Decision",
        "Kernel",
        "KernelChoice",
        "ModuleParameter",
        "ModuleBuildRequirements",
        "RegionDeclaration",
        "NetworkEdge",
        "NetworkBoundary",
    ):
        assert not hasattr(root, name), name


def test_artifact_projection_stays_one_way() -> None:
    """Layers project into artifacts; artifacts never reach back up."""

    upstream = (
        "finn.dataflow.model",
        "finn.dataflow.space",
        "finn.dataflow.kernels",
        "finn.dataflow.ops",
        "finn.dataflow._engine",
        "onnx",
        "qonnx.core.modelwrapper",
    )
    for path in (DATAFLOW / "artifacts").rglob("*.py"):
        named = _imported_modules(path)
        assert not any(_within(name, package) for name in named for package in upstream), path
    _assert_fresh_import_avoids("finn.dataflow.artifacts.packaging", upstream)


def test_canonical_values_stay_importable_without_the_engine() -> None:
    _assert_fresh_import_avoids("finn.dataflow.model.logical.region", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids("finn.dataflow.model.logical.network", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids(
        "finn.dataflow.model.logical.refs",
        ("finn.dataflow._engine", "finn.dataflow.ops", "finn.dataflow.kernels"),
    )
    _assert_fresh_import_avoids(
        "finn.dataflow.model",
        (
            "finn.dataflow.space",
            "finn.dataflow._engine",
            "finn.dataflow.kernels",
            "finn.dataflow.ops",
            "finn.dataflow.artifacts",
        ),
    )
    _assert_fresh_import_avoids(
        "finn.dataflow.model.logical",
        (
            "finn.dataflow.space",
            "finn.dataflow._engine",
            "finn.dataflow.kernels",
            "finn.dataflow.ops",
            "finn.dataflow.artifacts",
        ),
    )
    _assert_fresh_import_avoids("finn.dataflow.kernels.matmul.regions", ("finn.dataflow._engine",))
    _assert_fresh_import_avoids("finn.dataflow.kernels.matmul.networks", ("finn.dataflow._engine",))


def test_the_space_facade_does_not_drag_in_the_dataflow_model() -> None:
    """The bridge is opt-in.  ``space.__init__`` does not import it."""

    _assert_fresh_import_avoids(
        "finn.dataflow.space",
        ("finn.dataflow.model",),
    )


def test_layer_facades_do_not_eagerly_load_their_implementations() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.kernels",
        (
            "finn.dataflow.kernels.dotp_axi",
            "finn.dataflow.kernels.matmul.dot_product",
            "finn.dataflow.kernels.replay",
            "finn.dataflow.kernels.replay_buffer",
        ),
    )
    _assert_fresh_import_avoids(
        "finn.dataflow.ops.mvau", ("finn.dataflow.kernels.matmul.dot_product",)
    )


def test_the_private_engine_imports_no_finn_module() -> None:
    forbidden: set[str] = set()
    for path in (DATAFLOW / "_engine").glob("*.py"):
        forbidden.update(
            name for name in _imported_modules(path) if name == "finn" or name.startswith("finn.")
        )
    assert forbidden == set()


def test_pure_dot_product_does_not_load_physical_or_composition_frameworks() -> None:
    _assert_fresh_import_avoids(
        "finn.dataflow.kernels.dot_product",
        (
            "finn.dataflow.model.physical.axi_stream_contract",
            "finn.dataflow.model.physical.view",
            "finn.dataflow.artifacts.build",
            "finn.dataflow.kernels.target",
            "finn.dataflow.model.logical.view",
            "finn.dataflow.model.logical.composition",
        ),
    )


def test_physical_component_path_does_not_load_logical_models_or_compiler_nodes() -> None:
    for module in (
        "finn.dataflow.kernels.dotp_axi_minimal",
        "finn.dataflow.kernels.streaming",
        "finn.dataflow.kernels.mvau",
    ):
        _assert_fresh_import_avoids(
            module,
            (
                "finn.dataflow.model.logical.region",
                "finn.dataflow.model.logical.network",
                "finn.dataflow.model.logical.contract_authoring",
                "finn.dataflow.model.logical.composition",
                "finn.dataflow.model.logical.view",
                "finn.dataflow.model.physical.interface",
                "finn.dataflow.ops.mvau.op",
                "qonnx.core.modelwrapper",
            ),
        )
