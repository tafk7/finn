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
#:
#: Two names the reset removed have since been *reused*, not restored:
#: ``finn.custom_op.dataflow`` and ``finn.dataflow.ops.mvau.op`` are U4's own
#: registration and operation, written against the occurrence lifecycle and
#: sharing nothing with what stood there before.  They are checked below for
#: what they now contain rather than for absence, because "this name exists
#: again" and "the old implementation came back" are different claims.
RETIRED_MODULES = (
    "finn.dataflow.computation",
    "finn.dataflow.parameters.cyclic.computation",
    "finn.dataflow.ops.association",
    "finn.dataflow.ops.state",
    "finn.dataflow.authoring",
    "finn.dataflow.kernel",
    "finn.dataflow.op",
    "finn.dataflow.op_contracts",
    "finn.dataflow.ops.mvau.associations",
    "finn.dataflow.ops.mvau.binding",
    "finn.dataflow.ops.mvau.elaboration",
    "finn.dataflow.ops.mvau.inventory",
    "finn.dataflow.ops.mvau.problem",
    "finn.dataflow.ops.mvau.semantics",
    "finn.dataflow.ops.mvau.source",
    "finn.dataflow.ops.mvau.kernels.supplied_dot_product",
    "finn.dataflow.designs",
    "finn.dataflow.designs.design",
    "finn.dataflow.designs.physical",
    "finn.dataflow.ops.legacy",
    "finn.dataflow.ops.mvau.designs",
    "finn.dataflow.ops.mvau.designs.base",
    "finn.dataflow.ops.mvau.designs.batch_interleaved",
    "finn.dataflow.ops.mvau.designs.dot_product",
    "finn.dataflow.ops.mvau.designs.supply",
    "finn.dataflow.ops.replay.design",
    "finn.dataflow.resolution",
    "finn.dataflow.selection",
    "finn.dataflow.spec_algebra",
    "finn.dataflow.testing",
    "finn.transformation.fpgadataflow.infer_mvau_dataflow",
    "finn.transformation.fpgadataflow.select_dataflow_design",
)

#: Every module path the C1.5 model/space migration retired.
#:
#: Two groups, and they moved in opposite directions.  The root-level modules
#: went *into* ``finn.dataflow.model``, which is now the canonical Region and
#: Network model; the generic declaration/compiler/occurrence modules that used
#: to occupy that name went out to ``finn.dataflow.space``.  Both directions are
#: destructive: no forwarding module, no alias, no second import path.
#:
#: ``input_service`` and its successor ``network_operands`` are here because the
#: reference and presentation halves they held now live separately under
#: ``model.refs`` and ``model.presentation``.  ``finn.dataflow.semantic`` never
#: existed and must not appear: the model package is the semantic authority, and
#: an intermediate package would be a third name for the same thing.
MIGRATED_MODULES = (
    "finn.dataflow.datatypes",
    "finn.dataflow.input_service",
    "finn.dataflow.model.branching",
    "finn.dataflow.model.compiler",
    "finn.dataflow.model.declarations",
    "finn.dataflow.model.domains",
    "finn.dataflow.model.occurrence",
    "finn.dataflow.model.semantics",
    "finn.dataflow.model.spec_algebra",
    "finn.dataflow.network",
    "finn.dataflow.network_operands",
    "finn.dataflow.network_validation",
    "finn.dataflow.region",
    "finn.dataflow.region_profiles",
    "finn.dataflow.region_validation",
    "finn.dataflow.semantic",
    "finn.dataflow.model.composition",
    "finn.dataflow.model.datatypes",
    "finn.dataflow.model.maps",
    "finn.dataflow.model.network",
    "finn.dataflow.model.network_validation",
    "finn.dataflow.model.presentation",
    "finn.dataflow.model.refs",
    "finn.dataflow.model.region",
    "finn.dataflow.model.region_profiles",
    "finn.dataflow.model.region_validation",
    "finn.dataflow.space.dataflow_value_semantics",
    "finn.dataflow.kernels.artifacts",
    "finn.dataflow.kernels.kernel",
    "finn.dataflow.kernels.physical",
    "finn.dataflow.kernels.physical_composition",
    "finn.dataflow.ops.mvau.kernels",
    "finn.dataflow.ops.mvau.networks",
    "finn.dataflow.ops.mvau.regions",
    "finn.dataflow.ops.replay.kernel",
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


def test_every_migrated_module_is_gone_and_unreferenced() -> None:
    """C1.5 moved these, and moved means moved."""

    for module in MIGRATED_MODULES:
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
            if any(name == retired for retired in MIGRATED_MODULES)
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
    assert set(import_module("finn.dataflow.kernels").__all__) == {
        "ActivationReplayKernel",
        "BatchInterleavedDotpAxiKernel",
        "BatchInterleavedKernel",
        "DotProductKernel",
        "DotpAxiKernel",
        "DspBlock",
        "EmbeddedDotpAxiKernel",
        "MemstreamKernel",
        "ReplayBufferKernel",
        "WeightSupply",
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


def test_s2b_removes_the_old_declarations_without_aliases() -> None:
    for module in ("finn.dataflow.kernels", "finn.dataflow.model"):
        for name in ("Kernels", "Boundary", "Connection", "Sink", "ComputationContract"):
            assert not hasattr(import_module(module), name), (module, name)
    for module in ("finn.dataflow.kernels", "finn.dataflow.model"):
        for name in ("KernelPhysicalResult", "Parameter", "ComputationContract"):
            assert not hasattr(import_module(module), name), (module, name)


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
    adapters = {"authoring.py", "semantics.py", "view.py"}
    for path in (DATAFLOW / "model" / "logical").rglob("*.py"):
        if path.name in adapters:
            continue
        named = _imported_modules(path)
        assert not any(_within(name, package) for name in named for package in forbidden), path


def test_pure_physical_and_relation_values_keep_one_way_dependencies() -> None:
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
        DATAFLOW / "model" / "relations" / "values.py",
        DATAFLOW / "model" / "relations" / "validation.py",
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
        "finn.dataflow.model.relations.values",
        ("finn.dataflow.space", "finn.dataflow._engine", "finn.dataflow.kernels"),
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


def test_the_reused_names_hold_u4s_implementation_and_not_the_retired_one() -> None:
    domain = import_module("finn.custom_op.dataflow")
    assert set(domain.custom_op) == {"MvauDataflowOp", "ActivationReplayOp"}
    operation = import_module("finn.dataflow.ops.mvau.op")
    base = import_module("finn.dataflow.ops.base")
    assert issubclass(domain.custom_op["MvauDataflowOp"], base.DataflowOp)
    assert operation.MvauDataflowOp is domain.custom_op["MvauDataflowOp"]
    # The retired stack's entry points are not what came back.
    assert not hasattr(operation, "MVAUDataflowBuildContext")
    assert not hasattr(operation, "MvauDataflowOp") or not hasattr(
        operation.MvauDataflowOp, "resolve_dataflow"
    )


def test_the_traditional_custom_op_oracle_survived_the_reset() -> None:
    """The reset removed experiments, not FINN's established implementation."""

    assert (SOURCE / "custom_op" / "fpgadataflow" / "rtlbackend.py").is_file()
    assert (
        SOURCE / "custom_op" / "fpgadataflow" / "rtl" / "matrixvectoractivation_rtl.py"
    ).is_file()
