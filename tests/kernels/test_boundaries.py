# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Enforce the independent physical library and its internal dependency layers."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/finn/kernels"


def imported_modules(path: Path) -> set[str]:
    """Resolve absolute, relative and literal dynamic imports for source checks."""
    relative = path.relative_to(ROOT / "src").with_suffix("")
    module = ".".join(relative.parts)
    package = (
        module.removesuffix(".__init__") if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    )
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, package)
            names.add(name)
            names.update(name + "." + alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            call = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            argument = node.args[0]
            if (
                call in ("import_module", "__import__")
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ):
                name = argument.value
                if call == "import_module" and name.startswith("."):
                    package_argument = next(
                        (item.value for item in node.keywords if item.arg == "package"),
                        node.args[1] if len(node.args) > 1 else None,
                    )
                    base = (
                        package_argument.value
                        if isinstance(package_argument, ast.Constant)
                        and isinstance(package_argument.value, str)
                        else package
                    )
                    name = importlib.util.resolve_name(name, base)
                names.add(name)
    return names


@pytest.mark.parametrize(
    "source",
    (
        'importlib.import_module("..artifacts", __package__)',
        'importlib.import_module("..artifacts", "finn.core.space")',
        'import_module("..artifacts", package="finn.core.space")',
    ),
)
def test_relative_dynamic_imports_cannot_escape_layer_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    path = tmp_path / "src/finn/core/space/example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert "finn.kernels.artifacts" in imported_modules(path)


def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_kernel_sources_and_tests_have_no_dataflow_dependency() -> None:
    forbidden = ("finn.dataflow", "finn.custom_op.dataflow", "qonnx.core.modelwrapper")
    for path in PACKAGE.rglob("*.py"):
        assert not any(
            within(name, prefix) for name in imported_modules(path) for prefix in forbidden
        ), path
    for path in (ROOT / "tests/kernels").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                within(name, prefix) for name in names for prefix in (*forbidden, "dataflow")
            ), path


@pytest.mark.parametrize(
    "layer,allowed",
    [
        ("space", ("finn.core.space",)),
        ("artifacts", ("finn.kernels.artifacts",)),
    ],
)
def test_internal_layers_are_independent(layer: str, allowed: tuple[str, ...]) -> None:
    paths = tuple((PACKAGE / layer).rglob("*.py"))
    assert paths
    for path in paths:
        invalid = {
            name
            for name in imported_modules(path)
            if within(name, "finn") and not any(within(name, prefix) for prefix in allowed)
        }
        assert not invalid, (path, invalid)


def test_cold_import_and_construction_with_dataflow_unavailable() -> None:
    script = r"""
import importlib.abc
import sys


class RejectDataflow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "finn.dataflow", "finn.custom_op.dataflow",
            "qonnx.core.modelwrapper", "finn.kernels._engine",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)


sys.meta_path.insert(0, RejectDataflow())
from finn.kernels import DotpAxiKernel, DspBlock, WeightDelivery, mvau_assembly
from finn.core.space import Available
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.physical.axi_stream import AxiStream
from qonnx.core.datatype import DataType


point = DotpAxiKernel(
    {
        DotpAxiKernel.pe: 2,
        DotpAxiKernel.simd: 2,
        DotpAxiKernel.activation.dtype: DataType["INT3"],
        DotpAxiKernel.weights.dtype: DataType["INT3"],
        DotpAxiKernel.result.dtype: DataType["INT8"],
        DotpAxiKernel.target_dsp: DspBlock.DSP48E2,
        DotpAxiKernel.segment_length: 0,
    }
).with_choices(compute_pumping=False)
answer = point.build_requirements().accepted_result
assert isinstance(answer, Available)
assert isinstance(answer.value, ModuleBuildRequirements)
assert isinstance(point.activation.stream, AxiStream)
assert point.activation.payload_bits == 6
for mode in WeightDelivery:
    options = (
        {"weights": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        if mode is WeightDelivery.CYCLIC
        else {}
    )
    assembly = mvau_assembly(
        repetitions=2,
        matrix_width=4,
        matrix_height=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        pe=2,
        simd=2,
        target_dsp=DspBlock.DSP48E2,
        segment_length=0,
        compute_pumping=False,
        weight_delivery=mode,
        **options,
    )
    assert assembly.result_dtype == DataType["INT8"]
    assert assembly.requirements.contributions
assert not any(
    name.startswith("finn.kernels.") and ("._engine" in name or "._next" in name)
    for name in sys.modules
)
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_kernel_runtime_has_no_legacy_or_staging_modules() -> None:
    assert not (PACKAGE / "_engine").exists()
    assert not (PACKAGE / "space/_next").exists()
    for path in PACKAGE.rglob("*.py"):
        assert not path.name.startswith("_next"), path
        for name in imported_modules(path):
            assert not within(name, "finn.kernels._engine"), (path, name)
            assert not any(part.startswith("_next") for part in name.split(".")), (path, name)


@pytest.mark.parametrize("layer", ("space", "artifacts"))
def test_cold_layer_import_loads_only_its_own_kernel_modules(layer: str) -> None:
    script = r"""
import importlib
import importlib.abc
import sys

target = "finn.kernels." + sys.argv[1]
class RejectOtherLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("qonnx", "onnx", "finn.dataflow", "finn.custom_op.dataflow")):
            raise AssertionError("unexpected dependency: " + fullname)

sys.meta_path.insert(0, RejectOtherLayers())
importlib.import_module(target)
loaded = {name for name in sys.modules if name.startswith("finn.kernels.")}
assert all(name == target or name.startswith(target + ".") for name in loaded), loaded
"""
    result = subprocess.run([sys.executable, "-c", script, layer], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_duplicate_shared_implementation() -> None:
    old = ROOT / "src/finn/dataflow"
    for relative in (
        "space",
        "_engine",
        "artifacts",
        "model/kernel_base.py",
        "kernels/dotp_axi_minimal.py",
        "kernels/mvau.py",
        "kernels/streaming.py",
        "kernels/target.py",
        "kernels/resources",
        "kernels/matmul/resources.py",
        "model/logical/datatypes.py",
        "model/logical/datatype_semantics.py",
        "model/logical/datatype_domains.py",
    ):
        assert not (old / relative).exists(), relative
    for name in ("axi_stream", "layout", "structure", "validation", "lowering"):
        assert not (old / "model/physical" / (name + ".py")).exists()
