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
        'importlib.import_module("..artifacts", "finn.kernels.space")',
        'import_module("..artifacts", package="finn.kernels.space")',
    ),
)
def test_relative_dynamic_imports_cannot_escape_layer_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    path = tmp_path / "src/finn/kernels/space/example.py"
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
        ("_engine", ("finn.kernels._engine",)),
        ("space", ("finn.kernels.space", "finn.kernels._engine")),
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
        forbidden = ("finn.dataflow", "finn.custom_op.dataflow", "qonnx.core.modelwrapper")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)


sys.meta_path.insert(0, RejectDataflow())
from finn.kernels import DotpAxiKernel, DspBlock, WeightDelivery, mvau_assembly
from finn.kernels.space import Problem, Space, Subspace
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS, QONNX_DATATYPE_CODEC
from finn.kernels.artifacts.requirements import ModuleBuildRequirements
from finn.kernels.physical.axi_stream import AxiStream
from finn.kernels._engine import Decided
from qonnx.core.datatype import DataType


class Request(Space):
    pe = Problem(int)
    simd = Problem(int)
    dtype = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    result = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    target = Problem(DspBlock)
    segment = Problem(int)
    compute = Subspace(
        DotpAxiKernel,
        pe=pe,
        simd=simd,
        activation_dtype=dtype,
        weights_dtype=dtype,
        result_dtype=result,
        target_dsp=target,
        segment_length=segment,
    )


point = Request.start(
    {
        Request.pe: 2,
        Request.simd: 2,
        Request.dtype: DataType["INT3"],
        Request.result: DataType["INT8"],
        Request.target: DspBlock.DSP48E2,
        Request.segment: 0,
    }
).compute.assign(DotpAxiKernel.compute_pumping, False)
assert isinstance(point.physical.accepted_answer, Decided)
assert isinstance(point.physical.accepted_answer.value, ModuleBuildRequirements)
assert isinstance(point.activation, AxiStream)
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
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
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
