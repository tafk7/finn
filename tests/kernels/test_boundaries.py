# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Enforce the independent physical library and its internal dependency layers."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/finn/kernels"
SPACE_PACKAGE = ROOT / "src/finn/core/space"


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
    assert "finn.core.artifacts" in imported_modules(path)


def within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_kernel_sources_and_tests_have_no_parked_dependency() -> None:
    """``finn.kernels`` builds on ``finn.dataflow``, never on parked code or on a graph."""
    forbidden = (
        "finn.parked",
        "finn.custom_op.kernels",
        "finn.transformation.kernels",
        "qonnx.core.modelwrapper",
        "finn.core.onnx_exec",
        "finn.core.rtlsim_exec",
    )
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
                within(name, prefix) for name in names for prefix in (*forbidden, "parked")
            ), path


@pytest.mark.parametrize(
    "directory,allowed",
    [
        (SPACE_PACKAGE, ("finn.core.space",)),
        (PACKAGE / "artifacts", ("finn.kernels.artifacts",)),
    ],
)
def test_internal_layers_are_independent(directory: Path, allowed: tuple[str, ...]) -> None:
    paths = tuple(directory.rglob("*.py"))
    assert paths
    for path in paths:
        invalid = {
            name
            for name in imported_modules(path)
            if within(name, "finn") and not any(within(name, prefix) for prefix in allowed)
        }
        assert not invalid, (path, invalid)


def test_generic_space_imports_only_generic_dependencies() -> None:
    allowed = {*sys.stdlib_module_names, "greenlet"}
    for path in SPACE_PACKAGE.rglob("*.py"):
        invalid = {
            name
            for name in imported_modules(path)
            if not within(name, "finn.core.space") and name.split(".")[0] not in allowed
        }
        assert not invalid, (path, invalid)


def test_cold_import_and_construction_with_parked_code_unavailable() -> None:
    script = r"""
import importlib.abc
import sys


class RejectParked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "finn.parked", "qonnx.core.modelwrapper", "finn.core.onnx_exec",
            "finn.core.rtlsim_exec", "onnx",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)


sys.meta_path.insert(0, RejectParked())
from finn.core.space import Space, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels import DspBlock, MatMulKernel, PackedDotpKernel
from finn.kernels.target import Platform
from finn.kernels.artifacts.module import Composed, Leaf
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.transport import AxiStream
from finn.kernels.streams import BufferedStream, Stream
from qonnx.core.datatype import DataType

PLATFORM = Platform(
    period_ns=5.0,
    dsp=DspBlock.DSP48E2,
    uram=True,
    uram_init=True,
    clk2x=True,
    control_ports=1,
    memory_ports=0,
    aie=False,
)


class Placed(Space):
    x = Stream(
        tensor=Tensor((1, 2), ScalarEncoding(DataType["INT3"])),
        port="in0_V",
        platform=PLATFORM,
    )
    w = Stream(
        tensor=Tensor((2, 2), ScalarEncoding(DataType["INT3"])),
        port="in1_V",
        platform=PLATFORM,
    )
    y = Stream(
        tensor=Tensor((1, 2), ScalarEncoding(DataType["INT8"])),
        port="out0_V",
        platform=PLATFORM,
    )
    compute = PackedDotpKernel(
        result_dtype=DataType["INT8"],
        x_stream=x,
        w_stream=w,
        y_stream=y,
        platform=PLATFORM,
    )


point = commit(
    design_space(Placed()),
    {"compute.pe": 2, "compute.simd": 2, "compute.compute_pumping": False},
).compute
answer = point.module
assert isinstance(answer, Leaf)
assert isinstance(point.x.axis, AxiStream)
assert point.x.axis.payload_bits == 6
class Root(Kernel):
    id = "test.root"


INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
for memory in ("none", "memstream"):
    facts = dict(
        m=2,
        k=4,
        n=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        platform=PLATFORM,
    )
    if memory == "memstream":
        facts["weights"] = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))

    class Placed(Root):
        x = Stream(tensor=Tensor((2, 4), INT3), port="in0_V", platform=PLATFORM)
        w = BufferedStream(tensor=Tensor((4, 4), INT3), port="in1_V", platform=PLATFORM)
        y = Stream(tensor=Tensor((2, 4), INT8), port="out0_V", platform=PLATFORM)
        matmul = MatMulKernel(**facts, x_stream=x, w_stream=w, y_stream=y)
        w.contents = matmul.weight_values

    choices = {"w.transport": "direct", "matmul.compute": "packed"}
    if memory == "memstream":
        choices |= {
            "w.source.memstream.ram_style": "auto",
            "w.source.memstream.pumped_memory": False,
        }
    root = commit(design_space(Placed()), choices)
    root = commit(
        root,
        {
            "matmul.compute.packed.pe": 2,
            "matmul.compute.packed.simd": 2,
            "matmul.compute.packed.compute_pumping": False,
            "x.adapter": "input_gen",
            "x.adapter.input_gen.input_gen.ram_style": "auto",
        },
    )
    assert root.matmul.result_type == DataType["INT8"]
    assert isinstance(root.module, Composed) and root.module.fragment.instances
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", ("finn.core.space", "finn.kernels.artifacts"))
def test_cold_layer_import_loads_only_its_own_modules(target: str) -> None:
    script = r"""
import importlib
import importlib.abc
import sys

package_name = sys.argv[1]
class RejectOtherLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            "qonnx", "onnx", "finn.dataflow", "finn.parked", "finn.core.onnx_exec",
            "finn.core.rtlsim_exec",
        )) or (package_name == "finn.core.space" and fullname.startswith("finn.kernels")):
            raise AssertionError("unexpected dependency: " + fullname)

sys.meta_path.insert(0, RejectOtherLayers())
api = importlib.import_module(package_name)
if package_name == "finn.core.space":
    class Generic(api.Space):
        value: int = api.Param()

        @api.derived
        def increment(self) -> int:
            return self.value + 1

        @api.view
        def output(self) -> int:
            return self.increment

    assert api.design_space(Generic(value=3)).output == 4
loaded = {name for name in sys.modules if name.startswith("finn.")}
parents = {package_name.rsplit(".", 1)[0]}
assert all(
    name in parents or name == package_name or name.startswith(package_name + ".")
    for name in loaded
), loaded
"""
    result = subprocess.run([sys.executable, "-c", script, target], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
