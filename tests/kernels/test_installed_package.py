# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build and use the wheel without checkout imports or graph dependencies."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from layering import ROOT

from finn import resources as finn_resources

# -I -S ignores PYTHONPATH, the current directory, user packages and .pth files.
# Only the wheel target and ordinary dependency site-packages (which provide
# QONNX) are added. No FINN source or test package is on this path.
INSTALLED_BUILD = r"""
import importlib.abc
import json
from pathlib import Path
import sys

config = json.loads(sys.argv[1])
installed = Path(config["installed"])
sys.path[:0] = [str(installed), *config["site_packages"]]
sys.dont_write_bytecode = True

class RejectGraphDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("onnx", "finn.core.onnx_exec")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)
        if fullname.startswith("qonnx") and fullname not in (
            "qonnx", "qonnx.core", "qonnx.core.datatype"
        ):
            raise AssertionError("QONNX graph dependency: " + fullname)

sys.meta_path.insert(0, RejectGraphDependencies())

from finn.kernels.dotp import PackedDotpKernel
from finn.kernels.matmul import MatMulKernel
from finn.platform.capabilities import DspBlock, Fabric, Platform
from finn.dataflow.datatypes import ordinary_integer_bounds
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.channels import Channel, end_of
from finn.kernels.values.semantics import INTEGER_TENSOR, IntegerTensor
from finn.core.space import (
    Available, Decision, Param, Space, design_space, derived, divisors_of, view,
)
import greenlet  # the Space engine's declared native dependency

class Tiles(Space):
    extent: int = Param()
    lanes: int = Decision(domain=divisors_of(extent))

    @derived
    def cycles(self) -> int:
        return self.extent // self.lanes

    @view
    def shape(self) -> tuple[int, int]:
        return self.lanes, self.cycles

tile = design_space(Tiles(extent=12)).with_choices(lanes=3)
assert tile.shape == (3, 4)
assert tile.field(Tiles.shape).get() == (3, 4)
assert tile.inspect(Tiles.shape).accepted_result == Available((3, 4))
assert tile.field(Tiles.cycles).get() == 4
from finn.kernels.artifacts import build, module as built
from qonnx.core.datatype import DataType

assert (installed / "finn/kernels/py.typed").is_file()
assert (installed / "finn/core/space/py.typed").is_file()
assert (installed / "finn/dataflow/py.typed").is_file()
assert (installed / "finn/platform/py.typed").is_file()
assert "finn.dataflow.datatypes" in sys.modules
PLATFORM = Platform(
    period_ns=5.0,
    dsp=DspBlock.DSP48E2,
    fabric=Fabric.ULTRASCALE,
    uram=True,
    uram_init=True,
    resources=None,
)


X2 = Tensor((1, 2), ScalarEncoding(DataType["INT3"]))
W2 = Tensor((2, 2), ScalarEncoding(DataType["INT3"]))
Y2 = Tensor((1, 2), ScalarEncoding(DataType["INT8"]))

class PlacedDotp(Space):
    x_end = end_of("compute", "x_end")
    w_end = end_of("compute", "w_end")
    y_end = end_of("compute", "y_end")
    x = Channel(tensor=X2, port="in0_V", consumer=x_end, platform=PLATFORM)
    w = Channel(tensor=W2, port="in1_V", consumer=w_end, platform=PLATFORM)
    y = Channel(tensor=Y2, port="out0_V", producer=y_end, platform=PLATFORM)
    compute = PackedDotpKernel(
        result_range=ordinary_integer_bounds(DataType["INT8"]),
        x_tensor=X2,
        w_tensor=W2,
        y_tensor=Y2,
        platform=PLATFORM,
    )


dotp = commit(
    design_space(PlacedDotp()),
    {"compute.pe": 2, "compute.simd": 2, "compute.compute_pumping": False,
     "compute.reducer": "tree"},
).compute
answer = dotp.module
assert isinstance(answer, built.Leaf), answer
assert dict(answer.parameters)["ACCU_WIDTH"] == 8
assert dotp.x.element.dtype.name == "INT3"
assert dotp.x.axis.payload_bits == 6

dotp_sources = {
    "rtl/arith/add_multi_pkg.sv", "rtl/arith/compressor_pkg.sv",
    "rtl/arith/compressor_counters.sv", "rtl/arith/compress_core.sv",
    "rtl/arith/schedule_core.svh", "rtl/arith/sched_chunks.svh",
    "rtl/arith/add_multi_sched.svh", "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp.sv", "rtl/linalg/dotp_axi.sv",
}
roots = {"finnlib": Path(config["finnlib"])}
emitted_count = 0

def materialize(module, expected):
    global emitted_count
    emitted_count += 1
    directory = Path(config["output"]) / str(emitted_count)
    emitted = build.emit_module(module, directory, roots=roots)
    expected = expected | (
        {emitted.entry_point + ".sv"} if emitted.entry_point != "dotp_axi" else set()
    )
    assert set(emitted.sources) | set(emitted.data) == expected, emitted
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    assert actual == expected
    leaves = (module,) if isinstance(module, built.Leaf) else (
        leaf for _, leaf in module.fragment.instances
    )
    for leaf in leaves:
        for source in leaf.sources:
            original = (roots[source.root] / source.path).read_bytes()
            assert (directory / source.path).read_bytes() == original
    return directory / (emitted.entry_point + ".sv")

materialize(answer, dotp_sources)
class Root(Kernel):
    id = "test.root"

INT3, INT8 = ScalarEncoding(DataType["INT3"]), ScalarEncoding(DataType["INT8"])
for memory in ("none", "memstream"):
    facts = dict(
        m=3, k=4, n=4, activation_dtype=DataType["INT3"], weights_dtype=DataType["INT3"],
        platform=PLATFORM,
    )
    choices = {
        "x.transport": "direct",
        "w.transport": "direct",
        "y.transport": "direct",
        "matmul.compute": "packed",
    }
    expected = dotp_sources | {"rtl/shape/input_gen.sv"}
    # Unknown weights: the datatypes' result, INT8. Stored ones: their columns', INT7.
    weights, bits = {"w_tensor": Tensor((4, 4), INT3)}, 8
    if memory == "memstream":
        # Stored (k, n): the columns of the by-output rows; MatMul's, and the weight
        # channel's value.
        weights = {
            "w_contents": ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))
        }
        bits = 7
        choices |= {
            "w.source.memstream.ram_style": "auto",
            "w.source.memstream.pumped_memory": False,
        }
        expected |= {"rtl/infra/axilite.sv", "rtl/infra/memstream.sv", "rtl/infra/memstream_axi.sv"}

    x_tensor = Tensor((3, 4), INT3)
    y_tensor = Tensor((3, 4), ScalarEncoding(DataType[f"INT{bits}"]))

    # The MatMul in a root that declares its streams; the weight channel carries what
    # MatMul states for it.
    class Placed(Root):
        known: bool = Param()
        x_end = end_of("matmul", "x_end")
        w_end = end_of("matmul", "w_end")
        y_end = end_of("matmul", "y_end")

        @derived
        def w_tensor(self) -> Tensor:
            return self.matmul.w_carried

        @derived(when=known, semantics=INTEGER_TENSOR)
        def w_contents(self) -> IntegerTensor:
            return self.matmul.w_stored

        x = Channel(tensor=x_tensor, port="in0_V", consumer=x_end, platform=PLATFORM)
        w = Channel(
            tensor=w_tensor, contents=w_contents, port="in1_V", consumer=w_end, platform=PLATFORM
        )
        y = Channel(tensor=y_tensor, port="out0_V", producer=y_end, platform=PLATFORM)
        matmul = MatMulKernel(**facts, x_tensor=x_tensor, y_tensor=y_tensor, **weights)

    root = commit(design_space(Placed(known=memory == "memstream")), choices)
    root = commit(root, {
        "matmul.compute.packed.pe": 2,
        "matmul.compute.packed.simd": 2,
        "matmul.compute.packed.compute_pumping": False,
        "matmul.compute.packed.reducer": "tree",
        "x.adapter": "input_gen",
        "x.adapter.input_gen.input_gen.ram_style": "auto",
    })
    matmul = root.matmul
    beats = (
        root.x.ends.source.sequence.form.beats,
        matmul.compute.w.presented.form.beats,
        matmul.compute.y.presented.form.beats,
    )
    assert beats == (6, 12, 6), beats
    composed = root.module
    # A memory image ships as generated data, named by its contents.
    expected |= {item.path for _, leaf in composed.fragment.instances for item in leaf.data}
    wrapper = materialize(composed, expected).read_text()
    assert f".ACCU_WIDTH({bits})" in wrapper
    assert ".olst(n__u_x_adapter_input_gen_input_gen__olst)" in wrapper
    if memory == "memstream":
        assert '.INIT_FILE("memstream_' in wrapper
        assert root.w.source.image == (0x22C, 0x6BE, 0xDD3, 0x941)

# Catch namespace or editable-install leakage even if the import was permitted.
for name, module in tuple(sys.modules.items()):
    if name == "finn" or name.startswith("finn."):
        location = getattr(module, "__file__", None)
        if location is not None:
            assert Path(location).resolve().is_relative_to(installed), (name, location)
        for location in getattr(module, "__path__", ()):
            assert Path(location).resolve().is_relative_to(installed), (name, location)
print("installed dotp, external and memstream MatMul builds verified")
"""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


# Builds a wheel and installs it, about 12 s.
@pytest.mark.slow
def test_installed_wheel_materializes_independent_kernel_builds(tmp_path: Path) -> None:
    finnlib = Path(finn_resources.path("finnlib")).resolve()
    # Build from a clean temporary source snapshot: setuptools must neither
    # reuse a checkout build tree nor write generated metadata into the checkout.
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    for name in ("setup.py", "pyproject.toml", "VERSION", "README.md"):
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    wheels = tmp_path / "wheels"
    _run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--disable-pip-version-check",
            "--wheel-dir",
            str(wheels),
            str(snapshot),
        ],
        tmp_path,
    )
    (wheel,) = wheels.glob("finn-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert {
            "finn/kernels/py.typed",
            "finn/core/space/py.typed",
            "finn/dataflow/py.typed",
            "finn/platform/py.typed",
            "finn/dataflow/datatypes.py",
            "finn/dataflow/traversal.py",
            "finn/kernels/artifacts/module.py",
        } <= set(archive.namelist())
        (metadata_name,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        names = {value.split(";")[0].replace(" ", "") for value in requirements}
        for dependency in ("greenlet", "pyslang"):
            assert any(name.startswith(dependency) for name in names), dependency
        assert not any(name.startswith("jinja2") for name in names)
        assert not any(name.startswith("typing") for name in names)
    installed = tmp_path / "installed"
    _run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--target",
            str(installed),
            str(wheel),
        ],
        tmp_path,
    )
    # Include real dependency directories from this interpreter, including an
    # explicitly shared environment. Do not execute .pth files or add source
    # roots. Preserve overlay precedence so the native dependency comes from
    # the same environment as the test. The child verifies every loaded FINN
    # module came from the wheel.
    dependency_paths = list(
        dict.fromkeys(
            str(Path(path).resolve()) for path in sys.path if Path(path).name == "site-packages"
        )
    )
    assert dependency_paths
    config = {
        "installed": str(installed),
        "finnlib": str(finnlib),
        "site_packages": dependency_paths,
        "output": str(tmp_path / "output"),
    }
    result = _run([sys.executable, "-I", "-S", "-c", INSTALLED_BUILD, json.dumps(config)], tmp_path)
    assert "installed dotp, external and memstream MatMul builds verified" in result.stdout
