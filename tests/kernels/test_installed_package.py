# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build and use the wheel without checkout imports, parked code or graph dependencies."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from finn import resources as finn_resources

ROOT = Path(__file__).resolve().parents[2]
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
        forbidden = (
            "finn.parked", "finn.custom_op.dataflow", "onnx",
            "finn.kernels.space", "finn.core.modelwrapper", "finn.core.onnx_exec",
            "finn.core.rtlsim_exec",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)
        if fullname.startswith("qonnx") and fullname not in (
            "qonnx", "qonnx.core", "qonnx.core.datatype"
        ):
            raise AssertionError("QONNX graph dependency: " + fullname)

sys.meta_path.insert(0, RejectGraphDependencies())

from finn.kernels import DspBlock, MatMulKernel, PackedDotpKernel
from finn.kernels.target import Platform
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.base import Kernel
from finn.kernels.configure import commit
from finn.kernels.streams import BufferedStream, Stream
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
assert not (installed / "finn/kernels/_engine").exists()
assert (installed / "finn/core/space/py.typed").is_file()
assert (installed / "finn/dataflow/py.typed").is_file()
assert not (installed / "finn/parked").exists()
assert "finn.dataflow.datatypes" in sys.modules
assert not (installed / "finn/kernels/space").exists()
assert not (installed / "finn/kernels/resources").exists()
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


class PlacedDotp(Space):
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


dotp = commit(
    design_space(PlacedDotp()),
    {"compute.pe": 2, "compute.simd": 2, "compute.compute_pumping": False},
).compute
answer = dotp.module
assert isinstance(answer, built.Leaf), answer
assert dict(answer.parameters)["ACCU_WIDTH"] == 8
assert dotp.x.element.dtype.name == "INT3"
assert dotp.x.axis.payload_bits == 6

dotp_sources = {
    "rtl/arith/add_multi_pkg.sv", "rtl/arith/add_multi.sv",
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
    choices = {"w.transport": "direct", "matmul.compute": "packed"}
    expected = dotp_sources | {"rtl/shape/input_gen.sv"}
    if memory == "memstream":
        # Stored (k, n): the columns of the by-output rows.
        facts["weights"] = ((-4, 0, 3, -1), (-3, 1, 2, -2), (-2, 2, 1, -3), (-1, 3, 0, -4))
        choices |= {
            "w.source.memstream.ram_style": "auto",
            "w.source.memstream.pumped_memory": False,
        }
        expected |= {"rtl/infra/axilite.sv", "rtl/infra/memstream.sv", "rtl/infra/memstream_axi.sv"}

    # The MatMul in a root that declares its streams.
    class Placed(Root):
        x = Stream(tensor=Tensor((3, 4), INT3), port="in0_V", platform=PLATFORM)
        w = BufferedStream(tensor=Tensor((4, 4), INT3), port="in1_V", platform=PLATFORM)
        y = Stream(tensor=Tensor((3, 4), INT8), port="out0_V", platform=PLATFORM)
        matmul = MatMulKernel(**facts, x_stream=x, w_stream=w, y_stream=y)
        w.contents = matmul.weight_values

    root = commit(design_space(Placed()), choices)
    root = commit(root, {
        "matmul.compute.packed.pe": 2,
        "matmul.compute.packed.simd": 2,
        "matmul.compute.packed.compute_pumping": False,
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
    assert ".ACCU_WIDTH(8)" in wrapper
    assert ".olst(n__u_x_adapter_input_gen_input_gen__olst)" in wrapper
    if memory == "memstream":
        assert '.INIT_FILE("memstream_' in wrapper
        assert root.w.source.image == (0x22C, 0x6BE, 0xDD3, 0x941)

# Catch namespace or editable-install leakage even if the import was permitted.
for name, module in tuple(sys.modules.items()):
    if name == "finn" or name.startswith("finn."):
        assert "._engine" not in name and "._next" not in name, name
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
            "finn/dataflow/datatypes.py",
            "finn/dataflow/traversal.py",
            "finn/kernels/artifacts/module.py",
        } <= set(archive.namelist())
        assert not any(
            name.startswith("finn/kernels/_engine/")
            or name.startswith("finn/kernels/space/")
            or name.startswith("finn/parked/")
            or name.startswith("finn/custom_op/dataflow/")
            or name.startswith("finn/kernels/resources/")
            or name.startswith("finn/kernels/physical/")
            or any(part.startswith("_next") for part in name.split("/"))
            for name in archive.namelist()
        )
        assert not any(name.startswith("finn/dataflow/artifacts/") for name in archive.namelist())
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
