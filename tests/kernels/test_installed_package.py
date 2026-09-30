# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build and use the wheel without checkout imports, parked code or graph dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from email.parser import BytesParser
import zipfile


ROOT = Path(__file__).resolve().parents[2]
# -I -S ignores PYTHONPATH, the current directory, user packages and .pth files.
# Only the wheel target, a copied QONNX dependency and ordinary dependency
# site-packages are added. No FINN source or test package is on this path.
INSTALLED_BUILD = r"""
import importlib.abc
import json
from hashlib import sha256
from pathlib import Path
import sys

config = json.loads(sys.argv[1])
installed = Path(config["installed"])
sys.path[:0] = [str(installed), config["qonnx"], *config["site_packages"]]
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

from finn.kernels import DspBlock, PackedDotpKernel, WeightDelivery, matmul_assembly
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.configure import commit
from finn.kernels.streams import Stream
from finn.core.space import (
    Available, Decision, Param, Space, design_space, derived, divisors_of, view,
)
import greenlet

assert greenlet.__version__ == "3.2.4"

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
from finn.kernels.artifacts import build, contributions, contribution_types, requirements
from finn.kernels.artifacts.manifest import decode
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.resources import resource_root, template_root
from qonnx.core.datatype import DataType

resources = resource_root()
assert resources == template_root()
assert resources.is_relative_to(installed)
assert (installed / "finn/kernels/py.typed").is_file()
assert not (installed / "finn/kernels/_engine").exists()
assert (installed / "finn/core/space/py.typed").is_file()
assert (installed / "finn/dataflow/py.typed").is_file()
assert not (installed / "finn/parked").exists()
assert "finn.dataflow.datatypes" in sys.modules
assert not (installed / "finn/kernels/space").exists()
assert not (resources / "dotp_axi.sv").exists()
assert build.ModuleBuildRequirements is requirements.ModuleBuildRequirements
assert contributions.CopiedSource is contribution_types.CopiedSource
assert requirements.ModuleBuildRequirements.__module__ == "finn.kernels.artifacts.build"
assert contribution_types.CopiedSource.__module__ == "finn.kernels.artifacts.contributions"

class PlacedDotp(Space):
    x = Stream(tensor=Tensor((1, 2), ScalarEncoding(DataType["INT3"])), port="in0_V")
    w = Stream(tensor=Tensor((2, 2), ScalarEncoding(DataType["INT3"])), port="in1_V")
    y = Stream(tensor=Tensor((1, 2), ScalarEncoding(DataType["INT8"])), port="out0_V")
    compute = PackedDotpKernel(
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        result_dtype=DataType["INT8"],
        x_stream=x,
        w_stream=w,
        y_stream=y,
    )


dotp = commit(
    design_space(PlacedDotp()),
    {"compute.pe": 2, "compute.simd": 2, "compute.compute_pumping": False},
).compute
answer = dotp.build_requirements
assert isinstance(answer, requirements.ModuleBuildRequirements), answer
assert dict(answer.parameters)["ACCU_WIDTH"] == 8
assert dotp.x.element.dtype.name == "INT3"
assert dotp.x.axis.payload_bits == 6

dotp_sources = {
    "rtl/arith/add_multi_pkg.sv", "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp.sv", "rtl/linalg/dotp_axi.sv",
}
store = ArtifactStore(Path(config["store"]))
roots = {"kernels": resources, "finnlib": Path(config["finnlib"])}

def materialize(module, expected):
    prepared = build.prepare_module_build(
        module, roots=roots, template_roots=(template_root(),), blobs=store,
    )
    assert prepared.slots == (), "an initializer was left unresolved"
    artifact = build.materialize_module_sources(prepared, store)
    build.portable_module_component(prepared, artifact)
    directory = Path(artifact.directory)
    manifest = decode((directory / "artifact.json").read_bytes())
    expected = expected | (
        {prepared.abi.entry_point + ".sv"} if prepared.abi.entry_point != "dotp_axi" else set()
    )
    assert set(artifact.files) == expected, artifact.files
    assert tuple(item.path for item in manifest.files) == artifact.files
    assert manifest.entry_points == (prepared.abi.entry_point,)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    assert actual == expected | {"artifact.json"}
    for item in manifest.files:
        data = (directory / item.path).read_bytes()
        assert len(data) == item.size
        assert sha256(data).hexdigest() == item.digest
    for source in module.contributions:
        if isinstance(source, contributions.CopiedSource):
            original = (roots[source.root] / source.path).read_bytes()
            assert (directory / source.path).read_bytes() == original
    return directory / (prepared.abi.entry_point + ".sv")

materialize(answer, dotp_sources)
for delivery in WeightDelivery:
    options = {}
    expected = dotp_sources | {"rtl/shape/input_gen.sv"}
    if delivery is not WeightDelivery.EXTERNAL:
        # Stored (k, n): the columns of the by-output rows.
        options["weights"] = [[-4, 0, 3, -1], [-3, 1, 2, -2], [-2, 2, 1, -3], [-1, 3, 0, -4]]
    if delivery is WeightDelivery.MEMSTREAM:
        expected |= {"rtl/infra/axilite.sv", "rtl/infra/memstream.sv", "rtl/infra/memstream_axi.sv"}
    assembly = matmul_assembly(
        m=3, k=4, n=4, pe=2, simd=2,
        activation_dtype=DataType["INT3"], weights_dtype=DataType["INT3"],
        target_dsp=DspBlock.DSP48E2, weight_delivery=delivery, **options,
    )
    assert (assembly.activation_beats, assembly.weight_beats, assembly.result_beats) == (6, 12, 6)
    # A memory image ships as generated data, named by its contents.
    expected |= {
        item.path
        for item in assembly.requirements.contributions
        if isinstance(item, contributions.GeneratedData)
    }
    wrapper = materialize(assembly.requirements, expected).read_text()
    assert ".ACCU_WIDTH(8)" in wrapper
    assert ".olst(n__u_activations_input_gen__olst)" in wrapper
    if delivery is WeightDelivery.MEMSTREAM:
        assert '.INIT_FILE("memstream_' in wrapper
    if delivery is WeightDelivery.EXTERNAL:
        assert assembly.initializer == ()
    else:
        assert assembly.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)

# Catch namespace or editable-install leakage even if the import was permitted.
for name, module in tuple(sys.modules.items()):
    if name == "finn" or name.startswith("finn."):
        assert "._engine" not in name and "._next" not in name, name
        location = getattr(module, "__file__", None)
        if location is not None:
            assert Path(location).resolve().is_relative_to(installed), (name, location)
        for location in getattr(module, "__path__", ()):
            assert Path(location).resolve().is_relative_to(installed), (name, location)
print("installed dotp, external, cyclic and memstream MatMul manifests verified")
"""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_wheel_materializes_independent_kernel_builds(tmp_path: Path) -> None:
    finnlib = Path(os.environ.get("FINNLIB_ROOT", str(ROOT / "deps/finnlib"))).resolve()
    qonnx = (ROOT / "deps/qonnx").resolve()
    # Build from a clean temporary source snapshot: setuptools must neither
    # reuse a checkout build tree nor write generated metadata into the checkout.
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    for name in ("setup.py", "setup.cfg", "README.md"):
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
            "finn/kernels/resources/decomposed_wrapper.sv.j2",
        } <= set(archive.namelist())
        assert not any(
            name.startswith("finn/kernels/_engine/")
            or name.startswith("finn/kernels/space/")
            or name.startswith("finn/parked/")
            or name.startswith("finn/custom_op/dataflow/")
            or any(part.startswith("_next") for part in name.split("/"))
            for name in archive.namelist()
        )
        assert not any(name.startswith("finn/dataflow/artifacts/") for name in archive.namelist())
        (metadata_name,) = (
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        assert "greenlet==3.2.4" in {value.replace(" ", "") for value in requirements}
        assert any(value.startswith("typing_extensions") for value in requirements)
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
    dependency_root = tmp_path / "dependency"
    shutil.copytree(qonnx / "src/qonnx", dependency_root / "qonnx")
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
        "qonnx": str(dependency_root),
        "finnlib": str(finnlib),
        "site_packages": dependency_paths,
        "store": str(tmp_path / "store"),
    }
    result = _run([sys.executable, "-I", "-S", "-c", INSTALLED_BUILD, json.dumps(config)], tmp_path)
    assert (
        "installed dotp, external, cyclic and memstream MatMul manifests verified" in result.stdout
    )
