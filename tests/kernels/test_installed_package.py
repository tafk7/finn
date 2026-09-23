# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build and use the wheel without checkout imports or dataflow dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile


ROOT = Path(__file__).resolve().parents[2]
FINNLIB_REVISION = "dfeafac81cd2a6da27e647ee03915ade5532186e"
QONNX_REVISION = "21d4c1a72334002aaf80ae099dbf5d167f001001"
CORRECTED_WRAPPER_SHA256 = "466397881624b4d0a4884762700620d548271541dc97bb265ea4b80129a5cd90"

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
        forbidden = ("finn.dataflow", "finn.custom_op.dataflow", "onnx")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("forbidden dependency: " + fullname)
        if fullname.startswith("qonnx") and fullname not in (
            "qonnx", "qonnx.core", "qonnx.core.datatype"
        ):
            raise AssertionError("QONNX graph dependency: " + fullname)

sys.meta_path.insert(0, RejectGraphDependencies())

from finn.kernels import DotpAxiKernel, DspBlock, WeightDelivery, mvau_assembly
from finn.kernels._engine import Decided
from finn.kernels.artifacts import build, contributions, contribution_types, requirements
from finn.kernels.artifacts.manifest import decode
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_CODEC, QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.resources import resource_root, template_root
from finn.kernels.space import Problem, Space, Subspace
from qonnx.core.datatype import DataType

resources = resource_root()
assert resources == template_root()
assert resources.is_relative_to(installed)
assert (installed / "finn/kernels/py.typed").is_file()
assert sha256((resources / "dotp_axi.sv").read_bytes()).hexdigest() == config["wrapper_sha256"]
assert build.ModuleBuildRequirements is requirements.ModuleBuildRequirements
assert contributions.CopiedSource is contribution_types.CopiedSource
assert requirements.ModuleBuildRequirements.__module__ == "finn.kernels.artifacts.build"
assert contribution_types.CopiedSource.__module__ == "finn.kernels.artifacts.contributions"

class DotpRequest(Space):
    pe = Problem(int)
    simd = Problem(int)
    dtype = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    result = Problem(QONNX_DATATYPE_VALUE_SEMANTICS, canonical=QONNX_DATATYPE_CODEC)
    target = Problem(DspBlock)
    segment = Problem(int)
    compute = Subspace(
        DotpAxiKernel, pe=pe, simd=simd, activation_dtype=dtype, weights_dtype=dtype,
        result_dtype=result, target_dsp=target, segment_length=segment,
    )

dotp = DotpRequest.start({
    DotpRequest.pe: 2, DotpRequest.simd: 2, DotpRequest.dtype: DataType["INT3"],
    DotpRequest.result: DataType["INT8"], DotpRequest.target: DspBlock.DSP48E2,
    DotpRequest.segment: 0,
}).compute.assign(DotpAxiKernel.compute_pumping, False)
answer = dotp.physical.accepted_answer
assert isinstance(answer, Decided), answer
assert dict(answer.value.parameters)["ACCU_WIDTH"] == 8

dotp_sources = {
    "rtl/arith/add_multi_pkg.sv", "rtl/arith/add_multi.sv",
    "rtl/linalg/dotp_8sx9_dsp58.sv", "rtl/linalg/dotp.sv", "dotp_axi.sv",
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
    assert sha256((directory / "dotp_axi.sv").read_bytes()).hexdigest() == config["wrapper_sha256"]
    for source in module.contributions:
        if isinstance(source, contributions.CopiedSource):
            original = (roots[source.root] / source.path).read_bytes()
            assert (directory / source.path).read_bytes() == original
    return directory / (prepared.abi.entry_point + ".sv")

materialize(answer.value, dotp_sources)
for delivery in WeightDelivery:
    options = {}
    expected = dotp_sources | {"rtl/infra/replay_buffer.sv"}
    if delivery is WeightDelivery.CYCLIC:
        options["weights"] = [[-4, -3, -2, -1], [0, 1, 2, 3], [3, 2, 1, 0], [-1, -2, -3, -4]]
        expected |= {"cyclic_stream.sv"}
    assembly = mvau_assembly(
        repetitions=3, matrix_width=4, matrix_height=4, pe=2, simd=2,
        activation_dtype=DataType["INT3"], weights_dtype=DataType["INT3"],
        target_dsp=DspBlock.DSP48E2, weight_delivery=delivery, **options,
    )
    assert (assembly.activation_beats, assembly.weight_beats, assembly.result_beats) == (6, 12, 6)
    wrapper = materialize(assembly.requirements, expected).read_text()
    assert ".ACCU_WIDTH(8)" in wrapper
    assert ".olast(n__u_replay__olast)" in wrapper
    if delivery is WeightDelivery.CYCLIC:
        assert assembly.initializer == (0x22C, 0x6BE, 0xDD3, 0x941)
        assert ".INIT_DATA(48'h941dd36be22c)" in wrapper
    else:
        assert assembly.initializer == ()

# Catch namespace or editable-install leakage even if the import was permitted.
for name, module in tuple(sys.modules.items()):
    if name == "finn" or name.startswith("finn."):
        location = getattr(module, "__file__", None)
        if location is not None:
            assert Path(location).resolve().is_relative_to(installed), (name, location)
        for location in getattr(module, "__path__", ()):
            assert Path(location).resolve().is_relative_to(installed), (name, location)
print("installed dotp, external MVAU and cyclic MVAU manifests verified")
"""


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_wheel_materializes_independent_kernel_builds(tmp_path: Path) -> None:
    finnlib = Path(os.environ.get("FINNLIB_ROOT", str(ROOT / "deps/finnlib"))).resolve()
    qonnx = (ROOT / "deps/qonnx").resolve()
    for dependency, revision in ((finnlib, FINNLIB_REVISION), (qonnx, QONNX_REVISION)):
        assert _run(["git", "rev-parse", "HEAD"], dependency).stdout.strip() == revision

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
            "finn/kernels/_engine/py.typed",
            "finn/kernels/resources/dotp_axi.sv",
            "finn/kernels/resources/cyclic_stream.sv",
            "finn/kernels/resources/decomposed_wrapper.sv.j2",
        } <= set(archive.namelist())
        assert not any(name.startswith("finn/dataflow/artifacts/") for name in archive.namelist())
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
    paths = sysconfig.get_paths()
    config = {
        "installed": str(installed),
        "qonnx": str(dependency_root),
        "finnlib": str(finnlib),
        "site_packages": sorted({paths["purelib"], paths["platlib"]}),
        "store": str(tmp_path / "store"),
        "wrapper_sha256": CORRECTED_WRAPPER_SHA256,
    }
    result = _run([sys.executable, "-I", "-S", "-c", INSTALLED_BUILD, json.dumps(config)], tmp_path)
    assert "installed dotp, external MVAU and cyclic MVAU manifests verified" in result.stdout
