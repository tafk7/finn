# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ZynqBuild over a model of KernelOps, up to its IP builds: the partitions it prepares,
and the toolchain it hands to each transformation that runs Vivado or Vitis HLS.

test_design's Chain, its choices saved, as the KernelOps' model; no Vivado and no
Vitis HLS (the build itself: the TFC_W2A2 build script in the scratchpad's
records/zynq-kernel-build-2026-10-04).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow import make_zynq_proj
from finn.transformation.fpgadataflow.create_stitched_ip import collect_ip_dirs
from finn.transformation.fpgadataflow.kernel_partitions import partition_facts
from finn.transformation.fpgadataflow.make_driver import get_driver_shapes
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.util import hls
from finn.util.toolchain import Selection, Toolchain
from kernel_ops.test_partition import configured, kernel_model


def test_a_model_of_kernel_ops_becomes_iodma_and_kernel_partitions(tmp_path: Path) -> None:
    model = kernel_model()
    configured(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path))
    parent = build.prepare_kernel_partitions(model)
    bodies = [ModelWrapper(getCustomOp(node).get_nodeattr("model")) for node in parent.graph.node]
    assert [node.op_type for node in parent.graph.node] == ["StreamingDataflowPartition"] * 3
    assert [[node.op_type for node in body.graph.node] for body in bodies] == [
        ["IODMA_hls"],
        ["MatMul", "Thresholding", "MatMul"],
        ["IODMA_hls"],
    ]
    # The KernelOps' partition states its facts; the IODMAs' widths came from them.
    inputs, outputs = partition_facts(bodies[1])
    assert (inputs[0]["tdata"], outputs[0]["tdata"]) == (8, 16)
    assert getCustomOp(bodies[0].graph.node[0]).get_nodeattr("streamWidth") == 8
    assert getCustomOp(bodies[2].graph.node[0]).get_nodeattr("streamWidth") == 16
    assert get_driver_shapes(parent)["ishape_folded"] == [(1, 6, 2)]
    # The packaged IP is self-contained: the shell adds only its directory.
    assert collect_ip_dirs(bodies[1], "/stitch") == ["/stitch/ip"]


#: The steps of a build that run a tool, each given the build's toolchain.
TOOL_STEPS = ("HLSSynthIP", "PackagePartition", "CreateStitchedIP", "MakeZYNQProject")
#: The order they run in: per IODMA partition HLS synthesis, then its stitched
#: IP; the KernelOps' partition packaged; then the project.
TOOL_ORDER = [
    "HLSSynthIP",
    "CreateStitchedIP",
    "PackagePartition",
    "HLSSynthIP",
    "CreateStitchedIP",
    "MakeZYNQProject",
]


def recorded_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    toolchain: object,
    replaced: tuple[str, ...] = (*TOOL_STEPS, "PrepareIP"),
) -> tuple[ModelWrapper, list[tuple[str, object]]]:
    """ZynqBuild over the Chain, the ``replaced`` steps replaced by recorders of the
    toolchain each tool step is given: the parent model, and the order and
    toolchain of each recorded tool step."""
    seen: list[tuple[str, object]] = []

    def recorder(name: str) -> type[Transformation]:
        class Recorded(Transformation):  # type: ignore[misc]
            def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
                super().__init__()
                if name in TOOL_STEPS:
                    seen.append((name, toolchain))

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                return model, False

        return Recorded

    for name in replaced:
        monkeypatch.setattr(make_zynq_proj, name, recorder(name))
    model = kernel_model()
    configured(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path), toolchain=toolchain)
    return model.transform(build), seen


def test_a_build_runs_its_tools_through_the_toolchain_it_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    given = object()
    monkeypatch.setattr(make_zynq_proj, "legacy_toolchain", lambda: pytest.fail("prepared"))
    _, seen = recorded_build(monkeypatch, tmp_path, given)
    assert seen == [(name, given) for name in TOOL_ORDER]


def test_a_build_prepares_its_default_toolchain_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared: list[object] = []

    def legacy_toolchain() -> object:
        prepared.append(object())
        return prepared[-1]

    monkeypatch.setattr(make_zynq_proj, "legacy_toolchain", legacy_toolchain)
    _, seen = recorded_build(monkeypatch, tmp_path, None)
    assert len(prepared) == 1
    assert [name for name, _ in seen] == TOOL_ORDER
    assert all(toolchain is prepared[0] for _, toolchain in seen)


#: A Vitis HLS that reports 2024.2 and, given a node's script, makes the IP
#: directory HLSBackend checks for and notes that it ran.
FAKE_VITIS_HLS = """
import os, sys
if sys.argv[1] == "-version":
    print("Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.2 (64-bit)")
    sys.exit()
name = os.path.basename(sys.argv[2])[len("hls_syn_") : -len(".tcl")]
os.makedirs(f"project_{name}/sol1/impl/ip")
open("synthesized_here", "w").close()
"""


def legacy_refused() -> object:
    # An Exception, not pytest.fail: it is raised in a pool worker, which passes
    # an Exception back to the parent and dies on a BaseException.
    raise AssertionError("the legacy toolchain was prepared")


def test_hls_synthesis_runs_in_the_builds_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PrepareIP and HLSSynthIP run for real on the IODMAs' partitions, in two
    workers (the toolchain reaches them pickled), each node's synthesis by the
    build's toolchain: the only Vitis HLS on its PATH is a fake one."""
    tools = tmp_path / "tools"
    tools.mkdir()
    vitis_hls = tools / "vitis_hls"
    vitis_hls.write_text("#!" + sys.executable + "\n" + FAKE_VITIS_HLS)
    vitis_hls.chmod(0o755)
    toolchain = Toolchain(Selection(), {"PATH": f"{tools}:{os.defpath}"})
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "2")
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setattr(make_zynq_proj, "legacy_toolchain", legacy_refused)
    monkeypatch.setattr(hls, "legacy_toolchain", legacy_refused)
    parent, seen = recorded_build(
        monkeypatch,
        tmp_path / "partitions",
        toolchain,
        replaced=("PackagePartition", "CreateStitchedIP", "MakeZYNQProject"),
    )
    assert [name for name, _ in seen] == [name for name in TOOL_ORDER if name != "HLSSynthIP"]
    dmas = [
        getCustomOp(body.graph.node[0])
        for body in (
            ModelWrapper(getCustomOp(node).get_nodeattr("model")) for node in parent.graph.node
        )
        if body.graph.node[0].op_type == "IODMA_hls"
    ]
    assert len(dmas) == 2
    for dma in dmas:
        code = Path(dma.get_nodeattr("code_gen_dir_ipgen"))
        assert (code / "synthesized_here").is_file()
        assert dma.get_nodeattr("ipgen_path") == f"{code}/project_{dma.onnx_node.name}"
