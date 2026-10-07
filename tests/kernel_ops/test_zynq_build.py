# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""ZynqBuild over a model of KernelOps, up to its IP builds: the partitions it prepares,
the target it reads from the model, and the toolchain it hands to each transformation
that runs Vivado or Vitis HLS.

The Chain (``kernels.chain``), its choices saved, stated for Ultra96 in the Zynq shell, as
the KernelOps' model; no Vivado and no Vitis HLS runs (the bitstream build itself is not a
test).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

import pytest
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.kernels.base import KernelOpError, write_target
from finn.transformation.fpgadataflow import make_zynq_proj
from finn.transformation.fpgadataflow.create_stitched_ip import collect_ip_dirs
from finn.transformation.fpgadataflow.kernel_partitions import (
    PARTITION_INPUTS,
    PARTITION_OUTPUTS,
    partition_facts,
)
from finn.transformation.fpgadataflow.make_driver import get_driver_shapes
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.kernels import resolve_target
from finn.util import hls
from finn.util.toolchain import Selection, Toolchain
from finn.util.vivado import vivado_jobs
from kernel_ops.models import TARGET, configure_partition, kernel_model
from kernel_ops.packaging import ReachedVivado

#: Ultra96 in the Zynq shell: the target a Zynq build of Ultra96 at 5 ns reads.
ZYNQ = resolve_target(TARGET.part, 5.0, "vivado_zynq")


def zynq_model() -> ModelWrapper:
    """The Chain as KernelOps, stated for Ultra96 in the Zynq shell, its choices saved."""
    model = kernel_model()
    write_target(model, ZYNQ)
    configure_partition(model)
    return model


def test_a_model_of_kernel_ops_becomes_iodma_and_kernel_partitions(tmp_path: Path) -> None:
    model = zynq_model()
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
    # They describe that partition only: neither the model it was cut from, nor the
    # parent graph, nor an IODMA's body carries them.
    for other in (model, parent, bodies[0], bodies[2]):
        assert (other.get(PARTITION_INPUTS), other.get(PARTITION_OUTPUTS)) == (None, None)
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
        class Recorded(Transformation):
            def __init__(self, *args: object, toolchain: object = None, **kwargs: object):
                super().__init__()
                if name in TOOL_STEPS:
                    seen.append((name, toolchain))

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                return model, False

        return Recorded

    for name in replaced:
        monkeypatch.setattr(make_zynq_proj, name, recorder(name))
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path), toolchain=toolchain)
    return zynq_model().transform(build), seen


@pytest.mark.parametrize(
    "board, period_ns, refused",
    [
        ("Ultra96", 10.0, "period_ns: the model states 5.0, the build 10.0"),
        ("ZCU104", 5.0, "part: the model states 'xczu3eg-sbva484-1-e', the build 'xczu7ev-"),
    ],
)
def test_a_build_for_another_target_than_the_models_is_refused(
    tmp_path: Path, board: str, period_ns: float, refused: str
) -> None:
    build = ZynqBuild(board, period_ns, partition_model_dir=str(tmp_path))
    with pytest.raises(KernelOpError, match=refused):
        zynq_model().transform(build)


def test_a_model_stated_for_another_shell_is_refused(tmp_path: Path) -> None:
    # The Chain as kernel_model states it: no shell, so a doubled clock and one AXI-Lite
    # port its kernels may use, neither of which the Zynq shell gives a partition.
    model = kernel_model()
    configure_partition(model)
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path))
    with pytest.raises(KernelOpError, match="clk2x: the model states True, the build False"):
        model.transform(build)


def test_a_build_of_kernel_ops_runs_at_the_models_clock_period(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    periods: list[tuple[str, object]] = []

    def recorder(name: str) -> type[Transformation]:
        class Recorded(Transformation):
            def __init__(self, *args: object, **kwargs: object):
                super().__init__()
                if name in ("PrepareIP", "MakeZYNQProject"):
                    periods.append((name, args[1]))

            def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
                return model, False

        return Recorded

    for name in (*TOOL_STEPS, "PrepareIP"):
        monkeypatch.setattr(make_zynq_proj, name, recorder(name))
    build = ZynqBuild("Ultra96", None, partition_model_dir=str(tmp_path), toolchain=object())
    zynq_model().transform(build)
    assert periods == [("PrepareIP", 5.0), ("PrepareIP", 5.0), ("MakeZYNQProject", 5.0)]


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


def project_parent(tmp_path: Path, ifnames: str) -> ModelWrapper:
    """The Chain's parent graph as ZynqBuild prepares it, each partition stated as built
    (its IP project ``tmp_path``) with the interface names ``ifnames``."""
    build = ZynqBuild("Ultra96", 5.0, partition_model_dir=str(tmp_path))
    parent = build.prepare_kernel_partitions(zynq_model())
    for node in parent.graph.node:
        body_file = getCustomOp(node).get_nodeattr("model")
        body = ModelWrapper(body_file)
        for inner in body.get_nodes_by_op_type("IODMA_hls"):
            getCustomOp(inner).set_nodeattr("ip_path", str(tmp_path))
        body.set_metadata_prop("vivado_stitch_proj", str(tmp_path))
        body.set_metadata_prop("vivado_stitch_vlnv", "xilinx.com:hls:partition:1.0")
        body.set_metadata_prop("vivado_stitch_ifnames", ifnames)
        body.save(body_file)
    return parent


def test_the_project_reads_interface_names_as_json_and_never_executes_them(
    tmp_path: Path,
) -> None:
    """MakeZYNQProject reads each partition's ``vivado_stitch_ifnames`` as the JSON its
    writers write: a value that is Python, not JSON, is refused unexecuted."""
    executed = tmp_path / "executed"
    parent = project_parent(tmp_path, f"__import__('pathlib').Path({str(executed)!r}).touch()")
    with pytest.raises(json.JSONDecodeError):
        parent.transform(make_zynq_proj.MakeZYNQProject("Ultra96", 5.0, toolchain=object()))
    assert not executed.exists()


class ProjectScript:
    """A toolchain double for MakeZYNQProject: it keeps the project's Tcl and stops
    where Vivado would start."""

    def __init__(self) -> None:
        self.script = ""

    def run(self, tool: str, args: list[str], *, cwd: str, **options: object) -> None:
        self.script = (Path(cwd) / args[-1]).read_text()
        raise ReachedVivado(tool)


@pytest.mark.parametrize("jobs, expected", [(3, 3), (None, min(os.cpu_count() or 1, 16))])
def test_vivados_jobs_are_the_builds_not_an_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int | None, expected: int
) -> None:
    """The project launches its runs with the jobs it is given, by default the machine's
    cores, at most 16; NUM_DEFAULT_WORKERS, unset or set, is not read."""
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "1")
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    ifnames = json.dumps(
        {"axilite": ["s_axi_control"], "aximm": [], "s_axis": [], "m_axis": [], "clk": []}
    )
    parent = project_parent(tmp_path, ifnames)
    toolchain = ProjectScript()
    with pytest.raises(ReachedVivado):
        parent.transform(
            make_zynq_proj.MakeZYNQProject(
                "Ultra96", 5.0, toolchain=cast(Toolchain, toolchain), jobs=jobs
            )
        )
    assert f"launch_runs -to_step write_bitstream impl_1 -jobs {expected}\n" in toolchain.script


def test_a_number_of_jobs_that_is_none_is_refused() -> None:
    with pytest.raises(ValueError, match="positive number, not 0"):
        vivado_jobs(0)
