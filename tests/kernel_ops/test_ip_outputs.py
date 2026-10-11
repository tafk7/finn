# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 - REPORT is a Vivado table, its rows as wide as Vivado writes them

"""The ``ip`` shell's outputs: the packaged IP's interface description, its resources
per member out of context, and its XSim testbench.

The Chain (``kernel_ops.models``) is packaged against a toolchain double that stands
for Vivado (``kernel_ops.packaging.PackagedByStub``), so its description and its
resources from a report are read in the fast gate, and so is the step that writes the
testbench and runs it only when the verification step is asked. TFC_W2A2 is built
through the builder on the ``ip`` shell to its packaged IP and testbench, which the
verification step runs and which then runs on its own (marker ``xsim``; it packages
with Vivado too).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from kernels.xsim import requires_xsim
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder import build_dataflow_steps
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.builder.build_dataflow_steps import ExploreOptions, step_create_stitched_ip
from finn.builder.build_dataflow_testbench import RUN_SCRIPT, TESTBENCH_DIR, run_testbench
from finn.core.rtlsim_bench import SimulationFailed
from finn.custom_op.kernels.shell import configured_root, shell_root
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_IP,
    partition_body,
)
from finn.kernels.artifacts.build import instance_name
from finn.kernels.artifacts.interface import STREAM_FACTS
from finn.kernels.utilization import Resources
from finn.platform import TargetRequest
from finn.toolchain import xsim
from finn.toolchain.launcher import (
    Selection,
    Toolchain,
    ToolFailed,
    ToolUnavailable,
    host_toolchain,
)
from finn.toolchain.vivado import hierarchical_utilization
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import (
    boundary_facts,
    free_side,
    ooc_member_resources,
    resources_of,
    stream_order,
)
from kernel_ops.models import configure_partition, kernel_model
from kernel_ops.packaging import PackagedByStub, read_back
from kernel_ops.tfc import SHAPE


def test_the_description_beside_the_ip_reads_back_against_the_modules_pins(
    tmp_path: Path,
) -> None:
    configure_partition(model := kernel_model())
    project = tmp_path / "project"
    stub = cast(Toolchain, PackagedByStub())
    model = model.transform(PackagePartition("sdp_1", directory=project, toolchain=stub))
    described = json.loads((project / "interface.json").read_text())
    point, boundary = configured_root(model, "sdp_1")
    read_back(described, point.module.abi.pins, 5.0)
    assert described["ip"] == {
        "name": "sdp_1",
        "vlnv": "xilinx_finn:finn:sdp_1:1.0",
        "top": described["ip"]["top"],
    }
    assert described["ip"]["top"].startswith("finn_partition__")
    assert (described["part"], described["period_ns"]) == ("xczu3eg-sbva484-1-e", 5.0)
    # Each stream states its port's boundary facts, as the configured root's channels
    # state them, and the order its free side presents the tensor in: the Chain's,
    # row-major.
    inputs, outputs = boundary_facts(model, point, boundary, "sdp_1")
    keys = [key for key in (*STREAM_FACTS, "tdata") if key != "order"]
    facts = [{key: port[key] for key in keys} for port in inputs + outputs]
    stated = [{key: stream[key] for key in keys} for stream in described["streams"]]
    assert stated == facts
    for stream, port in zip(described["streams"], inputs + outputs, strict=True):
        assert stream["order"] == stream_order(free_side(point, port["tensor"]).form)
        assert stream["order"]["row_major"] is True
    assert [(stream["name"], stream["direction"]) for stream in described["streams"]] == [
        ("s_axis_0", "in"),
        ("m_axis_0", "out"),
    ]
    assert (described["axilite"], described["aximm"]) == ([], [])


def test_the_description_states_each_streams_passes_a_frame(tmp_path: Path) -> None:
    """On ``ip``, whose integrator moves the streams, the Chain's streamed w2 crosses the
    boundary once per row of x, three times a frame, which no end refuses there: the
    description states each stream's passes (``order``), so that its integrator supplies
    w2 three times (SZ11 (f))."""
    configure_partition(model := kernel_model(second_weights=False))
    stub = cast(Toolchain, PackagedByStub())
    model.transform(PackagePartition("sdp_1", directory=tmp_path, toolchain=stub))
    described = json.loads((tmp_path / "interface.json").read_text())
    assert [
        (stream["name"], stream["tensor"], stream["beats"], stream["order"]["passes"])
        for stream in described["streams"]
    ] == [("s_axis_0", "x", 6, 1), ("s_axis_1", "w2", 12, 3), ("m_axis_0", "y", 6, 1)]


#: Vivado's hierarchical utilization table, an UltraScale+ part (its columns, Vivado
#: 2025.2; the cells narrowed); two levels below the top.
REPORT = """\
1. Utilization by Hierarchy
---------------------------

+------------+--------+------------+------------+---------+------+-----+--------+--------+------+------------+
|  Instance  | Module | Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs | RAMB36 | RAMB18 | URAM | DSP Blocks |
+------------+--------+------------+------------+---------+------+-----+--------+--------+------+------------+
| top        |  (top) |        120 |        100 |      12 |    8 |  90 |      1 |      1 |    0 |          2 |
|   u_first  |  first |         70 |         60 |       6 |    4 |  50 |      1 |      0 |    0 |          2 |
|     inner  |  inner |         30 |         30 |       0 |    0 |  10 |      0 |      0 |    0 |          0 |
|   u_second | second |         50 |         40 |       6 |    4 |  40 |      0 |      1 |    0 |          0 |
+------------+--------+------------+------------+---------+------+-----+--------+--------+------+------------+
"""


def test_the_hierarchical_report_is_read_by_depth_in_the_devices_units() -> None:
    rows = hierarchical_utilization(REPORT)
    assert rows[0] == (
        0,
        "top",
        {"lut": 120, "ff": 90, "bram36": 1, "bram18": 1, "uram": 0, "dsp": 2},
    )
    assert [(depth, name, resources_of(counts)) for depth, name, counts in rows] == [
        (0, "top", Resources(lut=120, ff=90, bram18=3, dsp=2)),
        (1, "u_first", Resources(lut=70, ff=50, bram18=2, dsp=2)),
        (2, "inner", Resources(lut=30, ff=10)),
        (1, "u_second", Resources(lut=50, ff=40, bram18=1)),
    ]


def test_out_of_context_resources_are_stated_per_member_of_the_shell_root(
    tmp_path: Path,
) -> None:
    # Its adapters' memories at their baselines: the module is the one packaged.
    configure_partition(model := kernel_model())
    point, _ = configured_root(model, "chain")
    labels = [label for label, _ in point.module.fragment.instances]
    # Each instance 10 LUTs and its index in FFs; synthesis flattened the last into the
    # top, which so has no row for it.
    rows = [f"| finn_partition | (top) | {len(labels) * 10} | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 |"]
    rows += [
        f"|   {instance_name(label)} | m | 10 | 0 | 0 | 0 | {index} | 0 | 0 | 0 | 0 |"
        for index, label in enumerate(labels[:-1])
    ]
    header = (
        "| Instance | Module | Total LUTs | Logic LUTs | LUTRAMs | SRLs | FFs | RAMB36 | RAMB18 "
        "| URAM | DSP Blocks |"
    )
    (tmp_path / "finn_partition_partition_util.rpt").write_text("\n".join([header, *rows]))
    found = ooc_member_resources(model, tmp_path)
    assert found["total"] == {"lut": 70, "ff": 21, "bram18": 0, "uram": 0, "dsp": 0}
    # Each instance is its member's: a boundary channel's adapter the channel's (x), a
    # weight's memory its channel's, a kernel's compute the kernel's.
    assert {path: counted["ff"] for path, counted in found["members"].items()} == {
        "x": 0,
        "w1": 1,
        "levels": 2,
        "w2": 3,
        "first": 4,
        "activate": 5,
    }
    assert set(found["members"]) <= set(shell_root(model, model.graph.node).members)
    # The flattened instance's resources are the top's, attributed to no member.
    assert found["unattributed"] == {"lut": 10, "ff": 6, "bram18": 0, "uram": 0, "dsp": 0}
    assert found["unreported_instances"] == ["second.compute.packed"]


def test_the_testbench_is_an_output_run_only_by_its_verification_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """STITCHED_IP writes the testbench beside the IP and simulates nothing (PRINCIPLES
    §8: no build re-verifies the computation); STITCHED_IP_TESTBENCH, asked, runs it
    once, in the build's toolchain."""
    ran: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        build_dataflow_steps,
        "run_testbench",
        lambda directory, *, toolchain: ran.append((directory, toolchain)),
    )
    parent = kernel_model().transform(CutKernelPartition(tmp_path / "partition"))
    stub = PackagedByStub()
    for verify_steps in ([], [VerificationStepType.STITCHED_IP_TESTBENCH]):
        output = tmp_path / f"output_{len(verify_steps)}"
        cfg = DataflowBuildConfig(
            output_dir=str(output),
            target=TargetRequest(part="xczu3eg-sbva484-1-e", period_ns=5.0),
            generate_outputs=[DataflowOutputType.STITCHED_IP],
            verify_steps=verify_steps,
            verify_input_npy=str(tmp_path / "absent.npy"),
        )
        cfg._toolchain = cast(Toolchain, stub)
        step_create_stitched_ip(parent, cfg)
        testbench = output / "stitched_ip" / TESTBENCH_DIR
        assert (testbench / "check.sv").read_text().startswith("`timescale 1ns/1ps\n")
        assert (testbench / RUN_SCRIPT).is_file()
        assert not (testbench / "xsim.log").exists()
        assert ran == ([(testbench, stub)] if verify_steps else [])


def test_a_testbench_runs_by_its_script_and_a_mismatch_fails(tmp_path: Path) -> None:
    """``run_testbench`` runs the directory's run.sh as its user would, the toolchain's
    command directory first on PATH, and returns what it printed when it ends in PASS;
    anything else is ``SimulationFailed`` with what it printed. A launcher route runs
    no local script."""
    tools = tmp_path / "tools"
    tools.mkdir()
    simulator = tools / "fake_xsim"
    simulator.write_text('#!/bin/sh\necho "STREAM_$1"\n')
    simulator.chmod(0o755)
    toolchain = Toolchain(Selection(command_dir=str(tools)), {"PATH": os.defpath})
    for verdict in ("PASS", "FAIL"):
        bench = tmp_path / verdict
        bench.mkdir()
        (bench / RUN_SCRIPT).write_text(
            f"set -e\nfake_xsim {verdict} > xsim.log\ngrep -q STREAM_PASS xsim.log\necho PASS\n"
        )
        if verdict == "PASS":
            assert run_testbench(bench, toolchain=toolchain) == "PASS\n"
        else:
            with pytest.raises(SimulationFailed):
                run_testbench(bench, toolchain=toolchain)
        assert (bench / "xsim.log").read_text() == f"STREAM_{verdict}\n"
    remote = Toolchain(Selection(launcher=("site",)), {"PATH": os.defpath})
    with pytest.raises(ValueError, match="launcher route runs tools elsewhere"):
        run_testbench(tmp_path / "PASS", toolchain=remote)


def test_a_steps_tools_write_their_transcripts_into_its_log_directory(tmp_path: Path) -> None:
    """step_create_stitched_ip passes the build's toolchain down with its log directory,
    ``output_dir/logs/step_create_stitched_ip``: packaging's Vivado writes its transcript
    there, not beside its replay in the packaging project."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "vivado").write_text(
        "#!/bin/sh\necho 'INFO: [IP_Flow 19-234] packaging'\nmkdir -p ip\n"
        "echo '<stub/>' > ip/component.xml\n"
    )
    (tools / "vivado").chmod(0o755)
    output = tmp_path / "output"
    cfg = DataflowBuildConfig(
        output_dir=str(output),
        target=TargetRequest(part="xczu3eg-sbva484-1-e", period_ns=5.0),
        generate_outputs=[DataflowOutputType.STITCHED_IP],
        verify_input_npy=str(tmp_path / "absent.npy"),
    )
    cfg._toolchain = Toolchain(Selection(command_dir=str(tools)), {"PATH": os.defpath})
    step_create_stitched_ip(kernel_model().transform(CutKernelPartition(tmp_path / "p")), cfg)
    logs = output / "logs" / "step_create_stitched_ip"
    assert sorted(path.name for path in logs.iterdir()) == [
        "package.sh.stderr.log",
        "package.sh.stdout.log",
    ]
    assert (logs / "package.sh.stdout.log").read_text() == "INFO: [IP_Flow 19-234] packaging\n"
    assert (output / "stitched_ip" / "package.sh").is_file()
    assert not (output / "stitched_ip" / "package.sh.stdout.log").exists()
    # The build's own toolchain is unchanged: each step states its directory.
    assert cfg._resolve_toolchain().transcripts is None


def test_a_testbench_without_its_simulator_is_the_tools_absence(tmp_path: Path) -> None:
    """run.sh's guard (``finn.toolchain.xsim.UNAVAILABLE_STATUS``), before it runs
    anything: an environment that names no Vivado, or one whose PATH lacks a command of
    the simulator, is ``ToolUnavailable``, saying which, not a tool's failure."""
    (tmp_path / RUN_SCRIPT).write_text(xsim.script(["a.sv"], "top", snapshot="s", threads="1"))
    bare = Toolchain(Selection(), {"PATH": os.defpath})
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO is not set"):
        run_testbench(tmp_path, toolchain=bare)
    named = Toolchain(Selection(), {"PATH": os.defpath, "XILINX_VIVADO": str(tmp_path)})
    with pytest.raises(ToolUnavailable, match="xvlog is not on PATH") as missing:
        run_testbench(tmp_path, toolchain=named)
    assert missing.value.tool == "xsim"
    assert not (tmp_path / "xvlog.log").exists()


@pytest.mark.parametrize("tool", ["xvlog", "xelab"])
def test_a_testbench_that_does_not_build_is_the_tools_failure(tmp_path: Path, tool: str) -> None:
    """A run.sh whose compilation or elaboration fails exits with that tool's status
    (``finn.toolchain.xsim.EXIT_STATUS``): ``ToolFailed`` naming the tool and the run's
    transcript (beside run.sh; in a build's step, in its log directory), not a verdict
    on the partition."""
    (tmp_path / RUN_SCRIPT).write_text(
        f'echo "ERROR: [{tool}] it failed" | tee {tool}.log\nexit {xsim.EXIT_STATUS[tool]}\n'
    )
    toolchain = Toolchain(Selection(), {"PATH": os.defpath})
    logs = tmp_path / "logs" / "step_create_stitched_ip"
    for tools, transcript in (
        (toolchain, tmp_path / "run.sh.stdout.log"),
        (toolchain.transcripts_in(logs), logs / "run.sh.stdout.log"),
    ):
        with pytest.raises(ToolFailed) as failed:
            run_testbench(tmp_path, toolchain=tools)
        assert not isinstance(failed.value, AssertionError)
        assert (failed.value.tool, failed.value.status) == (tool, xsim.EXIT_STATUS[tool])
        assert failed.value.transcript == transcript
        assert failed.value.errors == (f"ERROR: [{tool}] it failed",)


@requires_xsim
def test_tfc_on_the_ip_shell_packages_and_its_testbench_passes_on_its_own(
    tmp_path: Path, tfc_streamlined: Path
) -> None:
    """TFC_W2A2 through the builder on ``ip`` (Ultra96's part at 5 ns, the Z0 baseline's
    exploration), STITCHED_IP asked and STITCHED_IP_TESTBENCH: the IP packages, its
    description reads back against the module's pins, the verification step runs the
    testbench written beside it, on the first image of verify_input_npy, and it prints
    PASS when run by its own script."""
    source_file = tmp_path / "streamlined.onnx"
    shutil.copyfile(tfc_streamlined, source_file)
    images = np.random.default_rng(3).integers(0, 256, size=(2, *SHAPE[1:])).astype(np.float32)
    np.save(tmp_path / "input.npy", images)
    output = tmp_path / "output"
    cfg = DataflowBuildConfig(
        output_dir=str(output),
        target=TargetRequest(board="Ultra96", period_ns=5.0),
        generate_outputs=[DataflowOutputType.STITCHED_IP],
        step_options={
            "step_explore": ExploreOptions(
                kernel_exploration=[
                    {"strategy": "target_throughput", "fps": 1e6},
                    {"strategy": "size_fifos"},
                ]
            )
        },
        verify_steps=[VerificationStepType.STITCHED_IP_TESTBENCH],
        verify_input_npy=str(tmp_path / "input.npy"),
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source_file), cfg) == 0
    log = (output / "build_dataflow.log").read_text()
    assert "Verification for stitched_ip_testbench : SUCCESS" in log
    ip = output / "stitched_ip"
    assert (ip / "ip" / "component.xml").is_file()
    # The partition the build packaged, opened through the parent graph the step saved.
    parent = ModelWrapper(str(output / "intermediate_models" / "step_create_stitched_ip.onnx"))
    _, body, _ = partition_body(parent)
    point, _ = configured_root(body, "partition")
    described = json.loads((ip / "interface.json").read_text())
    read_back(described, point.module.abi.pins, 5.0)
    # The IP is the partition's, of its one name; the body states it.
    assert described["ip"]["name"] == "partition"
    assert body.get(OUTPUT_IP) == str(ip / "ip")
    # The exploration folds the input's thresholds at 4 lanes and the last layer at 1
    # (Z0's choices), not test_tfc's 16 lanes by hand.
    assert [(s["name"], s["tdata"], s["element"]) for s in described["streams"]] == [
        ("s_axis_0", 32, "UINT8"),
        ("m_axis_0", 8, "INT8"),
    ]
    # The testbench runs from its own directory with Vivado's tools on PATH and
    # XILINX_VIVADO set, nothing else of the machine.
    testbench = ip / TESTBENCH_DIR
    assert str(tmp_path) not in (testbench / "run.sh").read_text()
    environment = dict(host_toolchain().environment)
    vivado = environment["XILINX_VIVADO"]
    environment["PATH"] = f"{vivado}/bin:{os.environ['PATH']}"
    ran = subprocess.run(
        ["sh", str(testbench / "run.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert ran.stdout.splitlines()[-1] == "PASS"
