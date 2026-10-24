# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 - REPORT is a Vivado table, its rows as wide as Vivado writes them

"""The ``ip`` shell's outputs: the packaged IP's interface description, its resources
per member out of context, and its XSim testbench.

The Chain (``support.models``) is packaged against a toolchain double that stands
for Vivado (``support.packaging.PackagedByStub``), so its description and its
resources from a report are read in the fast gate, and so are the step that writes the
testbench and the utility that runs it. TFC_W2A2 is built through the builder on the
``ip`` shell to its packaged IP and testbench, which ``run_testbench`` runs and which
then runs on its own (marker ``xsim``; it packages with Vivado too).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.ranges import TensorRange
from support.configure import configured
from support.models import (
    ROW_MAJOR_W2,
    choose_memories,
    configure_partition,
    kernel_model,
    streamed_w2_model,
)
from support.packaging import PackagedByStub, read_back, verified, verified_parent
from support.xsim import requires_xsim

import finn.kernels.explore as explore_module
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_manifest import MANIFEST_FILE
from finn.builder.build_dataflow_resources import built_reports
from finn.builder.build_dataflow_steps import step_package_partition
from finn.builder.build_dataflow_testbench import (
    RUN_SCRIPT,
    TESTBENCH_DIR,
    generated_frame,
    run_testbench,
    write_testbench,
)
from finn.core.graph.annotations.choices import HELD
from finn.core.graph.annotations.outputs import OUTPUT_IP, OUTPUT_IP_KEY, OUTPUT_VERIFIED
from finn.core.graph.annotations.platform import write_target
from finn.core.graph.errors import KernelOpError
from finn.core.graph.partition import design_digest, partition_body
from finn.core.reporting.findings import FindingKind
from finn.core.reporting.manifest import STORED
from finn.core.rtlsim_bench import SimulationFailed
from finn.custom_op.kernels.network import network
from finn.custom_op.kernels.shell import configured_network, verify_partition
from finn.kernels.artifacts.build import instance_name
from finn.kernels.artifacts.interface import STREAM_FACTS
from finn.platform.capabilities import Resources
from finn.platform.request import TargetRequest
from finn.platform.resolve import resolve_target
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
from finn.transformation.kernels.integration import integration
from finn.transformation.kernels.package import (
    boundary_facts,
    free_side,
    ooc_member_resources,
    resources_of,
    stream_order,
)


def test_the_description_beside_the_ip_reads_back_against_the_modules_pins(
    tmp_path: Path,
) -> None:
    configure_partition(model := kernel_model())
    project = tmp_path / "project"
    stub = cast(Toolchain, PackagedByStub())
    model = model.transform(PackagePartition("sdp_1", directory=project, toolchain=stub))
    described = json.loads((project / "interface.json").read_text())
    point, boundary = configured_network(model, "sdp_1")
    read_back(described, point.module.abi.pins, 5.0)
    assert described["ip"] == {
        "name": "sdp_1",
        "vlnv": "xilinx_finn:finn:sdp_1:1.0",
        "top": described["ip"]["top"],
    }
    assert described["ip"]["top"].startswith("finn_partition__")
    assert (described["part"], described["period_ns"]) == ("xczu3eg-sbva484-1-e", 5.0)
    # Each stream states its port's boundary facts, as the configured network's channels
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
    """On ``ip``, whose integrator moves the streams and states nothing of them, the
    Chain's streamed w2 crosses the boundary once per row of x, three times a frame: the
    description states each stream's passes (``order``), so that its integrator supplies
    w2 three times."""
    configure_partition(model := kernel_model(second_weights=False))
    stub = cast(Toolchain, PackagedByStub())
    model.transform(PackagePartition("sdp_1", directory=tmp_path, toolchain=stub))
    described = json.loads((tmp_path / "interface.json").read_text())
    assert [
        (stream["name"], stream["tensor"], stream["beats"], stream["order"]["passes"])
        for stream in described["streams"]
    ] == [("s_axis_0", "x", 6, 1), ("s_axis_1", "w2", 12, 3), ("m_axis_0", "y", 6, 1)]


def test_on_pynq_the_description_states_what_the_movers_state(tmp_path: Path) -> None:
    """In the Zynq shell each mover states its side row-major, one pass a frame, so the
    description states each stream's order as stated: the same Chain, its w2 read
    row-major (``ROW_MAJOR_W2``), crosses the boundary once a frame, 4 beats, and an
    ``input_gen`` in the partition replays it; every stream row-major, one pass."""
    model = kernel_model(second_weights=False)
    configured(model, {"second": ROW_MAJOR_W2})
    write_target(model, ZYNQ)
    configure_partition(model)
    stub = cast(Toolchain, PackagedByStub())
    model.transform(PackagePartition("sdp_1", directory=tmp_path, toolchain=stub))
    described = json.loads((tmp_path / "interface.json").read_text())
    assert [
        (
            stream["name"],
            stream["beats"],
            stream["order"]["row_major"],
            stream["order"]["passes"],
        )
        for stream in described["streams"]
    ] == [("s_axis_0", 6, True, 1), ("s_axis_1", 4, True, 1), ("m_axis_0", 3, True, 1)]


def test_the_frame_draws_each_input_from_its_range() -> None:
    """An input stating a range is drawn from it; one stating none, from its datatype's."""
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 512]) for name in "ab"]
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 512])
    graph = helper.make_graph([helper.make_node("Add", ["a", "b"], ["y"])], "g", inputs, [out])
    body = ModelWrapper(helper.make_model(graph))
    body.set_tensor_datatype("a", DataType["INT8"], TensorRange(-3, 2))
    body.set_tensor_datatype("b", DataType["UINT4"])
    frame = generated_frame(body)
    assert set(np.unique(frame["a"]).tolist()) == set(range(-3, 3))
    assert set(np.unique(frame["b"]).tolist()) == set(range(16))


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


def test_out_of_context_resources_are_stated_per_member_of_the_network(
    tmp_path: Path,
) -> None:
    # Its adapters' memories at their baselines: the module is the one packaged.
    configure_partition(model := kernel_model())
    point, _ = configured_network(model, "chain")
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
    assert set(found["members"]) <= set(network(model, model.graph.node).members)
    # The flattened instance's resources are the top's, attributed to no member.
    assert found["unattributed"] == {"lut": 10, "ff": 6, "bram18": 0, "uram": 0, "dsp": 0}
    assert found["unreported_instances"] == ["second.compute.packed"]


class SynthesizedByStub(PackagedByStub):
    """``PackagedByStub``, and with ``run_synth`` the hierarchical utilization report the
    script writes beside the IP (``REPORT``)."""

    def run(
        self, tool: str, args: object, *, cwd: Path, **options: object
    ) -> subprocess.CompletedProcess[bytes]:
        if "launch_runs synth_1" in (Path(cwd) / "package.tcl").read_text():
            (Path(cwd) / "finn_partition_partition_util.rpt").write_text(REPORT)
        return super().run(tool, args, cwd=cwd, **options)


def test_the_resources_report_reads_the_synthesis_of_the_ip_the_body_states(
    tmp_path: Path,
) -> None:
    """The out-of-context column of the packaged partition is the synthesis report beside
    the IP its body states, packaged with synthesis and of the key the body has now: none
    before packaging, none packaged without synthesis, none once the key changed;
    packaging again removes an earlier synthesis's report."""
    parent = kernel_model().transform(CutKernelPartition(tmp_path / "partition"))
    stub = cast(Toolchain, SynthesizedByStub())
    assert "ooc_synth" not in built_reports(parent, toolchain=stub)
    node, body, body_file = partition_body(parent)
    verified(body, node.name)
    project = tmp_path / "stitched_ip"
    packaged = body.transform(PackagePartition(node.name, directory=project, toolchain=stub))
    packaged.save(body_file)
    assert "ooc_synth" not in built_reports(parent, toolchain=stub)
    synthesized = body.transform(
        PackagePartition(node.name, run_synth=True, directory=project, toolchain=stub)
    )
    synthesized.save(body_file)
    report = project / "finn_partition_partition_util.rpt"
    assert built_reports(parent, toolchain=stub)["ooc_synth"] == report
    synthesized.set(OUTPUT_IP_KEY, "sha256:" + "0" * 64)
    synthesized.save(body_file)
    assert "ooc_synth" not in built_reports(parent, toolchain=stub)
    body.transform(PackagePartition(node.name, directory=project, toolchain=stub))
    assert not report.exists()


def test_the_testbench_is_an_output_the_build_does_not_run(tmp_path: Path) -> None:
    """The packaging step writes the testbench beside the IP and simulates nothing
    (PRINCIPLES §8: no build re-verifies the computation): run_testbench runs it, a
    utility."""
    parent = verified_parent(kernel_model().transform(CutKernelPartition(tmp_path / "partition")))
    output = tmp_path / "output"
    cfg = DataflowBuildConfig(
        output_dir=str(output),
        target=TargetRequest(part="xczu3eg-sbva484-1-e", period_ns=5.0),
    )
    cfg.prepared_toolchain = cast(Toolchain, PackagedByStub())
    step_package_partition(parent, cfg)
    testbench = output / "stitched_ip" / TESTBENCH_DIR
    assert (testbench / "check.sv").read_text().startswith("`timescale 1ns/1ps\n")
    assert (testbench / RUN_SCRIPT).is_file()
    assert not (testbench / "xsim.log").exists()


def test_an_ip_the_store_holds_is_reused_and_the_step_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Packaged again from the same partition, the IP is the store's: the step's line says
    it reused it, not that it packaged it, and the store records it reused."""
    parent = verified_parent(kernel_model().transform(CutKernelPartition(tmp_path / "partition")))
    cfg = DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        target=TargetRequest(part="xczu3eg-sbva484-1-e", period_ns=5.0),
    )
    cfg.prepared_toolchain = cast(Toolchain, PackagedByStub())
    step_package_partition(parent, cfg)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="finn"):
        step_package_partition(parent, cfg)
    assert "Reused the stored IP partition (" in caplog.text
    assert "Packaged IP partition" not in caplog.text
    statuses = [
        getattr(record, STORED)["status"]
        for record in caplog.records
        if getattr(record, STORED, {}).get("store") == "ip"
    ]
    assert statuses == ["reused"]


def _levels_short(channel: Any, neighbours: Any, period: int, words: int) -> bool:
    """A double of the seam's check (the Chain needs no FIFO at any folding): ``levels``
    (activate's output) needs more words than its transport holds; every other channel's
    holds."""
    return not str(channel.sides.source_owner).startswith("activate.")


def test_a_fifo_too_small_for_its_stream_is_refused_at_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FIFO, not held, that no longer keeps its producer within its idle time at the
    network's bottleneck, however it got there (stated by hand after the cut, here), is
    refused when the partition is verified, before anything is emitted, named, saying
    where to size it again, and nothing is recorded; binding alone builds it. Held, it is
    verified, a limitation naming it, and built as held."""
    parent = kernel_model().transform(CutKernelPartition(tmp_path / "partition"))
    node, body, _ = partition_body(parent)
    fifo = {
        "transport": "fifo",
        "transport.fifo.depth": 8,
        "transport.fifo.buffer.ram_style": "auto",
    }
    configured(body, {"levels": fifo})
    monkeypatch.setattr(explore_module, "holds", _levels_short)
    configured_network(body, node.name)
    with pytest.raises(KernelOpError) as refused:
        verify_partition(body, node.name)
    (finding,) = refused.value.findings
    assert (finding.kind, finding.code, finding.owner) == (
        FindingKind.REJECTION,
        "fifo-stale",
        "levels.transport.fifo.depth",
    )
    assert str(refused.value).startswith(
        f"{node.name}: FIFOs too small for their streams, refused at verification (run "
        "step_size_fifos): levels: levels.transport.fifo.depth: fifo-stale: depth 8 "
        "(8 words) no longer keeps its producer"
    )
    assert body.get(OUTPUT_VERIFIED) is None
    body.set(HELD, {"levels": ["transport.fifo.depth"]})
    _, check = verify_partition(body, node.name)
    assert [(f.kind, f.code) for f in check.findings] == [
        (FindingKind.LIMITATION, "fifo-held-stale")
    ]
    body.transform(
        PackagePartition(
            node.name, directory=tmp_path / "ip", toolchain=cast(Toolchain, PackagedByStub())
        )
    )


def test_verification_records_the_design_it_verified_and_a_save_keeps_it(
    tmp_path: Path,
) -> None:
    """``verify_partition`` records the body's design digest under ``finn.outputs``
    (``verified``), beside the IP's key and not in the design: the digest is unchanged by
    it, and the record survives a save and a reload."""
    parent = kernel_model().transform(CutKernelPartition(tmp_path / "partition"))
    node, body, body_file = partition_body(parent)
    before = design_digest(body)
    point, check = verify_partition(body, node.name)
    assert body.get(OUTPUT_VERIFIED) == design_digest(body) == before
    assert check.stale == {} and check.findings == ()
    assert point.network.members == configured_network(body, node.name)[0].network.members
    body.save(body_file)
    _, again, _ = partition_body(parent)
    assert again.get(OUTPUT_VERIFIED) == design_digest(again) == before


#: Ultra96 in the Zynq shell, whose integration reads the partition's movers.
ZYNQ = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")


def test_packaging_and_the_integration_refuse_a_design_not_verified_as_it_is(
    tmp_path: Path,
) -> None:
    """Emission compares the body's design with the one its verification recorded
    (``refuse_unverified``): packaging, before anything is staged, and the shell's
    integration refuse a body never verified (``design-unverified``) and one changed since
    (``design-changed``: here a FIFO stated by hand); verified again, both take it."""
    model = streamed_w2_model()
    write_target(model, ZYNQ)
    choose_memories(model)
    parent = model.transform(CutKernelPartition(tmp_path / "partition"))
    node, body, body_file = partition_body(parent)
    project = tmp_path / "ip"
    packaging = PackagePartition(
        node.name, directory=project, toolchain=cast(Toolchain, PackagedByStub())
    )

    def refused(code: str) -> None:
        for emit in (lambda: body.transform(packaging), lambda: integration(parent)):
            with pytest.raises(KernelOpError) as found:
                emit()
            assert [(f.code, f.owner) for f in found.value.findings] == [
                (code, "refuse_unverified")
            ]
        assert not project.exists()

    refused("design-unverified")
    verified_parent(parent)
    _, body, body_file = partition_body(parent)
    fifo = {
        "transport": "fifo",
        "transport.fifo.depth": 32,
        "transport.fifo.buffer.ram_style": "auto",
    }
    configured(body, {"levels": fifo})
    body.save(body_file)
    refused("design-changed")
    verified(body, node.name)
    body.save(body_file)
    integration(parent)
    assert body.transform(packaging).get(OUTPUT_IP) == str(project / "ip")


def test_the_testbench_refuses_a_design_not_verified_as_it_is(tmp_path: Path) -> None:
    """The testbench, like packaging, is written only for the design the partition's
    verification recorded (``refuse_unverified``): a body never verified
    (``design-unverified``) and one changed since (``design-changed``: here a FIFO stated
    by hand) are refused before anything is written; verified again, it is written."""
    parent = kernel_model().transform(CutKernelPartition(tmp_path / "partition"))
    node, body, _ = partition_body(parent)
    testbench = tmp_path / TESTBENCH_DIR
    stub = cast(Toolchain, PackagedByStub())

    def write() -> None:
        write_testbench(body, testbench, generated_frame(body), toolchain=stub, label=node.name)

    def refused(code: str) -> None:
        with pytest.raises(KernelOpError) as found:
            write()
        assert [(f.code, f.owner) for f in found.value.findings] == [(code, "refuse_unverified")]
        assert not testbench.exists()

    refused("design-unverified")
    verified(body, node.name)
    fifo = {
        "transport": "fifo",
        "transport.fifo.depth": 32,
        "transport.fifo.buffer.ram_style": "auto",
    }
    configured(body, {"levels": fifo})
    refused("design-changed")
    verified(body, node.name)
    write()
    assert (testbench / RUN_SCRIPT).is_file()


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
    """step_package_partition passes the build's toolchain down with its log directory,
    ``output_dir/logs/step_package_partition``: packaging's Vivado writes its transcript
    there, not beside its replay in the packaging project (nor in the store's entry)."""
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
    )
    cfg.prepared_toolchain = Toolchain(
        Selection(command_dir=str(tools)), {"PATH": os.defpath, "XILINX_VIVADO": str(tools)}
    )
    parent = verified_parent(kernel_model().transform(CutKernelPartition(tmp_path / "p")))
    step_package_partition(parent, cfg)
    logs = output / "logs" / "step_package_partition"
    assert sorted(path.name for path in logs.iterdir()) == [
        "package.sh.stderr.log",
        "package.sh.stdout.log",
    ]
    assert (logs / "package.sh.stdout.log").read_text() == "INFO: [IP_Flow 19-234] packaging\n"
    assert (output / "stitched_ip" / "package.sh").is_file()
    assert not (output / "stitched_ip" / "package.sh.stdout.log").exists()
    # The build's own toolchain is unchanged: each step states its directory.
    assert cfg.prepared_toolchain.transcripts is None


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
    logs = tmp_path / "logs" / "step_package_partition"
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
    exploration), listed to its packaged IP: the IP packages, its description reads back against
    the module's pins, ``run_testbench`` runs the testbench written beside it, on a
    generated frame, in the machine's toolchain, and it prints PASS when run by its own
    script."""
    source_file = tmp_path / "streamlined.onnx"
    shutil.copyfile(tfc_streamlined, source_file)
    output = tmp_path / "output"
    cfg = DataflowBuildConfig(
        output_dir=str(output),
        target=TargetRequest(board="Ultra96", period_ns=5.0),
        settings={"target_fps": 1e6},
        steps=[
            "phase_prepare_model",
            "phase_convert_to_hardware",
            "phase_create_accelerator",
            "step_target_throughput",
            "phase_optimize_hardware",
            "phase_emit_hardware",
            "step_resources_report",
        ],
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source_file), cfg) == 0
    manifest = json.loads((output / MANIFEST_FILE).read_text())
    # Packaged by its step, its transcript that step's.
    (packaged,) = [each for each in manifest["steps"] if each["name"] == "step_package_partition"]
    tools = {Path(run["transcript"]).name: run["status"] for run in packaged["tools"]}
    assert tools["package.sh.stdout.log"] == "completed"
    ip = output / "stitched_ip"
    ran_by_utility = run_testbench(ip / TESTBENCH_DIR, toolchain=host_toolchain())
    assert ran_by_utility.splitlines()[-1] == "PASS"
    assert (ip / "ip" / "component.xml").is_file()
    # The partition the build packaged, opened through the parent graph the step saved.
    parent = ModelWrapper(str(output / "intermediate_models" / "step_package_partition.onnx"))
    _, body, _ = partition_body(parent)
    point, _ = configured_network(body, "partition")
    described = json.loads((ip / "interface.json").read_text())
    read_back(described, point.module.abi.pins, 5.0)
    # The IP is the partition's, of its one name; the body states it.
    assert described["ip"]["name"] == "partition"
    assert body.get(OUTPUT_IP) == str(ip / "ip")
    # The exploration folds the input's thresholds at 4 lanes and the last layer at 1
    # (Z0's choices), not test_tfc's 16 lanes by hand.
    assert [(s["name"], s["tdata"], s["element"]) for s in described["streams"]] == [
        ("s_axis_0", 32, "UINT8"),
        ("m_axis_0", 8, "INT7"),
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
