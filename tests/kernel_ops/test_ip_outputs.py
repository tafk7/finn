# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E501 - REPORT is a Vivado table, its rows as wide as Vivado writes them

"""The ``ip`` shell's outputs: the packaged IP's interface description, its resources
per member out of context, and its XSim testbench.

The Chain (``kernel_ops.models``) is packaged against a toolchain double that stands
for Vivado (``kernel_ops.packaging.PackagedByStub``), so its description and its
resources from a report are read in the fast gate. TFC_W2A2 is built through the
builder on the ``ip`` shell to its packaged IP and testbench, which is then run on its
own (marker ``xsim``; it packages with Vivado too).
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

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.kernel_build_config import KernelBuildConfig, KernelOutputType
from finn.builder.kernel_testbench import TESTBENCH_DIR
from finn.custom_op.kernels.shell import shell_root
from finn.kernels.artifacts.build import instance_name
from finn.kernels.artifacts.interface import STREAM_FACTS
from finn.kernels.utilization import Resources
from finn.platform import TargetRequest
from finn.transformation.fpgadataflow.kernel_partitions import (
    OUTPUT_IP,
    partition_body,
)
from finn.transformation.kernels import PackagePartition
from finn.transformation.kernels.package import (
    boundary_facts,
    configured_root,
    free_side,
    hierarchical_utilization,
    ooc_member_resources,
    stream_order,
)
from finn.util.toolchain import Toolchain, machine_toolchain
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
    assert hierarchical_utilization(REPORT) == [
        (0, "top", Resources(lut=120, ff=90, bram18=3, dsp=2)),
        (1, "u_first", Resources(lut=70, ff=50, bram18=2, dsp=2)),
        (2, "inner", Resources(lut=30, ff=10)),
        (1, "u_second", Resources(lut=50, ff=40, bram18=1)),
    ]


def test_out_of_context_resources_are_stated_per_member_of_the_shell_root(
    tmp_path: Path,
) -> None:
    # Its adapters' memories left open: the module is the completed one, as packaged.
    model = kernel_model()
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
        "partition.w1": 1,
        "partition.levels": 2,
        "partition.w2": 3,
        "partition.first": 4,
        "partition.activate": 5,
    }
    assert set(found["members"]) <= set(shell_root(model, model.graph.node).members)
    # The flattened instance's resources are the top's, attributed to no member.
    assert found["unattributed"] == {"lut": 10, "ff": 6, "bram18": 0, "uram": 0, "dsp": 0}
    assert found["unreported_instances"] == ["second.compute.packed"]


@requires_xsim
@pytest.mark.skipif(shutil.which("vivado") is None, reason="Vivado is not selected")
def test_tfc_on_the_ip_shell_packages_and_its_testbench_passes_on_its_own(
    tmp_path: Path, tfc_streamlined: Path
) -> None:
    """TFC_W2A2 through the builder on ``ip`` (Ultra96's part at 5 ns, the Z0 baseline's
    exploration), STITCHED_IP asked: the IP packages, its description reads back against
    the module's pins, and the testbench written beside it, on the first image of
    verify_input_npy, prints PASS when run by its own script."""
    source_file = tmp_path / "streamlined.onnx"
    shutil.copyfile(tfc_streamlined, source_file)
    images = np.random.default_rng(3).integers(0, 256, size=(2, *SHAPE[1:])).astype(np.float32)
    np.save(tmp_path / "input.npy", images)
    output = tmp_path / "output"
    cfg = KernelBuildConfig(
        output_dir=str(output),
        target=TargetRequest(board="Ultra96", period_ns=5.0),
        generate_outputs=[KernelOutputType.STITCHED_IP],
        kernel_exploration=[
            {"strategy": "target_throughput", "fps": 1e6},
            {"strategy": "size_fifos"},
        ],
        verify_input_npy=str(tmp_path / "input.npy"),
        enable_build_pdb_debug=False,
    )
    assert build_dataflow_cfg(str(source_file), cfg) == 0
    ip = output / "stitched_ip"
    assert (ip / "ip" / "component.xml").is_file()
    # The partition the build packaged, opened through the parent graph the step saved.
    parent = ModelWrapper(str(output / "intermediate_models" / "step_kernel_stitched_ip.onnx"))
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
    environment = dict(machine_toolchain().environment)
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
