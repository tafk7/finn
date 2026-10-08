# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MakeZYNQProject, the HWCustomOp flow's block design, as this series changed it: it reads
each partition's interface names as JSON, and Vivado's jobs are the build's.

A link graph of one IODMA partition, built by hand and stated as built; no Vivado runs.
The kernel path's block design is the pynq shell's runner's (test_pynq_runner).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow import make_zynq_proj
from finn.transformation.fpgadataflow.create_dataflow_partition import PARTITION_DOMAIN
from finn.util.toolchain import Toolchain
from finn.util.vivado import vivado_jobs
from kernel_ops.packaging import ReachedVivado

#: The domain of the IODMA_hls node the link graph's partition holds.
IODMA_DOMAIN = "finn.custom_op.fpgadataflow.hls"


def project_parent(tmp_path: Path, ifnames: str) -> ModelWrapper:
    """A link graph of one IODMA partition, stated as built: its IODMA_hls's IP and the
    stitched IP in ``tmp_path``, the interface names ``ifnames``."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    dma = helper.make_node(
        "IODMA_hls",
        ["x"],
        ["y"],
        name="IODMA_hls_0",
        domain=IODMA_DOMAIN,
        backend="fpgadataflow",
        ip_path=str(tmp_path),
    )
    body = ModelWrapper(qonnx_make_model(helper.make_graph([dma], "body", [x], [y])))
    body.set_metadata_prop("vivado_stitch_proj", str(tmp_path))
    body.set_metadata_prop("vivado_stitch_vlnv", "xilinx_finn:finn:dma:1.0")
    body.set_metadata_prop("vivado_stitch_ifnames", ifnames)
    body.save(str(tmp_path / "body.onnx"))
    node = helper.make_node(
        "StreamingDataflowPartition",
        ["x"],
        ["y"],
        name="dma",
        domain=PARTITION_DOMAIN,
        model=str(tmp_path / "body.onnx"),
    )
    return ModelWrapper(qonnx_make_model(helper.make_graph([node], "link", [x], [y])))


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
