# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The site tool route (ci/README.md, "Running tools on LSF"): a transformation
called without a toolchain, as FINN's CI tests call them, runs each tool by the
machine's toolchain, under the command directory FINN_TOOL_DIR_OVERRIDE names.
The tools are fakes that record their calls (tests/util/conftest.py); no
Vivado, no HLS. That a stated selection wins over the machine's is
tests/util/test_build_toolchain.py's."""

from __future__ import annotations

import pytest

import numpy as np
import onnx.helper as oh
import os
from onnx import TensorProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFO_rtl
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.toolchain import Selection, machine_selection, machine_toolchain
from finn.xsi.compile import compile_sim_obj

pytestmark = pytest.mark.util

PART = "xcu250-figd2104-2L-e"


#: No machine file: the environment alone states the machine's settings.
NO_FILE = {"FINN_XILINX_ENV": ""}


def test_the_machine_selection_is_the_configured_environment_under_the_site_directory(
    monkeypatch, tmp_path
):
    assert machine_selection(NO_FILE) == Selection(hls_frontend="vitis_hls")
    site = {**NO_FILE, "FINN_TOOL_DIR_OVERRIDE": "/site/tools"}
    assert machine_selection(site) == Selection(command_dir="/site/tools", hls_frontend="vitis_hls")
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.delenv("FINN_XILINX_VERSION", raising=False)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SELECTED_BY_THE_MACHINE", "1")
    toolchain = machine_toolchain()
    assert toolchain.selection == Selection(command_dir=str(tmp_path), hls_frontend="vitis_hls")
    assert toolchain.environment["SELECTED_BY_THE_MACHINE"] == "1"


@pytest.mark.parametrize(
    "version, frontend",
    [
        (None, "vitis_hls"),
        ("2022.2", "vitis_hls"),
        ("2024.2", "vitis_hls"),
        ("2025.1", "vitis-run"),
    ],
)
def test_the_machine_hls_frontend_follows_the_machine_files_release(tmp_path, version, frontend):
    machine = tmp_path / "xilinx.env"
    machine.write_text("FINN_XILINX_PATH=/opt/Xilinx\n")
    if version:
        machine.write_text(f"FINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION={version}\n")
    assert machine_selection({"FINN_XILINX_ENV": str(machine)}).hls_frontend == frontend
    # The environment's release wins over the file's, as for every machine setting.
    environ = {"FINN_XILINX_ENV": str(machine), "FINN_XILINX_VERSION": "2025.2"}
    assert machine_selection(environ).hls_frontend == "vitis-run"


def test_a_machine_release_that_is_no_release_is_refused():
    with pytest.raises(ValueError, match="FINN_XILINX_VERSION=latest"):
        machine_selection({**NO_FILE, "FINN_XILINX_VERSION": "latest"})


def fifo_model(tmp_path):
    """One RTL FIFO, its HDL generated (no vendor tool)."""
    node = oh.make_node(
        "StreamingFIFO_rtl",
        ["inp"],
        ["outp"],
        name="fifo",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=4,
        folded_shape=[1, 4],
        normal_shape=[1, 4],
        dataType="INT8",
        impl_style="rtl",
        code_gen_dir_ipgen=str(tmp_path / "fifo"),
    )
    (tmp_path / "fifo").mkdir()
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(
                [node],
                "fifo",
                [oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
                [oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])],
            )
        )
    )
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_tensor_datatype("outp", DataType["INT8"])
    StreamingFIFO_rtl(model.graph.node[0]).generate_hdl(model, PART, 5.0)
    return model


def test_bare_simulation_and_stitching_run_under_the_site_directory(fake_tools, tmp_path):
    """compile_sim_obj, PrepareRTLSim, CreateStitchedIP and xsi_fifosim (its
    library and its C++ driver), each called without a toolchain."""
    site = fake_tools("site", machine=True)
    source = tmp_path / "top.v"
    source.write_text("module top(); endmodule")
    (tmp_path / "sim").mkdir()
    compile_sim_obj("top", [source], tmp_path / "sim")
    assert site.calls == ["vivado", "xelab"]  # the identity probe, then the compile
    model = fifo_model(tmp_path).transform(PrepareRTLSim())
    assert site.calls[2:] == ["xelab"]
    model = model.transform(CreateStitchedIP(PART, 5.0))
    assert site.calls[3:] == ["vivado"]
    assert xsi_fifosim(model, 1)["cycles"] == 100
    assert site.calls[4:] == ["xelab", "g++"]
    # The C++ driver ran with the simulation kernel on its loader path.
    driver = Path(model.get_metadata_prop("rtlsim_so").split("xsim.dir")[0])
    kernel = Path(os.environ["XILINX_VIVADO"]) / "lib/lnx64.o"
    assert (driver / "loader_path.txt").read_text().split(":")[0].strip() == str(kernel)


def mvau_model():
    """One MVAU, 8 inputs to 4 outputs, INT2 weights and inputs, no activation."""
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 8])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])
    node = oh.make_node(
        "MVAU",
        ["inp", "weights"],
        ["outp"],
        name="MVAU_0",
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=8,
        MH=4,
        SIMD=2,
        PE=2,
        inputDataType="INT2",
        weightDataType="INT2",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        preferred_impl_style="hls",
        mem_mode="internal_embedded",
    )
    model = ModelWrapper(qonnx_make_model(oh.make_graph([node], "mvau", [inp], [outp])))
    model.set_tensor_datatype("inp", DataType["INT2"])
    model.set_tensor_datatype("outp", DataType["INT32"])
    model.set_initializer("weights", np.ones((8, 4), dtype=np.float32))
    model.set_tensor_datatype("weights", DataType["INT2"])
    return model


def test_bare_cppsim_compiles_under_the_site_directory(fake_tools):
    site = fake_tools("site", machine=True)
    model = mvau_model().transform(SpecializeLayers(PART)).transform(PrepareCppSim())
    model.transform(CompileCppSim())
    assert site.calls == ["g++"]


#: A machine's release, and the calls its HLS synthesis makes: the version probe,
#: for vitis-run the capability probe (--help), then the synthesis.
HLS_CALLS = {"2024.2": ["vitis_hls"] * 2, "2025.2": ["vitis-run"] * 3}


@pytest.mark.parametrize("version", sorted(HLS_CALLS))
def test_bare_hls_synthesis_runs_the_machine_releases_frontend_under_the_site_directory(
    fake_tools, monkeypatch, version
):
    """HLSSynthIP called without a toolchain, as the CI's tests call it: the
    frontend the machine's release names, from the site directory."""
    site = fake_tools("site", machine=True)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_XILINX_VERSION", version)
    model = mvau_model().transform(SpecializeLayers(PART)).transform(PrepareIP(PART, 5.0))
    model.transform(HLSSynthIP())
    assert site.calls == HLS_CALLS[version]
