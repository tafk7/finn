# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The site tool route (docs/environment.md, "Running tools on LSF"): a transformation
called without a toolchain, as the tests call them, runs each tool by the
machine's toolchain, under the command directory FINN_TOOL_DIR_OVERRIDE names.
The tools are fakes that record their calls (tests/util/conftest.py); no
Vivado, no HLS. That a stated selection wins over the machine's is
tests/util/test_build_toolchain.py's."""

from __future__ import annotations

import pytest

from finn.util.hls import CallHLS
from finn.util.toolchain import Selection, machine_selection, machine_toolchain
from finn.xsi.compile import compile_sim_obj

pytestmark = pytest.mark.util

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


def test_bare_simulation_compiles_under_the_site_directory(fake_tools, tmp_path):
    """compile_sim_obj, called without a toolchain."""
    site = fake_tools("site", machine=True)
    source = tmp_path / "top.v"
    source.write_text("module top(); endmodule")
    (tmp_path / "sim").mkdir()
    compile_sim_obj("top", [source], tmp_path / "sim")
    assert site.calls == ["vivado", "xelab"]  # the identity probe, then the compile


#: A machine's release, and the calls its HLS synthesis makes: the version probe,
#: for vitis-run the capability probe (--help), then the synthesis.
HLS_CALLS = {"2024.2": ["vitis_hls"] * 2, "2025.2": ["vitis-run"] * 3}


@pytest.mark.parametrize("version", sorted(HLS_CALLS))
def test_bare_hls_synthesis_runs_the_machine_releases_frontend_under_the_site_directory(
    fake_tools, monkeypatch, tmp_path, version
):
    """CallHLS called without a toolchain, as the pynq shell's IODMA calls it: the
    frontend the machine's release names, from the site directory."""
    site = fake_tools("site", machine=True)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_XILINX_VERSION", version)
    build = tmp_path / "iodma"
    build.mkdir()
    (build / "hls_syn_idma0.tcl").write_text("exit\n")
    caller = CallHLS()
    caller.append_tcl(str(build / "hls_syn_idma0.tcl"))
    caller.build(str(build))
    assert site.calls == HLS_CALLS[version]
