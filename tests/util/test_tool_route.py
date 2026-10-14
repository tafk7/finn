# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The site tool route (docs/environment.md, "Running tools on LSF"): the machine's
toolchain (``host_toolchain``) runs each tool under the command directory
FINN_TOOL_DIR_OVERRIDE names. And the rule for a missing toolchain
(``finn.toolchain.launcher``): a function that runs a tool requires one, and only the
entry points a user calls prepare the machine's, once, when they are made or run.
The tools are fakes that record their calls (tests/util/conftest.py); no
Vivado, no HLS. That a stated selection wins over the machine's is
tests/util/test_build_toolchain.py's."""

from __future__ import annotations

import pytest

import inspect
import re
import sys
from pathlib import Path

from finn.builder.build_dataflow_testbench import run_testbench, write_testbench
from finn.core import rtlsim_bench
from finn.core.reporting.findings import Refusal
from finn.core.rtlsim_exec import XSim
from finn.harness.hls import c_simulate
from finn.platform import generate
from finn.shells.pynq import ipgen
from finn.shells.pynq.iodma import IODMA_hls
from finn.shells.pynq.runner import BuildPynqBitfile, build_pynq
from finn.toolchain import hls, launcher
from finn.toolchain.launcher import Selection, host_selection, host_toolchain
from finn.transformation.kernels.hls import built_hls
from finn.transformation.kernels.package import ElaboratePartition, PackagePartition
from finn.xsi import compile as xsi_compile
from finn.xsi import load_sim_obj
from finn.xsi import setup as xsi_setup
from finn.xsi.compile import compile_sim_obj

pytestmark = pytest.mark.util

#: No host file: the environment alone states the machine's settings.
NO_FILE = {"FINN_XILINX_ENV": ""}


def test_the_host_selection_is_the_configured_environment_under_the_site_directory(
    monkeypatch, tmp_path
):
    assert host_selection(NO_FILE) == Selection(hls_frontend="vitis_hls")
    site = {**NO_FILE, "FINN_TOOL_DIR_OVERRIDE": "/site/tools"}
    assert host_selection(site) == Selection(command_dir="/site/tools", hls_frontend="vitis_hls")
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.delenv("FINN_XILINX_VERSION", raising=False)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tmp_path))
    monkeypatch.setenv("SELECTED_BY_THE_MACHINE", "1")
    toolchain = host_toolchain()
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
def test_the_machine_hls_frontend_follows_the_host_files_release(tmp_path, version, frontend):
    machine = tmp_path / "xilinx.env"
    machine.write_text("FINN_XILINX_PATH=/opt/Xilinx\n")
    if version:
        machine.write_text(f"FINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION={version}\n")
    assert host_selection({"FINN_XILINX_ENV": str(machine)}).hls_frontend == frontend
    # The environment's release wins over the file's, as for every machine setting.
    environ = {"FINN_XILINX_ENV": str(machine), "FINN_XILINX_VERSION": "2025.2"}
    assert host_selection(environ).hls_frontend == "vitis-run"


def test_a_machine_release_that_is_no_release_is_refused():
    refused = "FINN_XILINX_VERSION: host-setting-invalid: latest is not a release"
    with pytest.raises(Refusal, match=refused):
        host_selection({**NO_FILE, "FINN_XILINX_VERSION": "latest"})


def test_simulation_compiles_under_the_site_directory(fake_tools, tmp_path):
    """compile_sim_obj, given the machine's toolchain; refused without one."""
    site = fake_tools("site", machine=True)
    source = tmp_path / "top.v"
    source.write_text("module top(); endmodule")
    (tmp_path / "sim").mkdir()
    with pytest.raises(TypeError, match="toolchain"):
        compile_sim_obj("top", [source], tmp_path / "sim")
    compile_sim_obj("top", [source], tmp_path / "sim", toolchain=host_toolchain())
    assert site.calls == ["xelab"]  # the identity reads the installation, runs nothing


#: A machine's release, and the calls its HLS synthesis makes: the version probe,
#: for vitis-run the capability probe (--help), then the synthesis.
HLS_CALLS = {"2024.2": ["vitis_hls"] * 2, "2025.2": ["vitis-run"] * 3}


@pytest.mark.parametrize("version", sorted(HLS_CALLS))
def test_hls_synthesis_runs_the_machine_releases_frontend_under_the_site_directory(
    fake_tools, monkeypatch, tmp_path, version
):
    """HLS synthesis given the machine's toolchain, as the pynq shell's build gives it: the
    frontend the machine's release names, from the site directory."""
    site = fake_tools("site", machine=True)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_XILINX_VERSION", version)
    request = tmp_path / "iodma"
    request.mkdir()
    (request / "script.tcl").write_text("exit\n")
    hls.synthesize(
        request, "xc7z020clg400-1", name="idma0", toolchain=host_toolchain(), cache=tmp_path / "c"
    )
    assert site.calls == HLS_CALLS[version]


#: Every function that runs a tool (or reads which installation it would run), each
#: with no default toolchain: called without one, a ``TypeError``.
DRIVERS = [
    rtlsim_bench.materialize,
    rtlsim_bench.simulate,
    rtlsim_bench.stream_bench,
    rtlsim_bench.stream_through,
    rtlsim_bench.stream_out,
    rtlsim_bench.measure,
    hls.synthesize,
    built_hls,
    ipgen.hls_synth_ip,
    ipgen.create_stitched_ip,
    IODMA_hls.ipgen_singlenode_code,
    build_pynq,
    c_simulate,
    write_testbench,
    run_testbench,
    compile_sim_obj,
    xsi_compile.get_simkernel_so,
    xsi_setup.check_prerequisites,
    xsi_setup.ensure_built,
    xsi_setup.build_xsi,
    load_sim_obj,
    generate.extract,
]


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda driver: driver.__qualname__)
def test_a_function_that_runs_a_tool_requires_a_toolchain(driver):
    toolchain = inspect.signature(driver).parameters["toolchain"]
    assert toolchain.default is inspect.Parameter.empty


def test_a_driver_called_without_a_toolchain_is_refused_by_name(tmp_path):
    for call in (
        lambda: rtlsim_bench.simulate([], "", tmp_path),
        lambda: hls.synthesize(tmp_path, "", name=""),
        lambda: xsi_compile.get_simkernel_so(),
        lambda: xsi_setup.ensure_built(),
    ):
        with pytest.raises(TypeError, match="toolchain"):
            call()


class Counted:
    """The machine's selection, counting what it prepares."""

    def __init__(self):
        self.prepared = []

    def prepare(self):
        self.prepared.append(object())
        return self.prepared[-1]


@pytest.fixture
def machine(monkeypatch):
    """The machine's selection replaced by one that counts what it prepares
    (``host_toolchain`` reads ``host_selection`` as it runs)."""
    counted = Counted()
    monkeypatch.setattr(launcher, "host_selection", lambda: counted)
    return counted


@pytest.mark.parametrize(
    "make",
    [
        lambda tmp_path, **given: PackagePartition("p", **given),
        lambda tmp_path, **given: ElaboratePartition(**given),
        lambda tmp_path, **given: XSim(**given),
        lambda tmp_path, **given: BuildPynqBitfile(tmp_path, **given),
    ],
    ids=["PackagePartition", "ElaboratePartition", "XSim", "BuildPynqBitfile"],
)
def test_an_entry_point_prepares_the_machines_toolchain_once_when_made(machine, tmp_path, make):
    entry = make(tmp_path)
    assert len(machine.prepared) == 1
    assert entry.toolchain is machine.prepared[0]
    given = object()
    assert make(tmp_path, toolchain=given).toolchain is given
    assert len(machine.prepared) == 1


def test_the_catalogs_command_line_extracts_by_the_machines_toolchain(
    machine, monkeypatch, tmp_path
):
    seen = []

    def extract(work, jobs, toolchain):
        seen.append(toolchain)
        raise generate.GenerationError("extraction", "stopped here")

    monkeypatch.setattr(generate, "extract", extract)
    with pytest.raises(generate.GenerationError):
        generate.main(["check", "--work", str(tmp_path)])
    assert seen == machine.prepared and len(seen) == 1


def test_the_xsi_command_line_checks_by_the_machines_toolchain(machine, monkeypatch):
    seen = []
    monkeypatch.setattr(xsi_setup, "check_prerequisites", lambda toolchain: seen.append(toolchain))
    monkeypatch.setattr(sys, "argv", ["setup", "--check", "--quiet"])
    assert xsi_setup.main() == 0
    assert seen == machine.prepared and len(seen) == 1


def test_no_tool_installation_is_read_from_this_processs_environment():
    """Only a toolchain's environment names an installation (``XILINX_*``): no module
    reads one from ``os.environ``."""
    root = Path(launcher.__file__).parents[2]
    found = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"(?<![A-Z_])XILINX_", line) and re.search(r"os\.environ|getenv", line)
    ]
    assert found == []
