# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import sys
from pathlib import Path

from finn.util.toolchain import machine_toolchain


@pytest.fixture(scope="session")
def hls_toolchain():
    """The machine's toolchain, as HLS C++ simulation takes it by default; the
    test is skipped when it names no HLS installation."""
    toolchain = machine_toolchain()
    try:
        toolchain.hls_installation()
    except LookupError as exc:
        pytest.skip(str(exc))
    return toolchain


#: Vivado, xelab, g++, Vitis HLS and vitis-run in one script: each notes its
#: name in the calls.log beside it and leaves what its caller checks for. Vivado
#: a stitched IP's wrapper and source list; xelab a simulation library; g++ an executable that
#: reports a finished XSI run; the HLS frontends a node's IP and Verilog.
FAKE_TOOL = """
import os, sys
tool = os.path.basename(sys.argv[0])
with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "calls.log"), "a") as log:
    log.write(tool + "\\n")
args = sys.argv[1:]
if args[:1] in (["-version"], ["--version"]):
    print(tool + (" v2025.2" if tool == "vitis-run" else " v2024.2") + " (64-bit)")
elif args[:1] == ["--help"]:
    print("--mode hls")
elif tool == "vivado":
    wrapper = "finn_vivado_stitch_proj.srcs/sources_1/bd/finn_design/hdl/finn_design_wrapper.v"
    os.makedirs(os.path.dirname(wrapper))
    with open(wrapper, "w") as f:
        f.write("module finn_design_wrapper(); endmodule")
    with open("all_verilog_srcs.txt", "w") as f:
        f.write(os.path.abspath(wrapper))
elif tool == "xelab":
    top = args[args.index("-s") + 1]
    os.makedirs(f"xsim.dir/{top}")
    open(f"xsim.dir/{top}/xsimk.so", "w").close()
elif tool == "g++":
    out = args[args.index("-o") + 1]
    with open(out, "w") as executable:
        executable.write(
            "#!/bin/sh\\nprintf 'cycles\\\\t100\\\\nlatency_cycles\\\\t60\\\\nTIMEOUT\\\\t0\\\\n'"
            " > results.txt\\n"
        )
    os.chmod(out, 0o755)
elif tool in ("vitis_hls", "vitis-run"):
    script = args[args.index("--tcl") + 1] if "--tcl" in args else args[1]
    name = os.path.basename(script)[len("hls_syn_") : -len(".tcl")]
    os.makedirs(f"project_{name}/sol1/impl/ip")
    os.makedirs(f"project_{name}/sol1/impl/verilog")
    open(f"project_{name}/sol1/impl/verilog/{name}.v", "w").close()
"""


class FakeTools:
    """A command directory of fake vendor tools (FAKE_TOOL)."""

    TOOLS = ("vivado", "xelab", "g++", "vitis_hls", "vitis-run")

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir()
        for tool in self.TOOLS:
            script = directory / tool
            script.write_text("#!" + sys.executable + "\n" + FAKE_TOOL)
            script.chmod(0o755)

    @property
    def calls(self) -> list[str]:
        """The tools called, in order."""
        log = self.directory / "calls.log"
        return log.read_text().split() if log.exists() else []


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    """Makes command directories of fake tools, by name; ``machine=True`` names
    the machine's (FINN_TOOL_DIR_OVERRIDE). Builds go under tmp_path, one worker
    each, and the toolchain's installations are empty directories under it:
    HLS (headers for cppsim) and a Vivado whose path gives its release."""
    monkeypatch.delenv("FINN_TOOL_DIR_OVERRIDE", raising=False)
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "1")
    (tmp_path / "hls").mkdir()
    monkeypatch.setenv("XILINX_HLS", str(tmp_path / "hls"))
    (tmp_path / "2024.2/Vivado").mkdir(parents=True)
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path / "2024.2/Vivado"))

    def make(name: str, *, machine: bool = False) -> FakeTools:
        tools = FakeTools(tmp_path / name)
        if machine:
            monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tools.directory))
        return tools

    return make
