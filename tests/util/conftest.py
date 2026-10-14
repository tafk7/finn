# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import sys
from pathlib import Path

#: Vivado, xelab, Vitis HLS and vitis-run in one script: each notes its name in
#: the calls.log beside it and leaves what its caller checks for. Vivado answers
#: its version probe; xelab leaves a simulation library; the HLS frontends a
#: solution's exported Verilog, reports and IP.
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
elif tool == "xelab":
    top = args[args.index("-s") + 1]
    os.makedirs(f"xsim.dir/{top}")
    open(f"xsim.dir/{top}/xsimk.so", "w").close()
elif tool in ("vitis_hls", "vitis-run"):
    for made in ("syn/verilog", "syn/report", "impl/ip"):
        os.makedirs(f"project/sol1/{made}")
    open("project/sol1/syn/verilog/top.v", "w").close()
"""


def fake_vivado(directory: Path, release: str = "2024.2") -> Path:
    """A Vivado installation in ``directory`` as far as FINN reads one without running
    it: its release (``data/version.sh``), its version file and its ``glbl.v``."""
    data = directory / "data"
    (data / "verilog/src").mkdir(parents=True, exist_ok=True)
    (data / "verilog/src/glbl.v").write_text("module glbl(); endmodule\n")
    (data / "version.sh").write_text(f"XILINX_VERSION_DEFAULT={release}\n")
    (data / "version.dat").write_text(f"Vivado {release}\n")
    return directory


@pytest.fixture
def vivado_installation():
    """Makes a Vivado installation in a directory (``fake_vivado``)."""
    return fake_vivado


class FakeTools:
    """A command directory of fake vendor tools (FAKE_TOOL)."""

    TOOLS = ("vivado", "xelab", "vitis_hls", "vitis-run")

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
    each, and the toolchain's installations are directories under it: HLS (headers
    for cppsim) and a Vivado with a simulation kernel directory and the ``glbl.v`` a
    simulation elaborates."""
    monkeypatch.delenv("FINN_TOOL_DIR_OVERRIDE", raising=False)
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setenv("NUM_DEFAULT_WORKERS", "1")
    (tmp_path / "hls").mkdir()
    monkeypatch.setenv("XILINX_HLS", str(tmp_path / "hls"))
    (tmp_path / "2024.2/Vivado/lib/lnx64.o").mkdir(parents=True)
    fake_vivado(tmp_path / "2024.2/Vivado")
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path / "2024.2/Vivado"))

    def make(name: str, *, machine: bool = False) -> FakeTools:
        tools = FakeTools(tmp_path / name)
        if machine:
            monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(tools.directory))
        return tools

    return make
