# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The RTL harness (``finn.harness.rtl``): it refuses a stream the module does not present,
before simulating, and runs the simulator through FINN's toolchain."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from finn.harness.rtl import SimulationFailed, simulate, stream_through
from finn.harness.toolchain import SIMULATOR_TOOLS, vivado_simulator
from finn.util.toolchain import Selection, Toolchain
from kernels.chain import chain


@pytest.mark.parametrize(
    "inputs, outputs, complaint",
    [
        # A name the module does not present (its ports take the shells' names).
        (
            {"in0_V": ([1], 6)},
            {"m_axis_0": ([1], 16)},
            "no input stream in0_V; its inputs: s_axis_0",
        ),
        # A name on the wrong side.
        (
            {"s_axis_0": ([1], 6)},
            {"s_axis_0": ([1], 6)},
            "no output stream s_axis_0; its outputs: m_axis_0",
        ),
    ],
)
def test_a_stream_the_module_lacks_is_refused_naming_its_ports(
    tmp_path: Path, inputs: dict, outputs: dict, complaint: str
) -> None:
    with pytest.raises(ValueError, match=complaint):
        stream_through(chain().module, tmp_path, inputs=inputs, outputs=outputs)
    assert not any(tmp_path.iterdir())  # refused before anything was built


def _toolchain(tmp_path: Path, xsim_prints: str) -> Toolchain:
    """A toolchain whose simulator tools are scripts in a command directory: each
    records its arguments, and xsim prints ``xsim_prints``."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in SIMULATOR_TOOLS:
        printed = xsim_prints if tool == "xsim" else ""
        script = tools / tool
        script.write_text(f'#!/bin/sh\necho "$0 $*" >> "{tmp_path}/calls"\necho "{printed}"\n')
        script.chmod(0o755)
    vivado = tmp_path / "vivado"
    environment = {"PATH": os.defpath, "XILINX_VIVADO": str(vivado)}
    return Toolchain(Selection(command_dir=str(tools)), environment)


def test_simulate_runs_the_tools_of_its_toolchain(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    (tmp_path / "sim").mkdir()
    output = simulate(["a.sv"], "module check; endmodule\n", tmp_path / "sim", toolchain=toolchain)
    assert "PASS" in output
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [Path(call.split()[0]).name for call in calls] == list(SIMULATOR_TOOLS)
    assert f"{tmp_path}/vivado/data/verilog/src/glbl.v" in calls[0]  # the toolchain's Vivado
    assert vivado_simulator(toolchain)


def test_simulate_without_pass_fails_as_an_assertion(tmp_path: Path) -> None:
    (tmp_path / "sim").mkdir()
    with pytest.raises(SimulationFailed, match="watchdog") as failed:
        simulate([], "", tmp_path / "sim", toolchain=_toolchain(tmp_path, "watchdog"))
    assert isinstance(failed.value, AssertionError)  # conformance collects these


def test_a_toolchain_naming_no_vivado_has_no_simulator(tmp_path: Path) -> None:
    toolchain = _toolchain(tmp_path, "PASS")
    bare = Toolchain(toolchain.selection, {"PATH": os.defpath})
    assert not vivado_simulator(bare)
    with pytest.raises(LookupError, match="XILINX_VIVADO"):
        simulate([], "", tmp_path, toolchain=bare)
