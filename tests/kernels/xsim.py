# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""XSim a composed module with one AXIS input and one AXIS output.

``stream_through`` builds ``requirements``, drives ``in0_V`` with ``words_in``
under stalls on both sides, and checks that ``out0_V`` presents ``words_out``,
each compared on its low ``out_bits`` (an AXIS word is padded to bytes).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from finn.kernels.artifacts.build import (
    ModuleBuildRequirements,
    materialize_module_sources,
    prepare_module_build,
)
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.resources import resource_root, template_root

ROOT = Path(__file__).resolve().parents[2]

requires_xsim = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)


def pack(values: Sequence[int], bits: int) -> int:
    """Lanes low first, each ``bits`` wide."""
    mask = (1 << bits) - 1
    return sum((value & mask) << (index * bits) for index, value in enumerate(values))


def stream_through(
    requirements: ModuleBuildRequirements,
    directory: Path,
    *,
    words_in: Sequence[int],
    in_bits: int,
    words_out: Sequence[int],
    out_bits: int,
) -> None:
    store = ArtifactStore(directory / "store")
    prepared = prepare_module_build(
        requirements,
        roots={"kernels": resource_root(), "finnlib": ROOT / "deps/finnlib"},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    sources = [str(Path(materialized.directory) / path) for path in materialized.files]
    in_width, out_width = (in_bits + 7) // 8 * 8, (out_bits + 7) // 8 * 8
    table_in = ", ".join(f"{in_width}'h{word:x}" for word in words_in)
    table_out = ", ".join(f"{out_bits}'h{word:x}" for word in words_out)
    testbench = directory / "check.sv"
    testbench.write_text(f"""`timescale 1ns/1ps
module check;
    logic ap_clk = 0, ap_rst_n = 0;
    logic [{in_width - 1}:0] in0_V_tdata; logic in0_V_tvalid = 0; wire in0_V_tready;
    wire [{out_width - 1}:0] out0_V_tdata; wire out0_V_tvalid; logic out0_V_tready = 0;
    logic [{in_width - 1}:0] words_in [{len(words_in)}] = '{{{table_in}}};
    logic [{out_bits - 1}:0] words_out [{len(words_out)}] = '{{{table_out}}};
    always #5 ap_clk = !ap_clk;
    {prepared.abi.entry_point} dut (.*);
    int sent = 0, received = 0, cycle = 0;
    always @(posedge ap_clk) begin
        cycle <= cycle + 1;
        if (ap_rst_n) begin
            if (in0_V_tvalid && in0_V_tready) sent <= sent + 1;
            if (out0_V_tvalid && out0_V_tready) begin
                if (out0_V_tdata[{out_bits - 1}:0] !== words_out[received])
                    $fatal(1, "word %0d: %h != %h", received, out0_V_tdata, words_out[received]);
                received <= received + 1;
            end
        end
    end
    always @(negedge ap_clk) begin
        in0_V_tvalid = ap_rst_n && sent < {len(words_in)} && (cycle % 3 != 0);
        in0_V_tdata = words_in[sent < {len(words_in)} ? sent : 0];
        out0_V_tready = cycle % 4 != 1;
    end
    initial begin
        repeat (16) @(posedge ap_clk);  // past the DSP models' startup recovery (GSR)
        ap_rst_n = 1;
        wait (received == {len(words_out)});
        repeat (4) @(posedge ap_clk);
        $display("STREAM_PASS");
        $finish;
    end
    initial begin #400000; $fatal(1, "watchdog"); end
endmodule
""")
    vivado = Path(os.environ.get("XILINX_VIVADO", str(Path(str(shutil.which("xelab"))).parents[1])))
    commands = (
        ["xvlog", "--sv", *sources, str(vivado / "data/verilog/src/glbl.v"), str(testbench)],
        [
            "xelab",
            "work.check",
            "work.glbl",
            "--mt",
            "2",
            "-L",
            "unisims_ver",
            "--snapshot",
            "check",
            "--timescale",
            "1ns/1ps",
        ],
        ["xsim", "check", "--runall"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "STREAM_PASS" in result.stdout, result.stdout + result.stderr
