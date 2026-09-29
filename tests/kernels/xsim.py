# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""XSim harness: build a module's sources, run a testbench, stream words through a Design.

``materialize`` builds ``requirements`` into a directory and places each
memory's INIT_FILE where ``$readmemh`` reads it. ``simulate`` elaborates a
testbench module ``check`` against sources and requires it to display
``PASS``. ``stream_through`` drives a composed module's AXIS inputs with
words (under stalls on both sides unless ``stalled`` is False) and checks
that each AXIS output presents its words, compared on their payload bits (an
AXIS word is padded to bytes); every other top input is held at zero. A
``repeating`` design (fed by a cyclic source) never stops producing: each
output then takes exactly its words and holds its ready low after them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from finn.kernels.artifacts.abi import Clock, Direction, Reset
from finn.kernels.artifacts.build import (
    ModuleBuildRequirements,
    materialize_module_sources,
    prepare_module_build,
)
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.physical.validation import abi_pins
from finn.kernels.resources import resource_root, template_root

ROOT = Path(__file__).resolve().parents[2]

requires_xsim = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("xvlog", "xelab", "xsim")),
    reason="Vivado simulator tools are unavailable",
)

Words = tuple[Sequence[int], int]
"""A port's words, and the payload bits of each."""


def pack(values: Sequence[int], bits: int) -> int:
    """Lanes low first, each ``bits`` wide."""
    mask = (1 << bits) - 1
    return sum((value & mask) << (index * bits) for index, value in enumerate(values))


def materialize(
    requirements: ModuleBuildRequirements, directory: Path
) -> tuple[str, list[str], dict[str, str]]:
    """The top module, its HDL sources, and each INIT_FILE's name and contents.

    FinnLib is ``FINNLIB_ROOT``, or the pinned checkout under ``deps``.
    """
    store = ArtifactStore(directory / "store")
    finnlib = Path(os.environ.get("FINNLIB_ROOT", str(ROOT / "deps/finnlib")))
    prepared = prepare_module_build(
        requirements,
        roots={"kernels": resource_root(), "finnlib": finnlib},
        template_roots=(template_root(),),
        blobs=store,
    )
    materialized = materialize_module_sources(prepared, store)
    files = [Path(materialized.directory) / path for path in materialized.files]
    sources = [str(path) for path in files if path.suffix != ".dat"]
    data = {path.name: path.read_text() for path in files if path.suffix == ".dat"}
    return str(prepared.abi.entry_point), sources, data


def simulate(sources: Sequence[str | Path], testbench: str, directory: Path) -> None:
    """Elaborate the testbench module ``check`` over ``sources``; it must display PASS."""
    bench = directory / "check.sv"
    bench.write_text("`timescale 1ns/1ps\n" + testbench)
    vivado = Path(os.environ.get("XILINX_VIVADO", str(Path(str(shutil.which("xelab"))).parents[1])))
    commands = (
        ["xvlog", "--sv", *map(str, sources), str(vivado / "data/verilog/src/glbl.v"), str(bench)],
        [
            "xelab",
            "work.check",
            "work.glbl",
            "--mt",
            "2",
            "-L",
            "unisims_ver",
            "-L",
            "unimacro_ver",
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
    assert "PASS" in result.stdout, result.stdout + result.stderr


def _table(name: str, bits: int, words: Sequence[int]) -> str:
    values = ", ".join(f"{bits}'h{word:x}" for word in words)
    return f"logic [{bits - 1}:0] {name} [{len(words)}] = '{{{values}}};"


def stream_through(
    requirements: ModuleBuildRequirements,
    directory: Path,
    *,
    inputs: Mapping[str, Words],
    outputs: Mapping[str, Words],
    stalled: bool = True,
    repeating: bool = False,
) -> None:
    top, sources, data = materialize(requirements, directory)
    for name, text in data.items():  # $readmemh reads an INIT_FILE from the simulator's directory
        (directory / name).write_text(text)
    valid, ready = ("cycle % 3 != 0", "cycle % 4 != 1") if stalled else ("1", "1")
    streams = {**inputs, **outputs}
    lines: list[str] = []
    for name, info in abi_pins(requirements.abi).items():
        if isinstance(info.role, (Clock, Reset)) or info.bus_id in streams:
            continue
        width = "" if info.width == 1 else f"[{info.width - 1}:0] "
        held = info.direction is Direction.IN
        lines.append(f"logic {width}{name} = 0;" if held else f"wire {width}{name};")
    drive: list[str] = []
    count: list[str] = []
    done: list[str] = []
    for port, (words, bits) in inputs.items():
        carrier, total = (bits + 7) // 8 * 8, len(words)
        lines += [
            f"logic [{carrier - 1}:0] {port}_tdata; logic {port}_tvalid = 0; wire {port}_tready;",
            _table(f"{port}_words", carrier, words),
            f"int {port}_sent = 0;",
        ]
        count.append(f"if ({port}_tvalid && {port}_tready) {port}_sent <= {port}_sent + 1;")
        drive += [
            f"{port}_tvalid = ap_rst_n && {port}_sent < {total} && ({valid});",
            f"{port}_tdata = {port}_words[{port}_sent < {total} ? {port}_sent : 0];",
        ]
    for port, (words, bits) in outputs.items():
        carrier = (bits + 7) // 8 * 8
        lines += [
            f"wire [{carrier - 1}:0] {port}_tdata; wire {port}_tvalid; logic {port}_tready = 0;",
            _table(f"{port}_words", bits, words),
            f"int {port}_received = 0;",
        ]
        count.append(
            f"""if ({port}_tvalid && {port}_tready) begin
                if ({port}_tdata[{bits - 1}:0] !== {port}_words[{port}_received])
                    $fatal(1, "{port} word %0d: %h != %h", {port}_received,
                        {port}_tdata, {port}_words[{port}_received]);
                {port}_received <= {port}_received + 1;
            end"""
        )
        taken = f"{port}_received < {len(words)} && " if repeating else ""
        drive.append(f"{port}_tready = {taken}({ready});")
        done.append(f"{port}_received == {len(words)}")
    newline = "\n    "
    simulate(
        sources,
        f"""module check;
    logic ap_clk = 0, ap_rst_n = 0;
    {newline.join(lines)}
    always #5 ap_clk = !ap_clk;
    {top} dut (.*);
    int cycle = 0;
    always @(posedge ap_clk) begin
        cycle <= cycle + 1;
        if (ap_rst_n) begin
            {(newline + "        ").join(count)}
        end
    end
    always @(negedge ap_clk) begin
        {(newline + "    ").join(drive)}
    end
    initial begin
        repeat (16) @(posedge ap_clk);  // past the DSP models' startup recovery (GSR)
        ap_rst_n = 1;
        wait ({" && ".join(done)});
        repeat (4) @(posedge ap_clk);
        $display("STREAM_PASS");
        $finish;
    end
    initial begin #400000; $fatal(1, "watchdog"); end
endmodule
""",
        directory,
    )
