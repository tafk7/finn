# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The core unit in XSim: a core presenting three AXI-Lite thresholdings (built by hand:
``support.presenting``), behind its one control port, ``s_axi_control``. Each thresholding's
registers are written through the port at its offset in the grouper's map and read back
there, and an address in the unit's window that no bus takes is answered DECERR."""

from __future__ import annotations

from pathlib import Path

from support.presenting import thresholdings
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import PERIOD_NS, RESET_CYCLES, materialize, simulate
from finn.core.space import Available
from finn.custom_op.kernels.shell import core_grouping
from finn.kernels.artifacts.abi import CLOCK, RESET, Direction, abi_pins
from finn.kernels.artifacts.unit import CoreUnit, emit_unit
from finn.kernels.dataflow_core import DataflowCoreKernel
from finn.platform.shells import shell_row
from finn.resources import finnlib_root
from finn.toolchain.launcher import host_toolchain

CHANNELS, THRESHOLDS = 2, 3
"""Each thresholding's registers (``support.presenting.TABLE``): a word a threshold of a
channel, at ``(channel * 4 + threshold) * 4`` (the thresholding's address, its threshold
index the low two bits of the word's)."""
DECERR = 3
"""The decoder's answer to an address no bus takes."""


def registers(offset: int) -> list[int]:
    """The byte addresses of a thresholding's registers at ``offset`` in the unit."""
    return [
        offset + (channel * 4 + threshold) * 4
        for channel in range(CHANNELS)
        for threshold in range(THRESHOLDS)
    ]


def testbench(name: str, unit: CoreUnit, offsets: tuple[int, ...], hole: int) -> str:
    """A testbench ``check`` of the unit ``name``: after reset, each bus's registers written
    through the unit's port, a pattern of the bus's own (a 2-bit threshold: ``(bus +
    register) % 4``), then every register read back; a write and a read at ``hole`` answered
    DECERR, the read's data zero. Every other input held idle (no stream runs)."""
    physical = abi_pins(unit.abi.pins)
    port = {member.logical: member.physical for member in unit.port.signals}
    lines = [f"    logic [{info.width - 1}:0] {pin};" for pin, info in physical.items()]
    idle = [
        f"        {pin} = 0;"
        for pin, info in physical.items()
        if info.direction is Direction.IN and pin not in (CLOCK, RESET)
    ]
    clock, reset = CLOCK, RESET
    writes = [
        (address, (bus + index) % 4)
        for bus, offset in enumerate(offsets)
        for index, address in enumerate(registers(offset))
    ]
    steps = [f"        write({address}, {value}, 0);" for address, value in writes]
    steps += [f"        check_read({address}, {value}, 0);" for address, value in writes]
    steps += [f"        write({hole}, 1, {DECERR});", f"        check_read({hole}, 0, {DECERR});"]
    connections = ",\n".join(f"        .{pin}({pin})" for pin in physical)
    p = port
    done = len(steps)
    return f"""
module check;
{chr(10).join(lines)}
    int errors = 0;

    {name} dut (
{connections}
    );

    initial {clock} = 0;
    always #{PERIOD_NS / 2} {clock} = !{clock};
    initial begin
        #({PERIOD_NS} * 200000);
        $display("FAIL: the watchdog fired");
        $finish;
    end

    task automatic write(input int address, input int data, input int response);
        @(posedge {clock});
        {p["awaddr"]} <= address; {p["awvalid"]} <= 1;
        {p["wdata"]} <= data; {p["wstrb"]} <= '1; {p["wvalid"]} <= 1;
        {p["bready"]} <= 1;
        forever begin
            @(posedge {clock});
            if ({p["awvalid"]} && {p["awready"]}) {p["awvalid"]} <= 0;
            if ({p["wvalid"]} && {p["wready"]}) {p["wvalid"]} <= 0;
            if ({p["bvalid"]}) begin
                if ({p["bresp"]} != response) begin
                    $display("write at %0d answered %0d, not %0d", address, {p["bresp"]},
                        response);
                    errors++;
                end
                {p["bready"]} <= 0;
                break;
            end
        end
    endtask

    task automatic check_read(input int address, input int data, input int response);
        @(posedge {clock});
        {p["araddr"]} <= address; {p["arvalid"]} <= 1; {p["rready"]} <= 1;
        forever begin
            @(posedge {clock});
            if ({p["arvalid"]} && {p["arready"]}) {p["arvalid"]} <= 0;
            if ({p["rvalid"]}) begin
                if ({p["rresp"]} != response || {p["rdata"]} !== data) begin
                    $display("read at %0d gave %0h answered %0d, not %0h answered %0d",
                        address, {p["rdata"]}, {p["rresp"]}, data, response);
                    errors++;
                end
                {p["rready"]} <= 0;
                break;
            end
        end
    endtask

    initial begin
{chr(10).join(idle)}
        {reset} = 0;
        repeat ({RESET_CYCLES}) @(posedge {clock});
        {reset} <= 1;
        repeat (4) @(posedge {clock});
{chr(10).join(steps)}
        if (errors == 0) $display("PASS: {done} writes and reads");
        else $display("FAIL: %0d errors", errors);
        $finish;
    end
endmodule
"""


@requires_xsim
def test_each_bus_is_written_and_read_at_its_offset_through_the_unit_s_port(
    tmp_path: Path,
) -> None:
    """Three thresholdings of 32 bytes at 0, 32 and 64 in a 4 KiB unit: what is written at
    each one's offset reads back there, each its own pattern (so no write reaches another
    bus), and an address past the map is the decoder's DECERR."""
    point = thresholdings(3, shell_row("pynq", "Ultra96"))
    module = point.core_answer(DataflowCoreKernel.module)
    grouping = core_grouping(point)
    assert isinstance(module, Available) and grouping == ((0, 32, 64), 4096)
    offsets, window = grouping
    toolchain = host_toolchain()
    top, sources, data = materialize(module.value, tmp_path / "core", toolchain=toolchain)
    unit = CoreUnit(top, f"finn:core:{top}:1.0", module.value.abi, offsets, window)
    emitted = emit_unit(unit, tmp_path / "unit", roots={"finnlib": finnlib_root()})
    run = tmp_path / "run"
    run.mkdir()
    for path, text in data.items():  # $readmemh reads an INIT_FILE from the simulator's
        (run / path).write_text(text)
    unit_sources = [emitted.directory / path for path in emitted.sources]
    output = simulate(
        [*sources, *unit_sources],
        testbench(emitted.entry_point, unit, offsets, offsets[-1] + 32),
        run,
        toolchain=toolchain,
    )
    assert f"PASS: {2 * 3 * CHANNELS * THRESHOLDS + 2} writes and reads" in output
