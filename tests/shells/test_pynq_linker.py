# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's linker (``finn.shells.pynq.linker``): the cells, the wires and the
placement it gives the Chain's description on ``pynq``, its second weights streamed (two
inputs and one output); what it refuses.
No Vivado runs: the runner's tests link through a fake (``test_pynq_runner``)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from support.models import zynq_parent
from support.packaging import (
    TFC_TIMING,
    TIMING_REPORT,
    UNMET_REPORT,
    UNMET_SLASHKIT_REPORT,
    FakeVivado,
)
from support.refusals import refused

from finn.kernels.shell_rules import placed_windows
from finn.platform.shells import shell_row
from finn.shells.integration import (
    Endpoint,
    Integration,
    IntegrationError,
    IpRef,
    describe,
)
from finn.shells.pynq.linker import (
    TIMING,
    Address,
    BlockDesignLinker,
    LinkerRefused,
    Wire,
    admitted,
    block_design,
    linked_row,
    placement,
    project_script,
    report,
    unit_bases,
    wire_one_bus_each,
    wiring,
)
from finn.toolchain.launcher import Toolchain
from finn.toolchain.vivado import Check, routed_timing

#: The facts the ``pynq`` shell declares of Ultra96 for its linker.
BOARD = {"board_part": "avnet.com:ultra96v1:part0:1.2"}


def packaged(described: Integration) -> Integration:
    """``described``, each unit's IP named and found where the test states."""
    ips = {unit.name: IpRef(f"v:{unit.name}", (f"/{unit.name}",)) for unit in described.units}
    return described.packaged(ips)


@pytest.fixture
def described(tmp_path: Path) -> Integration:
    return describe(zynq_parent(tmp_path))


def test_the_chain_is_wired_as_its_block_design(described: Integration) -> None:
    """The streams, each memory port to the memory interconnect's next slave port, the
    processor's control master into the control tree's one SmartConnect, each control bus
    from its next master port (input movers, the core, output movers), the SmartConnect's
    clock and reset, and every cell's clock and reset: each cell its unit's name, each pin
    the description's, its IP's own."""
    row = linked_row(described)
    assert wiring(described, row) == (
        Wire("axis", "in0/out0_V", "DataflowCore_0/s_axis_0"),
        Wire("axis", "in1/out0_V", "DataflowCore_0/s_axis_1"),
        Wire("axis", "DataflowCore_0/m_axis_0", "out0/in0_V"),
        Wire("aximm", "in0/m_axi_gmem", "smartconnect_0/S00_AXI"),
        Wire("aximm", "in1/m_axi_gmem", "smartconnect_0/S01_AXI"),
        Wire("aximm", "out0/m_axi_gmem", "smartconnect_0/S02_AXI"),
        Wire("axilite", "zynq_ps/M_AXI_HPM0_FPD", "control_0_0/S00_AXI"),
        Wire("axilite", "control_0_0/M00_AXI", "in0/s_axi_control"),
        Wire("axilite", "control_0_0/M01_AXI", "in1/s_axi_control"),
        Wire("axilite", "control_0_0/M02_AXI", "out0/s_axi_control"),
        Wire("clock", "zynq_ps/pl_clk0", "control_0_0/aclk"),
        Wire("reset", "rst_ps/peripheral_aresetn", "control_0_0/aresetn"),
        *(
            Wire(kind, f"smartconnect_0/{source}", f"{instance}/{pin}")
            for instance in ("in0", "in1", "DataflowCore_0", "out0")
            for kind, source, pin in (
                ("clock", "aclk", "ap_clk"),
                ("reset", "aresetn", "ap_rst_n"),
            )
        ),
    )


def test_each_control_bus_is_placed_from_the_processors_window(
    described: Integration,
) -> None:
    """Each control map from the fixed shell's control base, in order, 4 KiB apart: the
    row's least window, each aligned to its range."""
    assert placement(described, linked_row(described)) == (
        Address("in0/s_axi_control", 0xA000_0000, 4096),
        Address("in1/s_axi_control", 0xA000_1000, 4096),
        Address("out0/s_axi_control", 0xA000_2000, 4096),
    )
    stated = report(described)
    assert sorted(stated) == ["addresses", "units", "wires"]
    # Each unit's IP is the build's, none before it is packaged.
    assert stated["units"][0] == {"name": "in0", "ip": None}
    assert stated["addresses"][2] == {
        "interface": "out0/s_axi_control",
        "offset": "0xa0002000",
        "range": 4096,
    }


def grouped(described: Integration, offsets: Iterable[int]) -> Integration:
    """``described``, its core one unit of a 4 KiB window presenting a 32-byte bus at each
    of ``offsets`` (``t<i>_s_axilite``), as a core is placed behind its grouper."""
    (core,) = (unit for unit in described.units if unit.window is None)
    buses = (
        Endpoint(unit=core.name, bus=f"t{index}_s_axilite", offset=offset, range=32)
        for index, offset in enumerate(offsets)
    )
    return replace(
        described,
        units=tuple(
            replace(each, window=4096) if each == core else each for each in described.units
        ),
        endpoints=(*described.endpoints, *buses),
    )


def test_a_core_unit_is_one_control_port_its_buses_at_their_offsets(
    described: Integration,
) -> None:
    """A core presenting buses is one unit: wired by its one control port in the units'
    order, its window placed once, as a mover's; each bus at its unit's base and its offset
    in the unit's window, as the grouper maps it."""
    core = grouped(described, (0, 32, 64))
    row = linked_row(core)
    assert [each for each in wiring(core, row) if each.kind == "axilite"] == [
        Wire("axilite", "zynq_ps/M_AXI_HPM0_FPD", "control_0_0/S00_AXI"),
        *(
            Wire("axilite", f"control_0_0/M{index:02}_AXI", f"{unit}/s_axi_control")
            for index, unit in enumerate(("in0", "in1", "DataflowCore_0", "out0"))
        ),
    ]
    assert placement(core, row) == (
        Address("in0/s_axi_control", 0xA000_0000, 4096),
        Address("in1/s_axi_control", 0xA000_1000, 4096),
        Address("DataflowCore_0/s_axi_control", 0xA000_2000, 4096),
        Address("out0/s_axi_control", 0xA000_3000, 4096),
    )
    assert unit_bases(core, row) == {
        "in0": 0xA000_0000,
        "in1": 0xA000_1000,
        "DataflowCore_0": 0xA000_2000,
        "out0": 0xA000_3000,
    }
    assigned = [
        line
        for line in block_design(packaged(core), row).splitlines()
        if line.startswith("assign_bd") and "DataflowCore_0" in line
    ]
    assert assigned == [
        "assign_bd_address [get_bd_addr_segs DataflowCore_0/s_axi_control/Reg*] "
        "-offset 0xa0002000 -range 4096"
    ]


def test_a_mover_the_block_design_cannot_wire_is_refused_by_name(
    described: Integration,
) -> None:
    """The block design wires one AXI-Lite bus and one memory port a mover: a mover stating
    other counts is refused."""
    edge = described.edges[0]
    wire_one_bus_each(edge.contract, "x")
    for control, memory in ((2, 1), (1, 2), (1, 0)):
        contract = replace(edge.contract, control_buses=control, memory_ports=memory)
        with refused(
            "mover-buses-unwired",
            error=IntegrationError,
            owner="vivado-block-design",
            message="x: its iodma_hls mover states .* AXI-Lite",
        ):
            wire_one_bus_each(contract, "x")


def test_a_unit_whose_ip_is_not_packaged_is_refused(described: Integration) -> None:
    """The block design instantiates every unit's IP, each as a cell of its unit's name:
    the movers' are built and the core's packaged before it links."""
    with refused("unit-unpackaged", error=LinkerRefused, message="^in0: its IP is not packaged"):
        block_design(described, linked_row(described))
    ips = {unit.name: IpRef(f"v:{unit.name}", (f"/{unit.name}",)) for unit in described.units}
    design = block_design(described.packaged(ips), linked_row(described))
    assert "create_bd_cell -type ip -vlnv v:in0 in0\n" in design
    assert "create_bd_cell -type ip -vlnv v:DataflowCore_0 DataflowCore_0\n" in design


def test_a_row_without_a_fixed_shell_is_not_linked(described: Integration) -> None:
    """The block design places its units in the fixed shell's control window: a row with
    none (``ip``'s) is refused."""
    assert linked_row(described) == shell_row("pynq", "Ultra96")
    with refused(
        "integration-unexported",
        error=LinkerRefused,
        owner="vivado-block-design",
        message="the 'ip' shell states no fixed shell or no control window",
    ):
        linked_row(replace(described, row=("ip", "Ultra96")))


def ranged(described: Integration, ranges: Iterable[int]) -> Integration:
    """``described``, its endpoints of ``ranges`` in order, each mover's window its control
    bus's range."""
    endpoints = tuple(replace(each, range=size) for each, size in zip(described.endpoints, ranges))
    sizes = {each.unit: each.range for each in endpoints}
    units = tuple(
        unit if unit.window is None else replace(unit, window=sizes[unit.name])
        for unit in described.units
    )
    return replace(described, units=units, endpoints=endpoints)


def test_the_block_design_assigns_each_control_bus_the_address_it_returns(
    described: Integration,
) -> None:
    """One map: the block design assigns each control bus the offset and range ``placement``
    gives, which the link returns, whatever the window: here 64 KiB a bus, which the
    template's own procedure would have placed 4 KiB apart."""
    wide = ranged(packaged(described), [0x1_0000] * len(described.endpoints))
    row = linked_row(wide)
    assigned = [
        line for line in block_design(wide, row).splitlines() if line.startswith("assign_bd")
    ]
    assert assigned == [
        f"assign_bd_address [get_bd_addr_segs {each.interface}/Reg*] "
        f"-offset {each.offset:#010x} -range {each.range}"
        for each in placement(wide, row)
    ]
    assert [each.offset for each in placement(wide, row)] == [
        0xA000_0000,
        0xA001_0000,
        0xA002_0000,
    ]


def test_the_block_design_places_each_window_as_the_shell_s_rules_do(
    described: Integration,
) -> None:
    """A 64 KiB window between two of 4 KiB: each is placed after the last, aligned to
    itself, at the offsets the shell's rules span (``placed_windows``), the one function
    both read."""
    ranges = (0x1000, 0x1_0000, 0x1000)
    mixed = ranged(described, ranges)
    row = linked_row(mixed)
    assert row.fixed is not None
    offsets = [each.offset for each in placement(mixed, row)]
    assert offsets == [0xA000_0000, 0xA001_0000, 0xA002_0000]
    assert tuple(offsets) == placed_windows(ranges, row.fixed.control_base)


def test_a_control_window_below_the_row_s_least_window_is_refused(described: Integration) -> None:
    """A description built by hand whose unit takes a control window less than the row's
    least window (4096 bytes on ``pynq``, its linker's least range 128 at most) is refused
    before any tool runs, as the row's declaration refuses a least window below its
    linker's."""
    narrow = ranged(described, [64] * len(described.endpoints))
    with refused(
        "range-below-least",
        error=LinkerRefused,
        owner="vivado-block-design",
        message="^in0: 64 bytes; the row assigns a control window 4096 at least$",
    ):
        wiring(narrow, linked_row(narrow))


def many(described: Integration, count: int) -> Integration:
    """``described``, its first mover repeated as ``count`` units of their own, each with
    its control bus, beside the core."""
    unit, endpoint = described.units[0], described.endpoints[0]
    names = [f"{unit.name}_{index}" for index in range(count)]
    return replace(
        described,
        units=(
            *(replace(unit, name=name) for name in names),
            *(each for each in described.units if each.window is None),
        ),
        endpoints=tuple(replace(endpoint, unit=name) for name in names),
    )


def test_more_units_than_a_smartconnect_takes_are_a_tree_of_them(
    described: Integration,
) -> None:
    """Sixteen units take one SmartConnect; eighteen a tree: the root's two ports into two
    SmartConnects, the first giving sixteen units their control ports and the second two,
    in the units' order, each on the row's clock and the template's reset, each made
    LOW_AREA with its own count of ports. No count is refused."""
    sixteen = many(described, 16)
    row = linked_row(sixteen)
    control = [each for each in wiring(sixteen, row) if each.kind == "axilite"]
    assert control[0] == Wire("axilite", "zynq_ps/M_AXI_HPM0_FPD", "control_0_0/S00_AXI")
    assert {each.source.split("/")[0] for each in control[1:]} == {"control_0_0"}
    eighteen = packaged(many(described, 18))
    names = [unit.name for unit in eighteen.units if unit.window is not None]
    wires = wiring(eighteen, row)
    assert [each for each in wires if each.kind == "axilite"] == [
        Wire("axilite", "zynq_ps/M_AXI_HPM0_FPD", "control_0_0/S00_AXI"),
        Wire("axilite", "control_0_0/M00_AXI", "control_1_0/S00_AXI"),
        Wire("axilite", "control_0_0/M01_AXI", "control_1_1/S00_AXI"),
        *(
            Wire(
                "axilite", f"control_1_{index // 16}/M{index % 16:02}_AXI", f"{name}/s_axi_control"
            )
            for index, name in enumerate(names)
        ),
    ]
    tree = block_design(eighteen, row).split("\nset_property ip_repo_paths")[0]
    assert tree.splitlines() == [
        "create_bd_cell -type ip -vlnv $smartconnect_vlnv control_0_0",
        "set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 2 CONFIG.STRATEGY LOW_AREA] "
        "[get_bd_cells control_0_0]",
        "create_bd_cell -type ip -vlnv $smartconnect_vlnv control_1_0",
        "set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 16 CONFIG.STRATEGY LOW_AREA] "
        "[get_bd_cells control_1_0]",
        "create_bd_cell -type ip -vlnv $smartconnect_vlnv control_1_1",
        "set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 2 CONFIG.STRATEGY LOW_AREA] "
        "[get_bd_cells control_1_1]",
        "connect_bd_intf_net [get_bd_intf_pins control_0_0/S00_AXI] "
        "[get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD]",
        "connect_bd_intf_net [get_bd_intf_pins control_1_0/S00_AXI] "
        "[get_bd_intf_pins control_0_0/M00_AXI]",
        "connect_bd_intf_net [get_bd_intf_pins control_1_1/S00_AXI] "
        "[get_bd_intf_pins control_0_0/M01_AXI]",
        *(
            line
            for instance in ("control_0_0", "control_1_0", "control_1_1")
            for line in (
                f"connect_bd_net [get_bd_pins {instance}/aclk] [get_bd_pins zynq_ps/pl_clk0]",
                f"connect_bd_net [get_bd_pins {instance}/aresetn] "
                "[get_bd_pins rst_ps/peripheral_aresetn]",
            )
        ),
    ]


def test_a_selection_the_block_design_does_not_wire_is_refused_by_name(
    described: Integration,
) -> None:
    """Its template wires memory into the processor's HP0 and every cell on the row's main
    clock: a memory port bound elsewhere, a unit on another clock and a description with
    no memory port are refused, not wired as if they were HP0's and the main clock's."""
    row = linked_row(described)
    elsewhere = replace(described.bindings[1], target="HP1")
    with refused(
        "memory-target-unsupported",
        error=LinkerRefused,
        owner="vivado-block-design",
        message="^in1/m_axi_gmem: bound to 'HP1'; the block design wires memory into 'HP0'",
    ):
        wiring(replace(described, bindings=(described.bindings[0], elsewhere)), row)
    other = replace(described.clocks[2], clock="fast")
    with refused(
        "clock-unsupported",
        error=LinkerRefused,
        message="^DataflowCore_0/ap_clk: on the clock 'fast'; the block design runs every cell",
    ):
        wiring(replace(described, clocks=(*described.clocks[:2], other)), row)
    with refused("memory-unbound", error=LinkerRefused, message="binds no memory port"):
        wiring(replace(described, bindings=()), row)


def test_the_project_selects_the_declared_board_part_and_the_processors_family(
    described: Integration,
) -> None:
    """The board is selected by its declared part, not looked up by its name, and the
    processor's family is the fixed shell's processor's: a board with no part and a
    processor the template does not instantiate are refused before any tool runs."""
    row = linked_row(described)
    script = project_script(described, "", {"board_part": "example.org:probe:part0:1.0"}, row, 1)
    assert "set_property board_part example.org:probe:part0:1.0 [current_project]\n" in script
    assert "set ZYNQ_TYPE zynq_us+\n" in script and "Unrecognized board" not in script
    with refused("board-part-unstated", error=LinkerRefused, message="the board 'Ultra96'"):
        project_script(described, "", {}, row, 1)
    assert row.fixed is not None
    other = replace(row, fixed=replace(row.fixed, processor="versal_cips"))
    with refused(
        "processor-unsupported",
        error=LinkerRefused,
        message="the processor 'versal_cips' is none the block design instantiates",
    ):
        project_script(described, "", BOARD, other, 1)


def test_the_link_is_made_under_the_directory_it_is_given(
    described: Integration, tmp_path: Path
) -> None:
    """Its project, its script, its bitfile and its reports, all under ``directory``."""
    vivado = FakeVivado()
    directory = tmp_path / "linked"
    linked = BlockDesignLinker().link(
        packaged(described), BOARD, directory, cast(Toolchain, vivado)
    )
    ((_, args, _),) = vivado.runs
    (project,) = directory.iterdir()
    assert args[-1] == f"{project}/ip_config.tcl"
    assert linked.image.is_relative_to(project)
    assert all(path.is_relative_to(project) for path in linked.reports.values())
    assert linked.placement == {"in0": 0xA000_0000, "in1": 0xA000_1000, "out0": 0xA000_2000}
    # The fake's routed timing report is TFC's on Ultra96: its PL clock as routed, the
    # design's worst setup slack there.
    assert linked.clocks_mhz == {"main": 187.512}
    assert linked.timing == TFC_TIMING and TFC_TIMING.report == TIMING


def test_the_routed_timing_is_read_from_each_linker_s_real_report() -> None:
    """Each timing check of the design (setup, hold, pulse width: worst and total slack,
    failing and timed endpoints), each clock as routed, each clock's own checks and the
    ``check_timing`` counts, from real routed reports read once: TFC's on Ultra96, which
    meets its 5.333 ns; a design routed at 1 ns that misses it, reported as the ``pynq``
    template and as ``slashkit`` report a routed design, read alike. A report with no
    summary, or one cut short, is not read."""
    tfc = routed_timing(TIMING_REPORT)
    assert tfc.state == "Routed"
    assert tfc.design == {
        "setup": Check(1.667, 0.0, 0, 63298),
        "hold": Check(0.014, 0.0, 0, 63298),
        "pulse_width": Check(1.166, 0.0, 0, 19432),
    }
    assert tfc.clocks == {"clk_pl_0": {"period_ns": 5.333, "mhz": 187.512}}
    assert tfc.intra == {"clk_pl_0": tfc.design}
    assert (tfc.unchecked["no_clock"], tfc.unchecked["unconstrained_internal_endpoints"]) == (0, 0)
    unmet = routed_timing(UNMET_REPORT)
    assert unmet.design == {
        "setup": Check(-0.906, -43.505, 48, 56),
        "hold": Check(0.125, 0.0, 0, 56),
        "pulse_width": Check(-0.499, -0.499, 1, 10),
    }
    assert unmet.clocks == {"clk_pl_0": {"period_ns": 1.0, "mhz": 1000.0}}
    assert routed_timing(UNMET_SLASHKIT_REPORT) == unmet
    with pytest.raises(ValueError, match="no design state"):
        routed_timing("no summary")
    with pytest.raises(ValueError, match="'Clock Summary' table has no header"):
        routed_timing(_cut_after("| Clock Summary"))


def _cut_after(line: str) -> str:
    """TFC's real report (``TIMING_REPORT``) cut after its first line ``line``."""
    lines = TIMING_REPORT.splitlines(keepends=True)
    return "".join(lines[: [each.strip() for each in lines].index(line) + 1])


@pytest.mark.parametrize(
    "case, cause",
    [
        ("cut", r"not a whole timing summary \(the 'Clock Summary' table has no header\)"),
        ("encoding", r"not read \(UnicodeDecodeError: 'utf-8' codec can't decode"),
        ("unreadable", r"not read \(PermissionError: \[Errno 13\]"),
    ],
)
def test_a_report_cut_short_or_unreadable_is_refused_naming_its_cause(
    tmp_path: Path, case: str, cause: str
) -> None:
    """A routed timing report cut short in its clock summary, one not UTF-8 and one that
    cannot be opened are each refused as unread (``timing-unread``), naming the report and
    the cause, not escaping as the reader's error."""
    report = tmp_path / "post_route_timing.rpt"
    if case == "cut":
        report.write_text(_cut_after("| Clock Summary"))
    elif case == "encoding":
        report.write_bytes(b"\xff" + TIMING_REPORT.encode())
    else:
        report.write_text(TIMING_REPORT)
        report.chmod(0)
    message = re.escape(str(report)) + ": " + cause
    with refused("timing-unread", error=LinkerRefused, message=message):
        admitted(report, 5.0)


#: TFC's real Ultra96 report (``TIMING_REPORT``) with its design's worst hold slack
#: rewritten from 0.014 to -0.021 ns, in the design summary and the clock's own row: no
#: real routed report on hand misses hold (Vivado's router fixes hold where it can).
HOLD_UNMET_REPORT = TIMING_REPORT.replace(
    "1.667        0.000                      0                63298        0.014",
    "1.667        0.000                      0                63298       -0.021",
)

#: TFC's real report with its pulse-width check missed: worst and total slack -0.050 ns,
#: one failing endpoint, in the design summary and the clock's row.
PULSE_WIDTH_UNMET_REPORT = TIMING_REPORT.replace(
    "1.166        0.000                       0                 19432",
    "-0.050       -0.050                       1                 19432",
)

#: TFC's real report whose check_timing counts a register with no clock and two internal
#: endpoints unconstrained, its slacks unchanged.
UNTIMED_CORE_REPORT = TIMING_REPORT.replace(
    "checking no_clock (0)", "checking no_clock (1)"
).replace(
    "checking unconstrained_internal_endpoints (0)",
    "checking unconstrained_internal_endpoints (2)",
)

#: TFC's real report with its PL clock timing no endpoint and another clock, ``static``,
#: giving the design's slacks.
UNTIMED_PL_CLOCK_REPORT = TIMING_REPORT.replace(
    "clk_pl_0  {0.000 2.667}        5.333           187.512",
    "clk_pl_0  {0.000 2.667}        5.333           187.512\n"
    "static    {0.000 5.000}        10.000          100.000",
).replace(
    "clk_pl_0            1.667        0.000                      0                63298",
    "clk_pl_0              inf        0.000                      0                    0"
    "          inf        0.000                      0                    0"
    "          inf        0.000                       0                     0\n"
    "static              1.667        0.000                      0                63298",
)

#: TFC's real report whose PL clock states a period and frequency of zero.
ZERO_PERIOD_REPORT = TIMING_REPORT.replace(
    "clk_pl_0  {0.000 2.667}        5.333           187.512",
    "clk_pl_0  {0.000 2.667}        0.000           0.000",
)

#: TFC's real report stating its design synthesized, not routed.
UNROUTED_REPORT = TIMING_REPORT.replace(
    "| Design State      : Routed", "| Design State      : Synthesized"
)

#: TFC's real report with one failing setup endpoint whose slack rounds to -0.000 ns.
SETUP_ROUNDED_REPORT = TIMING_REPORT.replace(
    "1.667        0.000                      0                63298",
    "-0.000       -0.000                      1                63298",
)


@pytest.mark.parametrize(
    "timing, code, message",
    [
        (
            UNMET_REPORT,
            "timing-unmet",
            r"top_wrapper_timing_summary_routed\.rpt: the routed design's worst setup slack is "
            r"-0\.906 ns, 48 of 56 endpoints failing, at its clk_pl_0 clock's 1\.0 ns "
            r"\(1000\.0 MHz; 5\.0 ns asked\): it misses timing; ask a longer period",
        ),
        (None, "timing-unreported", r"no routed timing report \(.*routed\.rpt\)"),
        (
            "no summary",
            "timing-unread",
            r"routed\.rpt: not a whole timing summary \(no design state\)",
        ),
        (
            HOLD_UNMET_REPORT,
            "timing-hold-unmet",
            r"routed\.rpt: the routed design's worst hold slack is -0\.021 ns, 0 of 63298 "
            r"endpoints failing, .*: it misses timing; a hold time is missed at any period",
        ),
        (
            PULSE_WIDTH_UNMET_REPORT,
            "timing-pulse-width-unmet",
            r"routed\.rpt: the routed design's worst pulse width slack is -0\.05 ns, 1 of "
            r"19432 endpoints failing, .*: it misses timing; a clock's period or pulse width "
            "is below a primitive's minimum",
        ),
        (
            SETUP_ROUNDED_REPORT,
            "timing-unmet",
            r"routed\.rpt: the routed design's worst setup slack is -0\.0 ns, 1 of 63298 "
            "endpoints failing",
        ),
    ],
    ids=["unmet", "unreported", "unread", "hold-unmet", "pulse-width-unmet", "rounded-unmet"],
)
def test_an_image_that_misses_timing_is_refused_naming_its_report_and_slack(
    described: Integration, tmp_path: Path, timing: str | None, code: str, message: str
) -> None:
    """Vivado writes the bitfile of a routed design that misses timing: the linker reads
    the routed timing report it collects and refuses an image whose worst setup slack is
    negative at the period its PL clock was routed at (``timing-unmet``: a real report,
    -0.906 ns at 1 ns), naming the report, the slack and the period; as it does an image
    whose worst hold slack is negative (``timing-hold-unmet``), one with no report
    (``timing-unreported``) and one that states no PL clock or no slack
    (``timing-unread``); and an image whose pulse-width check fails, or with a failing
    setup endpoint whose slack rounds to ``-0.000``. No real routed report on hand misses
    hold or pulse width, or rounds a violation to zero: those cases are TFC's Ultra96
    report with its rows rewritten (``HOLD_UNMET_REPORT``, ``PULSE_WIDTH_UNMET_REPORT``,
    ``SETUP_ROUNDED_REPORT``)."""
    vivado = FakeVivado(timing=timing)
    with refused(code, error=LinkerRefused, owner="vivado-block-design", message=message):
        BlockDesignLinker().link(packaged(described), BOARD, tmp_path, cast(Toolchain, vivado))


@pytest.mark.parametrize(
    "timing, code, message",
    [
        (
            UNTIMED_CORE_REPORT,
            "timing-uncovered",
            r"routed\.rpt: the routed design leaves paths untimed \(check_timing: no_clock 1, "
            r"unconstrained_internal_endpoints 2\)",
        ),
        (
            UNTIMED_PL_CLOCK_REPORT,
            "timing-uncovered",
            r"routed\.rpt: its clk_pl_0 clock, the accelerator's, times no setup endpoint",
        ),
        (
            UNROUTED_REPORT,
            "timing-unread",
            r"routed\.rpt: it states the design 'Synthesized', not routed",
        ),
        (
            ZERO_PERIOD_REPORT,
            "timing-unread",
            r"routed\.rpt: its clk_pl_0 clock's period, 0\.0 ns, or 0\.0 MHz is not positive",
        ),
    ],
    ids=["untimed-paths", "untimed-pl-clock", "unrouted", "zero-period"],
)
def test_an_image_whose_timing_does_not_cover_it_is_refused(
    described: Integration, tmp_path: Path, timing: str, code: str, message: str
) -> None:
    """A finite slack proves nothing of paths it does not time: the linker refuses a
    routed design whose ``check_timing`` counts registers with no clock or internal
    endpoints unconstrained (``timing-uncovered``), one whose PL clock, the accelerator's,
    times no setup endpoint while another clock gives the design's slacks
    (``timing-uncovered``), a report of a design not routed and one whose PL clock's
    period is not positive (``timing-unread``). No
    real routed report on hand leaves paths untimed: these are TFC's Ultra96 report
    rewritten (``UNTIMED_CORE_REPORT``, ``UNTIMED_PL_CLOCK_REPORT``, ``UNROUTED_REPORT``)."""
    vivado = FakeVivado(timing=timing)
    with refused(code, error=LinkerRefused, owner="vivado-block-design", message=message):
        BlockDesignLinker().link(packaged(described), BOARD, tmp_path, cast(Toolchain, vivado))
