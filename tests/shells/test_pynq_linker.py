# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's linker (``finn.shells.pynq.linker``): the cells, the movers'
``IODMA_hls`` nodes, the wires and the placement it gives the Chain's description on
``pynq``, its second weights streamed (two inputs and one output); what it refuses.
No Vivado runs: the runner's tests link through a fake (``test_pynq_runner``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from support.models import zynq_parent
from support.packaging import FakeVivado
from support.refusals import refused

from finn.platform.shells import shell_row
from finn.shells.integration import (
    CORE,
    Integration,
    IntegrationError,
    IpRef,
    Unit,
    describe,
)
from finn.shells.pynq.linker import (
    Address,
    BlockDesignLinker,
    LinkerRefused,
    Wire,
    block_design,
    cell,
    iodma_configuration,
    linked_row,
    pin,
    placement,
    project_script,
    report,
    wire_one_bus_each,
    wiring,
)
from finn.toolchain.launcher import Toolchain

#: The facts the ``pynq`` shell declares of Ultra96 for its linker.
BOARD = {"board_part": "avnet.com:ultra96v1:part0:1.2"}


def packaged(described: Integration) -> Integration:
    """``described``, each unit's IP named and found where the test states."""
    ips = {unit.name: IpRef(f"v:{unit.name}", (f"/{unit.name}",)) for unit in described.units}
    return described.packaged(ips)


@pytest.fixture
def described(tmp_path: Path) -> Integration:
    return describe(zynq_parent(tmp_path))


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("in0", "idma0"),
        ("in12", "idma12"),
        ("out3", "odma3"),
        ("DataflowCore_0", "DataflowCore_0"),
        ("input", "input"),
    ],
)
def test_a_mover_unit_is_a_cell_by_its_direction_and_a_core_by_its_name(
    unit: str, expected: str
) -> None:
    assert cell(unit) == expected


def test_the_chain_is_wired_as_its_block_design(described: Integration) -> None:
    """Each mover's ``IODMA_hls`` configured from its facts: a frame's beats, the stream's
    bytes a beat in the UINT8 container, the memory port and the stream's width; the
    streams, each memory port to the memory interconnect's next slave port, each control
    bus from the control interconnect's next master port (input movers, the core, output
    movers), and every cell's clock and reset."""
    row = linked_row(described)
    assert [iodma_configuration(edge).attributes for edge in described.edges] == [
        {
            "numInputVectors": [1, 2],
            "NumChannels": 1,
            "dataType": "UINT8",
            "intfWidth": 16,
            "streamWidth": 8,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 4],
            "NumChannels": 2,
            "dataType": "UINT8",
            "intfWidth": 64,
            "streamWidth": 16,
            "direction": "in",
        },
        {
            "numInputVectors": [1, 1],
            "NumChannels": 4,
            "dataType": "UINT8",
            "intfWidth": 32,
            "streamWidth": 32,
            "direction": "out",
        },
    ]
    assert pin("in0/s_axi_control", described) == "idma0/s_axi_control_0"
    assert pin("DataflowCore_0/s_axis_1", described) == "DataflowCore_0/s_axis_1"
    assert wiring(described, row) == (
        Wire("axis", "idma0/m_axis_0", "DataflowCore_0/s_axis_0"),
        Wire("axis", "idma1/m_axis_0", "DataflowCore_0/s_axis_1"),
        Wire("axis", "DataflowCore_0/m_axis_0", "odma0/s_axis_0"),
        Wire("aximm", "idma0/m_axi_gmem0", "smartconnect_0/S00_AXI"),
        Wire("aximm", "idma1/m_axi_gmem0", "smartconnect_0/S01_AXI"),
        Wire("aximm", "odma0/m_axi_gmem0", "smartconnect_0/S02_AXI"),
        Wire("axilite", "axi_interconnect_0/M00_AXI", "idma0/s_axi_control_0"),
        Wire("axilite", "axi_interconnect_0/M01_AXI", "idma1/s_axi_control_0"),
        Wire("axilite", "axi_interconnect_0/M02_AXI", "odma0/s_axi_control_0"),
        *(
            Wire(kind, f"smartconnect_0/{source}", f"{instance}/{pin}")
            for instance in ("idma0", "idma1", "DataflowCore_0", "odma0")
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
        Address("idma0/s_axi_control_0", 0xA000_0000, 4096),
        Address("idma1/s_axi_control_0", 0xA000_1000, 4096),
        Address("odma0/s_axi_control_0", 0xA000_2000, 4096),
    )
    stated = report(described)
    assert stated["addresses"][2] == {
        "interface": "odma0/s_axi_control_0",
        "offset": "0xa0002000",
        "range": 4096,
    }
    assert stated["cells"] == {
        "in0": "idma0",
        "in1": "idma1",
        "DataflowCore_0": "DataflowCore_0",
        "out0": "odma0",
    }


def test_a_mover_the_block_design_cannot_wire_is_refused_by_name(
    described: Integration,
) -> None:
    """The block design wires one AXI-Lite bus and one memory port a mover, each an
    ``IODMA_hls``: a mover stating other counts, or of another kind, is refused."""
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
    other = replace(edge, contract=replace(edge.contract, kind="other"))
    with refused(
        "mover-kind-unsupported",
        error=IntegrationError,
        message="a mover of kind 'other' is not an iodma_hls mover",
    ):
        iodma_configuration(other)


def test_a_unit_whose_ip_is_not_packaged_is_refused(described: Integration) -> None:
    """The block design instantiates every unit's IP: the movers' are generated and the
    core's packaged before it links."""
    with refused("unit-unpackaged", error=LinkerRefused, message="^in0: its IP is not packaged"):
        block_design(described, linked_row(described))
    ips = {unit.name: IpRef(f"v:{unit.name}", (f"/{unit.name}",)) for unit in described.units}
    design = block_design(described.packaged(ips), linked_row(described))
    assert "create_bd_cell -type ip -vlnv v:in0 idma0\n" in design


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


def test_the_block_design_assigns_each_control_bus_the_address_it_returns(
    described: Integration,
) -> None:
    """One map: the block design assigns each control bus the offset and range ``placement``
    gives, which the link returns, whatever the window: here 64 KiB a bus, which the
    template's own procedure would have placed 4 KiB apart."""
    wide = replace(
        packaged(described),
        endpoints=tuple(replace(each, range=0x1_0000) for each in described.endpoints),
    )
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
        message="^in1/m_axi_gmem0: bound to 'HP1'; the block design wires memory into 'HP0'",
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


def test_two_units_given_one_cell_are_refused(described: Integration) -> None:
    """A core named as a mover's cell (``idma0``) would share it with the mover."""
    core = Unit(name="idma0", kind=CORE, ip=None, window=None)
    units = (described.units[0], described.units[1], core, described.units[3])
    with refused(
        "cell-name-collides",
        error=LinkerRefused,
        message="the units in0, idma0 would share the cells idma0",
    ):
        wiring(replace(described, units=units), linked_row(described))


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
    assert linked.placement == {
        "in0/s_axi_control": 0xA000_0000,
        "in1/s_axi_control": 0xA000_1000,
        "out0/s_axi_control": 0xA000_2000,
    }
