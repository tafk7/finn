# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's linker (``finn.shells.pynq.linker``): the cells, the movers'
``IODMA_hls`` nodes, the wires and the placement it gives the Chain's description on
``pynq``, its second weights streamed (two inputs and one output); what it refuses.
No Vivado runs: the runner's tests link through a fake (``test_pynq_runner``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from support.models import zynq_parent
from support.refusals import refused

from finn.platform.shells import shell_row
from finn.shells.integration import Integration, IntegrationError, IpRef, describe
from finn.shells.pynq.linker import (
    Address,
    LinkerRefused,
    Wire,
    block_design,
    cell,
    iodma_configuration,
    linked_row,
    pin,
    placement,
    report,
    wire_one_bus_each,
    wiring,
)


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
