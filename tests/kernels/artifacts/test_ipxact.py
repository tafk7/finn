# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""IP-XACT packaging: the Tcl and ``vivado_stitch_ifnames`` from a module's ABI, as text.

Nothing here runs Vivado: ``tests/kernel_ops/test_package.py`` packages a
partition with it (marker ``vivado``) and reads the IP back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Data,
    Derived,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import EmittedModule
from finn.kernels.artifacts.ipxact import (
    IpxactError,
    interface_names,
    interface_tcl,
    package_tcl,
    vlnv,
)


def stream(name: str, width: int, endpoint: Endpoint) -> Bus:
    members = [Member("tdata", f"{name}_tdata", width)]
    members += [Member(logical, f"{name}_{logical}") for logical in ("tvalid", "tready")]
    return Bus(
        name,
        StandardProtocol.AXIS,
        members,
        endpoint,
        associated_clock="ap_clk",
        associated_reset="ap_rst_n",
    )


AXILITE = Bus(
    "s_axilite",
    StandardProtocol.AXILITE,
    [
        Member(logical, f"s_axilite_{logical.upper()}", width)
        for logical, width in (("awaddr", 5), ("awvalid", 1), ("wdata", 32), ("rdata", 32))
    ],
    Endpoint.TARGET,
    associated_clock="ap_clk",
    associated_reset="ap_rst_n",
)
PORTS = (
    Signal("ap_clk", Direction.IN, 1, Clock(Free())),
    Signal("ap_clk2x", Direction.IN, 1, Clock(Derived("ap_clk", 2))),
    Signal("ap_rst_n", Direction.IN, 1, Reset(True, True, ("ap_clk", "ap_clk2x"))),
    stream("s_axis_0", 12, Endpoint.TARGET),
    stream("m_axis_0", 8, Endpoint.INITIATOR),
    AXILITE,
)


def test_the_interfaces_are_declared_from_the_abi_and_none_inferred() -> None:
    tcl = interface_tcl(PORTS, 5.0)
    text = "\n".join(tcl)
    # What package_project inferred goes first.
    assert tcl[1:7] == [
        "foreach item [ipx::get_bus_interfaces -of_objects $core] {",
        "    ipx::remove_bus_interface [get_property NAME $item] $core",
        "}",
        "foreach item [ipx::get_memory_maps -of_objects $core] {",
        "    ipx::remove_memory_map [get_property NAME $item] $core",
        "}",
    ]
    clock = text.split("set bus [ipx::add_bus_interface ap_clk $core]")[1].split("set bus")[0]
    assert "set_property value 200000000 [ipx::add_bus_parameter FREQ_HZ $bus]" in clock
    assert (
        "set_property value s_axis_0:m_axis_0:s_axilite [ipx::add_bus_parameter ASSOCIATED_BUSIF"
        in clock
    )
    assert "set_property value ap_rst_n [ipx::add_bus_parameter ASSOCIATED_RESET $bus]" in clock
    # A derived clock: its rate a multiple of its reference's, no buses, no reset.
    doubled = text.split("set bus [ipx::add_bus_interface ap_clk2x $core]")[1].split("set bus")[0]
    assert "set_property value 400000000 [ipx::add_bus_parameter FREQ_HZ $bus]" in doubled
    assert "ASSOCIATED" not in doubled
    assert "set_property value ACTIVE_LOW [ipx::add_bus_parameter POLARITY $bus]" in text
    source = text.split("set bus [ipx::add_bus_interface s_axis_0 $core]")[1].split("set bus")[0]
    assert "set_property interface_mode slave $bus" in source
    assert "set_property physical_name s_axis_0_tdata [ipx::add_port_map TDATA $bus]" in source
    assert "set_property value 2 [ipx::add_bus_parameter TDATA_NUM_BYTES $bus]" in source
    sink = text.split("set bus [ipx::add_bus_interface m_axis_0 $core]")[1].split("set bus")[0]
    assert "set_property interface_mode master $bus" in sink
    control = text.split("set bus [ipx::add_bus_interface s_axilite $core]")[1]
    assert "xilinx.com:interface:aximm:1.0" in control
    assert "set_property physical_name s_axilite_AWADDR [ipx::add_port_map AWADDR $bus]" in control
    assert "set_property value AXI4LITE [ipx::add_bus_parameter PROTOCOL $bus]" in control
    assert "set block [ipx::add_address_block Reg0 $map]" in control
    assert "set_property range 4096 $block" in control
    assert "set_property usage register $block" in control
    assert "set_property slave_memory_map_ref s_axilite $bus" in control
    assert tcl[-1].startswith("set_property value_resolve_type user")


def test_an_active_high_reset_says_so() -> None:
    reset = Signal("rst", Direction.IN, 1, Reset(False, True, ("ap_clk",)))
    text = "\n".join(interface_tcl((PORTS[0], reset), 5.0))
    assert "set_property value ACTIVE_HIGH [ipx::add_bus_parameter POLARITY $bus]" in text


def test_the_interface_names_are_the_stitched_ip_metadata() -> None:
    assert interface_names(PORTS) == {
        "clk": ["ap_clk"],
        "clk2x": ["ap_clk2x"],
        "rst": ["ap_rst_n"],
        "s_axis": [["s_axis_0", 12]],
        "m_axis": [["m_axis_0", 8]],
        "aximm": [],
        "axilite": ["s_axilite"],
        "ap_none": [],
    }
    assert "clk2x" not in interface_names((PORTS[0], PORTS[2]))


def test_the_script_packages_sources_or_a_checkpoint() -> None:
    emitted = EmittedModule("top__0", Path("/p/src"), ("a.sv", "top__0.sv"), ("m.dat",))
    plain = package_tcl(
        emitted, PORTS, part="xczu3eg-sbva484-1-e", clock_ns=5.0, ip_name="sdp_1", run_synth=False
    )
    assert "add_files -norecurse $project/src/m.dat" in plain
    assert "set_property top top__0 [current_fileset]" in plain
    assert "-import_files" in plain and "launch_runs" not in plain
    assert "set_property name sdp_1 $core" in plain
    assert plain.index("ipx::package_project") < plain.index("ipx::remove_bus_interface")
    synthesized = package_tcl(
        emitted, PORTS, part="xczu3eg-sbva484-1-e", clock_ns=5.0, ip_name="sdp_1", run_synth=True
    )
    assert "-mode out_of_context" in synthesized
    assert "create_clock -period 2.5 -name ap_clk2x" in synthesized
    assert "ipx::add_file dcp/top__0.dcp [ipx::get_file_groups xilinx_synthesischeckpoint]" in (
        synthesized
    )
    assert "set_property sdx_kernel true $core" in synthesized
    assert synthesized.index("write_checkpoint") < synthesized.index("ipx::package_project")
    assert vlnv("sdp_1") == "xilinx_finn:finn:sdp_1:1.0"


def test_the_script_searches_the_headers_directories_and_only_with_headers() -> None:
    def script(sources: tuple[str, ...]) -> str:
        emitted = EmittedModule("top__0", Path("/p/src"), sources, ())
        return package_tcl(
            emitted,
            PORTS,
            part="xczu3eg-sbva484-1-e",
            clock_ns=5.0,
            ip_name="sdp_1",
            run_synth=False,
        )

    with_headers = script(("arith/s.svh", "arith/a.sv", "top__0.sv"))
    assert "add_files -norecurse $project/src/arith/s.svh" in with_headers
    assert "set_property include_dirs [list $project/src/arith] [current_fileset]" in with_headers
    assert "include_dirs" not in script(("a.sv", "top__0.sv"))


def test_a_pin_outside_every_bus_interface_is_refused() -> None:
    loose = Signal("dout", Direction.OUT, 8, Data())
    with pytest.raises(IpxactError, match="dout: a pin outside every bus interface"):
        interface_tcl((PORTS[0], loose), 5.0)
    with pytest.raises(IpxactError, match="dout: a pin outside every bus interface"):
        interface_names((PORTS[0], loose))
