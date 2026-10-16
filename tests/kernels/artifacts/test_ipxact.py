# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""IP-XACT packaging: the Tcl and ``vivado_stitch_ifnames`` from a module's ABI, as text.

Nothing here runs Vivado: ``tests/kernel_ops/test_package.py`` packages a
partition with it (marker ``vivado``) and reads the IP back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.ipxact_ports import PORTS
from support.ipxact_script import BusInterface, interfaces

from finn.kernels.artifacts.abi import (
    Bus,
    Data,
    Direction,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import EmittedModule
from finn.kernels.artifacts.interface import STREAM_FACTS, describe_interface
from finn.kernels.artifacts.ipxact import (
    IpxactError,
    interface_names,
    interface_tcl,
    package_tcl,
    vlnv,
)


def test_the_interfaces_are_declared_from_the_abi_and_none_inferred() -> None:
    tcl = interface_tcl(PORTS, 5.0)
    declared = interfaces(tcl)
    assert list(declared) == ["ap_clk", "ap_clk2x", "ap_rst_n", "s_axis_0", "m_axis_0", "s_axilite"]
    # Order is the fact: what package_project inferred goes first, interfaces and memory
    # maps, or the declarations would be removed with it.
    removed = [
        position
        for position, line in enumerate(tcl)
        if "ipx::remove_bus_interface" in line or "ipx::remove_memory_map" in line
    ]
    added = [position for position, line in enumerate(tcl) if "ipx::add_bus_interface" in line]
    assert len(removed) == 2 and max(removed) < min(added)
    clock = declared["ap_clk"]
    assert clock.mode == "slave"
    assert clock.port_maps == {"CLK": "ap_clk"}
    assert clock.parameters["FREQ_HZ"] == "200000000"
    assert clock.parameters["ASSOCIATED_BUSIF"] == "s_axis_0:m_axis_0:s_axilite"
    assert clock.parameters["ASSOCIATED_RESET"] == "ap_rst_n"
    # A derived clock: its rate a multiple of its reference's, no buses, no reset.
    doubled = declared["ap_clk2x"]
    assert doubled.parameters == {"FREQ_HZ": "400000000"}
    assert declared["ap_rst_n"].parameters["POLARITY"] == "ACTIVE_LOW"
    source = declared["s_axis_0"]
    assert source.mode == "slave"
    assert source.port_maps["TDATA"] == "s_axis_0_tdata"
    assert source.parameters["TDATA_NUM_BYTES"] == "2"
    assert declared["m_axis_0"].mode == "master"
    control = declared["s_axilite"]
    assert control.bus_type == "xilinx.com:interface:aximm:1.0"
    assert control.port_maps["AWADDR"] == "s_axilite_AWADDR"
    assert control.parameters["PROTOCOL"] == "AXI4LITE"
    assert control.memory_map is not None
    [block] = control.memory_map.blocks
    assert (block.name, block.range, block.usage) == ("Reg0", 4096, "register")
    assert control.memory_map_ref == "s_axilite"
    # Order is the fact: the last statement makes every parameter above it the user's.
    assert tcl[-1].startswith("set_property value_resolve_type user")


def test_an_active_high_reset_says_so() -> None:
    reset = Signal("rst", Direction.IN, 1, Reset(False, True, ("ap_clk",)))
    declared = interfaces(interface_tcl((PORTS[0], reset), 5.0))
    assert declared["rst"].parameters["POLARITY"] == "ACTIVE_HIGH"


@pytest.mark.parametrize("clock_ns", [5.0, 3.3])
def test_the_script_and_the_interface_description_state_the_same_interface(
    clock_ns: float,
) -> None:
    declared = interfaces(interface_tcl(PORTS, clock_ns))
    # Stream facts the description needs of the caller: only the tdata width is the pin's.
    facts = {
        pin.name: {
            **dict.fromkeys(STREAM_FACTS),
            "tdata": next(member.width for member in pin.signals if member.logical == "tdata"),
        }
        for pin in PORTS
        if isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXIS
    }
    described = describe_interface(PORTS, facts, part="xczu3eg-sbva484-1-e", period_ns=clock_ns)

    def kind(bus_type: str) -> list[str]:
        return [name for name, bus in declared.items() if bus.bus_type == bus_type]

    assert kind("xilinx.com:signal:clock:1.0") == [clock["name"] for clock in described["clocks"]]
    assert kind("xilinx.com:signal:reset:1.0") == [reset["name"] for reset in described["resets"]]
    assert kind("xilinx.com:interface:axis:1.0") == [s["name"] for s in described["streams"]]
    assert kind("xilinx.com:interface:aximm:1.0") == [a["name"] for a in described["axilite"]]
    for clock in described["clocks"]:
        parameters = declared[clock["name"]].parameters
        assert int(parameters["FREQ_HZ"]) == clock["freq_hz"]
        if "derived" in clock:
            # The derived clock: a multiple of its reference's, and no reset of its own.
            of = clock["derived"]
            assert clock["freq_hz"] == of["ratio"] * int(declared[of["of"]].parameters["FREQ_HZ"])
            assert "ASSOCIATED_RESET" not in parameters
        else:
            # A reset is associated with the free clocks it is synchronous to.
            resets = {
                reset["name"]
                for reset in described["resets"]
                if clock["name"] in reset["synchronous_to"]
            }
            assert set(parameters["ASSOCIATED_RESET"].split(":")) == resets
    for reset in described["resets"]:
        assert declared[reset["name"]].parameters["POLARITY"] == reset["polarity"]
    for stream in described["streams"]:
        bus = declared[stream["name"]]
        assert bus.mode == ("slave" if stream["direction"] == "in" else "master")
        assert int(bus.parameters["TDATA_NUM_BYTES"]) == -(-stream["tdata"] // 8)
        assert stream["name"] in declared[stream["clock"]].parameters["ASSOCIATED_BUSIF"].split(":")
    for lite in described["axilite"]:
        bus = declared[lite["name"]]
        assert bus.mode == "slave"
        assert associated_clock(declared, lite["name"]) == lite["clock"]
        assert bus.memory_map is not None and bus.memory_map.name == lite["name"]
        [block] = bus.memory_map.blocks
        [window] = lite["register_map"]
        assert (block.name, block.base, block.range, block.width, block.usage) == (
            window["name"],
            window["base"],
            window["range"],
            window["width"],
            window["usage"],
        )
        assert block.range >= 1 << lite["address_width"]


def associated_clock(declared: dict[str, BusInterface], bus: str) -> str:
    """The one clock whose ``ASSOCIATED_BUSIF`` lists ``bus``."""
    [clock] = [
        name
        for name, interface in declared.items()
        if bus in interface.parameters.get("ASSOCIATED_BUSIF", "").split(":")
    ]
    return clock


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
