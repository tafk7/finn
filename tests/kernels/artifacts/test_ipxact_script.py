# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The reader of the IP-XACT interface script (``support.ipxact_script``), on a script by hand."""

from __future__ import annotations

import pytest
from support.ipxact_script import AddressBlock, MemoryMap, ScriptError, interfaces

SCRIPT = [
    "set core [ipx::current_core]",
    "foreach item [ipx::get_bus_interfaces -of_objects $core] {",
    "    ipx::remove_bus_interface [get_property NAME $item] $core",
    "}",
    "set bus [ipx::add_bus_interface clk $core]",
    "set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $bus",
    "set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $bus",
    "set_property interface_mode slave $bus",
    "set_property physical_name clk [ipx::add_port_map CLK $bus]",
    "set_property value 100000000 [ipx::add_bus_parameter FREQ_HZ $bus]",
    "set bus [ipx::add_bus_interface ctl $core]",
    "set_property interface_mode master $bus",
    "set_property physical_name ctl_awaddr [ipx::add_port_map AWADDR $bus]",
    "set map [ipx::add_memory_map ctl $core]",
    "set block [ipx::add_address_block Reg0 $map]",
    "set_property range 4096 $block",
    "set_property width 32 $block",
    "set_property usage register $block",
    "set_property slave_memory_map_ref ctl $bus",
    "set_property value_resolve_type user [ipx::get_bus_parameters -of_objects $core]",
]


def test_a_script_is_read_as_its_interfaces() -> None:
    read = interfaces(SCRIPT)
    assert list(read) == ["clk", "ctl"]
    clock = read["clk"]
    assert (clock.mode, clock.bus_type, clock.abstraction_type) == (
        "slave",
        "xilinx.com:signal:clock:1.0",
        "xilinx.com:signal:clock_rtl:1.0",
    )
    assert clock.port_maps == {"CLK": "clk"}
    assert clock.parameters == {"FREQ_HZ": "100000000"}
    assert clock.memory_map is None and clock.memory_map_ref is None
    control = read["ctl"]
    assert control.mode == "master"
    assert control.port_maps == {"AWADDR": "ctl_awaddr"}
    assert control.memory_map == MemoryMap("ctl", [AddressBlock("Reg0", 0, 4096, 32, "register")])
    assert control.memory_map_ref == "ctl"


def test_a_script_with_no_interface_reads_as_none() -> None:
    assert interfaces([]) == {}
    assert interfaces(SCRIPT[:4]) == {}


@pytest.mark.parametrize(
    "statement",
    [
        "set_property display_name x $bus",  # an unknown property of an interface
        "set_property physical_name clk [ipx::add_port_map CLK $bus] ;",  # a trailing word
        "ipx::infer_bus_interface clk xilinx.com:signal:clock_rtl:1.0 $core",  # a new command
    ],
)
def test_a_statement_it_does_not_know_is_refused(statement: str) -> None:
    with pytest.raises(ScriptError, match="a statement it does not know") as info:
        interfaces([*SCRIPT[:8], statement, *SCRIPT[8:]])
    assert statement in str(info.value)


def test_a_statement_before_every_interface_that_is_not_the_removal_is_refused() -> None:
    with pytest.raises(ScriptError, match="outside any interface"):
        interfaces(["set_property interface_mode slave $bus", *SCRIPT])


def test_the_removal_of_what_was_inferred_after_an_interface_is_refused() -> None:
    with pytest.raises(ScriptError, match="removed after a declaration"):
        interfaces([*SCRIPT[:5], *SCRIPT[1:4], *SCRIPT[5:]])


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("set bus [ipx::add_bus_interface clk $core]", "declared twice"),
        ("set_property physical_name other [ipx::add_port_map CLK $bus]", "mapped twice"),
        ("set_property value 1 [ipx::add_bus_parameter FREQ_HZ $bus]", "set twice"),
    ],
)
def test_a_name_declared_twice_is_refused(line: str, why: str) -> None:
    with pytest.raises(ScriptError, match=why):
        interfaces([*SCRIPT[:10], line, *SCRIPT[10:]])


def test_a_block_property_outside_its_block_is_refused() -> None:
    with pytest.raises(ScriptError, match="no address block"):
        interfaces([*SCRIPT[:10], "set_property range 4096 $block"])
    with pytest.raises(ScriptError, match="no memory map"):
        interfaces([*SCRIPT[:10], "set block [ipx::add_address_block Reg0 $map]"])
