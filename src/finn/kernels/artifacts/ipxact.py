# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""IP-XACT packaging of an emitted module: the Vivado Tcl and the interface names.

Text only, over a module's ABI pins (``abi``) and its emitted files
(``build.EmittedModule``); nothing here runs Vivado. ``package_tcl`` is the
batch script that packages a module as an IP (from its sources, or from an
out-of-context checkpoint with ``run_synth``); ``interface_tcl`` declares every
bus interface from the pins, none left to Vivado's inference; ``interface_names``
is the ``vivado_stitch_ifnames`` metadata CreateStitchedIP writes; ``vlnv`` is a
packaged IP's VLNV. ``finn.transformation.kernels.package`` (PackagePartition)
packages a partition with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from finn.kernels.artifacts.abi import (
    Bus,
    Clock,
    Derived,
    Endpoint,
    Pin,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.kernels.artifacts.build import EmittedModule
from finn.kernels.artifacts.sources import include_directories


class IpxactError(Exception):
    """A pin the IP-XACT packaging cannot declare."""


VENDOR, LIBRARY, VERSION = "xilinx_finn", "finn", "1.0"

# IP-XACT bus and abstraction types per ABI protocol.
_INTERFACES = {
    StandardProtocol.AXIS: ("xilinx.com:interface:axis:1.0", "xilinx.com:interface:axis_rtl:1.0"),
    StandardProtocol.AXILITE: (
        "xilinx.com:interface:aximm:1.0",
        "xilinx.com:interface:aximm_rtl:1.0",
    ),
}


def vlnv(ip_name: str) -> str:
    """The VLNV of the IP ``package_tcl`` packages as ``ip_name``."""
    return f"{VENDOR}:{LIBRARY}:{ip_name}:{VERSION}"


def _width(bus: Bus, logical: str) -> int:
    return next(member.width for member in bus.signals if member.logical == logical)


def _frequency(pin: Signal, clock_ns: float, pins: Sequence[Pin]) -> int:
    rate = pin.role.rate if isinstance(pin.role, Clock) else None
    if isinstance(rate, Derived):
        reference = next(p for p in pins if isinstance(p, Signal) and p.name == rate.of)
        return rate.ratio * _frequency(reference, clock_ns, pins)
    return round(1e9 / clock_ns)


def interface_tcl(pins: Sequence[Pin], clock_ns: float) -> list[str]:
    """Tcl that declares every bus interface of ``pins`` on ``[ipx::current_core]``.

    It first removes what ``ipx::package_project`` inferred (interfaces and memory
    maps). A clock's ``FREQ_HZ`` is its rate at ``clock_ns`` (a derived clock's a
    multiple of its reference's) and, as every bus parameter, resolvable by the
    user: the shell's frequency wins.
    """
    tcl = [
        "set core [ipx::current_core]",
        "foreach item [ipx::get_bus_interfaces -of_objects $core] {",
        "    ipx::remove_bus_interface [get_property NAME $item] $core",
        "}",
        "foreach item [ipx::get_memory_maps -of_objects $core] {",
        "    ipx::remove_memory_map [get_property NAME $item] $core",
        "}",
    ]
    buses = [pin for pin in pins if isinstance(pin, Bus)]
    for pin in pins:
        if isinstance(pin, Signal) and isinstance(pin.role, Clock):
            tcl += [
                f"set bus [ipx::add_bus_interface {pin.name} $core]",
                "set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $bus",
                "set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $bus",
                "set_property interface_mode slave $bus",
                f"set_property physical_name {pin.name} [ipx::add_port_map CLK $bus]",
                f"set_property value {_frequency(pin, clock_ns, pins)}"
                " [ipx::add_bus_parameter FREQ_HZ $bus]",
            ]
            associated = [bus.name for bus in buses if bus.associated_clock == pin.name]
            if associated:
                tcl.append(
                    f"set_property value {':'.join(associated)}"
                    " [ipx::add_bus_parameter ASSOCIATED_BUSIF $bus]"
                )
            resets = [
                reset.name
                for reset in pins
                if isinstance(reset, Signal)
                and isinstance(reset.role, Reset)
                and pin.name in (reset.role.synchronous_to or ())
            ]
            if resets and not isinstance(pin.role.rate, Derived):
                tcl.append(
                    f"set_property value {':'.join(resets)}"
                    " [ipx::add_bus_parameter ASSOCIATED_RESET $bus]"
                )
        elif isinstance(pin, Signal) and isinstance(pin.role, Reset):
            polarity = "ACTIVE_LOW" if pin.role.active_low else "ACTIVE_HIGH"
            tcl += [
                f"set bus [ipx::add_bus_interface {pin.name} $core]",
                "set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $bus",
                "set_property bus_type_vlnv xilinx.com:signal:reset:1.0 $bus",
                "set_property interface_mode slave $bus",
                f"set_property physical_name {pin.name} [ipx::add_port_map RST $bus]",
                f"set_property value {polarity} [ipx::add_bus_parameter POLARITY $bus]",
            ]
        elif isinstance(pin, Bus):
            bus_type, abstraction = _INTERFACES[pin.protocol]
            mode = "slave" if pin.endpoint is Endpoint.TARGET else "master"
            tcl += [
                f"set bus [ipx::add_bus_interface {pin.name} $core]",
                f"set_property abstraction_type_vlnv {abstraction} $bus",
                f"set_property bus_type_vlnv {bus_type} $bus",
                f"set_property interface_mode {mode} $bus",
                *(
                    f"set_property physical_name {member.physical}"
                    f" [ipx::add_port_map {member.logical.upper()} $bus]"
                    for member in pin.signals
                ),
            ]
            if pin.protocol is StandardProtocol.AXIS:
                data_bytes = -(-_width(pin, "tdata") // 8)
                tcl.append(
                    f"set_property value {data_bytes} [ipx::add_bus_parameter TDATA_NUM_BYTES $bus]"
                )
            else:
                window = max(2 ** _width(pin, "awaddr"), 4096)
                tcl += [
                    "set_property value AXI4LITE [ipx::add_bus_parameter PROTOCOL $bus]",
                    f"set map [ipx::add_memory_map {pin.name} $core]",
                    "set block [ipx::add_address_block Reg0 $map]",
                    f"set_property range {window} $block",
                    "set_property width 32 $block",
                    "set_property usage register $block",
                    f"set_property slave_memory_map_ref {pin.name} $bus",
                ]
        else:
            raise IpxactError(f"{pin.name}: a pin outside every bus interface")
    tcl.append(
        "set_property value_resolve_type user"
        " [ipx::get_bus_parameters -of_objects [ipx::get_bus_interfaces -of_objects $core]]"
    )
    return tcl


def interface_names(pins: Sequence[Pin]) -> dict[str, list[Any]]:
    """``vivado_stitch_ifnames`` as CreateStitchedIP writes it: each stream with its tdata
    width, ``clk2x`` only when a clock is derived."""
    names: dict[str, list[Any]] = {
        "clk": [],
        "rst": [],
        "s_axis": [],
        "m_axis": [],
        "aximm": [],
        "axilite": [],
        "ap_none": [],
    }
    for pin in pins:
        if isinstance(pin, Signal) and isinstance(pin.role, Clock):
            key = "clk2x" if isinstance(pin.role.rate, Derived) else "clk"
            names.setdefault(key, []).append(pin.name)
        elif isinstance(pin, Signal) and isinstance(pin.role, Reset):
            names["rst"].append(pin.name)
        elif isinstance(pin, Bus) and pin.protocol is StandardProtocol.AXIS:
            key = "s_axis" if pin.endpoint is Endpoint.TARGET else "m_axis"
            names[key].append([pin.name, _width(pin, "tdata")])
        elif isinstance(pin, Bus):
            names["axilite"].append(pin.name)
        else:
            raise IpxactError(f"{pin.name}: a pin outside every bus interface")
    return names


def package_tcl(
    emitted: EmittedModule,
    pins: Sequence[Pin],
    *,
    part: str,
    clock_ns: float,
    ip_name: str,
    run_synth: bool,
) -> str:
    """The Vivado batch script that packages ``emitted`` in ``emitted.directory``'s parent."""
    top = emitted.entry_point
    # A header is found where its includers read it from.
    includes = " ".join(f"$project/src/{d}" for d in include_directories(emitted.sources))
    tcl = [
        "set project [file normalize [file dirname [info script]]]",
        f"create_project -force {top} $project/project -part {part}",
        *(f"add_files -norecurse $project/src/{path}" for path in emitted.sources),
        *(f"add_files -norecurse $project/src/{path}" for path in emitted.data),
        *([f"set_property include_dirs [list {includes}] [current_fileset]"] if includes else []),
        f"set_property top {top} [current_fileset]",
        "update_compile_order -fileset sources_1",
    ]
    if run_synth:
        doubled = [
            pin.name
            for pin in pins
            if isinstance(pin, Signal)
            and isinstance(pin.role, Clock)
            and isinstance(pin.role.rate, Derived)
        ]
        tcl += [
            "set xdc [open $project/clocks.xdc w]",
            f'puts $xdc "create_clock -period {clock_ns} -name ap_clk \\[get_ports ap_clk\\]"',
            *(
                f'puts $xdc "create_clock -period {clock_ns / 2} -name {name}'
                f' \\[get_ports {name}\\]"'
                for name in doubled
            ),
            "close $xdc",
            "add_files -fileset constrs_1 -norecurse $project/clocks.xdc",
            "set_property USED_IN {synthesis out_of_context} [get_files $project/clocks.xdc]",
            "set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}"
            " -value {-mode out_of_context} -objects [get_runs synth_1]",
            "launch_runs synth_1 -jobs 8",
            "wait_on_run [get_runs synth_1]",
            'if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {',
            '    error "synthesis of the partition failed"',
            "}",
            "open_run synth_1 -name synth_1",
            f"write_verilog -force -mode synth_stub $project/{top}_stub.v",
            f"write_checkpoint -force $project/{top}.dcp",
            f"write_xdc -force $project/{top}.xdc",
            f"report_utilization -hierarchical -file $project/{top}_partition_util.rpt",
            "close_design",
            "remove_files -fileset constrs_1 [get_files $project/clocks.xdc]",
        ]
    tcl += [
        f"ipx::package_project -root_dir $project/ip -vendor {VENDOR} -library {LIBRARY}"
        " -taxonomy /UserIP -import_files -set_current true",
        *interface_tcl(pins, clock_ns),
        f"set_property name {ip_name} $core",
        f"set_property display_name {ip_name} $core",
        f"set_property description {{FINN partition {ip_name} ({top})}} $core",
        f"set_property version {VERSION} $core",
        "set_property core_revision 1 $core",
        "set_property ipi_drc {ignore_freq_hz true} $core",
    ]
    if run_synth:
        tcl += [
            "set_property sdx_kernel true $core",
            "set_property sdx_kernel_type rtl $core",
            "set_property supported_families { } $core",
            "set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core",
            "set_property auto_family_support_level level_2 $core",
            "foreach group {xilinx_anylanguagebehavioralsimulation xilinx_anylanguagesynthesis} {",
            "    ipx::remove_all_file [ipx::get_file_groups $group]",
            "    ipx::remove_file_group $group $core",
            "}",
            "file delete -force $project/ip/sim $project/ip/src",
            "file mkdir $project/ip/dcp $project/ip/impl",
            f"file copy -force $project/{top}.dcp $project/ip/dcp",
            f"file copy -force $project/{top}.xdc $project/ip/impl",
            "ipx::add_file_group xilinx_implementation $core",
            f"set_property used_in [list implementation] [ipx::add_file impl/{top}.xdc"
            " [ipx::get_file_groups xilinx_implementation]]",
            "ipx::add_file_group xilinx_synthesischeckpoint $core",
            f"ipx::add_file dcp/{top}.dcp [ipx::get_file_groups xilinx_synthesischeckpoint]",
            "ipx::add_file_group xilinx_simulationcheckpoint $core",
            f"ipx::add_file dcp/{top}.dcp [ipx::get_file_groups xilinx_simulationcheckpoint]",
        ]
    tcl += [
        "ipx::create_xgui_files $core",
        "ipx::update_checksums $core",
        "if {![ipx::check_integrity -quiet $core]} {",
        "    ipx::check_integrity $core",
        '    error "the packaged partition fails its integrity check"',
        "}",
        "ipx::save_core $core",
        "",
    ]
    return "\n".join(tcl)


__all__ = [
    "IpxactError",
    "interface_names",
    "interface_tcl",
    "package_tcl",
    "vlnv",
]
