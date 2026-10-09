# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2023-2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An end's IODMA as the IP the pynq shell's block design instantiates: its scratch model
of one ``IODMA_hls`` node (``finn.shells.pynq.iodma``) through code generation
(``prepare_ip``), Vitis HLS (``hls_synth_ip``) and a block design of its own, packaged
(``create_stitched_ip``), whose IP repositories ``ip_repositories`` names.

This is a frozen extraction of the HWCustomOp flow's ``PrepareIP``, ``HLSSynthIP`` and
``CreateStitchedIP`` (and ``ReplaceVerilogRelPaths``, ``collect_ip_dirs``) for one IODMA
node, which write the same project. Two things of theirs no IODMA reads are left out:
CreateStitchedIP's memstreamer repository (rtllib's ``memstream``, which the block
design does not instantiate) and the Vitis HLS script's custom HLS include directory
(which holds none of the headers an IODMA includes). The simulation-control module
(``sim_ctrl.v``) and the IP's rudimentary driver (``mdd/``) are the shell's own copies
(``data/``).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from importlib.resources import files
from pathlib import Path
from shutil import copytree

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.shells.pynq.iodma import IODMA_hls
from finn.util.basic import make_build_dir
from finn.util.resources import tcl_quote
from finn.util.toolchain import Toolchain, machine_toolchain


def data_path(*parts: str) -> str:
    """An existing path in the shell's data directory (``data/``)."""
    resource = files(__package__).joinpath("data", *parts)
    if not isinstance(resource, Path):
        raise RuntimeError("FINN's data requires an unpacked installation; install with pip")
    if not resource.exists():
        raise FileNotFoundError(resource)
    return str(resource.resolve())


def _iodma(model: ModelWrapper) -> IODMA_hls:
    (node,) = model.graph.node
    op = getCustomOp(node)
    if not isinstance(op, IODMA_hls):
        raise TypeError(f"{node.name}: an end's scratch model holds one IODMA_hls, not {op!r}")
    return op


def prepare_ip(model: ModelWrapper, fpgapart: str, clk: float) -> ModelWrapper:
    """The IODMA's HLS code and Vitis HLS script, for ``fpgapart`` at ``clk`` ns, in a new
    build directory its node states (``code_gen_dir_ipgen``)."""
    op = _iodma(model)
    code_gen_dir = make_build_dir(prefix="code_gen_ipgen_" + str(op.onnx_node.name) + "_")
    op.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
    op.code_generation_ipgen(model, fpgapart, clk)
    return model


def hls_synth_ip(model: ModelWrapper, toolchain: Toolchain | None = None) -> ModelWrapper:
    """The IODMA's IP, by Vitis HLS run in ``toolchain`` (by default the machine's) on
    the script ``prepare_ip`` wrote; its node states the project, the IP and its VLNV."""
    _iodma(model).ipgen_singlenode_code(toolchain=toolchain)
    return model


def _replace_verilog_relpaths(ipgen_path: str) -> None:
    """Convert ./ relative file paths to absolute ones for generated Verilog"""
    if ipgen_path is not None and os.path.isdir(ipgen_path):
        for dname, dirs, filenames in os.walk(ipgen_path):
            for fname in filenames:
                if fname.endswith(".v"):
                    fpath = os.path.join(dname, fname)
                    with open(fpath, "r") as f:
                        s = f.read()
                    old = '$readmemh(".'
                    new = '$readmemh("%s' % dname
                    s = s.replace(old, new)
                    old = '"./'
                    new = '"%s/' % dname
                    s = s.replace(old, new)
                    with open(fpath, "w") as f:
                        f.write(s)


#: The core cleanup the packaged IP gets after ``ipx::package_project``.
_CORE_CLEANUP = """
set core [ipx::current_core]

# Add rudimentary driver
file copy -force data ip/
set file_group [ipx::add_file_group -type software_driver {} $core]
set_property type mdd       [ipx::add_file data/finn_design.mdd $file_group]
set_property type tclSource [ipx::add_file data/finn_design.tcl $file_group]

# Remove all XCI references to subcores
set impl_files [ipx::get_file_groups xilinx_implementation -of $core]
foreach xci [ipx::get_files -of $impl_files {*.xci}] {
    ipx::remove_file [get_property NAME $xci] $impl_files
}

# Construct a single flat memory map for each AXI-lite interface port
foreach port [get_bd_intf_ports -filter {CONFIG.PROTOCOL==AXI4LITE}] {
    set pin $port
    set awidth ""
    while { $awidth == "" } {
        set pins [get_bd_intf_pins -of [get_bd_intf_nets -boundary_type lower -of $pin]]
        set kill [lsearch $pins $pin]
        if { $kill >= 0 } { set pins [lreplace $pins $kill $kill] }
        if { [llength $pins] != 1 } { break }
        set pin [lindex $pins 0]
        set awidth [get_property CONFIG.ADDR_WIDTH $pin]
    }
    if { $awidth == "" } {
       puts "CRITICAL WARNING: Unable to construct address map for $port."
    } {
       set range [expr 2**$awidth]
       set range [expr $range < 4096 ? 4096 : $range]
       puts "INFO: Building address map for $port: 0+:$range"
       set name [get_property NAME $port]
       set addr_block [ipx::add_address_block Reg0 [ipx::add_memory_map $name $core]]
       set_property range $range $addr_block
       set_property slave_memory_map_ref $name [ipx::get_bus_interfaces $name -of $core]
    }
}

# Finalize and Save
ipx::update_checksums $core
ipx::save_core $core

# Remove stale subcore references from component.xml
file rename -force ip/component.xml ip/component.bak
set ifile [open ip/component.bak r]
set ofile [open ip/component.xml w]
set buf [list]
set kill 0
while { [eof $ifile] != 1 } {
    gets $ifile line
    if { [string match {*<spirit:fileSet>*} $line] == 1 } {
        foreach l $buf { puts $ofile $l }
        set buf [list $line]
    } elseif { [llength $buf] > 0 } {
        lappend buf $line

        if { [string match {*</spirit:fileSet>*} $line] == 1 } {
            if { $kill == 0 } { foreach l $buf { puts $ofile $l } }
            set buf [list]
            set kill 0
        } elseif { [string match {*<xilinx:subCoreRef>*} $line] == 1 } {
            set kill 1
        }
    } else {
        puts $ofile $line
    }
}
close $ifile
close $ofile
"""


def _block_design(op: IODMA_hls) -> tuple[list[str], list[str], dict[str, list]]:
    """The IODMA's block design (its cell and the simulation-control module) and its
    connections, every port of the IODMA's external, as CreateStitchedIP makes them of a
    graph of one IODMA node; and the external interfaces' names by protocol."""
    name = op.onnx_node.name
    intf = op.get_verilog_top_module_intf_names()
    names: dict[str, list] = {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"],
        "s_axis": [],
        "m_axis": [],
        "aximm": [],
        "axilite": [],
        "ap_none": [],
    }
    create = op.code_generation_ipi()
    connect = [
        "make_bd_pins_external [get_bd_pins %s/%s]" % (name, intf["clk"][0]),
        "set_property name ap_clk [get_bd_ports ap_clk_0]",
        "make_bd_pins_external [get_bd_pins %s/%s]" % (name, intf["rst"][0]),
        "set_property name ap_rst_n [get_bd_ports ap_rst_n_0]",
    ]
    for axilite in intf["axilite"]:
        connect.append("make_bd_intf_pins_external [get_bd_intf_pins %s/%s]" % (name, axilite))
        names["axilite"].append("%s_%d" % (axilite, len(names["axilite"])))
    for index, (aximm, width) in enumerate(intf["aximm"]):
        # A generic AXI-MM master accessing global memory, its segment's range 4G.
        external = "m_axi_gmem%d" % index
        segment = "%s/Data_m_axi_gmem/SEG_%s_Reg" % (name, external)
        connect += [
            "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]" % (name, aximm),
            "set_property name %s [get_bd_intf_ports m_axi_gmem_0]" % external,
            "assign_bd_address",
            "set_property offset 0 [get_bd_addr_segs {%s}]" % segment,
            "set_property range 4G [get_bd_addr_segs {%s}]" % segment,
        ]
        names["aximm"].append((external, width))
    for kind, prefix in (("s_axis", "s_axis"), ("m_axis", "m_axis")):
        for index, (stream, width) in enumerate(intf[kind]):
            connect += [
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]" % (name, stream),
                "set_property name %s_%d [get_bd_intf_ports %s_0]" % (prefix, index, stream),
            ]
            names[kind].append(("%s_%d" % (prefix, index), width))
    # The simulation-control module, as CreateStitchedIP inserts it in every design.
    create += [
        "add_files -norecurse %s" % tcl_quote(data_path("sim_ctrl.v")),
        "create_bd_cell -type module -reference sim_ctrl sim_ctrl_0",
    ]
    connect += [
        "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins sim_ctrl_0/ap_clk]",
        "make_bd_pins_external [get_bd_pins sim_ctrl_0/sim_finish]",
        "set_property name sim_finish [get_bd_ports sim_finish_0]",
    ]
    return create, connect, names


def create_stitched_ip(
    model: ModelWrapper,
    fpgapart: str,
    clk_ns: float,
    ip_name: str,
    toolchain: Toolchain | None = None,
) -> ModelWrapper:
    """The IODMA's IP (``hls_synth_ip``'s) in a Vivado block design of its own, named
    ``ip_name``, packaged as the IP ``xilinx_finn:finn:<ip_name>:1.0`` by Vivado run in
    ``toolchain`` (by default the machine's). The scratch model states the project
    (``vivado_stitch_proj``), the VLNV, the interface names and the wrapper HDL."""
    op = _iodma(model)
    # ensure non-relative readmemh .dat files
    _replace_verilog_relpaths(op.get_nodeattr("ipgen_path"))
    ip_dir = op.get_nodeattr("ip_path")
    assert os.path.isdir(ip_dir), "IP generation directory doesn't exist."
    create, connect, names = _block_design(op)
    prjname = "finn_vivado_stitch_proj"
    vivado_stitch_proj_dir = make_build_dir(prefix="vivado_stitch_proj_")
    model.set_metadata_prop("vivado_stitch_proj", vivado_stitch_proj_dir)
    block_name = ip_name
    tcl = [
        "create_project %s %s -part %s" % (prjname, vivado_stitch_proj_dir, fpgapart),
        # no warnings on long module names
        "set_msg_config -id {[BD 41-1753]} -suppress",
        "set_property ip_repo_paths [list %s] [current_project]" % tcl_quote(ip_dir),
        "update_ip_catalog",
        'create_bd_design "%s"' % block_name,
        *create,
        *connect,
    ]
    fclk_mhz = 1 / (clk_ns * 0.001)
    fclk_hz = fclk_mhz * 1000000
    tcl.append("set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk]" % round(fclk_hz))
    tcl.append("save_bd_design")
    tcl.append("validate_bd_design")
    tcl.append("save_bd_design")
    # create wrapper hdl (for rtlsim later on)
    bd_base = "%s/%s.srcs/sources_1/bd/%s" % (vivado_stitch_proj_dir, prjname, block_name)
    bd_filename = "%s/%s.bd" % (bd_base, block_name)
    tcl.append("make_wrapper -files [get_files %s] -top" % bd_filename)
    wrapper_filename = "%s/hdl/%s_wrapper.v" % (bd_base, block_name)
    tcl.append("add_files -norecurse %s" % tcl_quote(wrapper_filename))
    model.set_metadata_prop("wrapper_filename", wrapper_filename)
    tcl.append("set_property top %s_wrapper [current_fileset]" % block_name)
    # export block design itself as an IP core
    block_vendor = "xilinx_finn"
    block_library = "finn"
    block_vlnv = "%s:%s:%s:1.0" % (block_vendor, block_library, block_name)
    model.set_metadata_prop("vivado_stitch_vlnv", block_vlnv)
    model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(names))
    tcl.append(
        (
            "ipx::package_project -root_dir %s/ip -vendor %s "
            "-library %s -taxonomy /UserIP -module %s -import_files"
        )
        % (vivado_stitch_proj_dir, block_vendor, block_library, block_name)
    )
    # Allow user to customize clock in deployment of stitched IP
    tcl.append("set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]")
    # in some cases, the IP packager seems to infer an aperture of 64K or 4G,
    # preventing address assignment of the DDR_LOW and/or DDR_HIGH segments
    # the following is a hotfix to remove this aperture during IODMA packaging
    for aximm_name, _ in names["aximm"]:
        tcl.append(
            "ipx::remove_segment -quiet %s:APERTURE_0 "
            "[ipx::get_address_spaces %s -of_objects [ipx::current_core]]"
            % (aximm_name, aximm_name)
        )
    tcl.append("set_property core_revision 2 [ipx::find_open_core %s]" % block_vlnv)
    tcl.append("ipx::create_xgui_files [ipx::find_open_core %s]" % block_vlnv)
    # mark bus interface params as user-resolvable to avoid FREQ_MHZ mismatches
    tcl.append(
        "set_property value_resolve_type user [ipx::get_bus_parameters "
        "-of [ipx::get_bus_interfaces -of [ipx::current_core ]]]"
    )
    # add a rudimentary driver mdd to get correct ranges in xparameters.h later on
    copytree(data_path("mdd"), vivado_stitch_proj_dir + "/data")
    tcl.append(_CORE_CLEANUP)
    # export list of used Verilog files (for rtlsim later on)
    tcl.append(
        "set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 "
        + "&& (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog "
        + '|| FILE_TYPE == "Verilog Header" || FILE_TYPE == VHDL '
        + "|| FILE_TYPE == XCI)}]"
    )
    v_file_list = "%s/all_verilog_srcs.txt" % vivado_stitch_proj_dir
    tcl.append("set fp [open %s w]" % v_file_list)
    # write each verilog filename to all_verilog_srcs.txt
    tcl.append("foreach vf $all_v_files {puts $fp $vf}")
    tcl.append("close $fp")
    # write the project creator tcl script
    tcl_string = "\n".join(tcl) + "\n"
    with open(vivado_stitch_proj_dir + "/make_project.tcl", "w") as f:
        f.write(tcl_string)
    # create a shell script and call Vivado
    make_project_sh = vivado_stitch_proj_dir + "/make_project.sh"
    toolchain = toolchain or machine_toolchain()
    toolchain.probe("vivado")
    args = ["-mode", "batch", "-source", "make_project.tcl"]
    with open(make_project_sh, "w") as f:
        f.write("#!/bin/bash\nset -e\ncd " + shlex.quote(vivado_stitch_proj_dir) + "\n")
        f.write("exec " + shlex.join(toolchain.command("vivado", *args)) + "\n")
    result = toolchain.run("vivado", args, cwd=vivado_stitch_proj_dir, check=False)
    sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
    sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
    result.check_returncode()
    # wrapper may be created in different location depending on Vivado version
    if not os.path.isfile(wrapper_filename):
        # check in alternative location (.gen instead of .srcs)
        wrapper_filename_alt = wrapper_filename.replace(".srcs", ".gen")
        if os.path.isfile(wrapper_filename_alt):
            model.set_metadata_prop("wrapper_filename", wrapper_filename_alt)
        else:
            raise Exception(
                """CreateStitchedIP failed, no wrapper HDL found under %s or %s.
                Please check logs under the parent directory."""
                % (wrapper_filename, wrapper_filename_alt)
            )
    return model


def ip_repositories(model: ModelWrapper) -> list[str]:
    """The IP repositories a design that instantiates the IODMA's packaged IP needs: its
    HLS IP and the packaged IP itself."""
    project = model.get_metadata_prop("vivado_stitch_proj")
    return [_iodma(model).get_nodeattr("ip_path"), project + "/ip"]


__all__ = ["create_stitched_ip", "data_path", "hls_synth_ip", "ip_repositories", "prepare_ip"]
