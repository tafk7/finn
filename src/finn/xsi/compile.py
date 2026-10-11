# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure XSI design compilation/discovery; importing this module never loads the bridge.

Which Vivado compiles and simulates is the toolchain's (``finn.toolchain.vivado``), the
machine's unless one is given; without one, ``ToolUnavailable``.
"""
import json
import os
from pathlib import Path

from finn.toolchain import vivado
from finn.toolchain.launcher import ToolFailed, host_toolchain, identity
from finn.toolchain.xsim import xelab_threads
from finn.xsi._artifacts import write_record
from finn_xsi.srcutil import order_pkg_first


def compile_sim_obj(
    top_module_name,
    source_list,
    sim_out_dir,
    debug=False,
    behav=False,
    *,
    toolchain=None,
    timeout=None,
    cancel=None,
):
    toolchain = toolchain or host_toolchain()
    glbl = str(vivado.glbl(toolchain))
    tool = identity(toolchain, "xelab")
    sim_out_dir = os.fspath(sim_out_dir)
    source_list = list(map(os.fspath, source_list))
    # create a .prj file with the source files
    with open(sim_out_dir + "/rtlsim.prj", "w") as f:
        f.write(f"verilog work {json.dumps(glbl, ensure_ascii=False)}\n")

        # extract (unique, by using a set) verilog headers for inclusion
        verilog_headers = {
            os.path.dirname(x) for x in source_list if x.endswith(".vh") or x.endswith(".svh")
        }
        verilog_header_incl_str = " ".join(
            "--include " + json.dumps(x, ensure_ascii=False) for x in sorted(verilog_headers)
        )

        # packages must come before the modules that import them
        srcs_list = order_pkg_first(source_list)
        for src_line in srcs_list:
            quoted = json.dumps(src_line, ensure_ascii=False)
            if src_line.endswith(".v"):
                f.write(f"verilog work {verilog_header_incl_str} {quoted}\n")
            elif src_line.endswith(".vhd"):
                # note that Verilog header incls are not added for VHDL
                f.write(f"vhdl2008 work {quoted}\n")
            elif src_line.endswith(".sv"):
                f.write(f"sv work {verilog_header_incl_str} {quoted}\n")
            elif src_line.endswith(".vh") or src_line.endswith(".svh"):
                # skip adding Verilog headers directly (see verilog_header_incl_str)
                continue
            else:
                raise Exception(f"Unknown extension for .prj file sources: {src_line}")

    # now call xelab to generate the .so for the design to be simulated
    # list of libs for xelab retrieved from Vitis HLS cosim cmdline
    # the particular lib version used depends on the Vivado/Vitis version being used
    # but putting in multiple (nonpresent) versions seems to pose no problem as long
    # as the correct one is also in there. at least this is how Vitis HLS cosim is
    # handling it.
    # TODO make this an optional param instead of hardcoding
    xelab_libs = [
        "smartconnect_v1_0",
        "axi_protocol_checker_v1_1_12",
        "axi_protocol_checker_v1_1_13",
        "axis_protocol_checker_v1_1_11",
        "axis_protocol_checker_v1_1_12",
        "xil_defaultlib",
        "unisims_ver",
        "xpm",
        "floating_point_v7_1_16",
        "floating_point_v7_0_21",
        "floating_point_v7_1_18",
        "floating_point_v7_1_15",
        "floating_point_v7_1_19",
        "floating_point_v7_1_21",
        "floating_point_v7_0_26",
    ]

    cmd_xelab = [
        "work.glbl",
        "work." + top_module_name,
        "-relax",
        "-prj",
        "rtlsim.prj",
        "-dll",
        "-s",
        top_module_name,
    ]
    cmd_xelab.extend(["--mt", xelab_threads(toolchain.environment)])
    if debug:
        cmd_xelab.append("-debug")
        cmd_xelab.append("all")
    if behav:
        cmd_xelab.append("-define")
        cmd_xelab.append("FINN_SIMULATION")
    for lib in xelab_libs:
        cmd_xelab.append("-L")
        cmd_xelab.append(lib)

    # check=True so an xelab failure is raised here (ToolFailed), not later as a missing
    # xsimk.so
    toolchain.run(
        "xelab",
        cmd_xelab,
        cwd=sim_out_dir,
        timeout=timeout,
        cancel=cancel,
        replay=Path(sim_out_dir) / "compile_sim.sh",
    )
    out_so_relative_path = "xsim.dir/%s/xsimk.so" % top_module_name
    out_so_full_path = sim_out_dir + "/" + out_so_relative_path

    if not os.path.isfile(out_so_full_path):
        raise ToolFailed.of(
            "xelab",
            transcript=Path(sim_out_dir) / "compile_sim.sh.stdout.log",
            what=f"it left no {out_so_full_path}",
        )

    write_record(
        out_so_full_path,
        kind="design",
        tool=tool,
        sources=[*source_list, glbl],
        arguments=cmd_xelab,
    )
    return (sim_out_dir, out_so_relative_path)


def get_simkernel_so(toolchain=None):
    """The simulation kernel library XSI loads, by the release of the toolchain's Vivado
    (the machine's by default; ``finn.toolchain.vivado.release``): renamed in 2024.2."""
    if vivado.release(toolchain or host_toolchain()) > (2024, 1):
        simkernel_so = "libxv_simulator_kernel.so"
    else:
        simkernel_so = "librdi_simulator_kernel.so"
    return simkernel_so
