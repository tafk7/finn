"""Pure XSI design compilation/discovery; importing this module never loads the bridge."""
import errno
import json
import os
import re
from pathlib import Path
from typing import Optional

from finn.util.toolchain import machine_toolchain
from finn.xsi._artifacts import tool_identity, write_record
from finn_xsi.srcutil import order_pkg_first


def locate_glbl(environ=None) -> Optional[str]:
    """
    Tries to determine the glbl.v file path from environment variables.
    Returns None if it cannot be found.
    """
    # Get GLBL from the Vitis environment variable
    vivado_path = (os.environ if environ is None else environ).get("XILINX_VIVADO", "")
    if vivado_path:
        glbl_path = os.path.join(vivado_path, "data", "verilog", "src", "glbl.v")
        if os.path.isfile(glbl_path):
            return glbl_path
    return None


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
    toolchain = toolchain or machine_toolchain()
    identity = tool_identity(toolchain)
    sim_out_dir = os.fspath(sim_out_dir)
    source_list = list(map(os.fspath, source_list))
    # create a .prj file with the source files
    with open(sim_out_dir + "/rtlsim.prj", "w") as f:
        glbl = locate_glbl(toolchain.environment)
        if glbl is not None:
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
        "work." + top_module_name,
        "-relax",
        "-prj",
        "rtlsim.prj",
        "-dll",
        "-s",
        top_module_name,
    ]
    # Xelab defaults to "auto" threading, which can expand to hundreds of
    # workers on shared servers. Large stitched FINNLoop designs have shown
    # intermittent elaborator SIGABRTs in that mode, so keep the default
    # bounded while still allowing explicit override.
    xelab_mt = toolchain.environment.get(
        "FINN_XELAB_MT", toolchain.environment.get("NUM_DEFAULT_WORKERS", "8")
    )
    if xelab_mt == "1":
        xelab_mt = "off"
    cmd_xelab.extend(["--mt", xelab_mt])
    if debug:
        cmd_xelab.append("-debug")
        cmd_xelab.append("all")
    if behav:
        cmd_xelab.append("-define")
        cmd_xelab.append("FINN_SIMULATION")
    for lib in xelab_libs:
        cmd_xelab.append("-L")
        cmd_xelab.append(lib)

    if locate_glbl(toolchain.environment) is not None:
        cmd_xelab.insert(0, "work.glbl")

    # check=True so an xelab failure is raised here, not later as a missing xsimk.so
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
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_so_full_path)

    sources = list(source_list)
    if locate_glbl(toolchain.environment):
        sources.append(locate_glbl(toolchain.environment))
    write_record(
        out_so_full_path, kind="design", tool=identity, sources=sources, arguments=cmd_xelab
    )
    return (sim_out_dir, out_so_relative_path)


def get_simkernel_so(environ=None):
    vivado_path = (os.environ if environ is None else environ).get("XILINX_VIVADO", "")
    # xsi kernel lib name depends on Vivado version (renamed in 2024.2)
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    if match is None:
        raise ValueError("Select XILINX_VIVADO with a recognizable version, or pass a kernel path")
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) > (2024, 1):
        simkernel_so = "libxv_simulator_kernel.so"
    else:
        simkernel_so = "librdi_simulator_kernel.so"
    return simkernel_so
