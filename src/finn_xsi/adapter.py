#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	rtlsim_multi_io interface adapter for FINN XSI
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import os
import os.path

from finn.util.basic import get_rtlsim_timeout_error_message
from finn.xsi.compile import (  # noqa: F401
    compile_sim_obj,
    get_simkernel_so,
    locate_glbl,
)


def load_sim_obj(sim_out_dir, out_so_relative_path, tracefile=None, simkernel_so=None):
    from finn_xsi.sim_engine import SimEngine  # noqa: PLC0415

    if simkernel_so is None:
        simkernel_so = get_simkernel_so()
    oldcwd = os.getcwd()
    try:
        os.chdir(sim_out_dir)
        sim = SimEngine(simkernel_so, out_so_relative_path, "finnxsi_rtlsim.log", tracefile)
        if tracefile:
            sim.top.trace_all()
        return sim
    finally:
        os.chdir(oldcwd)


def reset_rtlsim(
    sim, rst_name="ap_rst_n", active_low=True, clk_name="ap_clk", clk2x_name="ap_clk2x", n_cycles=16
):
    sim.do_reset(n_cycles=n_cycles)
    sim.run()


def close_rtlsim(sim):
    sim_finish = sim.top.getPort("sim_finish")
    if sim_finish is not None:
        sim_finish.set(1).write_back()
        sim.cycle({})
    # Explicitly finalize the design (calls xsi_close -> flushes+closes the .wdb)
    # instead of relying on GC of sim.top. Port back-refs and the pybind use_map
    # can keep the Design alive past `del sim`, so without this the waveform can
    # be left unflushed on a timeout/pdb exit, corrupting its trace tail.
    close = getattr(sim.top, "close", None)
    if close is not None:
        close()
    del sim


def rtlsim_multi_io(
    sim,
    io_dict,
    num_out_values,
    sname="_V_V",
    liveness_threshold=10000,
    liveness_estimate=None,
):
    if len(io_dict["outputs"]) > 1:
        assert isinstance(
            num_out_values, dict
        ), "num_out_values must be dict for multiple output streams"
    else:
        # num_out_values is provided as integer (indicating the expected
        # outputs from the single output stream) - make into dict
        oname = list(io_dict["outputs"].keys())[0]
        num_out_values = {oname: num_out_values}

    # FINN XSI expects hex strings, while rtlsim_multi_io uses
    # lists of arbitrary-precision integers, so need to convert
    # inputs and outputs to appropriate format
    # TODO: refactor components&data packing to directly generate and consume
    # hex strings instead of arb-prec Python integers
    for inp in io_dict["inputs"]:
        arbprec_int_input = io_dict["inputs"][inp]
        hexstring_input = map(lambda var: f"{var:0x}", arbprec_int_input)
        stream_name = inp + sname
        sim.stream_input(stream_name, hexstring_input)

    hex_output_streams = {}
    watchdogs = []
    for out in io_dict["outputs"]:
        stream_name = out + sname
        watchdog = sim.create_watchdog(f"{stream_name} timeout", liveness_threshold)
        watchdogs.append(watchdog)
        hex_output_streams[out] = sim.collect_output(
            stream_name,
            num_out_values[out],
            watchdog=watchdog,
        )

    start_ticks = sim.ticks
    try:
        ret = sim.run()
        if len(ret) > 0:
            assert False, (
                get_rtlsim_timeout_error_message(liveness_threshold, liveness_estimate)
                + f" Triggered watchdogs: {str(ret)}. Check rtlsim_trace if any."
            )
        end_ticks = sim.ticks
        for out in io_dict["outputs"]:
            io_dict["outputs"][out] = list(
                map(lambda var: int(var, base=16), hex_output_streams[out])
            )
    finally:
        # Remove the per-output watchdogs so they do not outlive this data pass.
        # sim.watchdogs is persistent; a leftover (already-exhausted) watchdog
        # would keep ticking and prematurely abort any later sim.run(), e.g. a
        # post-hook AXI-Lite weight read-back (see test_fpgadataflow_mvau).
        for watchdog in watchdogs:
            if watchdog in sim.watchdogs:
                sim.remove_watchdog(watchdog)

    return end_ticks - start_ticks
