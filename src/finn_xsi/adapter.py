#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	load, reset and close an XSI simulation for FINN
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import os
import os.path

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
