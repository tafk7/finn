############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""Phase 4 — rtlsim the COMPOSED, stitched MVAU end-to-end (THE PRIZE).

Runs INSIDE the FINN Docker container (needs the pre-built finn_xsi/xsi.so). This is
the moment composition goes from "the pieces are byte-equivalent" to "the composition
produces a WORKING kernel": it emits a decoupled MVAU as compute wrapper + memstream
delivery wrapper + the STITCH connectivity, builds the Vivado block design from the
stitch Tcl (Phase 3), then drives the composed top through FINN's own proven XSI rtlsim
harness (finn.xsi.compile_sim_obj / rtlsim_multi_io) with real activation stimulus and
asserts the output equals the golden matmul.

Weights flow memstream -> in1_V purely through the role-derived stitch net (no pin
literal / mem_mode anywhere in the resolver). A PASS proves the composability thesis
functionally, not just structurally.

Usage (from finn/):
    bash run-docker.sh bash src/finn/kernels/tests/hardware/run_rtlsim_composed.sh
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from qonnx.core.datatype import DataType

from finn import xsi
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import get_liveness_threshold_cycles, make_build_dir

from finn.kernels.space import Context, resolve
from finn.kernels.compute.mvau import mvau_schema, MVAU_DSP_SOFTVEC
from finn.kernels.compute.mvau.compose_emit import emit_composed
from finn.kernels.dataflow.memory.names import DECOUPLED, WEIGHTS
from finn.kernels.model.param_names import ram_style_key, topology_key

RAM_STYLE = ram_style_key(WEIGHTS)
TOPOLOGY = topology_key(WEIGHTS)

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
CLK_NS = 5.0
MODULE = "mvau_top"
FINN_ROOT = os.environ["FINN_ROOT"]

MW, MH, PE, SIMD = 6, 8, 2, 2  # matmul + fold geometry


def _setup():
    rng = np.random.RandomState(0)
    W = rng.randint(-7, 7, size=(MW, MH)).astype(np.float32)
    X = rng.randint(-7, 7, size=(1, MW)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    ctx = Context(
        shapes={"weights": (MW, MH), "inp": (1, MW), "out": (1, MH)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_schema(), ctx, {
        "backend": MVAU_DSP_SOFTVEC, "PE": PE, "SIMD": SIMD, "resType": "dsp",
        TOPOLOGY: DECOUPLED, RAM_STYLE: "block",
    })
    return ctx, point, W, X, idt, odt


def _write_sources(arts, d):
    paths = []
    for gf in arts.generated:
        p = os.path.join(d, gf.filename)
        with open(p, "w") as f:
            f.write(gf.content())
        paths.append(p)
    for df in arts.data_files:
        with open(os.path.join(d, df.filename), "w") as f:
            f.write(df.content)  # memblock.dat next to wrapper for $readmemh
    for sf in arts.static_files:
        sp = os.path.join(FINN_ROOT, sf.resource)
        assert os.path.isfile(sp), f"missing static source {sf.resource}"
        paths.append(sp)
    # sibling DSP58 core: the un-split mvu_vvu_axi.sv references both cores (see the
    # elaborate harness note); softvec bundle ships only mvu.sv.
    sibling = os.path.join(FINN_ROOT, "finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv")
    if sibling not in paths:
        paths.append(sibling)
    return paths


def _build_bd(paths, stitch_commands, d):
    """Build + validate the bd from the stitch Tcl, generate the region netlist, and
    return the list of all verilog sources the composed top needs for rtlsim."""
    add = "\n".join(f"add_files -norecurse {{{p}}}" for p in paths)
    stitch = "\n".join(stitch_commands)
    tcl = f"""
create_project composed_prj {d}/prj -part {FPGAPART} -force
{add}
update_compile_order -fileset sources_1
create_bd_design "{MODULE}_region"
create_bd_port -dir I -type clk ap_clk
create_bd_port -dir I -type rst ap_rst_n
set_property CONFIG.ASSOCIATED_RESET {{ap_rst_n}} [get_bd_ports ap_clk]
{stitch}
regenerate_bd_layout
validate_bd_design
save_bd_design
set bd_file [get_files {d}/prj/composed_prj.srcs/sources_1/bd/{MODULE}_region/{MODULE}_region.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
make_wrapper -files $bd_file -top
puts "BD_DONE"
"""
    tcl_path = os.path.join(d, "build_bd.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl)
    r = subprocess.run(
        ["vivado", "-mode", "batch", "-source", tcl_path, "-nojournal", "-nolog"],
        cwd=d, capture_output=True, text=True,
    )
    if "BD_DONE" not in (r.stdout + r.stderr):
        print("\n".join((r.stdout + r.stderr).splitlines()[-40:]))
        raise RuntimeError("bd build failed")

    # collect all rtl sources for the composed top: the leaves/wrappers + the
    # bd-generated region netlist + per-IP synth stubs + the region wrapper.
    gen_root = os.path.join(d, "prj", "composed_prj.gen", "sources_1", "bd",
                            f"{MODULE}_region")
    srcs = list(paths)
    srcs.append(os.path.join(gen_root, "synth", f"{MODULE}_region.v"))
    srcs.append(os.path.join(gen_root, "hdl", f"{MODULE}_region_wrapper.v"))
    ip_dir = os.path.join(gen_root, "ip")
    for ip in sorted(os.listdir(ip_dir)) if os.path.isdir(ip_dir) else []:
        synth = os.path.join(ip_dir, ip, "synth")
        if os.path.isdir(synth):
            srcs += [os.path.join(synth, f) for f in os.listdir(synth)
                     if f.endswith((".v", ".sv"))]
    return [s for s in srcs if os.path.isfile(s)]


def _golden(W, X, odt):
    acc = (X.astype(np.int64) @ W.astype(np.int64)).astype(np.int64)  # (1, MH)
    return acc


def main():
    if not xsi.is_available():
        print("finn_xsi/xsi.so not available — build it first")
        return 1
    finnxsi = xsi

    ctx, point, W, X, idt, odt = _setup()
    arts = emit_composed(point, ctx, module_name=MODULE)
    print(f"composed emit: {len(arts.generated)} gen, {len(arts.ipi.commands)} stitch cmds")

    # the memblock.dat must sit in the rtlsim working dir for $readmemh("memblock.dat").
    work = make_build_dir("rtlsim_composed_mvau_")

    with tempfile.TemporaryDirectory() as d:
        all_srcs = _build_bd(paths=_write_sources(arts, d),
                             stitch_commands=arts.ipi.commands, d=d)
        print(f"bd built; {len(all_srcs)} rtl sources for composed top")

        # copy memblock.dat into the sim working dir (relative $readmemh path)
        dat = [f for f in arts.data_files if f.filename == "memblock.dat"][0]
        with open(os.path.join(work, "memblock.dat"), "w") as f:
            f.write(dat.content)

        top = f"{MODULE}_region_wrapper"
        sim_base, sim_rel = finnxsi.compile_sim_obj(top, all_srcs, work, behav=True)
        sim = finnxsi.load_sim_obj(sim_base, sim_rel, None)

    # --- drive: fold the activation, pack, stream in; collect + unpack output ---
    x_folded = X.reshape(1, MW // SIMD, SIMD)
    packed_in = npy_to_rtlsim_input(x_folded, idt, idt.bitwidth() * SIMD)

    io_dict = {
        "inputs": {"in0_V_0": packed_in},
        "outputs": {"out0_V_0": []},
    }
    num_out = MH // PE  # PE-folded output beats

    finnxsi.reset_rtlsim(sim)
    finnxsi.rtlsim_multi_io(
        sim, io_dict, num_out, sname="",
        liveness_threshold=get_liveness_threshold_cycles(),
    )
    finnxsi.close_rtlsim(sim)

    # The MVU output stream is PE accumulators of accDataType (noActivation =>
    # outputDataType := accDataType), NOT the graph's declared INT16 output. Unpack with
    # the resolved accumulator dtype + its PE-folded stream width (stream_width.out
    # = PE * acc_bits), else the per-element bit boundary is wrong.
    acc_dt = point.accDataType
    o_stream_w = point["stream_width.out"]  # PE * acc_dt.bitwidth() — MVAU publishes the
    # tiling-engine-namespaced key, not a bare `outstream_width` axis (that is a thresholding/
    # layernorm resolved axis). Same key the emit reads, so it matches the simulated hardware.
    packed_out = io_dict["outputs"]["out0_V_0"]
    o_folded = rtlsim_output_to_npy(
        packed_out, None, acc_dt, (1, MH // PE, PE), o_stream_w, acc_dt.bitwidth()
    )
    got = o_folded.reshape(1, MH).astype(np.int64)
    exp = _golden(W, X, odt)

    print("golden:", exp.flatten().tolist())
    print("rtlsim:", got.flatten().tolist())
    ok = np.array_equal(got, exp)
    print("\nRESULT:", "COMPOSED RTLSIM MATMUL PASS" if ok else "FAIL (output mismatch)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
