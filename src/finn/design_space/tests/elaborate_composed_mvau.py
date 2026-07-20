############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""Phase 3 — elaborate the STITCHED composed MVAU through the real Vivado bd flow.

Runs INSIDE the FINN Docker container (needs Vivado). This is the first proof that
composition produces a structurally valid block design, not just two byte-equivalent
islands: it takes a resolved decoupled MVAU, runs ``emit_composed`` to get the compute
wrapper + memstream wrapper + the STITCH IPICommands, then drives the actual Vivado
block-design flow — ``create_bd_design``, add the emitted + static RTL as project
sources, create the region-level clk/rst/AXIS ports the stitch broadcast/export
reference, run the stitch Tcl verbatim, then ``validate_bd_design`` + generate the bd
wrapper + ``synth_design -rtl`` (elaboration-only).

The stitch Tcl is emitted by ``space/stitch.py`` from the declared ports alone (no pin
literal / op type / mem_mode) — so a PASS here is the role-driven wiring proven to
build a real Vivado bd. Phase 4 adds rtlsim on top for the functional proof.

Usage (from finn/):
    bash run-docker.sh bash src/finn/design_space/tests/run_elaborate_composed.sh
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from qonnx.core.datatype import DataType

from finn.design_space.space import Context, resolve
from finn.design_space.fixtures.mvau import mvau_schema, MVAU_DSP_SOFTVEC
from finn.design_space.fixtures.mvau.compose_emit import emit_composed
from finn.design_space.fixtures.parameters.names import (
    DECOUPLED,
    RAM_STYLE,
    TOPOLOGY,
)

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
CLK_NS = 5.0
MODULE = "mvau_top"
FINN_ROOT = os.environ["FINN_ROOT"]


def _resolve_decoupled():
    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    ctx = Context(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=CLK_NS,
    )
    point = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_DSP_SOFTVEC, "PE": 2, "SIMD": 2, "resType": "dsp",
        "noActivation": 1, TOPOLOGY: DECOUPLED, RAM_STYLE: "block",
    })
    return ctx, point


def _write_sources(arts, d):
    """Write emitted wrappers + .dat and resolve static .sv paths. Returns (list of
    absolute source paths to add, list of missing sources)."""
    paths, missing = [], []
    for gf in arts.generated:
        p = os.path.join(d, gf.filename)
        with open(p, "w") as f:
            f.write(gf.content())
        paths.append(p)
    for df in arts.data_files:
        with open(os.path.join(d, df.filename), "w") as f:
            f.write(df.content)  # memblock.dat next to the wrapper for $readmemh
    for sf in arts.static_files:
        sp = os.path.join(FINN_ROOT, sf.resource)
        (paths if os.path.isfile(sp) else missing).append(sp if os.path.isfile(sp) else sf.resource)

    # The softvec bundle ships only mvu.sv — the ASPIRATIONAL post-2c-split source set
    # (see fixtures/mvau/__init__.py "2c forward requirement"). But the real, unsplit
    # finn-rtllib/mvu/mvu_vvu_axi.sv still references BOTH cores in its generate fork
    # (mvu_vvu_axi.sv:315 genINT8 -> mvu_vvu_8sx9_dsp58), so FINN's own RTL file list
    # ships both (matrixvectoractivation_rtl.py:200-201). Until the physical wrapper
    # split lands, elaboration needs the sibling DSP58 core present. Add it here (a
    # harness-level accommodation of the un-split RTL, NOT a change to the emit).
    sibling = os.path.join(FINN_ROOT, "finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv")
    if os.path.isfile(sibling) and sibling not in paths:
        paths.append(sibling)
    return paths, missing


def _bd_tcl(source_paths, stitch_commands, d):
    """Assemble the Vivado bd script: project + sources, region ports, the stitch Tcl,
    then VALIDATE + generate the netlist. Validation is the real proof the stitch's
    role-derived connectivity is accepted by Vivado's block-design engine; the netlist
    it generates is elaborated flat afterwards (xelab --relax) to prove it builds. We do
    NOT run project-mode synth_design here — its front-end rejects finn-rtllib
    axilite.sv's legal use-before-declare (a Vivado-2025.2 strictness vs FINN's 2022.x
    baseline), which xelab --relax tolerates. The region ports (clk/rst) are what the
    stitch broadcast (get_bd_ports ap_clk/ap_rst_n) binds to."""
    add = "\n".join(f"add_files -norecurse {{{p}}}" for p in source_paths)
    stitch = "\n".join(stitch_commands)
    return f"""
create_project composed_prj {d}/prj -part {FPGAPART} -force
{add}
update_compile_order -fileset sources_1

create_bd_design "{MODULE}_region"
create_bd_port -dir I -type clk ap_clk
create_bd_port -dir I -type rst ap_rst_n
set_property CONFIG.ASSOCIATED_RESET {{ap_rst_n}} [get_bd_ports ap_clk]

# --- the stitch: instantiate cells, bind roles, broadcast clk/rst, export boundary ---
{stitch}

regenerate_bd_layout
validate_bd_design
save_bd_design
puts "BD_VALIDATE: PASS"

set bd_file [get_files {d}/prj/composed_prj.srcs/sources_1/bd/{MODULE}_region/{MODULE}_region.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
make_wrapper -files $bd_file -top
puts "BD_GENERATE: PASS"
"""


def _flat_elaborate(leaf_paths, d):
    """xvlog + xelab --relax the composed region flat: all leaf RTL + our wrappers + the
    bd-generated region netlist/wrappers, elaborated top-down from the region wrapper.
    --relax tolerates axilite.sv's use-before-declare that 2025.2's project synth
    rejects. This is the structural proof the composed, stitched design elaborates."""
    gen_root = os.path.join(d, "prj", "composed_prj.gen", "sources_1", "bd",
                            f"{MODULE}_region")
    region_wrapper = os.path.join(gen_root, "hdl", f"{MODULE}_region_wrapper.v")
    region_synth = os.path.join(gen_root, "synth", f"{MODULE}_region.v")
    ip_stubs = []
    ip_dir = os.path.join(gen_root, "ip")
    for ip in sorted(os.listdir(ip_dir)) if os.path.isdir(ip_dir) else []:
        synth = os.path.join(ip_dir, ip, "synth")
        if os.path.isdir(synth):
            ip_stubs += [os.path.join(synth, f) for f in os.listdir(synth)
                         if f.endswith((".v", ".sv"))]

    # leaves + our wrappers (leaf_paths) first, then bd-generated netlist layers.
    srcs = list(leaf_paths) + ip_stubs + [region_synth, region_wrapper]
    srcs = [p for p in srcs if os.path.isfile(p)]

    log = subprocess.run(
        ["xvlog", "-sv", "--relax", "--define", "FINN_SIMULATION"] + srcs,
        cwd=d, capture_output=True, text=True,
    )
    if log.returncode != 0:
        print("--- xvlog stderr tail ---")
        print("\n".join((log.stdout + log.stderr).splitlines()[-20:]))
        return False
    top = f"{MODULE}_region_wrapper"
    el = subprocess.run(
        ["xelab", "--relax", "-debug", "typical", top, "-s", "composed_elab"],
        cwd=d, capture_output=True, text=True,
    )
    print("--- xelab tail ---")
    print("\n".join((el.stdout + el.stderr).splitlines()[-12:]))
    return el.returncode == 0


def main():
    ctx, point = _resolve_decoupled()
    arts = emit_composed(point, ctx, module_name=MODULE)

    print(f"composed emit: {len(arts.generated)} generated, "
          f"{len(arts.static_files)} static, {len(arts.ipi.commands)} stitch cmds, "
          f"{len(arts.ports)} ports")
    print("--- stitch commands ---")
    for c in arts.ipi.commands:
        print("  " + c)

    with tempfile.TemporaryDirectory() as d:
        paths, missing = _write_sources(arts, d)
        if missing:
            print("MISSING static sources:", missing)
            return 1
        print(f"resolved {len(paths)} source files")

        tcl = _bd_tcl(paths, arts.ipi.commands, d)
        tcl_path = os.path.join(d, "build_bd.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl)

        r = subprocess.run(
            ["vivado", "-mode", "batch", "-source", tcl_path, "-nojournal", "-nolog"],
            cwd=d, capture_output=True, text=True,
        )
        out = r.stdout + "\n" + r.stderr
        bd_ok = "BD_VALIDATE: PASS" in out and "BD_GENERATE: PASS" in out
        if not bd_ok:
            print("--- vivado output tail ---")
            print("\n".join(out.splitlines()[-40:]))
            print("\nRESULT: FAIL (bd validate/generate)")
            return 1
        print("BD VALIDATE + GENERATE: PASS (stitch connectivity accepted by Vivado)")

        elab_ok = _flat_elaborate(paths, d)
        print("\nRESULT:", "COMPOSED ELABORATION PASS" if elab_ok else "FAIL (flat elaborate)")
        return 0 if elab_ok else 1


if __name__ == "__main__":
    sys.exit(main())
