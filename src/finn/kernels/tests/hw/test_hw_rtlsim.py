############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""rtlsim functional correctness — the slow hardware tier (Vivado + XSI).

Two flagship guarantees, gated behind --runslow and self-skipping when xsi.so/Vivado is
absent:

  * composed MVAU: emit compute wrapper + memstream delivery + the role-derived STITCH,
    build the Vivado block design, drive the composed top through FINN's XSI rtlsim, and
    assert the output equals the golden matmul. Weights flow memstream→in1_V purely via the
    stitch net (no pin literal in the resolver). Proves composability FUNCTIONALLY.
  * split-equiv: the 2c-split per-core wrappers are BIT-IDENTICAL to the original fused
    wrapper for a representative config per core — the behavioural oracle that guards the
    intentional forked-HDL divergence no byte-diff can cover.
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest
from qonnx.core.datatype import DataType

pytestmark = pytest.mark.slow_hw


def _xsi_or_skip():
    from finn import xsi

    if not xsi.is_available():
        pytest.skip("finn_xsi/xsi.so not available")
    return xsi


# ===========================================================================
# Composed MVAU — the prize.
# ===========================================================================

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
CLK_NS = 5.0
MODULE = "mvau_top"
MW, MH, PE, SIMD = 6, 8, 2, 2


def _composed_setup():
    from finn.kernels.engine.context import Context
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.mvau import MvauDataflowOp, MVAU_DSP_SOFTVEC
    from finn.kernels.dataflow.parameters.names import DECOUPLED, WEIGHTS
    from finn.kernels.model.param_names import ram_style_key, topology_key

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
    point = resolve(MvauDataflowOp.compile(), ctx, {
        "backend": MVAU_DSP_SOFTVEC, "PE": PE, "SIMD": SIMD, "resType": "dsp",
        topology_key(WEIGHTS): DECOUPLED, ram_style_key(WEIGHTS): "block",
    })
    return ctx, point, W, X, idt, odt


def _write_sources(arts, d):
    finn_root = os.environ["FINN_ROOT"]
    paths = []
    for gf in arts.generated:
        p = os.path.join(d, gf.filename)
        with open(p, "w") as f:
            f.write(gf.content())
        paths.append(p)
    for df in arts.data_files:
        with open(os.path.join(d, df.filename), "w") as f:
            f.write(df.content)
    for sf in arts.static_files:
        sp = os.path.join(finn_root, sf.resource)
        assert os.path.isfile(sp), f"missing static source {sf.resource}"
        paths.append(sp)
    sibling = os.path.join(finn_root, "finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv")
    if sibling not in paths:
        paths.append(sibling)
    return paths


def _build_bd(paths, stitch_commands, d):
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
        raise RuntimeError("bd build failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-40:]))
    gen_root = os.path.join(d, "prj", "composed_prj.gen", "sources_1", "bd", f"{MODULE}_region")
    srcs = list(paths)
    srcs.append(os.path.join(gen_root, "synth", f"{MODULE}_region.v"))
    srcs.append(os.path.join(gen_root, "hdl", f"{MODULE}_region_wrapper.v"))
    ip_dir = os.path.join(gen_root, "ip")
    for ip in sorted(os.listdir(ip_dir)) if os.path.isdir(ip_dir) else []:
        synth = os.path.join(ip_dir, ip, "synth")
        if os.path.isdir(synth):
            srcs += [os.path.join(synth, f) for f in os.listdir(synth) if f.endswith((".v", ".sv"))]
    return [s for s in srcs if os.path.isfile(s)]


def test_composed_mvau_rtlsim_matches_golden_matmul():
    xsi = _xsi_or_skip()
    from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
    from finn.util.basic import get_liveness_threshold_cycles, make_build_dir
    from finn.kernels.compute.mvau.compose_emit import emit_composed

    ctx, point, W, X, idt, odt = _composed_setup()
    arts = emit_composed(point, ctx, module_name=MODULE)
    work = make_build_dir("rtlsim_composed_mvau_")

    with tempfile.TemporaryDirectory() as d:
        all_srcs = _build_bd(_write_sources(arts, d), arts.ipi.commands, d)
        dat = [f for f in arts.data_files if f.filename == "memblock.dat"][0]
        with open(os.path.join(work, "memblock.dat"), "w") as f:
            f.write(dat.content)
        top = f"{MODULE}_region_wrapper"
        sim_base, sim_rel = xsi.compile_sim_obj(top, all_srcs, work, behav=True)
        sim = xsi.load_sim_obj(sim_base, sim_rel, None)

    x_folded = X.reshape(1, MW // SIMD, SIMD)
    packed_in = npy_to_rtlsim_input(x_folded, idt, idt.bitwidth() * SIMD)
    io_dict = {"inputs": {"in0_V_0": packed_in}, "outputs": {"out0_V_0": []}}
    num_out = MH // PE

    xsi.reset_rtlsim(sim)
    xsi.rtlsim_multi_io(sim, io_dict, num_out, sname="",
                        liveness_threshold=get_liveness_threshold_cycles())
    xsi.close_rtlsim(sim)

    acc_dt = point.accDataType
    o_stream_w = point["stream_width.out"]
    o_folded = rtlsim_output_to_npy(
        io_dict["outputs"]["out0_V_0"], None, acc_dt, (1, MH // PE, PE), o_stream_w, acc_dt.bitwidth()
    )
    got = o_folded.reshape(1, MH).astype(np.int64)
    exp = (X.astype(np.int64) @ W.astype(np.int64)).astype(np.int64)
    assert np.array_equal(got, exp), f"golden={exp.flatten().tolist()} rtlsim={got.flatten().tolist()}"


# ===========================================================================
# Split-equiv — split per-core wrappers bit-identical to the fused wrapper.
# ===========================================================================

_FUSED_SOURCES = (
    "mvu_pkg.sv", "mvu_vvu_axi.sv", "replay_buffer.sv", "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv", "add_multi.sv",
)
_LIVENESS = 200000


def _split_configs():
    from finn.kernels.compute.mvau import MVAU_DSP_SOFTVEC, MVAU_DSP_PACKED

    return [
        ("softvec", MVAU_DSP_SOFTVEC, "xczu7ev-ffvc1156-2-e", "INT8", "INT8", 2, 2, 4, 4),
        ("packed", MVAU_DSP_PACKED, "xcvc1902-vsva2197-2MP-e-S", "INT8", "INT8", 2, 2, 4, 4),
    ]


def _drive(xsi, top_module, sources, top_path, io_dict, num_out):
    with tempfile.TemporaryDirectory() as sd:
        srcs = list(sources) + [top_path]
        sim_dir, so_rel = xsi.compile_sim_obj(top_module, srcs, sd, behav=True)
        sim = xsi.load_sim_obj(sim_dir, so_rel)
        xsi.reset_rtlsim(sim)
        local = {"inputs": {k: list(v) for k, v in io_dict["inputs"].items()}, "outputs": {"out0": []}}
        xsi.rtlsim_multi_io(sim, local, num_out, sname="_V", liveness_threshold=_LIVENESS)
        xsi.close_rtlsim(sim)
        return local["outputs"]["out0"]


@pytest.mark.parametrize("cfg", _split_configs(), ids=lambda c: c[0])
def test_split_wrappers_bit_identical_to_fused(cfg):
    xsi = _xsi_or_skip()
    from finn.kernels.engine.context import Context
    from finn.kernels.engine.resolve import resolve
    from finn.kernels.compute.mvau import MvauDataflowOp
    from finn.kernels.compute.mvau.emit_rtl import emit_mvau_rtl
    from finn.kernels.dataflow.parameters.names import DECOUPLED, WEIGHTS
    from finn.kernels.model.param_names import topology_key

    label, backend, fpgapart, wdt_name, idt_name, pe, simd, mw, mh = cfg
    wdt, idt, odt = DataType[wdt_name], DataType[idt_name], DataType["INT16"]
    rng = np.random.RandomState(0)
    W = rng.randint(int(wdt.min()) + 1, int(wdt.max()) + 1, size=(mw, mh)).astype(np.float32)
    ctx = Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W}, fpgapart=fpgapart, clk_ns=5.0,
    )
    point = resolve(MvauDataflowOp.compile(), ctx, {
        "backend": backend, "PE": pe, "SIMD": simd, "resType": "dsp", topology_key(WEIGHTS): DECOUPLED,
    })

    split_arts = emit_mvau_rtl(point, ctx, module_name="mvau_split")
    split_top = split_arts.generated[0].content()
    core = MvauDataflowOp.selected_backend(point).rtl_core_module
    assert f"{core} #(" in split_top
    golden_top = split_top.replace("mvau_split", "mvau_fused").replace(core, "mvu_vvu_axi")
    assert "mvu_vvu_axi #(" in golden_top

    sf, nf = mw // simd, mh // pe
    in_word_bits = simd * idt.bitwidth()
    wt_word_bits = pe * simd * wdt.bitwidth()
    in0 = [int(rng.randint(0, 1 << in_word_bits)) for _ in range(sf)]
    in1 = [int(rng.randint(0, 1 << wt_word_bits)) for _ in range(sf * nf)]
    io_dict = {"inputs": {"in0": in0, "in1": in1}, "outputs": {"out0": []}}

    finn_root = os.environ["FINN_ROOT"]
    with tempfile.TemporaryDirectory() as d:
        split_path = os.path.join(d, "mvau_split.v")
        golden_path = os.path.join(d, "mvau_fused.v")
        with open(split_path, "w") as f:
            f.write(split_top)
        with open(golden_path, "w") as f:
            f.write(golden_top)
        split_srcs = [os.path.join(finn_root, sf_.resource) for sf_ in split_arts.static_files]
        golden_srcs = [os.path.join(finn_root, "finn-rtllib", "mvu", s) for s in _FUSED_SOURCES]
        out_split = _drive(xsi, "mvau_split", split_srcs, split_path, io_dict, nf)
        out_golden = _drive(xsi, "mvau_fused", golden_srcs, golden_path, io_dict, nf)

    assert out_split == out_golden and len(out_split) == nf, (
        f"{label}: split={out_split} golden={out_golden}"
    )
