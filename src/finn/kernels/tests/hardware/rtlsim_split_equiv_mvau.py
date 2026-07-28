############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""rtlsim bit-equivalence: split per-core wrappers vs the original fused wrapper.

Runs INSIDE the FINN Docker container (needs the pre-built finn_xsi/xsi.so). The
2c-split replaced the fused ``mvu_vvu_axi.sv`` (internal genINT8/genSoftVec fork) with
two standalone per-core wrappers built from a shared ``.svh`` base. This test proves the
split changed NO behaviour: for a representative config per core it drives the split
wrapper and the original fused wrapper with the SAME random weight+activation stream
and asserts BIT-IDENTICAL output streams. Because we fork vendored HDL, this behavioural
oracle REPLACES the byte-``run_diff`` check on the instantiation line.

Construction guarantees the only difference under test is the compute path:
  * The split top is our real emit (instantiates mvu_vvu_axi_softvec / _packed).
  * The golden top is that SAME emitted wrapper with the instantiated module name
    string-replaced back to the fused ``mvu_vvu_axi`` (whose internal fork then selects
    the matching core). Identical ports, params and stimulus — only the instantiated
    module differs.

Because we compare two RTL modules to EACH OTHER (not to a numeric golden), the stimulus
is raw random integers packed to the stream widths: any value is fed identically to both
DUTs, so a mismatch can only mean the split altered behaviour.

Two configs:
  * softvec — DSP48E2 part (VERSION=2), so the fused fork takes genSoftVec.
  * packed  — DSP58 part, INT8/INT8 (w<=8, a<=9, lanes<=3), so the fork takes genINT8.

Usage (from finn/):
    bash run-docker.sh bash src/finn/kernels/tests/hardware/run_rtlsim_equiv.sh
"""

import os
import sys
import tempfile

import numpy as np
from qonnx.core.datatype import DataType

from finn import xsi
from finn.kernels.space import Context, resolve
from finn.kernels.ops.mvau import (
    mvau_schema,
    MVAU_DSP_SOFTVEC,
    MVAU_DSP_PACKED,
)
from finn.kernels.ops.mvau.emit_rtl import emit_mvau_rtl
from finn.kernels.ops.parameters.names import DECOUPLED, WEIGHTS
from finn.kernels.space.param_names import topology_key

TOPOLOGY = topology_key(WEIGHTS)

# The full fused source set (what the ORIGINAL wrapper compiles against): the retired
# fused wrapper + BOTH cores + plumbing. This is exactly FINN's ship-everything list.
FUSED_SOURCES = (
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
)

LIVENESS = 200000

# (label, implementation, fpgapart, weight dtype, input dtype, PE, SIMD, MW, MH)
CONFIGS = [
    # DSP48E2 -> VERSION=2 -> fused fork selects genSoftVec
    ("softvec", MVAU_DSP_SOFTVEC, "xczu7ev-ffvc1156-2-e", "INT8", "INT8", 2, 2, 4, 4),
    # DSP58, INT8/INT8 (w<=8, a<=9, lanes<=3) -> fused fork selects genINT8
    ("packed", MVAU_DSP_PACKED, "xcvc1902-vsva2197-2MP-e-S", "INT8", "INT8", 2, 2, 4, 4),
]


def _context(fpgapart, wdt, idt, odt, mw, mh, W):
    return Context(
        shapes={"weights": (mw, mh), "inp": (1, mw), "out": (1, mh)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=fpgapart, clk_ns=5.0,
    )


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(text)
    return p


def _resolve_sources(arts):
    """Absolute paths of the bundle's static sources under FINN_ROOT.

    Keeps ``.svh`` entries: compile_sim_obj derives the ``--include`` dir from their
    dirname and skips them as compile units itself.
    """
    out = []
    for sf in arts.static_files:
        p = os.path.join(os.environ["FINN_ROOT"], sf.resource)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MISSING static source: {sf.resource}")
        out.append(p)
    return out


def _drive(top_module, sources, top_path, io_dict, num_out):
    """Compile+load one DUT, drive io_dict, return the out0 list."""
    with tempfile.TemporaryDirectory() as sd:
        srcs = list(sources) + [top_path]
        sim_dir, so_rel = xsi.compile_sim_obj(top_module, srcs, sd, behav=True)
        sim = xsi.load_sim_obj(sim_dir, so_rel)
        xsi.reset_rtlsim(sim)
        # fresh copies of the value iterators per DUT
        local = {
            "inputs": {k: list(v) for k, v in io_dict["inputs"].items()},
            "outputs": {"out0": []},
        }
        xsi.rtlsim_multi_io(sim, local, num_out, sname="_V", liveness_threshold=LIVENESS)
        xsi.close_rtlsim(sim)
        return local["outputs"]["out0"]


def _run_config(cfg):
    label, impl, fpgapart, wdt_name, idt_name, pe, simd, mw, mh = cfg
    print(f"\n========== rtlsim equivalence: {label} ({impl}) ==========")
    wdt, idt, odt = DataType[wdt_name], DataType[idt_name], DataType["INT16"]

    rng = np.random.RandomState(0)
    W = rng.randint(int(wdt.min()) + 1, int(wdt.max()) + 1, size=(mw, mh)).astype(np.float32)
    ctx = _context(fpgapart, wdt, idt, odt, mw, mh, W)
    # DSP core is streamed-weight-only (embedded illegal); decoupled topology. The
    # compute-half emit under test is topology-independent.
    point = resolve(mvau_schema(), ctx, {
        "backend": impl, "PE": pe, "SIMD": simd, "resType": "dsp",
        TOPOLOGY: DECOUPLED,
    })

    # SPLIT top: our real emit (instantiates the per-core wrapper).
    split_arts = emit_mvau_rtl(point, ctx, module_name="mvau_split")
    split_top = split_arts.generated[0].content()
    core = point.rtl_core_module  # mvu_vvu_axi_softvec / mvu_vvu_axi_packed
    assert f"{core} #(" in split_top, f"emit did not instantiate {core}"

    # GOLDEN top: same wrapper, instantiated core string-replaced back to the fused
    # module. Only the instantiated module name differs.
    golden_top = split_top.replace("mvau_split", "mvau_fused").replace(core, "mvu_vvu_axi")
    assert "mvu_vvu_axi #(" in golden_top

    # Stimulus geometry (IS_MVU embedded, numInputVectors=[1]):
    sf, nf = mw // simd, mh // pe
    act_w, w_w = idt.bitwidth(), wdt.bitwidth()
    in_word_bits = simd * act_w            # one in0 beat packs SIMD activations
    wt_word_bits = pe * simd * w_w         # one in1 beat packs PE*SIMD weights
    in0 = [int(rng.randint(0, 1 << in_word_bits)) for _ in range(sf)]
    in1 = [int(rng.randint(0, 1 << wt_word_bits)) for _ in range(sf * nf)]
    io_dict = {"inputs": {"in0": in0, "in1": in1}, "outputs": {"out0": []}}
    print(f"  geometry: SF={sf} NF={nf} in0_beats={len(in0)} in1_beats={len(in1)} "
          f"expect {nf} outputs")

    with tempfile.TemporaryDirectory() as d:
        split_path = _write(d, "mvau_split.v", split_top)
        golden_path = _write(d, "mvau_fused.v", golden_top)

        split_srcs = _resolve_sources(split_arts)  # .svh kept -> --include dir
        golden_srcs = [
            os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "mvu", s) for s in FUSED_SOURCES
        ]

        out_split = _drive("mvau_split", split_srcs, split_path, io_dict, nf)
        out_golden = _drive("mvau_fused", golden_srcs, golden_path, io_dict, nf)

    print(f"  split : {out_split}")
    print(f"  golden: {out_golden}")
    if out_split == out_golden and len(out_split) == nf:
        print(f"  {label.upper()} EQUIVALENCE: PASS (bit-identical, {nf} outputs)")
        return True
    print(f"  {label.upper()} EQUIVALENCE: FAIL")
    return False


def main():
    if not xsi.is_available():
        print("finn_xsi/xsi.so not available — build it first")
        return 1
    ok = True
    for cfg in CONFIGS:
        ok &= _run_config(cfg)
    print("\nRESULT:", "RTLSIM EQUIVALENCE PASS" if ok else "RTLSIM EQUIVALENCE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
