############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""RTL elaboration check for our emitted per-core MVAU wrappers.

Runs INSIDE the FINN Docker container (needs Vivado's xvlog). For EACH DSP bundle
(softvec + packed) it writes the emitted ``mvau_top.v`` to a temp dir, resolves the
finn-rtllib sources the ``StaticFile`` refs name, and runs ``xvlog -sv`` + ``xelab``
to confirm the emitted wrapper elaborates against its OWN (disjoint) source set — no
"module not found." This is the honest 2c-split gate: post-split each bundle ships only
its owned core + per-core wrapper + the shared base ``.svh``, so it must elaborate with
NO reference to the other core.

The base body lives in ``.svh`` fragments `include`d by the per-core ``.sv`` wrappers;
those are NOT standalone compile units — they are supplied via the ``-i`` include dir
(``finn-rtllib/mvu``) and excluded from the xvlog source list.

Usage (from finn/):
    bash run-docker.sh bash src/finn/kernels/tests/hardware/run_elaborate.sh
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from qonnx.core.datatype import DataType

from finn.kernels.space import Context, resolve, emit_point
from finn.kernels.ops.mvau import (
    mvau_schema,
    mvau_pool,
    MVAU_DSP_SOFTVEC,
    MVAU_DSP_PACKED,
)
from finn.kernels.ops.parameters.names import DECOUPLED, WEIGHTS
from finn.kernels.space.param_names import topology_key

TOPOLOGY = topology_key(WEIGHTS)

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"  # Versal / DSP58
RTLLIB = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib")
MVU_INCLUDE_DIR = os.path.join(RTLLIB, "mvu")  # where the base .svh live


def _make_context():
    rng = np.random.RandomState(0)
    # INT8 weights / INT8 activations on DSP58: packed-eligible (w<=8, a<=9, lanes<=3)
    # AND softvec-buildable — so the SAME context resolves for both bundles.
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    return Context(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=5.0,
    )


def _elaborate(impl_name, ctx):
    """Emit + xvlog + xelab one bundle. Returns 0 on success, 1 on failure."""
    print(f"\n========== elaborating {impl_name} ==========")
    # The DSP core is streamed-weight-only (embedded illegal); use decoupled. The
    # compute-half emit is topology-independent, so this elaboration is unaffected.
    point = resolve(mvau_schema(), ctx, {
        "backend": impl_name, "PE": 2, "SIMD": 2, "resType": "dsp",
        TOPOLOGY: DECOUPLED,
    })
    arts = emit_point(mvau_pool(), point, ctx)

    with tempfile.TemporaryDirectory() as d:
        top = arts.generated[0]
        top_path = os.path.join(d, top.filename)
        with open(top_path, "w") as f:
            f.write(top.content())

        # Resolve declared sources. `.svh` are include fragments, not compile units:
        # confirm they exist but keep them OUT of the xvlog source list (supplied via
        # the -i include dir instead).
        compile_srcs = []
        for sf in arts.static_files:
            p = os.path.join(os.environ["FINN_ROOT"], sf.resource)
            if not os.path.isfile(p):
                print(f"MISSING static source: {sf.resource}")
                return 1
            if p.endswith(".svh"):
                continue
            compile_srcs.append(p)
        print(f"resolved {len(compile_srcs)} compile sources + 1 generated wrapper "
              f"(+ base .svh via -i {MVU_INCLUDE_DIR})")

        # xvlog compile (SystemVerilog). Order: package first, wrapper last.
        srcs = sorted(compile_srcs, key=lambda p: (0 if p.endswith("mvu_pkg.sv") else 1)) + [top_path]
        cmd = ["xvlog", "-sv", "-i", MVU_INCLUDE_DIR, "--define", "FINN_SIMULATION"] + srcs
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        print("--- xvlog stdout tail ---")
        print("\n".join(r.stdout.splitlines()[-15:]))
        if r.returncode != 0:
            print("--- xvlog stderr tail ---")
            print("\n".join(r.stderr.splitlines()[-15:]))
            print(f"XVLOG: FAIL ({impl_name})")
            return 1
        print("XVLOG: PASS (all sources compiled)")

        # xelab elaborate the top module.
        top_module = top.filename[:-2]  # strip .v
        r2 = subprocess.run(
            ["xelab", "-debug", "typical", top_module, "-s", f"mvau_elab_{impl_name}"],
            cwd=d, capture_output=True, text=True,
        )
        print("--- xelab stdout tail ---")
        print("\n".join(r2.stdout.splitlines()[-20:]))
        if r2.returncode != 0:
            print("--- xelab stderr tail ---")
            print("\n".join(r2.stderr.splitlines()[-20:]))
            print(f"XELAB: FAIL ({impl_name})")
            return 1
        print(f"XELAB: PASS ({impl_name} wrapper elaborates)")
    return 0


def main():
    ctx = _make_context()
    for impl_name in (MVAU_DSP_SOFTVEC, MVAU_DSP_PACKED):
        if _elaborate(impl_name, ctx) != 0:
            print(f"\nRESULT: ELABORATION FAIL ({impl_name})")
            return 1
    print("\nRESULT: ELABORATION PASS (softvec + packed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
