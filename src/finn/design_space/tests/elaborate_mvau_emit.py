############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################

"""RTL elaboration check for our emitted MVAU wrapper.

Runs INSIDE the FINN Docker container (needs Vivado's xvlog). Writes our emitted
``mvau_top.v`` to a temp dir, resolves the finn-rtllib ``.sv`` sources our
``StaticFile`` refs name, and runs ``xvlog -sv`` + ``xelab`` to confirm the emitted
wrapper elaborates and the declared source set is complete. This is the cheap
insurance that our ``.sources`` list and wrapper are synthesizable — not just
string-equivalent to FINN.

Usage (from finn/):
    bash run-docker.sh bash src/finn/design_space/tests/_run_elaborate.sh
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from qonnx.core.datatype import DataType

from finn.design_space.space import Context, resolve, emit_point
from finn.design_space.fixtures.mvau import mvau_schema, mvau_pool, MVAU_DSP_SOFTVEC

FPGAPART = "xcvc1902-vsva2197-2MP-e-S"
RTLLIB = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib")


def main():
    rng = np.random.RandomState(0)
    W = rng.randint(int(DataType["INT8"].min()) + 1, int(DataType["INT8"].max()) + 1,
                    size=(6, 8)).astype(np.float32)
    wdt = idt = DataType["INT8"]
    odt = DataType["INT16"]
    ctx = Context(
        shapes={"weights": (6, 8), "inp": (1, 6), "out": (1, 8)},
        datatypes={"weights": wdt, "inp": idt, "out": odt},
        initializers={"weights": W},
        fpgapart=FPGAPART, clk_ns=5.0,
    )
    point = resolve(mvau_schema(), ctx, {
        "implementation": MVAU_DSP_SOFTVEC, "PE": 2, "SIMD": 2, "resType": "dsp",
        "mem_mode": "internal_embedded", "noActivation": 1,
    })
    arts = emit_point(mvau_pool(), point, ctx)

    with tempfile.TemporaryDirectory() as d:
        # write our generated wrapper
        top = arts.generated[0]
        top_path = os.path.join(d, top.filename)
        with open(top_path, "w") as f:
            f.write(top.content())

        # resolve the static .sv the bundle declares (StaticFile.resource is a repo
        # path under FINN_ROOT). Confirm each exists, collect for elaboration.
        sv_paths = []
        for sf in arts.static_files:
            p = os.path.join(os.environ["FINN_ROOT"], sf.resource)
            if not os.path.isfile(p):
                print(f"MISSING static source: {sf.resource}")
                return 1
            sv_paths.append(p)
        print(f"resolved {len(sv_paths)} static .sv sources + 1 generated wrapper")

        # xvlog compile (SystemVerilog). Order: package first, then the rest, wrapper last.
        srcs = sorted(sv_paths, key=lambda p: (0 if p.endswith("mvu_pkg.sv") else 1)) + [top_path]
        log = os.path.join(d, "xvlog.log")
        cmd = ["xvlog", "-sv", "--define", "FINN_SIMULATION"] + srcs
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        print("--- xvlog stdout tail ---")
        print("\n".join(r.stdout.splitlines()[-15:]))
        if r.returncode != 0:
            print("--- xvlog stderr tail ---")
            print("\n".join(r.stderr.splitlines()[-15:]))
            print("XVLOG: FAIL")
            return 1
        print("XVLOG: PASS (all sources compiled)")

        # xelab elaborate the top module.
        top_module = top.filename[:-2]  # strip .v
        r2 = subprocess.run(
            ["xelab", "-debug", "typical", top_module, "-s", "mvau_elab"],
            cwd=d, capture_output=True, text=True,
        )
        print("--- xelab stdout tail ---")
        print("\n".join(r2.stdout.splitlines()[-20:]))
        if r2.returncode != 0:
            print("--- xelab stderr tail ---")
            print("\n".join(r2.stderr.splitlines()[-20:]))
            print("XELAB: FAIL")
            return 1
        print("XELAB: PASS (wrapper elaborates)")

    print("\nRESULT: ELABORATION PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
