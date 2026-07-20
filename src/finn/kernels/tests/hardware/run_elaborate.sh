#!/bin/bash
# Runner for the MVAU emit RTL-elaboration check inside the FINN container.
set -e
cd "$FINN_ROOT"
python src/finn/kernels/tests/hardware/elaborate_mvau_emit.py 2>&1 | tee "$FINN_ROOT/_elab_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_elab_out.txt"
