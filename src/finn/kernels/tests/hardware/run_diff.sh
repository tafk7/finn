#!/bin/bash
# Runner for the MVAU emit differential test inside the FINN container.
# run-docker.sh word-splits its command, so we pass a single no-arg script path.
set -e
cd "$FINN_ROOT"
python src/finn/kernels/tests/hardware/diff_mvau_emit_vs_finn.py 2>&1 | tee "$FINN_ROOT/_diff_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_diff_out.txt"
