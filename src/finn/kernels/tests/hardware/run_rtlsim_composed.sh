#!/bin/bash
# Runner for the COMPOSED (stitched) MVAU end-to-end rtlsim inside the FINN container.
# run-docker.sh word-splits its command, so we pass a single no-arg script path.
set -e
cd "$FINN_ROOT"
python src/finn/kernels/tests/hardware/rtlsim_composed_mvau.py 2>&1 \
  | tee "$FINN_ROOT/_rtlsim_composed_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_rtlsim_composed_out.txt"
