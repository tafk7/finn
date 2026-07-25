#!/bin/bash
# Runner for the split-vs-fused MVAU rtlsim bit-equivalence test inside the FINN
# container. run-docker.sh word-splits its command, so we pass a single no-arg path.
set -e
cd "$FINN_ROOT"
python src/finn/kernels/tests/hardware/rtlsim_split_equiv_mvau.py 2>&1 \
  | tee "$FINN_ROOT/_rtlsim_equiv_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_rtlsim_equiv_out.txt"
