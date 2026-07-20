#!/bin/bash
# Runner for the COMPOSED (stitched) MVAU bd-elaboration check inside the FINN
# container. run-docker.sh word-splits its command, so we pass a single no-arg script.
set -e
cd "$FINN_ROOT"
python src/finn/design_space/tests/elaborate_composed_mvau.py 2>&1 \
  | tee "$FINN_ROOT/_elab_composed_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_elab_composed_out.txt"
