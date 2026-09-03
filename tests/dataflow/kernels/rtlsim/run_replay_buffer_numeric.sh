#!/bin/bash
set -euo pipefail
cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
log="${KERNEL_REPLAY_LOG:-$FINN_ROOT/kernel_replay_buffer_numeric.log}"
status=0
python tests/dataflow/kernels/rtlsim/replay_buffer_numeric.py \
    ${KERNEL_REPLAY_CASE:+--case "$KERNEL_REPLAY_CASE"} 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
