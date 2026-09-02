#!/bin/bash
set -euo pipefail
cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
log="${KERNEL_REPLAY_SYNTH_LOG:-$FINN_ROOT/kernel_replay_buffer_synth.log}"
status=0
python tests/dataflow/model/rtlsim/replay_buffer_synth.py \
    ${KERNEL_REPLAY_SYNTH_CASE:+--case "$KERNEL_REPLAY_SYNTH_CASE"} 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
