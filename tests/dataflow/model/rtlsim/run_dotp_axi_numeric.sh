#!/bin/bash
set -euo pipefail
cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
log="${KERNEL_DOTP_LOG:-$FINN_ROOT/kernel_dotp_axi_numeric.log}"
status=0
python tests/dataflow/model/rtlsim/dotp_axi_numeric.py \
    ${KERNEL_DOTP_CASE:+--case "$KERNEL_DOTP_CASE"} 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
