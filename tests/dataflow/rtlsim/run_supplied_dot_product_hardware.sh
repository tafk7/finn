#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
log="${D6A_HARDWARE_LOG:-$FINN_ROOT/d6a-memstream-hardware.log}"
status=0
python tests/dataflow/rtlsim/supplied_dot_product_hardware.py "$@" 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
