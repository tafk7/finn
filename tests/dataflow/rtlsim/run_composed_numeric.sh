#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Fixture 8 runner. run-docker.sh word-splits its command, so this takes no
# arguments and is passed as a single path.
# pipefail matters here: without it the exit status is tee's, so a failing
# comparison would be reported as a pass by anything checking $?.
set -euo pipefail
cd "$FINN_ROOT"
# The harness spawns rtl_transport per simulation and imports its own package
# path, so tests/ has to be importable.
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
# Vivado writes straight to fd 1 while Python buffers into a pipe, so without
# this the log reads as pages of xelab noise with the results appended at the
# end -- useless for watching a long run.
export PYTHONUNBUFFERED=1
log="${FIXTURE8_LOG:-$FINN_ROOT/fixture8.log}"
status=0
python tests/dataflow/rtlsim/composed_mvau_numeric.py 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
