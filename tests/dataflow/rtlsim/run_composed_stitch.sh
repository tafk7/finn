#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Fixture 7 runner. run-docker.sh word-splits its command, so this takes no
# arguments and is passed as a single path.
# pipefail matters here: without it the exit status is tee's, so a failing
# block design would be reported as a pass by anything checking $?.
set -euo pipefail
cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
# Vivado writes straight to fd 1 while Python buffers into a pipe; without this
# the results land at the end of pages of Vivado noise.
export PYTHONUNBUFFERED=1
log="${FIXTURE7_LOG:-$FINN_ROOT/fixture7.log}"
status=0
python tests/dataflow/rtlsim/composed_mvau_stitch.py 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
