#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Fixture 5 runner. run-docker.sh word-splits its command, so this takes no
# arguments and is passed as a single path.
# pipefail matters here: without it the exit status is tee's, so a failing
# comparison would be reported as a pass by anything checking $?.
set -euo pipefail
cd "$FINN_ROOT"
# A generated log, so it goes somewhere .gitignore already covers (*.log) and
# never into the tree.  Override to put it elsewhere.
log="${FIXTURE5_LOG:-$FINN_ROOT/fixture5.log}"
status=0
python tests/dataflow/rtlsim/composed_mvau_equiv.py 2>&1 | tee "$log" || status=$?
echo "EXIT=$status" | tee -a "$log"
exit "$status"
