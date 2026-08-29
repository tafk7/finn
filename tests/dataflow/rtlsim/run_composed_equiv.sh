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
status=0
python tests/dataflow/rtlsim/composed_mvau_equiv.py 2>&1 \
    | tee "$FINN_ROOT/_fixture5_out.txt" || status=$?
echo "EXIT=$status" | tee -a "$FINN_ROOT/_fixture5_out.txt"
exit "$status"
