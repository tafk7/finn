#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Fixture 5 runner. run-docker.sh word-splits its command, so this takes no
# arguments and is passed as a single path.
set -e
cd "$FINN_ROOT"
python tests/dataflow/rtlsim/composed_mvau_equiv.py 2>&1 | tee "$FINN_ROOT/_fixture5_out.txt"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$FINN_ROOT/_fixture5_out.txt"
