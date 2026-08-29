#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# The ~20 lines that matter out of a fixture log's ~1300.
#
# A green fixture 5 run is 1261 lines, of which 1256 are xelab boilerplate:
# INFO, Compiling, Copyright, and 176 identical "doesn't have a timescale"
# warnings.  Reading that to find out whether it passed is pure waste, and
# summarising it from memory is how a result nobody read gets reported as a
# pass.  This extracts the signal mechanically instead, so a report can only
# say what the log says.
#
# Usage: scripts/rtl-log-summary.sh fixture5.log
set -euo pipefail

log="${1:-}"
if [ -z "$log" ]; then
  echo "usage: $0 <fixture log>" >&2
  exit 2
fi
if [ ! -f "$log" ]; then
  # Distinguishable from a failing run: no log means the fixture never got far
  # enough to open one, which is "did not run", not "failed".
  echo "NO LOG AT: $log"
  exit 3
fi

now=$(date +%s)
modified=$(stat -c %Y "$log")

echo "== log =="
echo "path      $log"
echo "bytes     $(stat -c %s "$log")"
# Staleness is the only signal an XSI hang gives: the third load_sim_obj in a
# process hangs with no diagnostic and without the watchdog firing.  A run with
# no terminating EXIT= line and a log that has not grown in minutes is that.
echo "idle_for  $((now - modified))s"
if grep -qE '^EXIT=' "$log"; then
  echo "terminated yes"
else
  echo "terminated no  (still running, or died without reaching the runner's EXIT= line)"
fi

echo
echo "== signal =="
# Two-space indentation is how both fixtures print their own per-configuration
# results; Vivado's own indented output starts at four.  Everything else here
# is an anchored literal the fixtures emit.
grep -E \
  -e '^(finn|finnlib)[[:space:]]' \
  -e '^=+ fixture' \
  -e '^ {2}[A-Za-z]' \
  -e '^RESULT:' \
  -e '^EXIT=' \
  -e '^[0-9]+ passed' \
  "$log" || echo "(no fixture output -- the run never reached its own first print)"

echo
echo "== diagnostics =="
# Errors, plus the specific failure strings the fixtures raise themselves.
# SIGABRT is here because xelab's threaded elaborator aborts intermittently,
# which is a flake rather than a design failure and has to be named as one.
if ! grep -E \
  -e '^ERROR:' \
  -e '^[A-Za-z_.]*Error' \
  -e '^Traceback' \
  -e 'deadlock, watchdogs fired' \
  -e 'missing RTL source' \
  -e 'FinnLib RTL not found' \
  -e 'simulation subprocess failed' \
  -e 'A valid license was not found' \
  -e 'SIGABRT|SIGSEGV|Fatal' \
  "$log"; then
  echo "(none)"
fi

echo
echo "== warnings, deduplicated =="
# XSIM 43-4099 (no timescale) and 43-3431 (LIBRARY_PATH is set) fire in every
# green run -- 176 times in one -- and mean nothing here.  Counting the rest by
# message code says whether anything unfamiliar appeared without pasting it.
if ! grep -E '^WARNING:' "$log" \
  | grep -vE '\[XSIM 43-4099\]|\[XSIM 43-3431\]' \
  | sed -E 's/^WARNING: (\[[^]]+\]).*/\1/' \
  | sort | uniq -c | sort -rn; then
  echo "(none beyond the two known-benign XSIM codes)"
fi
