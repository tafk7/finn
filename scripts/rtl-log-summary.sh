#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# The lines that matter out of an RTL run, extracted mechanically so a report
# can only say what the logs say.
#
#   scripts/rtl-log-summary.sh OUT        an xsim-sweep.sh output directory
#   scripts/rtl-log-summary.sh LOG        one job's log (a numeric sweep or pytest)
#
# A green job log is mostly xelab boilerplate: INFO, Compiling, Copyright, and
# hundreds of identical "doesn't have a timescale" warnings. Reading that to
# find out whether it passed is pure waste, and summarising it from memory is
# how a result nobody read gets reported as a pass.
set -euo pipefail

target="${1:-}"
if [ -z "$target" ]; then
  echo "usage: $0 <xsim-sweep output directory | job log>" >&2
  exit 2
fi
if [ ! -e "$target" ]; then
  # Distinguishable from a failing run: nothing there means the run never got
  # far enough to open it, which is "did not run", not "failed".
  echo "NOTHING AT: $target"
  exit 3
fi
now=$(date +%s)

idle() { echo $((now - $(stat -c %Y "$1"))); }

if [ -d "$target" ]; then
  events="$target/events.log"
  echo "== sweep =="
  echo "path      $target"
  if [ ! -f "$events" ]; then
    echo "no events.log: not an xsim-sweep.sh output, or it died before starting a job"
    exit 3
  fi
  sed -n 's/^[0-9:]* SWEEP //p' "$events" | head -1
  started=$(grep -c ' START ' "$events" || true)
  done_=$(grep -c ' DONE ' "$events" || true)
  # Under the sweep's --jobs cap, a listed job that has not started (nor been skipped) waits.
  listed=$(grep -c . "$target/jobs.tsv" 2> /dev/null || true)
  skipped=$(grep -c ' SKIP ' "$events" || true)
  echo "jobs      $done_/$started done, $((${listed:-0} - started - skipped)) waiting, $skipped skipped"
  if grep -q ' END overall' "$events"; then
    echo "finished  $(sed -n 's/^\([0-9:]*\) END overall \(.*\)/\2 at \1/p' "$events")"
  else
    echo "finished  no  (still running, or the sweep itself died; idle_for $(idle "$events")s)"
  fi
  echo
  echo "== failed jobs =="
  grep ' DONE ' "$events" | grep -v ' exit=0 ' || echo "(none)"
  echo
  echo "== running =="
  # A job with a START and no DONE. XSI's hang gives no diagnostic at all, so a
  # log that has stopped growing is the only sign of one.
  running=0
  for name in $(sed -n 's/^[0-9:]* START //p' "$events"); do
    grep -q " DONE $name exit=" "$events" && continue
    running=1
    echo "$name  idle_for $(idle "$target/logs/$name.log")s  $(grep -cE '^PASS|PASSED' "$target/logs/$name.log" || true) passes so far"
  done
  [ "$running" = 1 ] || echo "(none)"
  if [ -f "$target/summary.log" ]; then
    echo
    echo "== summary.log =="
    cat "$target/summary.log"
  fi
  exit 0
fi

log="$target"
echo "== log =="
echo "path      $log"
echo "bytes     $(stat -c %s "$log")"
echo "idle_for  $(idle "$log")s"
if grep -qE '^(exit|EXIT)=' "$log"; then
  echo "terminated yes  $(grep -E '^(exit|EXIT)=' "$log" | tail -1)"
else
  echo "terminated no  (still running, or not launched through xsim-sweep.sh's runner)"
fi

echo
echo "== signal =="
# Identity (finn/finnlib), the numeric sweeps' own PASS/FAIL lines, pytest's
# per-test verdicts and its closing tally.
grep -E \
  -e '^(finn|finnlib)[[:space:]]' \
  -e '^Evidence:' \
  -e '^(PASS|FAIL)' \
  -e ' (PASSED|FAILED|SKIPPED|ERROR)( |$)' \
  -e '^=+ .*(passed|failed|skipped|error)' \
  -e '^(exit|EXIT)=' \
  "$log" || echo "(no harness output -- the run never reached its own first print)"

echo
echo "== diagnostics =="
# Errors, plus the specific failure strings the harness raises itself. SIGABRT
# is here because xelab's threaded elaborator aborts intermittently, which is a
# flake rather than a design failure and has to be named as one.
if ! grep -E \
  -e '^ERROR:' \
  -e '^E  ' \
  -e '^[A-Za-z_.]*Error' \
  -e '^Traceback' \
  -e 'deadlock, watchdogs fired' \
  -e 'missing RTL source' \
  -e 'FinnLib RTL not found' \
  -e 'simulation subprocess (failed|exit)' \
  -e 'simulation timed out after' \
  -e 'A valid license was not found' \
  -e 'SIGABRT|SIGSEGV|Fatal' \
  "$log" | head -40; then
  echo "(none)"
fi

echo
echo "== warnings, deduplicated =="
# XSIM 43-4099 (no timescale) and 43-3431 (LIBRARY_PATH is set) fire in every
# green run and mean nothing here. Counting the rest by message code says
# whether anything unfamiliar appeared without pasting it.
if ! grep -E '^WARNING:' "$log" \
  | grep -vE '\[XSIM 43-4099\]|\[XSIM 43-3431\]' \
  | sed -E 's/^WARNING: (\[[^]]+\]).*/\1/' \
  | sort | uniq -c | sort -rn | grep .; then
  echo "(none beyond the two known-benign XSIM codes)"
fi
