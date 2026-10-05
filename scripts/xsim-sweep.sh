#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Every XSim check of the kernel layer, run from this checkout in parallel:
#   - each conformance XSim test in its own pytest process,
#   - the rest of the kernel suite, the KernelOps' XSim tests and their Vivado tests
#     (packaging; marker vivado), with Vivado selected,
#   - the numeric XSI sweeps (MatMul, dotp, adapters), one simulation per process.
#
# For "XSim from a commit", run it in a worktree or clone at that commit: natively
# with its own .venv (uv sync), or in a sandbox with the image's environment.
# FinnLib is the `finnlib` resource, as in every run.
#
#   bash scripts/xsim-sweep.sh [--smoke] [OUT [TMP]]   exit 0 only if every job passed
#
# --smoke runs one conformance XSim test and one MatMul case: a few minutes, and
# enough to tell a broken harness or toolchain from a design result before the
# full sweep (~25 minutes) is spent on it.
#
# OUT keeps the evidence:
#   events.log    one line as each job starts and ends; read it for progress
#   collect.log   the collection of the conformance jobs
#   summary.log   the conformance count, per-job exit and pass counts, written when
#                 every job has ended (or at once, when collection failed)
#   summary.json  the same, for tools
#   logs/, sim-*  each job's log and simulation store
# pytest's temporary trees go to TMP (default OUT-tmp), which is scratch: they hold
# symlinks that point outside it.
#
# Vivado: this machine's ~/.config/finn/xilinx.env, or FINN_XILINX_PATH and
# FINN_XILINX_VERSION (applied through scripts/activate.sh; a variable wins over
# the file), or XILINX_VIVADO already selected.

set -u
SMOKE=0
if [ "${1:-}" = --smoke ]; then
    SMOKE=1
    shift
fi
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT" || exit 2
SHA=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
OUT=$(readlink -f "${1:-/tmp/xsim-sweep-$SHA-$(basename "$ROOT")}")
TMP=$(readlink -f "${2:-$OUT-tmp}")
# Two sweeps of one commit share the default OUT: never delete one in flight
# (an events.log without END that moved in the last two hours).
if [ -f "$OUT/events.log" ] && ! grep -q ' END overall' "$OUT/events.log" \
   && [ -n "$(find "$OUT/events.log" -mmin -120)" ]; then
    echo "a sweep is already running into $OUT; wait for it, or give another OUT" >&2
    exit 2
fi
rm -rf "$OUT" "$TMP" && mkdir -p "$OUT/logs" "$TMP"
EVENTS="$OUT/events.log"

if [ -x "$ROOT/.venv/bin/python" ] \
   && python3 "$ROOT/docker/xilinx_install.py" configured 2> "$OUT/activate.log"; then
    # Not under set -u: activation applies AMD's settings scripts, which read unset
    # variables, and an unset one would end this script silently.
    set +u
    # shellcheck source=/dev/null
    source "$ROOT/scripts/activate.sh" >> "$OUT/activate.log" 2>&1
    set -u
fi
if [ -z "${XILINX_VIVADO:-}" ]; then
    echo "no Vivado selected: configure ~/.config/finn/xilinx.env, or set XILINX_VIVADO" >&2
    exit 2
fi
# A selected but unapplied Vivado (an sbx sandbox, a bare shell): the sweeps load
# the simulation kernel in-process and need its library path. AMD's settings
# scripts read unset variables, so not under set -u.
if [ "${FINN_ENV_APPLIED:-}" != 1 ]; then
    set +u
    # shellcheck source=/dev/null
    . "$ROOT/docker/finn-toolchain.sh"
    set -u
fi
# The checkout's own environment natively; the image's active one in a container or sandbox.
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src:$ROOT/tests" FINN_ROOT="$ROOT"
export FINN_XELAB_MT="${FINN_XELAB_MT:-2}"
unset FORCE_COLOR

FINNLIB=$("$PY" -c 'from finn import resources; print(resources.path("finnlib"))' 2> /dev/null)
# A working clone by its commit; the cached pin, not a repository, by its verified digest.
if FINNLIB_SHA=$(git -C "$FINNLIB" rev-parse --short HEAD 2> /dev/null); then
    [ -z "$(git -C "$FINNLIB" status --porcelain --untracked-files=no | head -1)" ] \
        || FINNLIB_SHA="$FINNLIB_SHA dirty"
else
    FINNLIB_SHA="pinned $(head -c 19 "$FINNLIB/.finn-resource" 2> /dev/null || echo unknown)"
fi
IDENTITY="finn $SHA${DIRTY:+ dirty}  finnlib $FINNLIB_SHA $FINNLIB"

# Lines short enough that concurrent appends never interleave.
event() { echo "$(date +%T) $*" >> "$EVENTS"; }

job() {  # <log name> <command...>: run one job, its log ending in exit=N
    local name=$1 log="$OUT/logs/$1.log" start code; shift
    start=$(date +%s)
    event "START $name"
    "$@" > "$log" 2>&1
    code=$?
    echo "exit=$code" >> "$log"
    event "DONE $name exit=$code $(($(date +%s) - start))s"
}
pytest_run() {  # <log name> <pytest args...>
    local name=$1; shift
    job "$name" "$PY" -m pytest -v -p no:cacheprovider --basetemp="$TMP/$name" "$@"
}
sweep_run() {  # <log name> <module> [args...]
    local name=$1 module=$2; shift 2
    job "sweep-$name" "$PY" -m "$module" "$@" --output "$OUT/sim-$name"
}

# Writes summary.log and summary.json from the job logs, ends events.log and
# exits. <status>: 0, or non-zero when the sweep itself failed; a failed job
# fails it too.
finish() {
    local status=$1 rows=() log name code passes fails skips summary
    {
        echo "commit $(git rev-parse HEAD) checkout $ROOT"
        echo "$IDENTITY"
        echo "conformance collected=$collected collection exit=$collect_code"
        for log in "$OUT"/logs/*.log; do
            [ -e "$log" ] || continue
            name=$(basename "$log" .log)
            code=$(sed -n 's/^exit=//p' "$log" | tail -1)
            [ "$code" = 0 ] || status=1
            passes=$(grep -cE '^PASS|PASSED' "$log")
            fails=$(grep -cE '^FAIL|FAILED|^Traceback' "$log")
            skips=$(grep -c 'SKIPPED' "$log")
            case $name in
                sweep-*) summary="passes=$passes fails=$fails" ;;
                *) summary=$(grep -E ' passed| failed| skipped' "$log" | tail -1) ;;
            esac
            echo "$name exit=$code $summary"
            rows+=("{\"job\": \"$name\", \"exit\": ${code:-null}, \"passes\": $passes, \"fails\": $fails, \"skips\": $skips}")
        done
        echo "overall exit=$status"
    } > "$OUT/summary.log"
    {
        printf '{"commit": "%s", "dirty": %s, "smoke": %s, "exit": %s, "conformance_collected": %s, "jobs": [\n  ' \
            "$(git rev-parse HEAD)" "$([ -n "$DIRTY" ] && echo true || echo false)" \
            "$([ "$SMOKE" = 1 ] && echo true || echo false)" "$status" "$collected"
        (IFS=$'\n'; echo "${rows[*]}") | paste -sd ',' | sed 's/},{/},\n  {/g'
        printf ']}\n'
    } > "$OUT/summary.json"
    event "END overall exit=$status"
    cat "$OUT/summary.log"
    exit "$status"
}

event "SWEEP $IDENTITY smoke=$SMOKE"
# The conformance jobs, one per test id. Collected in an explicit mode: with the
# project's addopts cleared and one -q, pytest prints one path::id line per test,
# whatever .pytest.ini sets. A collection that fails or finds nothing ends the
# sweep: an empty list would run no conformance job and could still pass.
"$PY" -m pytest -o addopts= -q --collect-only -p no:cacheprovider --confcutdir=tests/kernels \
    tests/kernels/test_conformance.py -m xsim > "$OUT/collect.log" 2>&1
collect_code=$?
mapfile -t ids < <(grep '^tests/kernels/test_conformance\.py::' "$OUT/collect.log")
collected=${#ids[@]}
if [ "$collect_code" != 0 ] || [ "$collected" = 0 ]; then
    {
        echo "conformance collection failed (exit=$collect_code, $collected jobs); the end of $OUT/collect.log:"
        tail -n 15 "$OUT/collect.log"
    } >&2
    finish 1
fi
[ "$SMOKE" = 0 ] || ids=("${ids[@]:0:1}")
for id in "${ids[@]}"; do
    pytest_run "conformance-$(echo "${id#*::}" | tr -c 'A-Za-z0-9_\n-' '_')" \
        --confcutdir=tests/kernels "$id" &
done

if [ "$SMOKE" = 1 ]; then
    sweep_run packed kernels.rtlsim.matmul_numeric --case packed &
else
    pytest_run kernels-rest --confcutdir=tests/kernels tests/kernels \
        --ignore=tests/kernels/test_conformance.py &
    pytest_run kernel-ops-xsim --confcutdir=tests/kernel_ops tests/kernel_ops -m xsim &
    pytest_run kernel-ops-vivado --confcutdir=tests/kernel_ops tests/kernel_ops -m vivado &

    sweep_run dense kernels.rtlsim.matmul_numeric &
    sweep_run fifo-packed kernels.rtlsim.matmul_numeric --case packed --weight-fifo-depth 2 &
    sweep_run fifo-int8-pumped kernels.rtlsim.matmul_numeric --case int8_pumped --weight-fifo-depth 2 &
    sweep_run depthwise kernels.rtlsim.matmul_numeric --depthwise &
    sweep_run memstream kernels.rtlsim.matmul_numeric --delivery memstream &
    sweep_run memstream-depthwise kernels.rtlsim.matmul_numeric --depthwise --delivery memstream &
    sweep_run pumped-memory kernels.rtlsim.matmul_numeric --pumped-memory &
    sweep_run sets kernels.rtlsim.matmul_numeric --sets 3 &
    sweep_run dotp kernels.rtlsim.pure_dot_product_numeric &
    sweep_run dotp-stress kernels.rtlsim.pure_dot_product_numeric --stress &
    sweep_run adapters kernels.rtlsim.adapter_numeric &
fi
wait
finish 0
