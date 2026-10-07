#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Every XSim check of the kernel layer, run from this checkout in parallel:
#   - each conformance XSim test in its own pytest process,
#   - the rest of the kernel suite, the KernelOps' XSim tests and their Vivado tests
#     (packaging; marker vivado), with Vivado selected,
#   - the numeric XSI sweeps (MatMul, dotp, adapters, thresholds), one simulation per process.
#
# For "XSim from a commit", run it in a worktree or clone at that commit: natively
# with its own .venv (uv sync), or in a sandbox with the image's environment.
# FinnLib is the `finnlib` resource, as in every run.
#
#   bash scripts/xsim-sweep.sh [--smoke | --changed-since BASELINE] [OUT [TMP]]
#                                                       exit 0 only if every job passed
#
# --smoke runs one conformance XSim test and one MatMul case: a few minutes, and
# enough to tell a broken harness or toolchain from a design result before the
# full sweep (~25 minutes) is spent on it.
#
# Every job has a key: a digest of everything its simulation consumes (the designs
# it simulates, as materialized without Vivado; its harness and stimulus code; the
# XSI runtime; this script; the selected Vivado), computed by scripts/emitted_text.py.
# A full sweep records each job's key in summary.json. --changed-since BASELINE,
# the summary.json of a passed full sweep, runs only the jobs whose key differs
# from the baseline's or that the baseline did not pass, and records the rest as
# skipped (unchanged since the baseline's commit) with their keys. Its exit is 0
# only if every job it ran passed and the baseline itself passed (overall exit=0,
# not smoke); such a sweep can be a baseline in turn.
#
# OUT keeps the evidence:
#   events.log    one line as each job starts and ends; read it for progress
#   collect.log   the collection of the conformance jobs
#   jobs.tsv      the jobs: name, kind, arguments
#   keys.json     each job's key and the inputs it digests (keys.log: how they were computed)
#   selection.tsv with --changed-since: run or skip per job, and why
#   summary.log   the conformance count, per-job exit and pass counts (or skipped,
#                 and why), written when every job has ended (or at once, when
#                 collection failed)
#   summary.json  the same, with each job's key and inputs, for tools
#   logs/, sim-*  each job's log and simulation store
# pytest's temporary trees, and the designs the keys digest, go to TMP (default
# OUT-tmp), which is scratch: they hold symlinks that point outside it.
#
# Vivado: this machine's ~/.config/finn/xilinx.env, or FINN_XILINX_PATH and
# FINN_XILINX_VERSION (applied through scripts/activate.sh; a variable wins over
# the file), or XILINX_VIVADO already selected.

set -u
SMOKE=0
BASELINE=
while [ $# -gt 0 ]; do
    case $1 in
        --smoke) SMOKE=1; shift ;;
        --changed-since)
            BASELINE=$(readlink -f "${2:-}")
            if [ ! -f "$BASELINE" ]; then
                echo "--changed-since needs the summary.json of a passed full sweep" >&2
                exit 2
            fi
            shift 2 ;;
        *) break ;;
    esac
done
if [ "$SMOKE" = 1 ] && [ -n "$BASELINE" ]; then
    echo "--smoke and --changed-since exclude each other" >&2
    exit 2
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

TOOL="$ROOT/scripts/emitted_text.py"
collect_code=

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

# Writes summary.log and summary.json (scripts/emitted_text.py summarize), ends
# events.log and exits. <status>: 0, or non-zero when the sweep itself failed; a
# failed job fails it too, as does a baseline that did not pass.
finish() {
    local status
    "$PY" "$TOOL" summarize "$OUT" --status "$1" --identity "$IDENTITY" \
        --collect-code "$collect_code" ${BASELINE:+--baseline "$BASELINE"} \
        $([ "$SMOKE" = 1 ] && echo --smoke) $([ -n "$DIRTY" ] && echo --dirty)
    status=$?
    event "END overall exit=$status"
    exit "$status"
}

event "SWEEP $IDENTITY smoke=$SMOKE${BASELINE:+ changed-since=$BASELINE}"
# The jobs: one per conformance XSim test id (collected; a collection that fails
# or finds nothing ends the sweep, as an empty list could still pass), the pytest
# groups and the numeric sweeps.
"$PY" "$TOOL" jobs $([ "$SMOKE" = 1 ] && echo --smoke) --collect-log "$OUT/collect.log" \
    > "$OUT/jobs.tsv" 2> "$OUT/jobs.log"
collect_code=$?
if [ "$collect_code" != 0 ]; then
    cat "$OUT/jobs.log" >&2
    finish 1
fi

keys() {
    "$PY" "$TOOL" keys --jobs "$OUT/jobs.tsv" --work "$TMP/keys" > "$OUT/keys.json.part" \
        2> "$OUT/keys.log" && mv "$OUT/keys.json.part" "$OUT/keys.json"
}
if [ -n "$BASELINE" ]; then
    # Keys first: they decide what runs. Without keys, every job runs.
    if keys && "$PY" "$TOOL" select "$BASELINE" "$OUT/keys.json" > "$OUT/selection.tsv"; then
        event "SELECTED $(grep -c $'\trun\t' "$OUT/selection.tsv") of $(wc -l < "$OUT/jobs.tsv") jobs"
    else
        event "NO KEYS (keys.log): every job runs"
        : > "$OUT/selection.tsv"
    fi
else
    keys &  # beside the jobs: only the summary reads them
fi

# The list on descriptor 3: a job reading its standard input must not consume it.
while IFS=$'\t' read -r -u 3 -a fields; do
    name=${fields[0]} kind=${fields[1]} args=("${fields[@]:2}")
    if grep -q "^$name"$'\tskip\t' "$OUT/selection.tsv" 2> /dev/null; then
        event "SKIP $name"
        continue
    fi
    case $kind in
        sweep) job "$name" "$PY" -m "${args[@]}" --output "$OUT/sim-${name#sweep-}" & ;;
        *) pytest_run "$name" "${args[@]}" & ;;
    esac
done 3< "$OUT/jobs.tsv"
wait
finish 0
