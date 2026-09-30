#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Every XSim check of the kernel layer, run from this checkout in parallel:
#   - each conformance XSim test in its own pytest process,
#   - the rest of the kernel suite with Vivado selected,
#   - the numeric XSI sweeps (MatMul, dotp, adapters), one simulation per process.
#
# For "XSim from a commit", run it in a worktree or clone at that commit: natively
# with its own .venv (uv sync), or in a sandbox with the image's environment.
# FinnLib is the `finnlib` resource, as in every run.
#
#   bash scripts/xsim-sweep.sh [OUT]      -> OUT/summary.log, exit 0 only if all pass
#
# Vivado: set FINN_XILINX_PATH and FINN_XILINX_VERSION (applied through
# scripts/activate.sh), or have XILINX_VIVADO already selected.

set -u
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT" || exit 2
SHA=$(git rev-parse --short HEAD)
OUT=$(readlink -f "${1:-/tmp/xsim-sweep-$SHA-$(basename "$ROOT")}")
rm -rf "$OUT" && mkdir -p "$OUT/logs" "$OUT/tmp"

if [ -n "${FINN_XILINX_PATH:-}" ] && [ -n "${FINN_XILINX_VERSION:-}" ]; then
    # shellcheck source=/dev/null
    source "$ROOT/scripts/activate.sh" > "$OUT/activate.log" 2>&1
fi
if [ -z "${XILINX_VIVADO:-}" ]; then
    echo "no Vivado selected: set FINN_XILINX_PATH/FINN_XILINX_VERSION or XILINX_VIVADO" >&2
    exit 2
fi
# The checkout's own environment natively; the image's active one in a container or sandbox.
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src:$ROOT/tests" FINN_ROOT="$ROOT"
export FINN_XELAB_MT="${FINN_XELAB_MT:-2}"
unset FORCE_COLOR

pytest_run() {  # <log name> <pytest args...>
    local name=$1; shift
    "$PY" -m pytest -v -p no:cacheprovider --confcutdir=tests/kernels \
        --basetemp="$OUT/tmp/$name" "$@" > "$OUT/logs/$name.log" 2>&1
    echo "exit=$?" >> "$OUT/logs/$name.log"
}
sweep_run() {  # <log name> <module> [args...]
    local name=$1 module=$2; shift 2
    "$PY" -m "$module" "$@" --output "$OUT/sim-$name" > "$OUT/logs/sweep-$name.log" 2>&1
    echo "exit=$?" >> "$OUT/logs/sweep-$name.log"
}

mapfile -t ids < <("$PY" -m pytest -q --collect-only -p no:cacheprovider --confcutdir=tests/kernels \
    tests/kernels/test_conformance.py -k xsim \
    | sed -n 's#^ *<Function \(.*\)>$#tests/kernels/test_conformance.py::\1#p')
for id in "${ids[@]}"; do
    pytest_run "conformance-$(echo "${id#*::}" | tr -c 'A-Za-z0-9_\n-' '_')" "$id" &
done
pytest_run kernels-rest tests/kernels --ignore=tests/kernels/test_conformance.py &

sweep_run dense kernels.rtlsim.matmul_numeric &
sweep_run fifo-packed kernels.rtlsim.matmul_numeric --case packed --weight-fifo-depth 2 &
sweep_run fifo-int8-pumped kernels.rtlsim.matmul_numeric --case int8_pumped --weight-fifo-depth 2 &
sweep_run depthwise kernels.rtlsim.matmul_numeric --depthwise &
sweep_run memstream kernels.rtlsim.matmul_numeric --delivery memstream &
sweep_run memstream-depthwise kernels.rtlsim.matmul_numeric --depthwise --delivery memstream &
sweep_run pumped-memory kernels.rtlsim.matmul_numeric --pumped-memory &
sweep_run writable kernels.rtlsim.matmul_numeric --writable &
sweep_run sets kernels.rtlsim.matmul_numeric --sets 3 &
sweep_run dotp kernels.rtlsim.pure_dot_product_numeric &
sweep_run dotp-stress kernels.rtlsim.pure_dot_product_numeric --stress &
sweep_run adapters kernels.rtlsim.adapter_numeric &
wait

status=0
{
    echo "commit $(git rev-parse HEAD) checkout $ROOT"
    for log in "$OUT"/logs/*.log; do
        code=$(sed -n 's/^exit=//p' "$log" | tail -1)
        [ "$code" = 0 ] || status=1
        case $(basename "$log") in
            sweep-*) summary="passes=$(grep -c '^PASS' "$log") fails=$(grep -c '^FAIL' "$log")" ;;
            *) summary=$(grep -E ' passed| failed' "$log" | tail -1) ;;
        esac
        echo "$(basename "$log" .log) exit=$code $summary"
    done
    echo "overall exit=$status"
} > "$OUT/summary.log"
cat "$OUT/summary.log"
exit $status
