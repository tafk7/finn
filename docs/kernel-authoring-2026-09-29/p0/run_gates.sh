#!/bin/bash
# P0.1: every step of check-kernels.sh (Space gate included) and check-dataflow-design.sh,
# without `set -e`, so a step failing on the missing deps/ does not hide the later steps.
# This worktree has no deps/ (P0 needs none): tests reading FinnLib sources fail on both
# the base and the patched engine; compare the two runs' FAILED lists.
#   bash docs/kernel-authoring-2026-09-29/p0/run_gates.sh <label>   -> p0/gates-<label>.log
cd "$(dirname "$(readlink -f "$0")")/../../.." || exit 1
export PYTHONDONTWRITEBYTECODE=1
PY=/home/tkeller/prj-kernels/.kernel-venv/bin/python
LOG=docs/kernel-authoring-2026-09-29/p0/gates-$1.log
step() { echo "=== $*"; "$@"; echo "=== exit=$? :: $*"; }
{
    step env PYTHON_BIN=$PY bash scripts/check-space.sh
    step env PYTHONPATH=src:tests $PY -m pytest -q -p no:cacheprovider --confcutdir=tests/kernels tests/kernels
    step env PYTHONPATH=src:tests $PY -m pytest -q -p no:cacheprovider --confcutdir=tests/graph tests/graph
    step ruff format --check src/finn/kernels tests/kernels src/finn/graph tests/graph scripts/benchmark-space.py
    step ruff check src/finn/kernels tests/kernels src/finn/graph tests/graph scripts/benchmark-space.py
    step env -u PYTHONPATH MYPYPATH=src:tests mypy --no-incremental --strict --explicit-package-bases \
        -p finn.kernels -p finn.graph
    step env PYTHON_BIN=$PY bash scripts/check-dataflow-design.sh
} > "$LOG" 2>&1
grep -E "^=== exit|^FAILED|^ERROR|passed|failed" "$LOG" | grep -v "^E "
