#!/bin/bash
# A1: XSim from a snapshot of a commit (git archive, its own deps/finnlib and deps/qonnx).
# One pytest process per conformance XSim test, all in parallel, plus the rest of the
# kernel suite with Vivado on PATH. Each simulation runs the xsim tools in a subprocess.
#   bash docs/kernel-authoring-2026-09-29/a1/xsim.sh <commit>   -> /tmp/a1-xsim-<sha>/logs/
set -u
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse --short "$1")
SNAP=/tmp/a1-xsim-$SHA
PY=/home/tkeller/prj-kernels/.kernel-venv/bin/python
rm -rf "$SNAP" && mkdir -p "$SNAP/deps" "$SNAP/logs" "$SNAP/tmp"
git -C "$ROOT" archive "$SHA" | tar -x -C "$SNAP"
cp -r "$ROOT/deps/finnlib" "$ROOT/deps/qonnx" "$SNAP/deps/"
cd "$SNAP" || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:deps/qonnx/src
run() {  # <log name> <pytest args...>
    local name=$1; shift
    "$PY" -m pytest -v -p no:cacheprovider --confcutdir=tests/kernels --basetemp="$SNAP/tmp/$name" \
        "$@" > "logs/$name.log" 2>&1
    echo "exit=$?" >> "logs/$name.log"
}
mapfile -t ids < <("$PY" -m pytest -q --collect-only -p no:cacheprovider --confcutdir=tests/kernels \
    tests/kernels/test_conformance.py -k xsim \
    | sed -n 's#^ *<Function \(.*\)>$#tests/kernels/test_conformance.py::\1#p')
for id in "${ids[@]}"; do
    run "$(echo "${id#*::}" | tr -c 'A-Za-z0-9_\n-' '_')" "$id" &
done
run kernels-rest tests/kernels --ignore=tests/kernels/test_conformance.py -q &
wait
for log in logs/*.log; do echo "$log: $(grep -E '^(PASSED|FAILED)| passed| failed|^exit=' "$log" | tail -2 | tr '\n' ' ')"; done
