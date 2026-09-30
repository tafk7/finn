#!/bin/bash
# The numeric XSI sweeps (MatMul, dotp, adapters) from a snapshot of a commit, beside
# a1/xsim.sh's conformance and pytest run. Each simulation runs in its own process.
#   bash docs/kernel-authoring-2026-09-29/xsim-sweeps.sh <commit>   -> /tmp/ka-sweeps-<sha>/summary.log
set -u
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse --short "$1")
BASE=/tmp/ka-sweeps-$SHA; TREE=$BASE/tree
rm -rf "$BASE"; mkdir -p "$TREE/deps"
git -C "$ROOT" archive "$SHA" | tar -x -C "$TREE"
cp -r "$ROOT/deps/finnlib" "$ROOT/deps/qonnx" "$TREE/deps/"
# The built XSI extension is ignored build output; XSI loads it from FINN_ROOT.
cp /home/tkeller/prj-kernels/finn-kernels-extraction/finn_xsi/xsi.so "$TREE/finn_xsi/"
cd "$TREE" || exit 1
PY=/home/tkeller/prj-kernels/.kernel-venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$TREE/src:$TREE/tests:$TREE/deps/qonnx/src \
    FINN_ROOT=$TREE FINNLIB_ROOT=$TREE/deps/finnlib FINN_XELAB_MT=2 \
    LD_LIBRARY_PATH=/home/tkeller/Xilinx/2025.2/Vivado/lib/lnx64.o
unset FORCE_COLOR
LOG=$BASE/summary.log
run() {  # <name> <module> [args...]
    local name=$1 module=$2; shift 2
    "$PY" -m "$module" "$@" --output "$BASE/sim-$name" > "$BASE/$name.txt" 2>&1
    echo "exit=$?" >> "$BASE/$name.txt"
    echo "$name passes=$(grep -c '^PASS' "$BASE/$name.txt") fails=$(grep -c '^FAIL' "$BASE/$name.txt") $(tail -1 "$BASE/$name.txt")" >> "$LOG"
}
run dense kernels.rtlsim.matmul_numeric &
run fifo-packed kernels.rtlsim.matmul_numeric --case packed --weight-fifo-depth 2 &
run fifo-int8-pumped kernels.rtlsim.matmul_numeric --case int8_pumped --weight-fifo-depth 2 &
run depthwise kernels.rtlsim.matmul_numeric --depthwise &
run memstream kernels.rtlsim.matmul_numeric --delivery memstream &
run memstream-depthwise kernels.rtlsim.matmul_numeric --depthwise --delivery memstream &
run pumped-memory kernels.rtlsim.matmul_numeric --pumped-memory &
run writable kernels.rtlsim.matmul_numeric --writable &
run sets kernels.rtlsim.matmul_numeric --sets 3 &
run dotp kernels.rtlsim.pure_dot_product_numeric &
run dotp-stress kernels.rtlsim.pure_dot_product_numeric --stress &
run adapters kernels.rtlsim.adapter_numeric &
wait
echo done >> "$LOG"
cat "$LOG"
