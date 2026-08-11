#!/usr/bin/env bash
# Runs the slow hardware tier of the kernel test tree (Vivado/XSI required;
# finn_codegen tests need the FINN env but no Vivado). slow_hw tests self-skip when
# xsi.is_available()/Vivado is absent, so this is safe to run without a toolchain.
# Invoke via: ./run-docker.sh bash scripts/run_kernel_hw.sh
# Output location defaults to .agents/logs/ and is overridable with $KERNEL_LOG_DIR.
set -o pipefail
cd "${FINN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" || exit 1
LOG_DIR=${KERNEL_LOG_DIR:-.agents/logs}
mkdir -p "$LOG_DIR"
OUT="$LOG_DIR/kernel_hw_result.txt"
python -m pytest src/finn/kernels/tests/hw -q -p no:cacheprovider --strict-markers \
    --runslow </dev/null >"$OUT" 2>&1
echo "PYTEST_EXIT=$?" >>"$OUT"
tail -25 "$OUT"
