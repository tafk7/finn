#!/usr/bin/env bash
# Runs the slow hardware tier of the phase-2 kernel test tree (Vivado/XSI required;
# finn_codegen tests need the FINN env but no Vivado). slow_hw tests self-skip when
# xsi.is_available()/Vivado is absent, so this is safe to run without a toolchain.
# Invoke via: ./run-docker.sh bash .agents/tmp/run_kernel_hw.sh
set -o pipefail
cd "$FINN_ROOT" || cd /home/tkeller/prj-kernels/finn
OUT=.agents/tmp/kernel_hw_result.txt
python -m pytest src/finn/kernels/tests/hw -q -p no:cacheprovider --strict-markers \
    --runslow </dev/null >"$OUT" 2>&1
echo "PYTEST_EXIT=$?" >>"$OUT"
tail -25 "$OUT"
