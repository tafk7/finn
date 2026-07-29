#!/usr/bin/env bash
# Runs the NEW phase-2 kernel test tree (fast-unit + integration + finn_codegen,
# excluding the slow_hw tier) inside the FINN docker env and writes a clean result
# tail to a host-visible file.
# Invoke via: ./run-docker.sh bash .agents/tmp/run_kernel_tests.sh
set -o pipefail
cd "$FINN_ROOT" || cd /home/tkeller/prj-kernels/finn
OUT=.agents/tmp/kernel_tests_result.txt
python -m pytest src/finn/kernels/tests -q -p no:cacheprovider --strict-markers \
    -m "not slow_hw" </dev/null >"$OUT" 2>&1
echo "PYTEST_EXIT=$?" >>"$OUT"
tail -25 "$OUT"
