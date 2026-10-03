#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
SCRIPT=$(readlink -f "$0")
FINN_ROOT=$(dirname "$(dirname "$SCRIPT")")
export FINN_ROOT
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}
cd "$FINN_ROOT"
RUN_PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests"
# These are independent Space/kernel gates. Parked dataflow/graph tests remain
# outside this command; their compatibility is not claimed.
PYTHON_BIN="$PYTHON_BIN" RUFF_BIN="$RUFF_BIN" MYPY_BIN="$MYPY_BIN" bash scripts/check-space.sh
# XSim tests are deselected even when Vivado is selected (a FINN checkout's
# .envrc selects it): they take minutes each, and scripts/xsim-sweep.sh runs them.
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q -m "not xsim" --confcutdir=tests/kernels \
    tests/kernels
# The graph adapters, the layer above finn.kernels.
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q -m "not xsim" --confcutdir=tests/graph tests/graph
# The KernelOps (finn.custom_op.kernels), also above finn.kernels.
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q -m "not xsim" --confcutdir=tests/kernel_ops \
    tests/kernel_ops
"$RUFF_BIN" format --check src/finn/kernels tests/kernels src/finn/graph tests/graph \
    src/finn/custom_op/kernels src/finn/transformation/kernels tests/kernel_ops \
    scripts/benchmark-space.py
"$RUFF_BIN" check src/finn/kernels tests/kernels src/finn/graph tests/graph \
    src/finn/custom_op/kernels src/finn/transformation/kernels tests/kernel_ops \
    scripts/benchmark-space.py
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases -p finn.kernels -p finn.graph \
    -p finn.custom_op.kernels -p finn.transformation.kernels
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases \
    tests/kernels/typing tests/kernels/helpers.py \
    tests/kernels/test_boundaries.py tests/kernels/test_datatypes.py \
    tests/kernels/artifacts/conftest.py tests/kernels/artifacts/test_isolation.py \
    tests/kernels/test_installed_package.py
