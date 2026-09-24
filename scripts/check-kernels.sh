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
if [ -d "$FINN_ROOT/deps/qonnx/src" ]; then
    RUN_PYTHONPATH="$RUN_PYTHONPATH:$FINN_ROOT/deps/qonnx/src"
fi
"$PYTHON_BIN" --version
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q --confcutdir=tests/kernels tests/kernels
"$RUFF_BIN" format --check src/finn/kernels tests/kernels
"$RUFF_BIN" check src/finn/kernels tests/kernels
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases -p finn.kernels
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases \
    tests/kernels/space tests/kernels/typing tests/kernels/helpers.py \
    tests/kernels/test_boundaries.py tests/kernels/test_datatypes.py \
    tests/kernels/artifacts/conftest.py tests/kernels/artifacts/test_isolation.py \
    tests/kernels/test_installed_package.py
