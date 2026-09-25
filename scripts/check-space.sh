#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
SCRIPT=$(readlink -f "$0")
FINN_ROOT=$(dirname "$(dirname "$SCRIPT")")
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}
cd "$FINN_ROOT"

"$PYTHON_BIN" --version
PYTHONPATH=src:tests "$PYTHON_BIN" -m pytest -q --confcutdir=tests/core/space tests/core/space
# Documentation examples are checked separately in scratchpad/space/.
"$RUFF_BIN" format --check src/finn/core/space tests/core/space
"$RUFF_BIN" check --extend-select I src/finn/core/space tests/core/space
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases -p finn.core.space
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental --strict --explicit-package-bases tests/core/space
