#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FINN_ROOT=$(dirname "$SCRIPTPATH")
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}

cd "$FINN_ROOT"
export PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow

"$RUFF_BIN" format --check \
    src/finn/dataflow \
    tests/dataflow
"$RUFF_BIN" check \
    src/finn/dataflow \
    tests/dataflow

MYPYPATH=src:tests "$MYPY_BIN" \
    --strict \
    --explicit-package-bases \
    src/finn/dataflow/_engine \
    src/finn/dataflow/design \
    src/finn/dataflow/mvau_design.py \
    tests/dataflow/engine \
    tests/dataflow/design
