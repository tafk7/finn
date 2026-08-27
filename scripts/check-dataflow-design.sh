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
    src/finn/dataflow/region.py \
    src/finn/dataflow/region_profiles.py \
    src/finn/dataflow/region_validation.py \
    src/finn/dataflow/kernel.py \
    src/finn/dataflow/network.py \
    src/finn/dataflow/network_validation.py \
    src/finn/dataflow/mvau \
    src/finn/dataflow/parameters \
    src/finn/dataflow/ops \
    src/finn/dataflow/mvau_design.py \
    tests/dataflow/engine \
    tests/dataflow/design \
    tests/dataflow/mvau \
    tests/dataflow/parameters \
    tests/dataflow/test_kernel_authoring.py \
    tests/dataflow/test_network.py \
    tests/dataflow/test_network_validation.py \
    tests/dataflow/test_mvau_op.py
