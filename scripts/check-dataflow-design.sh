#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for the canonical logical values in finn.dataflow and their tests.
#
# finn.dataflow sits directly on finn.core.space and below finn.kernels, so this
# gate checks only the value layer. Run scripts/check-kernels.sh as well for a
# change that can reach the kernels built on it. Parked code under finn.parked
# and tests/parked is outside every gate.

set -euo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FINN_ROOT=$(dirname "$SCRIPTPATH")
export FINN_ROOT
export PYTHONDONTWRITEBYTECODE=1
if [ "$#" -ne 0 ]; then
    printf 'unexpected argument: %s\n' "$1" >&2
    exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}

cd "$FINN_ROOT"

printf 'finn     %s  %s\n' "$(git rev-parse HEAD)" "$FINN_ROOT"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pytest --version
"$RUFF_BIN" --version
"$MYPY_BIN" --version

# The tests need qonnx importable.  Prefer the pinned checkout fetch-repos.sh
# places under deps/, so the gate does not depend on the caller having installed
# it, and fall back to whatever the environment provides.
RUN_PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests"
if [ -d "$FINN_ROOT/deps/qonnx/src" ]; then
    RUN_PYTHONPATH="$RUN_PYTHONPATH:$FINN_ROOT/deps/qonnx/src"
fi
RUN_PYTHONPATH="$RUN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow

DATAFLOW_SOURCES=(
    src/finn/dataflow
    tests/dataflow
)

"$RUFF_BIN" format --check "${DATAFLOW_SOURCES[@]}"
"$RUFF_BIN" check "${DATAFLOW_SOURCES[@]}"

# Deliberately *not* on PYTHONPATH: mypy must not see qonnx.  It ships no
# py.typed, so with it importable every qonnx import changes error code and the
# existing `type: ignore[import-not-found]` comments read as unused.
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    -p finn.dataflow

# The typed canonical tests.  The remaining files under tests/dataflow predate
# strict typing of tests and are checked by pytest and ruff only.
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    tests/dataflow/parameters \
    tests/dataflow/model/test_composition.py \
    tests/dataflow/model/test_datatypes.py \
    tests/dataflow/model/test_facade.py \
    tests/dataflow/model/test_network.py \
    tests/dataflow/model/test_network_validation.py \
    tests/dataflow/test_package_boundaries.py
