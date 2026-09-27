#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for the canonical logical values in finn.dataflow and their tests.
#
# finn.dataflow sits directly on finn.core.space and below finn.kernels, so this
# gate checks only the value layer. Run scripts/check-kernels.sh as well for a
# change that can reach the kernels built on it. Parked code under finn.parked
# is reference only and outside every gate.

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
RUN_PYTHONPATH="$RUN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow

DATAFLOW_SOURCES=(
    src/finn/dataflow
    tests/dataflow
)

"$RUFF_BIN" format --check "${DATAFLOW_SOURCES[@]}"
"$RUFF_BIN" check "${DATAFLOW_SOURCES[@]}"

# mypy resolves installed packages from the environment; QONNX, which ships no
# type information, is covered by an override in pyproject.toml.
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    -p finn.dataflow

# The canonical tests are strictly typed too.
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    tests/dataflow
