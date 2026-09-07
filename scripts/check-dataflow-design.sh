#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FINN_ROOT=$(dirname "$SCRIPTPATH")
export FINN_ROOT
REQUIRE_PARITY=0
for argument in "$@"; do
    case "$argument" in
        --require-parity) REQUIRE_PARITY=1 ;;
        *) printf 'unknown argument: %s\n' "$argument" >&2; exit 2 ;;
    esac
done

PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}

cd "$FINN_ROOT"

FINN_REVISION=$(git rev-parse HEAD)
printf 'finn     %s  %s\n' "$FINN_REVISION" "$FINN_ROOT"
if git -C "$FINN_ROOT/deps/finnlib" rev-parse HEAD >/dev/null 2>&1; then
    FINNLIB_REVISION=$(git -C "$FINN_ROOT/deps/finnlib" rev-parse HEAD)
    printf 'finnlib  %s  %s\n' "$FINNLIB_REVISION" "$FINN_ROOT/deps/finnlib"
else
    printf 'finnlib  unavailable\n'
fi
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

# Deliberately *not* exported: mypy below must not see qonnx on PYTHONPATH.  It
# ships no py.typed, so with it importable every qonnx import changes error code
# and the existing `type: ignore[import-not-found]` comments read as unused --
# dozens of false positives that have nothing to do with the change under test.
# Oracle parity: skipped on an ordinary developer machine, *required* for a
# Gate run.  Which mode produced a report is part of the report -- a suite that
# silently skipped its own comparison would read as though it had made it.
if [ "$REQUIRE_PARITY" = "1" ]; then
    export FINN_DATAFLOW_REQUIRE_PARITY=1
    ORACLE=${FINN_DATAFLOW_ORACLE:-$(dirname "$FINN_ROOT")/finn}
    if ORACLE_REVISION=$(git -C "$ORACLE" rev-parse HEAD 2>/dev/null); then
        printf 'oracle   %s  %s  (parity REQUIRED)\n' "$ORACLE_REVISION" "$ORACLE"
    else
        printf 'oracle   unavailable at %s  (parity REQUIRED -- the run will fail)\n' "$ORACLE"
    fi
else
    printf 'oracle   parity optional; pass --require-parity for gate evidence\n'
fi

PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q tests/fpgadataflow/test_mvau_cycle_estimate.py

DATAFLOW_SOURCES=(
    src/finn/analysis/verify_custom_nodes.py
    src/finn/custom_op/dataflow
    src/finn/dataflow
    tests/dataflow
    tests/fpgadataflow/test_mvau_cycle_estimate.py
)

"$RUFF_BIN" format --check "${DATAFLOW_SOURCES[@]}"
"$RUFF_BIN" check "${DATAFLOW_SOURCES[@]}"

# One package, one invocation.  The U1.5 reset removed the experimental stacks
# that forced this into a hand-maintained file list, so a module added under
# src/finn/dataflow is now type-checked without editing this script.
env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    -p finn.dataflow \
    -p finn.custom_op.dataflow

env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --no-incremental \
    --strict \
    --explicit-package-bases \
    src/finn/analysis/verify_custom_nodes.py \
    tests/dataflow/engine \
    tests/dataflow/parameters \
    tests/dataflow/typing \
    tests/dataflow/kernels/test_module_build_spec.py \
    tests/dataflow/designs/test_projection_boundary.py \
    tests/dataflow/space/test_value_semantics.py \
    tests/dataflow/model/test_datatypes.py \
    tests/dataflow/model/test_facade.py \
    tests/dataflow/model/test_network.py \
    tests/dataflow/model/test_network_validation.py \
    tests/dataflow/test_package_boundaries.py
