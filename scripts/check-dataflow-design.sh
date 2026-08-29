#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FINN_ROOT=$(dirname "$SCRIPTPATH")
export FINN_ROOT
PYTHON_BIN=${PYTHON_BIN:-python3}
RUFF_BIN=${RUFF_BIN:-ruff}
MYPY_BIN=${MYPY_BIN:-mypy}

cd "$FINN_ROOT"

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
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow
PYTHONPATH="$RUN_PYTHONPATH" "$PYTHON_BIN" -m pytest -q tests/fpgadataflow/test_mvau_cycle_estimate.py

DATAFLOW_SOURCES=(
    src/finn/analysis/verify_custom_nodes.py
    src/finn/dataflow
    src/finn/transformation/fpgadataflow/infer_mvau_dataflow.py
    src/finn/transformation/fpgadataflow/select_dataflow_design.py
    tests/dataflow
    tests/fpgadataflow/test_mvau_cycle_estimate.py
)

"$RUFF_BIN" format --check "${DATAFLOW_SOURCES[@]}"
"$RUFF_BIN" check "${DATAFLOW_SOURCES[@]}"

env -u PYTHONPATH MYPYPATH=src:tests "$MYPY_BIN" \
    --strict \
    --explicit-package-bases \
    src/finn/analysis/verify_custom_nodes.py \
    src/finn/dataflow/_engine \
    src/finn/dataflow/design \
    src/finn/dataflow/authoring \
    src/finn/dataflow/op.py \
    src/finn/dataflow/resolution.py \
    src/finn/dataflow/testing \
    src/finn/dataflow/region.py \
    src/finn/dataflow/region_profiles.py \
    src/finn/dataflow/region_validation.py \
    src/finn/dataflow/kernel.py \
    src/finn/dataflow/kernels.py \
    src/finn/dataflow/hardware \
    src/finn/dataflow/spec_algebra.py \
    src/finn/dataflow/network.py \
    src/finn/dataflow/network_validation.py \
    src/finn/dataflow/mvau \
    src/finn/dataflow/parameters \
    src/finn/dataflow/ops \
    src/finn/dataflow/mvau_design.py \
    src/finn/custom_op/dataflow \
    src/finn/transformation/fpgadataflow/infer_mvau_dataflow.py \
    src/finn/transformation/fpgadataflow/select_dataflow_design.py \
    tests/dataflow/engine \
    tests/dataflow/design \
    tests/dataflow/hardware \
    tests/dataflow/mvau \
    tests/dataflow/parameters \
    tests/dataflow/synthetic_op.py \
    tests/dataflow/mvau_op_facts.py \
    tests/dataflow/normalized_structure.py \
    tests/dataflow/test_datatypes.py \
    tests/dataflow/test_kernel_admission.py \
    tests/dataflow/test_authoring_op_design.py \
    tests/dataflow/test_authoring_scope.py \
    tests/dataflow/test_mvau_narrow_weights.py \
    tests/dataflow/test_mvau_problem_fields.py \
    tests/dataflow/test_dataflow_op.py \
    tests/dataflow/test_kernel_authoring.py \
    tests/dataflow/test_kernel_class_authoring.py \
    tests/dataflow/test_network.py \
    tests/dataflow/test_network_validation.py \
    tests/dataflow/test_mvau_op.py \
    tests/dataflow/test_kernel_pool.py \
    tests/dataflow/test_mvau_inference.py \
    tests/dataflow/test_dataflow_selection.py \
    tests/dataflow/test_op_kernel_waterfall_acceptance.py
