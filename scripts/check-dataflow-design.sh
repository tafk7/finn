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
export PYTHONPATH="$FINN_ROOT/src:$FINN_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest -q --confcutdir=tests/dataflow tests/dataflow
"$PYTHON_BIN" -m pytest -q tests/fpgadataflow/test_mvau_cycle_estimate.py

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

MYPYPATH=src:tests "$MYPY_BIN" \
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
    tests/dataflow/mvau \
    tests/dataflow/parameters \
    tests/dataflow/synthetic_op.py \
    tests/dataflow/mvau_op_facts.py \
    tests/dataflow/test_authoring_op_design.py \
    tests/dataflow/test_authoring_scope.py \
    tests/dataflow/test_mvau_narrow_weights.py \
    tests/dataflow/test_dataflow_op.py \
    tests/dataflow/test_kernel_authoring.py \
    tests/dataflow/test_network.py \
    tests/dataflow/test_network_validation.py \
    tests/dataflow/test_mvau_op.py \
    tests/dataflow/test_kernel_pool.py \
    tests/dataflow/test_mvau_inference.py \
    tests/dataflow/test_dataflow_selection.py \
    tests/dataflow/test_op_kernel_waterfall_acceptance.py
