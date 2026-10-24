#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for finn.core.space and its tests, for finn.core.reporting, the layer below it, and for
# finn.core.search, the search's generic half above it.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

gate_pytest tests/core/reporting
gate_pytest tests/core/space
gate_pytest tests/core/search
# Documentation examples are checked separately in scratchpad/space/.
# The layer table (tests/support/layering.py) and the refusals' helper
# (tests/support/refusals.py), which every layer's tests use, the two cores the engine's tests
# share (tests/support/space_cores.py), the chain two engine tests walk
# (tests/support/space_pipeline.py), the walk an engine test and a kernel test share
# (tests/support/space_walk.py), the class-based oracle's engine-only modules
# (tests/support/space_oracle, but for its targets and tool), and the pre-commit check every
# gate's gate_ruff runs, are checked with the lowest layer.
gate_ruff src/finn/core/reporting tests/core/reporting src/finn/core/space tests/core/space \
    src/finn/core/search tests/core/search \
    tests/support/layering.py tests/support/refusals.py tests/support/value_classes.py \
    tests/support/space_cores.py tests/support/space_pipeline.py tests/support/space_walk.py \
    scripts/_gate_precommit.py \
    tests/support/space_oracle/__init__.py tests/support/space_oracle/forms.py \
    tests/support/space_oracle/public.py tests/support/space_oracle/compare.py \
    tests/support/space_oracle/toys.py
gate_mypy -p finn.core.reporting -p finn.core.space -p finn.core.search
gate_mypy tests/core/reporting tests/core/space tests/core/search tests/support/layering.py \
    tests/support/refusals.py tests/support/value_classes.py tests/support/space_cores.py \
    tests/support/space_pipeline.py tests/support/space_walk.py \
    tests/support/space_oracle/__init__.py tests/support/space_oracle/forms.py \
    tests/support/space_oracle/public.py tests/support/space_oracle/compare.py \
    tests/support/space_oracle/toys.py
