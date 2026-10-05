#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for the canonical logical values in finn.dataflow and their tests.
#
# finn.dataflow sits directly on finn.core.space and below finn.kernels, so this
# gate checks only the value layer. Run scripts/check-kernels.sh as well for a
# change that can reach the kernels built on it. Parked code under finn.parked
# is reference only and outside every gate.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

gate_pytest tests/dataflow
gate_ruff src/finn/dataflow tests/dataflow
# QONNX, which ships no type information, is covered by an override in
# pyproject.toml. The canonical tests are strictly typed too.
gate_mypy -p finn.dataflow
gate_mypy tests/dataflow
