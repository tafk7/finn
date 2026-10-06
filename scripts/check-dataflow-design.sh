#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for the canonical logical values in finn.dataflow and their tests.
#
# finn.dataflow sits below finn.kernels and imports no engine, so this gate
# checks only the value layer; scripts/check-kernels.sh runs it, and checks the
# kernels built on it too. Parked code under finn.parked
# is reference only and outside every gate.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

gate_pytest tests/dataflow
gate_ruff src/finn/dataflow tests/dataflow
# QONNX, which ships no type information, is covered by an override in
# .mypy.ini. The canonical tests are strictly typed too.
gate_mypy -p finn.dataflow
gate_mypy tests/dataflow
