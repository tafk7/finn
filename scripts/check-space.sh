#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for finn.core.space and its tests.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

gate_pytest tests/core/space
# Documentation examples are checked separately in scratchpad/space/.
gate_ruff src/finn/core/space tests/core/space
gate_mypy -p finn.core.space
gate_mypy tests/core/space
