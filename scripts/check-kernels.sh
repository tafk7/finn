#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for finn.kernels, the KernelOps and their transformations, and their tests.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

# The layers below, in order: finn.core.space, then finn.dataflow. Parked
# dataflow/graph tests remain outside this command; their compatibility is not
# claimed.
bash scripts/check-space.sh
bash scripts/check-dataflow-design.sh
# XSim and other Vivado tests (markers xsim, vivado) are deselected even when
# Vivado is selected (a FINN checkout's .envrc selects it): they take minutes
# each, and scripts/xsim-sweep.sh runs them.
gate_pytest tests/kernels -m "not xsim and not vivado"
# The KernelOps and their transformations, the layer above finn.kernels.
gate_pytest tests/kernel_ops -m "not xsim and not vivado"
gate_ruff src/finn/kernels tests/kernels \
    src/finn/custom_op/kernels src/finn/transformation/kernels tests/kernel_ops \
    scripts/benchmark-space.py
gate_mypy -p finn.kernels -p finn.custom_op.kernels -p finn.transformation.kernels
# Whole directories; the files not yet strictly typed are listed in pyproject.toml.
gate_mypy tests/kernels tests/kernel_ops
