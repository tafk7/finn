#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Gate for finn.kernels, the KernelOps and their transformations, and their tests.

# shellcheck source=scripts/_gate-common.sh
source "$(dirname "$(readlink -f "$0")")/_gate-common.sh"

# The layers below, independent of each other: finn.core.space and finn.dataflow.
bash scripts/check-space.sh "${GATE_ARGS[@]}"
bash scripts/check-dataflow-design.sh "${GATE_ARGS[@]}"
# XSim and other Vivado tests (markers xsim, vivado) are deselected even when
# Vivado is selected (a FINN checkout's .envrc selects it): they take minutes
# each, and scripts/xsim-sweep.sh runs them. --fast deselects the slow tests too:
# the TFC platform binding and the wheel build.
gate_pytest tests/kernels xsim vivado
# The KernelOps and their transformations, the layer above finn.kernels.
gate_pytest tests/kernel_ops xsim vivado
# The XSim sweep's tooling: the emitted text and each job's key (no Vivado).
gate_pytest tests/xsim_sweep
# finn.util and the rest of FINN's support code (toolchain, resources,
# installation, containers, CI tooling, the builder's CPU-only flows). Its tests
# use tests/conftest.py (the seeded RNG, ci/ on sys.path). The slow tests (wheel
# and editable-environment builds, a build child; tens of seconds each) and the
# builds to IP or bitfile (end2end) are deselected in either mode, so this step
# stays at a few minutes.
gate_pytest --conftest-root tests tests/util xsim vivado end2end slow
gate_ruff src/finn/kernels tests/kernels src/finn/platform \
    src/finn/custom_op/kernels src/finn/custom_op/partition src/finn/transformation/kernels \
    src/finn/harness src/finn/shells/*.py src/finn/shells/pynq/*.py tests/kernel_ops \
    src/finn/builder/kernel_testbench.py src/finn/builder/kernel_resources.py \
    scripts/benchmark-space.py scripts/emitted_text.py tests/xsim_sweep tests/oracle
# finn.util.toolchain: the toolchain packaging takes (PackagePartition's toolchain=).
# finn.shells.pynq's runner and driver; its IODMA and IP generation are frozen
# extractions of the HWCustomOp flow's, untyped as they were.
# finn.builder.kernel_testbench and kernel_resources: the ip shell's testbench and the
# resources per shell member, the builder modules typed.
gate_mypy -p finn.kernels -p finn.platform -p finn.custom_op.kernels -p finn.custom_op.partition \
    -p finn.transformation.kernels -p finn.harness -m finn.shells.pynq.runner \
    -m finn.shells.pynq.driver -m finn.util.toolchain \
    -m finn.builder.kernel_testbench -m finn.builder.kernel_resources
# Whole directories; the files not yet strictly typed are listed in .mypy.ini.
gate_mypy tests/kernels tests/kernel_ops tests/xsim_sweep tests/oracle
