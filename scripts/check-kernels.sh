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
# the TFC platform binding and the wheel build. An HLS top's C simulation (marker
# vitis) is a check to run by hand, which no gate decides on.
gate_pytest tests/kernels xsim vivado vitis
# The KernelOps and their transformations, the layer above finn.kernels.
gate_pytest tests/kernel_ops xsim vivado
# The XSim sweep's tooling: the emitted text and each job's key (no Vivado).
gate_pytest tests/xsim_sweep
# Graph preparation's front end (the preparation layer: finn.transformation.qonnx
# and .streamline, which the kernel path's graph-preparation phase runs; tests/
# kernel_ops tests the phase itself) and the Brevitas export it reads, under
# tests/conftest.py. Not the legacy ops' preparation, which their ports adapt
# (KT21): MLO's loop rolling, PWPolyF's export. One thread per worker: each of the
# workers would otherwise run torch over every CPU. About 1 700 tests, 7 minutes
# alone.
OMP_NUM_THREADS=1 gate_pytest --conftest-root tests \
    --ignore tests/transformation/test_loop_rolling.py \
    --ignore tests/brevitas/test_brevitas_pwpolyf.py \
    tests/transformation tests/brevitas xsim vivado
# finn.util and the rest of FINN's support code (toolchain, resources,
# installation, containers, CI tooling, the builder's CPU-only flows). Its tests
# use tests/conftest.py (the seeded RNG, ci/ on sys.path). The slow tests (wheel
# and editable-environment builds, a build child; tens of seconds each) and the
# builds to IP or bitfile (end2end) are deselected in either mode, so this step
# stays at a few minutes.
gate_pytest --conftest-root tests tests/util xsim vivado end2end slow
gate_ruff src/finn/kernels tests/kernels src/finn/platform src/finn/core/executors \
    src/finn/core/containers.py src/finn/core/onnx_exec.py \
    src/finn/custom_op/kernels src/finn/custom_op/partition src/finn/transformation/kernels \
    src/finn/transformation/prepare \
    src/finn/harness src/finn/shells/*.py src/finn/shells/pynq/*.py tests/kernel_ops \
    src/finn/builder/kernel_testbench.py src/finn/builder/kernel_resources.py \
    src/finn/builder/kernel_build_checks.py \
    src/finn/builder/kernel_build_config.py src/finn/builder/kernel_build_runner.py \
    src/finn/builder/kernel_build_steps.py \
    scripts/benchmark-space.py scripts/emitted_text.py tests/xsim_sweep tests/oracle
# finn.util.toolchain: the toolchain packaging takes (PackagePartition's toolchain=).
# finn.shells.pynq's runner and driver; its IODMA and IP generation are annotated, not
# strictly checked: frozen extractions of the HWCustomOp flow's.
# finn.builder's kernel path: its configuration, its checks, step runner and steps, the ip
# shell's testbench and the resources per shell member.
gate_mypy -p finn.kernels -p finn.platform -p finn.custom_op.kernels -p finn.custom_op.partition \
    -p finn.transformation.kernels -p finn.transformation.prepare -p finn.harness \
    -p finn.core.executors -m finn.core.containers -m finn.core.onnx_exec \
    -m finn.shells.pynq.runner -m finn.shells.pynq.driver -m finn.util.toolchain \
    -m finn.builder.kernel_build_config -m finn.builder.kernel_build_runner \
    -m finn.builder.kernel_build_steps -m finn.builder.kernel_testbench \
    -m finn.builder.kernel_resources -m finn.builder.kernel_build_checks
# Whole directories; the files not yet strictly typed are listed in .mypy.ini.
gate_mypy tests/kernels tests/kernel_ops tests/xsim_sweep tests/oracle
