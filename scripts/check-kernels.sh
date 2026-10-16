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
# each, and scripts/xsim-sweep.sh runs them. --fast deselects the slow tests too
# (marker slow: whole-network and builder-phase tests, the wheel build, the entry
# modules' interpreters, the XSim sweep tooling's text comparisons). An HLS top's C simulation (marker
# vitis) is a check to run by hand, which no gate decides on.
gate_pytest tests/kernels xsim vivado vitis
# The KernelOps and their transformations, the layer above finn.kernels.
gate_pytest tests/kernel_ops xsim vivado
# The builder's, the platform's and the analysis' tests, split from the KernelOps' by
# the packages they test (each its own conftest).
gate_pytest tests/builder xsim vivado
gate_pytest tests/platforms xsim vivado
gate_pytest tests/analysis xsim vivado
# The XSim sweep's tooling: the emitted text and each job's key (no Vivado).
gate_pytest tests/xsim_sweep
# Graph preparation's front end (the preparation layer: finn.transformation.qonnx
# and .streamline, which the builder's graph-preparation steps run; tests/builder
# tests the steps themselves) and the Brevitas export it reads, under
# tests/conftest.py. One thread per worker: each of the workers would otherwise run
# torch over every CPU. About 1 140 tests (collected without xsim and vivado).
OMP_NUM_THREADS=1 gate_pytest --conftest-root tests \
    tests/transformation tests/brevitas xsim vivado
# finn.util and the rest of FINN's support code (toolchain, resources,
# installation, containers, the build entry). Its tests use tests/conftest.py (the
# seeded RNG). The slow tests (wheel and editable-environment builds, a build child;
# tens of seconds each) are deselected in either mode, so this step stays at a few
# minutes.
gate_pytest --conftest-root tests tests/util xsim vivado slow
gate_ruff src/finn/kernels tests/kernels tests/support src/finn/platform src/finn/core/graph \
    src/finn/core/executor.py src/finn/core/onnx_exec.py \
    src/finn/core/rtlsim_bench.py src/finn/core/rtlsim_exec.py src/finn/core/rtlsim_pacing.py \
    src/finn/custom_op/kernels src/finn/custom_op/partition src/finn/transformation/kernels \
    src/finn/analysis src/finn/util/draws.py src/finn/util/typed_json.py \
    src/finn/harness src/finn/shells/*.py src/finn/shells/pynq/*.py tests/kernel_ops \
    tests/builder tests/platforms tests/analysis \
    src/finn/builder/build_dataflow_testbench.py src/finn/builder/build_dataflow_resources.py \
    src/finn/builder/build_dataflow_checks.py \
    src/finn/builder/build_dataflow_config.py src/finn/builder/build_dataflow_plan.py \
    src/finn/builder/build_dataflow_steps.py src/finn/builder/build_dataflow_verification.py \
    scripts/benchmark-space.py scripts/emitted_text.py tests/xsim_sweep tests/oracle
# finn.toolchain.launcher: the toolchain packaging takes (PackagePartition's toolchain=);
# finn.toolchain.ip: the packaged-IP store PackagePartition packages through;
# finn.toolchain.xsim: XSim's driver, which the testbench and packaging's elaboration run.
# finn.shells.pynq's runner, driver and deployment package; its IODMA and IP generation
# are annotated, not strictly checked: frozen extractions of the HWCustomOp flow's.
# finn.builder: its configuration, its plan, its checks
# and steps, the ip shell's testbench, the partition's Python verification and the
# resources per shell member.
gate_mypy -p finn.kernels -p finn.platform -p finn.core.graph -p finn.custom_op.kernels \
    -p finn.custom_op.partition \
    -p finn.transformation.kernels -p finn.analysis -p finn.harness \
    -m finn.util.draws -m finn.util.typed_json \
    -m finn.core.executor -m finn.core.rtlsim_bench -m finn.core.rtlsim_exec \
    -m finn.core.rtlsim_pacing -m finn.core.onnx_exec \
    -m finn.shells.pynq.runner -m finn.shells.pynq.driver -m finn.shells.pynq.deploy \
    -m finn.toolchain.launcher -m finn.toolchain.ip \
    -m finn.toolchain.xsim \
    -m finn.builder.build_dataflow_config -m finn.builder.build_dataflow_plan \
    -m finn.builder.build_dataflow_steps -m finn.builder.build_dataflow_testbench \
    -m finn.builder.build_dataflow_resources -m finn.builder.build_dataflow_checks \
    -m finn.builder.build_dataflow_verification \
    -m finn.builder.build_dataflow_manifest
# Whole directories; the files not yet strictly typed are listed in .mypy.ini.
gate_mypy tests/support tests/kernels tests/kernel_ops tests/builder tests/platforms \
    tests/analysis tests/xsim_sweep tests/oracle
