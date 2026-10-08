# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel harness: what checks a kernel's hardware, above the kernels and their
transformations and below the flow.

``finn.harness.rtl`` writes stream testbenches for a kernel module and runs them
in XSim; ``finn.harness.toolchain`` resolves what a simulation runs on (FinnLib as
FINN resolves it, Vivado through FINN's toolchain) and the revisions a run prints.
The tests that use the harness stay in ``tests/``; nothing here imports pytest or
the test tree.

The package re-exports nothing: each name is imported from the module that owns it.
"""
