# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel harness: what checks a kernel's hardware, above the kernels and their
transformations and below the flow.

The stream testbench and its pacing are the XSim executor's
(``finn.core.executors.xsim``); ``finn.harness.toolchain`` resolves what a
simulation runs on (FinnLib as FINN resolves it, Vivado through FINN's toolchain)
and the revisions a run prints; ``finn.harness.ops`` checks the hardware a model of
KernelOp nodes generates against their oracle, ``execute_node``, at the ops'
boundary (PRINCIPLES 8), as two runs of the model: in XSim and in Python;
``finn.harness.reference`` checks a KernelOp's pattern and
its reference against the ONNX it covers, every value equal, and which platform
rows' kernels admit it, a gap record where none does (the ONNX
entry); ``finn.harness.points`` draws a kernel's lean covering points and its
refused side from its design space (KT10); ``finn.harness.orders`` decodes output
words that differ as the order a stream was walked in, by beat and index tuple.
The tests that use the harness stay in ``tests/``; nothing here imports pytest or
the test tree.

The package re-exports nothing: each name is imported from the module that owns it.
"""
