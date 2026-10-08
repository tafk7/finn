# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shells' builds: what builds the kernel path's partition into a shell, for each
shell row of the platform registry that integrates the partition
(``finn.platform.shells``; the ``ip`` shell has none: its user integrates the IP).

- ``pynq``: the Zynq block design (``pynq.runner``), its ends' IODMAs (``pynq.iodma``,
  ``pynq.ipgen``) and its PYNQ driver (``pynq.driver``).

A shell's build reads the partition's integration export
(``finn.transformation.kernels.integration``) and the packaged partition
(``PackagePartition``); the builder (``finn.builder.kernel_build_steps``) runs it.
Nothing here imports the HWCustomOp flow (``tests/layering.py``).
"""
