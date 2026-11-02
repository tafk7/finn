# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shells' builds: what builds the partition into a shell, for each
shell row of the platform registry that integrates the partition
(``finn.platform.shells``; the ``ip`` shell has none: its user integrates the IP).

- ``integration``: the integration description every linker reads (``describe``), with
  no platform's names in it, and what a linker and a runtime take and give back
  (``Linker``, ``Linked``, ``Runtime``).
- ``build``: the bitfile, the driver and the deployment package of a build, each through
  the code its shell's declarations name (``BuildBitfile``, ``MakeDriver``,
  ``MakeDeployment``).
- ``runtime``: the shared driver every runtime ships.
- ``pynq``: the Zynq block design (``pynq.linker``) and the PYNQ runtime
  (``pynq.runtime``, ``pynq.device``).

A shell's build reads the partition's integration description and the packaged partition
(``PackagePartition``), and links them through its declared linker; the builder
(``finn.builder.build_dataflow_steps``) runs it.
"""
