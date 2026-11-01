# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The shells' builds: what builds the partition into a shell, for each
shell row of the platform registry that integrates the partition
(``finn.platform.shells``; the ``ip`` shell has none: its user integrates the IP).

- ``integration``: the integration description every linker reads (``describe``), with
  no platform's names in it, and what a linker takes and gives back (``Linker``,
  ``Linked``).
- ``pynq``: the Zynq block design (``pynq.linker``, ``pynq.runner``), its movers' IODMAs
  (``pynq.iodma``, ``pynq.ipgen``), its PYNQ driver (``pynq.driver``) and its deployment
  package (``pynq.deploy``).

A shell's build reads the partition's integration description and the packaged partition
(``PackagePartition``), and links them through its linker; the builder
(``finn.builder.build_dataflow_steps``) runs it.
"""
