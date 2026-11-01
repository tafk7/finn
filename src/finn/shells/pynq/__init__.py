# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build: the Zynq block design around the partition and its movers,
and the driver its ``pynq`` runtime writes.

- ``linker``: the ``vivado-block-design`` linker (``BlockDesignLinker``): the block
  design's cells, pins, wires and addresses for an integration description, its project,
  Vivado run on it, and what it made.
- ``runner``: each unit's IP packaged (a mover's, its kernel's HLS export:
  ``mover_ip``) and the description linked (``build_pynq``); the shell's options
  (``PynqOptions``); the bitfile's build as a transformation of the build's parent graph
  (``BuildPynqBitfile``).
- ``runtime``: the ``pynq`` runtime (``PynqRuntime``), which writes the shared driver
  (``finn.shells.runtime``), the device below and the runtime file beside them;
  ``device``: the PYNQ device the shared driver runs on (``PynqDevice``), shipped with it.
- ``driver``: the driver of the build's parent graph and its description, as a
  transformation (``MakePynqDriver``).
- ``deploy``: the deployment package, the bitfile, the driver and the host's model
  copied for the board (``MakePynqDeployment``).
- ``templates``: the block design's project template.
"""
