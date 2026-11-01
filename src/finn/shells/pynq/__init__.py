# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build: the Zynq block design around the partition and its movers,
and the PYNQ driver its host runtime runs.

- ``linker``: the ``vivado-block-design`` linker (``BlockDesignLinker``): the block
  design's cells, pins, wires and addresses for an integration description, its project,
  Vivado run on it, and what it made.
- ``runner``: each unit's IP packaged (a mover's, its kernel's HLS export:
  ``mover_ip``) and the description linked (``build_pynq``); the shell's options
  (``PynqOptions``); the bitfile's build as a transformation of the build's parent graph
  (``BuildPynqBitfile``).
- ``driver``: the PYNQ driver and its description, and their transformation
  (``MakePynqDriver``).
- ``deploy``: the deployment package, the bitfile, the driver and the host's model
  copied for the board (``MakePynqDeployment``).
- ``templates`` and ``data/``: the project and driver templates and the driver's base
  files.
"""
