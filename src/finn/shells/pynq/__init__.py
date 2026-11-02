# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's code, which its declarations name (``finn.platform``'s
``platform.toml``) and the build calls through them (``finn.shells.build``): the Zynq block
design around the cores and their movers, and the driver its ``pynq`` runtime writes.

- ``linker``: the ``vivado-block-design`` linker (``BlockDesignLinker``): the block
  design's cells, pins, wires and addresses for an integration description, its project,
  Vivado run on it, and what it made.
- ``runtime``: the ``pynq`` runtime (``PynqRuntime``), which writes the shared driver
  (``finn.shells.runtime``), the device below and the runtime file beside them;
  ``device``: the PYNQ device the shared driver runs on (``PynqDevice``), shipped with it.
- ``templates``: the block design's project template.
"""
