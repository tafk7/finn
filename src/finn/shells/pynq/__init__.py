# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``pynq`` shell's build: the Zynq block design around the partition, its ends'
IODMAs, and the PYNQ driver its host runtime runs.

- ``runner``: the block design from the integration export, Vivado run on it, and what
  it made (``build_pynq``); the shell's options (``PynqOptions``).
- ``iodma`` and ``ipgen``: each end's IODMA, an HLS IP packaged as its instance, a frozen
  extraction of the HWCustomOp flow's ``IODMA_hls`` and its IP generation.
- ``driver``: the PYNQ driver and its description.
- ``templates`` and ``data/``: the project, HLS and driver templates, the driver's base
  files, the IODMA IP's driver descriptor and simulation-control module.
"""
