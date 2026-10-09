# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform registry: what parts, boards and shells are, and the one resolution of
a build's target from them.

- ``finn.platform.catalog``: every part of the supported series, generated from
  Vivado's part database (``finn.platform.generate``): its device, the device's
  resources per SLR and totals (on a reduced die of several SLRs, each SLR's site
  capacity under the totals as a cap), and the devices that share them, with an
  overlay for parts FINN does not ship (``FINN_PLATFORM_CATALOG``);
- ``finn.platform.architectures``: what FINN builds for each of Vivado's
  architectures (fabric, DSP block, UltraRAM initialisation), each rule checked by
  the generator's site probe, and which caps of each capped device a fill proved;
- ``finn.platform.boards``: a board's part and Vivado preset;
- ``finn.platform.shells``: a shell's row for a board: its ends, budgets, doubled
  clock, integration, host runtime and static region;
- ``finn.platform.resolve``: ``resolve_target``, a part or a board, a clock period
  and a shell to the target (``finn.kernels.target.Target``), and ``refuse_drift``,
  which refuses a build whose target is not its model's;
- ``finn.platform.request``: ``TargetRequest``, what a build configuration states of
  its target, resolved by ``resolve_target``.

It sits above the kernels and below the KernelOps. Kernels never import it: they
read the capabilities it resolves (``finn.kernels.target.Platform``), which the model
states (the ``finn.platform`` graph metadata,
``finn.custom_op.kernels.base.read_target``). The shell root reads its target's
shell row (``finn.custom_op.kernels.shell``): the ends it offers and the budgets it
admits. Every refusal is named (``TargetRefused``).
"""

from finn.platform.boards import BOARDS, Board
from finn.platform.catalog import device, part, part_report, parts
from finn.platform.refusal import TargetRefused
from finn.platform.request import TargetRequest
from finn.platform.resolve import refuse_drift, resolve_target
from finn.platform.shells import (
    IP,
    IP_ROW,
    PYNQ,
    ROWS,
    VIVADO_BLOCK_DESIGN,
    ShellRow,
    StaticRegion,
    shell_row,
)

__all__ = [
    "BOARDS",
    "IP",
    "IP_ROW",
    "PYNQ",
    "ROWS",
    "VIVADO_BLOCK_DESIGN",
    "Board",
    "ShellRow",
    "StaticRegion",
    "TargetRefused",
    "TargetRequest",
    "device",
    "part",
    "part_report",
    "parts",
    "refuse_drift",
    "resolve_target",
    "shell_row",
]
