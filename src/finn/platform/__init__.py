# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The platform registry: what parts, boards and shells are, and the one resolution of
a build's target from them.

- ``finn.platform.parts``: a part's fabric, DSP block, UltraRAM and resource totals;
- ``finn.platform.boards``: a board's part and Vivado preset;
- ``finn.platform.shells``: a shell's row for a board: its ends, budgets, doubled
  clock, integration, host runtime and static region;
- ``finn.platform.resolve``: ``resolve_target``, a part or a board, a clock period
  and a shell to the target (``finn.kernels.target.Target``), and ``refuse_drift``,
  which refuses a build whose target is not its model's.

Kernels never import it: they read the capabilities it resolves
(``finn.kernels.target.Platform``), which the model states (the ``finn.platform``
graph metadata, ``finn.custom_op.kernels.base.read_target``). Every refusal is named
(``TargetRefused``).
"""

from finn.platform.boards import BOARDS, Board
from finn.platform.parts import FAMILIES, PARTS, PartFacts, part_facts
from finn.platform.refusal import TargetRefused
from finn.platform.resolve import refuse_drift, resolve_target
from finn.platform.shells import (
    IP,
    PYNQ,
    ROWS,
    SHELL_NAMES,
    SLASH,
    XRT,
    ShellRow,
    StaticRegion,
    shell_row,
)

__all__ = [
    "BOARDS",
    "FAMILIES",
    "IP",
    "PARTS",
    "PYNQ",
    "ROWS",
    "SHELL_NAMES",
    "SLASH",
    "XRT",
    "Board",
    "PartFacts",
    "ShellRow",
    "StaticRegion",
    "TargetRefused",
    "part_facts",
    "refuse_drift",
    "resolve_target",
    "shell_row",
]
