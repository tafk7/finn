# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Parts, boards and shells, and the one function that resolves a build's target from
them (``resolve_target``).

Kernels never import this package. They read the capabilities it resolves
(``finn.kernels.target.Platform``) from the model. Every refusal is a
``TargetRefused`` carrying its finding, by code.
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
