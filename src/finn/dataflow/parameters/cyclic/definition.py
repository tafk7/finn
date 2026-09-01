# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Facts shared by every cyclic parameter-supply Kernel.

This module owns the stable RAM-style vocabulary and target capability facts
used by cyclic parameter delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finn.dataflow.design import QualifiedPath


class CyclicRamStyle(str, Enum):
    """RAM implementations exposed by the FINN RTL memstream Kernel."""

    AUTO = "auto"
    BRAM = "block"
    LUTRAM = "distributed"
    URAM = "ultra"


@dataclass(frozen=True)
class CyclicTargetMemoryCapabilities:
    """Target facts used by the current on-chip supply constraints."""

    supports_initialized_uram: bool


class CyclicParameterKernelPaths:
    """Problem paths every cyclic parameter-supply Kernel reads."""

    RUNTIME_WRITABLE = QualifiedPath("problem.cyclic_parameter.runtime_writable")
    TARGET_MEMORY_CAPABILITIES = QualifiedPath(
        "problem.target.cyclic_parameter_memory_capabilities"
    )


__all__ = [
    "CyclicParameterKernelPaths",
    "CyclicRamStyle",
    "CyclicTargetMemoryCapabilities",
]
