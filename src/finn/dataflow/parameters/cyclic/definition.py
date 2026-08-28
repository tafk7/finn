# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Facts shared by every cyclic parameter-supply Kernel.

The selectable cyclic identities now live in
:mod:`finn.dataflow.parameters.supply_kernels` as ordinary Kernels.  What
remains here is the vocabulary those Kernels and the operation both read: the
RAM implementations one of them exposes, the target memory capability, and the
problem paths that carry build requirements.
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
