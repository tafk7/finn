# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic parameter-supply vocabulary and regions."""

from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterKernelPaths,
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region

__all__ = [
    "CyclicParameterKernelPaths",
    "CyclicRamStyle",
    "CyclicTargetMemoryCapabilities",
    "construct_cyclic_parameter_region",
]
