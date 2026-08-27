# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic on-chip parameter-delivery Kernel."""

from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL,
    CYCLIC_PARAMETER_KERNEL_SPEC,
    CyclicParameterBinding,
    CyclicParameterBindingSelection,
    CyclicParameterKernelPaths,
    CyclicRamStyle,
    CyclicTargetMemoryCapabilities,
    build_cyclic_parameter_kernel_spec,
)
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region

__all__ = [
    "CYCLIC_PARAMETER_KERNEL",
    "CYCLIC_PARAMETER_KERNEL_SPEC",
    "CyclicParameterBinding",
    "CyclicParameterBindingSelection",
    "CyclicParameterKernelPaths",
    "CyclicRamStyle",
    "CyclicTargetMemoryCapabilities",
    "build_cyclic_parameter_kernel_spec",
    "construct_cyclic_parameter_region",
]
