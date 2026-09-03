# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cyclic parameter-supply vocabulary and regions."""

from finn.dataflow.parameters.cyclic.computation import CYCLIC_PARAMETER_DELIVERY
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region

__all__ = [
    "CYCLIC_PARAMETER_DELIVERY",
    "construct_cyclic_parameter_region",
]
