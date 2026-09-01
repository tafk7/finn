# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Computation contract for cyclic parameter delivery."""

from finn.dataflow.computation import ComputationContract

CYCLIC_PARAMETER_DELIVERY = ComputationContract("parameter.cyclic_delivery")

__all__ = ["CYCLIC_PARAMETER_DELIVERY"]
